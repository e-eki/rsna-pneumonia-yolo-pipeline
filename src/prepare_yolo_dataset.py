"""
Подготовка YOLO-датасета: фильтрация разметки и сборка data/processed/.

Скрипт берёт уже размеченный датасет в YOLO-формате и:
  1. Проверяет, что каждому изображению соответствует .txt с разметкой.
  2. Отсеивает «плохие» боксы: невалидные координаты, слишком маленькие,
     слишком вытянутые или выходящие за границы кадра.
  3. Изображения с критичными проблемами в разметке убирает в
     data/samples/rejected/, чтобы модель не училась на испорченных данных.
  4. Сохраняет чистый датасет в data/processed/ (формат изображений
     задаётся в preprocessing.output_format: original / jpg / png).
  5. Пишет data.yaml — конфиг для Ultralytics.

Ожидаемая структура источника:
    data/raw/<raw_subdir>/images/{train,val}/*.png
    data/raw/<raw_subdir>/labels/{train,val}/*.txt

Куда что складывается:
    data/processed/{images,labels}/{train,val}/   — чистый датасет
    data/processed/data.yaml                      — конфиг для YOLO
    data/samples/rejected/rejected_images/        — отвергнутые изображения
    data/samples/rejected/rejected_annotations.csv — отвергнутые боксы
    data/staged/filtering_report.json             — сводка фильтрации

Пример запуска:
    python -m src.prepare_yolo_dataset --config configs/config.yaml
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from src.utils import ensure_dirs, get_logger, load_config, setup_logging

logger = get_logger(__name__)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# Причины отклонения, при которых изображение считается «испорченным»
# и убирается из обучения целиком (см. drop_images_with_critical_rejected_boxes).
DEFAULT_CRITICAL_REASONS = {
    "extends_out_of_bounds",
    "aspect_too_high",
    "aspect_too_low",
    "center_out_of_bounds",
    "size_over_1",
}


# ─────────────────────────── работа с изображениями ───────────────────────────

def find_images(images_dir: Path):
    """Возвращает отсортированный список изображений в директории."""
    return sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in IMG_EXTS)


def save_image(src: Path, dst_dir: Path, output_format: str,
               jpeg_quality: int = 92) -> Path:
    """
    Сохраняет изображение в dst_dir с нужным форматом.

    output_format:
      "original" — копировать как есть, сохраняя расширение источника;
      "jpg"      — конвертировать в JPEG (быстрее I/O, меньше размер);
      "png"      — конвертировать в PNG (без потерь, но тяжелее).

    Возвращает путь к сохранённому файлу.
    """
    fmt = (output_format or "original").lower()

    if fmt == "original":
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return dst

    if fmt == "jpg":
        dst = dst_dir / f"{src.stem}.jpg"
        with Image.open(src) as im:
            im.convert("RGB").save(dst, "JPEG",
                                   quality=jpeg_quality, optimize=True)
        return dst

    if fmt == "png":
        dst = dst_dir / f"{src.stem}.png"
        with Image.open(src) as im:
            im.save(dst, "PNG", optimize=True)
        return dst

    raise ValueError(f"Неизвестный output_format: {output_format!r} "
                     f"(ожидается 'original' | 'jpg' | 'png')")


# ─────────────────────────── работа с разметкой ───────────────────────────

def read_yolo_label(label_path: Path):
    """
    Читает YOLO-разметку из .txt.

    Возвращает список словарей. Валидные боксы имеют вид
    {"cls", "xc", "yc", "w", "h", "raw"}, невалидные строки —
    {"error": "<причина>", "raw": "<строка>"}.
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                boxes.append({"error": "bad_line_format", "raw": line})
                continue
            try:
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                boxes.append({"cls": cls, "xc": xc, "yc": yc,
                              "w": w, "h": h, "raw": line})
            except ValueError:
                boxes.append({"error": "bad_number_format", "raw": line})

    return boxes


def clip_yolo_box(b):
    """
    Подрезает YOLO-бокс к [0, 1] так, чтобы он остался внутри кадра.

    Возвращает (xc, yc, w, h) либо None, если после подрезки бокс
    полностью «схлопнулся» (не осталось площади).
    """
    xc, yc, w, h = b["xc"], b["yc"], b["w"], b["h"]

    x1 = max(0.0, xc - w / 2)
    y1 = max(0.0, yc - h / 2)
    x2 = min(1.0, xc + w / 2)
    y2 = min(1.0, yc + h / 2)

    if x2 <= x1 or y2 <= y1:
        return None

    new_w = x2 - x1
    new_h = y2 - y1
    return x1 + new_w / 2, y1 + new_h / 2, new_w, new_h


def box_reject_reasons(b, min_area, min_aspect, max_aspect, oob_tolerance=0.0):
    """
    Возвращает список причин отклонения бокса (пустой список — бокс хороший).

    oob_tolerance управляет поведением для боксов на границе кадра:
      0.0  — строгая проверка, боксы у края отклоняются (extends_out_of_bounds);
      > 0  — бокс в пределах допуска подрезается к [0, 1] вместо отклонения.

    Побочный эффект: при oob_tolerance > 0 координаты бокса `b` могут
    быть подрезаны (мутирует dict).
    """
    if "error" in b:
        return [b["error"]]

    xc, yc, w, h = b["xc"], b["yc"], b["w"], b["h"]

    if w <= 0 or h <= 0:
        return ["non_positive_size"]

    reasons = []

    if xc < 0 or xc > 1 or yc < 0 or yc > 1:
        reasons.append("center_out_of_bounds")

    left, right = xc - w / 2, xc + w / 2
    top, bottom = yc - h / 2, yc + h / 2
    eps = oob_tolerance

    if left < -eps or right > 1 + eps or top < -eps or bottom > 1 + eps:
        reasons.append("extends_out_of_bounds")
    elif eps > 0 and (left < 0 or right > 1 or top < 0 or bottom > 1):
        clipped = clip_yolo_box(b)
        if clipped is None:
            reasons.append("clipped_to_zero")
        else:
            # Обновляем и dict, и локальные переменные — дальше
            # все проверки используют уже подрезанный бокс.
            b["xc"], b["yc"], b["w"], b["h"] = clipped
            xc, yc, w, h = clipped

    if w > 1 or h > 1:
        reasons.append("size_over_1")

    area = w * h
    if area < min_area:
        reasons.append(f"area_too_small:{area:.6f}")

    if h > 0:
        aspect = w / h
        if aspect > max_aspect:
            reasons.append(f"aspect_too_high:{aspect:.2f}")
        elif aspect < min_aspect:
            reasons.append(f"aspect_too_low:{aspect:.2f}")

    return reasons


def write_yolo_label(label_path: Path, boxes):
    """Сохраняет очищенную YOLO-разметку в файл."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for b in boxes:
            f.write(f"{b['cls']} {b['xc']:.6f} {b['yc']:.6f} "
                    f"{b['w']:.6f} {b['h']:.6f}\n")


# ─────────────────────────── обработка одного split ───────────────────────────

def prepare_split(raw_root: Path, processed_root: Path,
                  samples_rejected_dir: Path, config: dict, split_name: str):
    """
    Обрабатывает один split (train или val).

    Возвращает (stats, rejected_rows):
      stats         — словарь со счётчиками для отчёта;
      rejected_rows — список записей для rejected_annotations.csv.
    """
    fconf = config["preprocessing"]["filtering"]
    pconf = config["preprocessing"]

    min_area = fconf["min_box_area_frac"]
    min_aspect = fconf["min_aspect_ratio"]
    max_aspect = fconf["max_aspect_ratio"]
    oob_tolerance = fconf.get("out_of_bounds_tolerance", 0.0)
    drop_all_rejected = fconf.get("drop_images_with_all_boxes_rejected", True)
    drop_critical = fconf.get("drop_images_with_critical_rejected_boxes", False)
    critical_reasons = set(fconf.get("critical_reject_reasons",
                                     DEFAULT_CRITICAL_REASONS))

    output_format = pconf.get("output_format", "original")
    jpeg_quality = pconf.get("jpeg_quality", 92)

    images_dir = raw_root / "images" / split_name
    labels_dir = raw_root / "labels" / split_name

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Нет {images_dir} или {labels_dir}")

    images = find_images(images_dir)
    logger.info(f"[{split_name}] Изображений: {len(images)} "
                f"(output_format={output_format})")

    out_images = processed_root / "images" / split_name
    out_labels = processed_root / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rejected_images_dir = samples_rejected_dir / "rejected_images"
    rejected_images_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "split": split_name,
        "images_total": len(images),
        "images_positive": 0,
        "images_negative": 0,
        "images_all_boxes_rejected": 0,
        "images_critical_rejected": 0,
        "images_dropped_to_rejected": 0,
        "images_missing_label": 0,
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_rejected": 0,
        "boxes_clipped": 0,
        "rejected_by_reason": {},
    }
    rejected_rows = []

    for img_path in images:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"

        if label_path.exists():
            boxes = read_yolo_label(label_path)
        else:
            logger.warning(f"[{split_name}] Нет label для {img_path.name}")
            stats["images_missing_label"] += 1
            boxes = []

        stats["boxes_total"] += len(boxes)

        try:
            with Image.open(img_path) as im:
                img_W, img_H = im.size
        except Exception as e:
            logger.warning(f"Не удалось прочитать {img_path}: {e}")
            continue

        kept_boxes = []
        any_critical_rejected = False

        for b in boxes:
            orig = {k: b.get(k) for k in ("xc", "yc", "w", "h")}
            reasons = box_reject_reasons(b, min_area, min_aspect, max_aspect,
                                         oob_tolerance)
            was_clipped = (
                "error" not in b
                and any(abs(b.get(k, 0) - orig[k]) > 1e-9
                        for k in ("xc", "yc", "w", "h"))
            )

            if not reasons:
                if was_clipped:
                    stats["boxes_clipped"] += 1
                kept_boxes.append(b)
                stats["boxes_kept"] += 1
                continue

            # Бокс отвергнут — учитываем причину и пишем строку для аудита.
            stats["boxes_rejected"] += 1
            reason_keys = {r.split(":")[0] for r in reasons}
            for key in reason_keys:
                stats["rejected_by_reason"][key] = \
                    stats["rejected_by_reason"].get(key, 0) + 1
            if reason_keys & critical_reasons:
                any_critical_rejected = True

            if "error" not in b:
                x_px = (orig["xc"] - orig["w"] / 2) * img_W
                y_px = (orig["yc"] - orig["h"] / 2) * img_H
                w_px = orig["w"] * img_W
                h_px = orig["h"] * img_H
            else:
                x_px = y_px = w_px = h_px = 0.0

            rejected_rows.append({
                "patientId": stem,
                "split": split_name,
                "x": x_px, "y": y_px,
                "width": w_px, "height": h_px,
                "x_norm": orig.get("xc"),
                "y_norm": orig.get("yc"),
                "w_norm": orig.get("w"),
                "h_norm": orig.get("h"),
                "reject_reason": "; ".join(reasons),
                "critical": bool(reason_keys & critical_reasons),
            })

        # Решаем судьбу изображения: оставить в processed или убрать в rejected.
        all_rejected = len(boxes) > 0 and len(kept_boxes) == 0
        should_drop = (
            (drop_all_rejected and all_rejected)
            or (drop_critical and any_critical_rejected)
        )

        if should_drop:
            stats["images_dropped_to_rejected"] += 1
            if all_rejected:
                stats["images_all_boxes_rejected"] += 1
            if any_critical_rejected:
                stats["images_critical_rejected"] += 1

            # Отвергнутые храним в оригинальном формате — так их удобнее
            # смотреть глазами при аудите.
            shutil.copy2(img_path, rejected_images_dir / img_path.name)
            logger.warning(
                f"[{split_name}] {stem}: "
                f"all_rejected={all_rejected}, critical={any_critical_rejected} "
                f"→ rejected_images/"
            )
            continue

        try:
            save_image(img_path, out_images, output_format, jpeg_quality)
        except Exception as e:
            logger.warning(f"Не удалось сохранить {img_path.name}: {e}")
            continue

        write_yolo_label(out_labels / f"{stem}.txt", kept_boxes)

        if len(boxes) == 0:
            stats["images_negative"] += 1
        else:
            stats["images_positive"] += 1

    logger.info(f"[{split_name}] "
                f"positive={stats['images_positive']}, "
                f"negative={stats['images_negative']}, "
                f"all_rejected={stats['images_all_boxes_rejected']}, "
                f"critical_rejected={stats['images_critical_rejected']}, "
                f"dropped_to_rejected={stats['images_dropped_to_rejected']}")
    logger.info(f"[{split_name}] "
                f"boxes: total={stats['boxes_total']}, "
                f"kept={stats['boxes_kept']}, "
                f"rejected={stats['boxes_rejected']}, "
                f"clipped={stats['boxes_clipped']}")

    return stats, rejected_rows


# ─────────────────────────── data.yaml ───────────────────────────

def make_data_yaml(processed_dir: Path):
    """Пишет data.yaml — конфиг датасета для Ultralytics."""
    data_yaml = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["pneumonia"],
    }
    out = processed_dir / "data.yaml"
    with open(out, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    logger.info(f"data.yaml создан: {out}")
    return data_yaml


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--raw-subdir", default=None,
                        help="Подпапка в data/raw с YOLO-датасетом. "
                             "По умолчанию берётся из config.dataset.raw_yolo_subdir.")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    raw_subdir = args.raw_subdir or config["dataset"].get("raw_yolo_subdir", "rsna_yolo")

    raw_root = Path(config["paths"]["raw_dir"]) / raw_subdir
    processed_root = Path(config["paths"]["processed_dir"])
    staged_root = Path(config["paths"]["staged_dir"])
    samples_root = Path(config["paths"].get("samples_dir", "data/samples"))
    samples_rejected_dir = samples_root / "rejected"
    samples_rejected_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(f"Нет папки {raw_root}")

    pconf = config["preprocessing"]
    output_format = pconf.get("output_format", "original")
    jpeg_quality = pconf.get("jpeg_quality", 92)

    logger.info(f"Источник:             {raw_root}")
    logger.info(f"Назначение processed: {processed_root}")
    logger.info(f"Назначение rejected:  {samples_rejected_dir}")
    logger.info(f"Формат вывода:        {output_format} "
                f"(jpeg_quality={jpeg_quality})")

    all_stats = []
    all_rejected = []

    for split_name in ("train", "val"):
        if not (raw_root / "images" / split_name).exists():
            logger.warning(f"Пропуск: {raw_root}/images/{split_name} не существует")
            continue
        stats, rejected = prepare_split(raw_root, processed_root,
                                        samples_rejected_dir, config,
                                        split_name)
        all_stats.append(stats)
        all_rejected.extend(rejected)

    make_data_yaml(processed_root)

    # Отчёт по отвергнутым боксам — только если они есть.
    if all_rejected:
        rej_path = samples_rejected_dir / "rejected_annotations.csv"
        pd.DataFrame(all_rejected).to_csv(rej_path, index=False)
        logger.info(f"Отвергнутые боксы: {rej_path} ({len(all_rejected)} строк)")
    else:
        logger.info("Отвергнутых боксов нет — rejected_annotations.csv не создаётся")

    # Сводный отчёт для аудита пайплайна.
    report = {
        "source": str(raw_root),
        "destination_processed": str(processed_root),
        "destination_rejected": str(samples_rejected_dir),
        "output_format": output_format,
        "jpeg_quality": jpeg_quality,
        "splits": all_stats,
    }
    report_path = staged_root / "filtering_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Отчёт фильтрации: {report_path}")

    logger.info("=" * 60)
    logger.info("ИТОГИ ПОДГОТОВКИ ДАТАСЕТА")
    logger.info("=" * 60)
    for s in all_stats:
        logger.info(f"[{s['split']}]")
        logger.info(f"  изображений всего:       {s['images_total']}")
        logger.info(f"  positive:                {s['images_positive']}")
        logger.info(f"  negative:                {s['images_negative']}")
        logger.info(f"  all_boxes_rejected:      {s['images_all_boxes_rejected']}")
        logger.info(f"  critical_rejected:       {s['images_critical_rejected']}")
        logger.info(f"  dropped_to_rejected:     {s['images_dropped_to_rejected']}")
        logger.info(f"  missing_label:           {s['images_missing_label']}")
        logger.info(f"  боксов всего:            {s['boxes_total']}")
        logger.info(f"  боксов сохранено:        {s['boxes_kept']}")
        logger.info(f"  боксов отвергнуто:       {s['boxes_rejected']}")
        logger.info(f"  боксов подрезано:        {s['boxes_clipped']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()