# src/prepare_yolo_dataset.py
"""
Подготовка датасета, УЖЕ размеченного в формате YOLO.

Вход:  data/raw/rsna_yolo/{images,labels}/{train,val}/*.{png,jpg,txt}
Выход: data/processed/{images,labels}/{train,val}/  + data.yaml
       data/staged/rejected_annotations.csv         (для collect_rejected.py)
       data/staged/rejected_images/                 (перемещённые картинки)
       data/staged/filtering_report.json

Что делает:
  1. Проверяет пары image <-> label и корректность имён.
  2. Читает YOLO-разметку (нормализованные xc, yc, w, h).
  3. Фильтрует «плохие» боксы:
       - w <= 0 или h <= 0
       - координаты вне [0, 1]
       - box выходит за границы изображения (xc ± w/2 не в [0,1])
       - площадь < min_box_area_frac
       - aspect ratio вне [min, max]
  4. Изображения, где все боксы отвергнуты, перемещает в staged/rejected_images/
     (остаются в processed как negative, если drop=false).
  5. Пишет data.yaml.

Пример:
  python -m src.prepare_yolo_dataset --config configs/config.yaml
"""
import argparse
import json
import shutil
from pathlib import Path

import yaml
from PIL import Image

from src.utils import load_config, ensure_dirs, setup_logging, get_logger

logger = get_logger(__name__)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
CLASS_NAMES = {0: "pneumonia"}


# ------------------------- утилиты -------------------------

def find_images(images_dir: Path):
    return sorted([p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS])


def read_yolo_label(label_path: Path):
    """Читает YOLO-разметку. Возвращает список dict {cls, xc, yc, w, h, raw}."""
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
                boxes.append({"cls": cls, "xc": xc, "yc": yc, "w": w, "h": h,
                              "raw": line})
            except ValueError:
                boxes.append({"error": "bad_number_format", "raw": line})
    return boxes


def box_reject_reasons(b, min_area, min_aspect, max_aspect,
                       img_W, img_H, oob_tolerance=0.0):
    """
    Возвращает список причин отклонения бокса (пустой — значит бокс хороший).

    Параметр `oob_tolerance`:
      - 0.0   — строгая проверка границ: боксы, упирающиеся в край кадра,
                отвергаются как extends_out_of_bounds;
      - > 0   — боксы в пределах tolerance считаются «касающимися края»
                и подрезаются к [0, 1] вместо отклонения.

    Побочный эффект: при tolerance > 0 бокс `b` может быть подрезан
    (мутирует dict).
    """
    reasons = []

    if "error" in b:
        return [b["error"]]

    cls, xc, yc, w, h = b["cls"], b["xc"], b["yc"], b["w"], b["h"]

    if w <= 0 or h <= 0:
        reasons.append("non_positive_size")
        return reasons

    if xc < 0 or xc > 1 or yc < 0 or yc > 1:
        reasons.append("center_out_of_bounds")

    # ── Проверка границ с допуском (по умолчанию 0.0 = строго) ──
    left   = xc - w / 2
    right  = xc + w / 2
    top    = yc - h / 2
    bottom = yc + h / 2
    eps = oob_tolerance

    if left < -eps or right > 1 + eps or top < -eps or bottom > 1 + eps:
        reasons.append("extends_out_of_bounds")
    elif eps > 0 and (left < 0 or right > 1 or top < 0 or bottom > 1):
        # В пределах допуска, но чуть вылезает — подрезаем.
        clipped = clip_yolo_box(b)
        if clipped is None:
            reasons.append("clipped_to_zero")
        else:
            b["xc"], b["yc"], b["w"], b["h"] = clipped

    if w > 1 or h > 1:
        reasons.append("size_over_1")

    area = b["w"] * b["h"]
    if area < min_area:
        reasons.append(f"area_too_small:{area:.6f}")

    if b["h"] > 0:
        aspect = b["w"] / b["h"]
        if aspect > max_aspect:
            reasons.append(f"aspect_too_high:{aspect:.2f}")
        elif aspect < min_aspect:
            reasons.append(f"aspect_too_low:{aspect:.2f}")

    return reasons


def write_yolo_label(label_path: Path, boxes):
    """Сохраняет очищенную YOLO-разметку."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for b in boxes:
            f.write(f"{b['cls']} {b['xc']:.6f} {b['yc']:.6f} "
                    f"{b['w']:.6f} {b['h']:.6f}\n")


# ------------------------- основная логика -------------------------

def prepare_split(raw_root: Path, processed_root: Path,
                  samples_rejected_dir: Path, config: dict, split_name: str):
    """
    Обрабатывает один split.

    Ожидаемая структура источника:
        raw_root/images/<split>/*.png
        raw_root/labels/<split>/*.txt

    Логика удаления изображений в samples/rejected/rejected_images/:
      - если ВСЕ боксы отвергнуты (drop_images_with_all_boxes_rejected);
      - если ЛЮБОЙ отвергнутый бокс имеет критичную причину
        (drop_images_with_critical_rejected_boxes).
    """
    fconf = config["preprocessing"]["filtering"]
    min_area = fconf["min_box_area_frac"]
    min_aspect = fconf["min_aspect_ratio"]
    max_aspect = fconf["max_aspect_ratio"]
    oob_tolerance = fconf.get("out_of_bounds_tolerance", 0.0)

    drop_all_rejected = fconf.get("drop_images_with_all_boxes_rejected", True)
    drop_critical = fconf.get("drop_images_with_critical_rejected_boxes", False)
    critical_reasons = set(fconf.get("critical_reject_reasons", [
        "extends_out_of_bounds", "aspect_too_high", "aspect_too_low",
        "center_out_of_bounds", "size_over_1",
    ]))

    images_dir = raw_root / "images" / split_name
    labels_dir = raw_root / "labels" / split_name

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Нет {images_dir} или {labels_dir}")

    images = find_images(images_dir)
    logger.info(f"[{split_name}] Изображений: {len(images)}")

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

        if not label_path.exists():
            logger.warning(f"[{split_name}] Нет label для {img_path.name}")
            stats["images_missing_label"] += 1
            boxes = []
        else:
            boxes = read_yolo_label(label_path)

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
                                         img_W, img_H, oob_tolerance)
            was_clipped = (
                "error" not in b
                and any(abs(b.get(k, 0) - orig[k]) > 1e-9
                        for k in ("xc", "yc", "w", "h"))
            )

            if reasons:
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
                    "critical": reason_keys & critical_reasons != set(),
                })
            else:
                if was_clipped:
                    stats["boxes_clipped"] += 1
                kept_boxes.append(b)
                stats["boxes_kept"] += 1

        # ── Определяем судьбу изображения ──
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

            shutil.copy2(img_path, rejected_images_dir / img_path.name)
            logger.warning(
                f"[{split_name}] {stem}: "
                f"all_rejected={all_rejected}, critical={any_critical_rejected} "
                f"→ rejected_images/"
            )
        else:
            shutil.copy2(img_path, out_images / img_path.name)
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


def make_data_yaml(processed_dir: Path):
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


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--raw-subdir", default="rsna_yolo")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    raw_root = Path(config["paths"]["raw_dir"]) / args.raw_subdir
    processed_root = Path(config["paths"]["processed_dir"])
    staged_root = Path(config["paths"]["staged_dir"])
    samples_root = Path(config["paths"].get("samples_dir", "data/samples"))
    samples_rejected_dir = samples_root / "rejected"
    samples_rejected_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(f"Нет папки {raw_root}")

    logger.info(f"Источник: {raw_root}")
    logger.info(f"Назначение processed: {processed_root}")
    logger.info(f"Назначение rejected:  {samples_rejected_dir}")

    all_stats = []
    all_rejected = []

    for split_name in ("train", "val"):
        images_dir = raw_root / "images" / split_name
        if not images_dir.exists():
            logger.warning(f"Пропуск: {images_dir} не существует")
            continue
        stats, rejected = prepare_split(raw_root, processed_root,
                                        samples_rejected_dir, config,
                                        split_name)
        all_stats.append(stats)
        all_rejected.extend(rejected)

    make_data_yaml(processed_root)

    # rejected_annotations.csv → samples/rejected/
    if all_rejected:
        import pandas as pd
        df_rej = pd.DataFrame(all_rejected)
        rej_path = samples_rejected_dir / "rejected_annotations.csv"
        df_rej.to_csv(rej_path, index=False)
        logger.info(f"Отвергнутые боксы: {rej_path} ({len(df_rej)} строк)")
    else:
        logger.info("Отвергнутых боксов нет — rejected_annotations.csv не создаётся")

    # filtering_report.json остаётся в staged/
    report = {
        "source": str(raw_root),
        "destination_processed": str(processed_root),
        "destination_rejected": str(samples_rejected_dir),
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