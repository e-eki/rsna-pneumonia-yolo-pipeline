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


def box_reject_reasons(b, min_area, min_aspect, max_aspect, img_W, img_H):
    """Возвращает список причин отклонения бокса (пустой — значит бокс хороший)."""
    reasons = []

    if "error" in b:
        return [b["error"]]

    cls, xc, yc, w, h = b["cls"], b["xc"], b["yc"], b["w"], b["h"]

    if w <= 0 or h <= 0:
        reasons.append("non_positive_size")
        return reasons  # дальше считать бессмысленно

    if xc < 0 or xc > 1 or yc < 0 or yc > 1:
        reasons.append("center_out_of_bounds")

    # Выходит ли бокс за границы кадра
    if xc - w / 2 < 0 or xc + w / 2 > 1 or yc - h / 2 < 0 or yc + h / 2 > 1:
        reasons.append("extends_out_of_bounds")

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
    """Сохраняет очищенную YOLO-разметку."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for b in boxes:
            f.write(f"{b['cls']} {b['xc']:.6f} {b['yc']:.6f} "
                    f"{b['w']:.6f} {b['h']:.6f}\n")


# ------------------------- основная логика -------------------------

def prepare_split(raw_split_dir: Path, processed_split_dir: Path,
                  rejected_images_dir: Path, config: dict, split_name: str):
    fconf = config["preprocessing"]["filtering"]
    min_area = fconf["min_box_area_frac"]
    min_aspect = fconf["min_aspect_ratio"]
    max_aspect = fconf["max_aspect_ratio"]

    images_dir = raw_split_dir / "images"
    labels_dir = raw_split_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Нет {images_dir} или {labels_dir}")

    images = find_images(images_dir)
    logger.info(f"[{split_name}] Изображений: {len(images)}")

    out_images = processed_split_dir / "images"
    out_labels = processed_split_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {
        "split": split_name,
        "images_total": len(images),
        "images_copied": 0,
        "images_positive": 0,     # остались хотя бы с одним боксом
        "images_negative": 0,     # пустой label (норма)
        "images_all_boxes_rejected": 0,
        "images_missing_label": 0,
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_rejected": 0,
        "rejected_by_reason": {},
    }

    rejected_rows = []  # для rejected_annotations.csv

    for img_path in images:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"

        if not label_path.exists():
            logger.warning(f"[{split_name}] Нет label для {img_path.name}")
            stats["images_missing_label"] += 1
            # Считаем за negative (пустая разметка)
            boxes = []
        else:
            boxes = read_yolo_label(label_path)

        stats["boxes_total"] += len(boxes)

        # Получаем размеры изображения (нужны для перевода в пиксели при отчёте)
        try:
            with Image.open(img_path) as im:
                img_W, img_H = im.size
        except Exception as e:
            logger.warning(f"Не удалось прочитать {img_path}: {e}")
            continue

        kept_boxes = []
        for b in boxes:
            reasons = box_reject_reasons(b, min_area, min_aspect, max_aspect,
                                         img_W, img_H)
            if reasons:
                stats["boxes_rejected"] += 1
                for r in reasons:
                    key = r.split(":")[0]
                    stats["rejected_by_reason"][key] = \
                        stats["rejected_by_reason"].get(key, 0) + 1

                # Сохраняем в отчёт (переводим YOLO -> пиксели)
                if "error" not in b:
                    x_px = (b["xc"] - b["w"] / 2) * img_W
                    y_px = (b["yc"] - b["h"] / 2) * img_H
                    w_px = b["w"] * img_W
                    h_px = b["h"] * img_H
                else:
                    x_px = y_px = w_px = h_px = 0.0

                rejected_rows.append({
                    "patientId": stem,
                    "split": split_name,
                    "x": x_px,
                    "y": y_px,
                    "width": w_px,
                    "height": h_px,
                    "x_norm": b.get("xc", None),
                    "y_norm": b.get("yc", None),
                    "w_norm": b.get("w", None),
                    "h_norm": b.get("h", None),
                    "reject_reason": "; ".join(reasons),
                })
            else:
                kept_boxes.append(b)

        # Копируем изображение
        dst_img = out_images / img_path.name
        shutil.copy2(img_path, dst_img)
        stats["images_copied"] += 1

        # Пишем очищенный label
        write_yolo_label(out_labels / f"{stem}.txt", kept_boxes)

        # Классифицируем
        if len(boxes) == 0:
            stats["images_negative"] += 1
        elif len(kept_boxes) > 0:
            stats["images_positive"] += 1
        else:
            # Были боксы, но все отвергнуты
            stats["images_all_boxes_rejected"] += 1
            if fconf.get("drop_images_with_all_boxes_rejected", True):
                # Перемещаем картинку в rejected_images/
                rejected_images_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst_img), str(rejected_images_dir / img_path.name))
                logger.warning(
                    f"[{split_name}] {stem}: все боксы отвергнуты → "
                    f"перемещено в rejected_images/"
                )

    logger.info(f"[{split_name}] "
                f"positive={stats['images_positive']}, "
                f"negative={stats['images_negative']}, "
                f"all_rejected={stats['images_all_boxes_rejected']}")
    logger.info(f"[{split_name}] "
                f"boxes: total={stats['boxes_total']}, "
                f"kept={stats['boxes_kept']}, "
                f"rejected={stats['boxes_rejected']}")
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
    parser.add_argument("--raw-subdir", default="rsna_yolo",
                        help="Подпапка в data/raw с YOLO-датасетом")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    raw_root = Path(config["paths"]["raw_dir"]) / args.raw_subdir
    processed_root = Path(config["paths"]["processed_dir"])
    staged_root = Path(config["paths"]["staged_dir"])
    rejected_images_dir = staged_root / "rejected_images"

    if not raw_root.exists():
        raise FileNotFoundError(f"Нет папки {raw_root}")

    logger.info(f"Источник: {raw_root}")
    logger.info(f"Назначение: {processed_root}")

    all_stats = []
    all_rejected = []

    for split_name in ("train", "val"):
        split_raw = raw_root / split_name
        if not split_raw.exists():
            logger.warning(f"Пропуск: {split_raw} не существует")
            continue
        split_processed = processed_root / split_name
        stats, rejected = prepare_split(split_raw, split_processed,
                                        rejected_images_dir, config,
                                        split_name)
        all_stats.append(stats)
        all_rejected.extend(rejected)

    # data.yaml
    make_data_yaml(processed_root)

    # rejected_annotations.csv
    if all_rejected:
        import pandas as pd
        df_rej = pd.DataFrame(all_rejected)
        rej_path = staged_root / "rejected_annotations.csv"
        rej_path.parent.mkdir(parents=True, exist_ok=True)
        df_rej.to_csv(rej_path, index=False)
        logger.info(f"Отвергнутые боксы: {rej_path} ({len(df_rej)} строк)")

    # filtering_report.json
    report = {
        "source": str(raw_root),
        "destination": str(processed_root),
        "splits": all_stats,
    }
    report_path = staged_root / "filtering_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Отчёт фильтрации: {report_path}")

    # Итоговая сводка
    logger.info("=" * 60)
    logger.info("ИТОГИ ПОДГОТОВКИ ДАТАСЕТА")
    logger.info("=" * 60)
    for s in all_stats:
        logger.info(f"[{s['split']}]")
        logger.info(f"  изображений всего:    {s['images_total']}")
        logger.info(f"  positive:             {s['images_positive']}")
        logger.info(f"  negative:             {s['images_negative']}")
        logger.info(f"  all_boxes_rejected:   {s['images_all_boxes_rejected']}")
        logger.info(f"  missing_label:        {s['images_missing_label']}")
        logger.info(f"  боксов всего:         {s['boxes_total']}")
        logger.info(f"  боксов отвергнуто:    {s['boxes_rejected']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()