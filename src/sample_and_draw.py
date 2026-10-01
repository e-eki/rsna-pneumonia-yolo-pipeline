# src/sample_and_draw.py
"""
Создание случайной проверочной выборки с отрисованной разметкой.

Поддерживает три стадии:
  --stage raw        : данные в data/raw (аннотации в CSV, изображения jpg/png)
  --stage staged     : данные в data/staged (CSV + jpg)
  --stage processed  : YOLO-датасет в data/processed (labels/*.txt + images/*.jpg)

Примеры:
  python src/sample_and_draw.py --stage raw       --num 100
  python src/sample_and_draw.py --stage staged    --num 100
  python src/sample_and_draw.py --stage processed --split val --num 100 --grid
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils import load_config, setup_logging, get_logger

logger = get_logger(__name__)

# ------------------------- Отрисовка -------------------------

BOX_COLOR = (0, 200, 0)       # зелёный (BGR)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 140, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_box_xyxy(img, x1, y1, x2, y2, label):
    H, W = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), TEXT_BG, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)


def _draw_corner_text(img, text):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    cv2.rectangle(img, (5, 5), (20 + tw, 15 + th), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 10 + th), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_boxes_xywh(img, boxes, class_names):
    """boxes: список dict {x, y, w, h, cls}."""
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        label = f"{class_names.get(b['cls'], 'obj')} #{b.get('id', '')}"
        _draw_box_xyxy(img, x, y, x + w, y + h, label)
    return img


def draw_boxes_yolo(img, boxes, class_names):
    """boxes: список dict {xc, yc, w, h, cls} (нормализованные)."""
    H, W = img.shape[:2]
    for b in boxes:
        x1 = (b["xc"] - b["w"] / 2) * W
        y1 = (b["yc"] - b["h"] / 2) * H
        x2 = (b["xc"] + b["w"] / 2) * W
        y2 = (b["yc"] + b["h"] / 2) * H
        label = class_names.get(b["cls"], str(b["cls"]))
        _draw_box_xyxy(img, x1, y1, x2, y2, label)
    return img


# ------------------------- Сбор данных по стадиям -------------------------

def _find_one(patterns, root):
    for p in patterns:
        found = list(Path(root).rglob(p))
        if found:
            return found
    return []


def collect_raw_or_staged(root: Path, class_names: dict):
    """Читает CSV-аннотации и изображения для стадий raw/staged."""
    csv_files = _find_one(["*.csv"], root)
    if not csv_files:
        raise FileNotFoundError(f"CSV с аннотациями не найден в {root}")
    df = pd.read_csv(csv_files[0])
    logger.info(f"Аннотаций загружено: {len(df)} из {csv_files[0]}")

    # Ищем изображения
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(Path(root).rglob(ext))
    if not images:
        raise FileNotFoundError(f"Изображений не найдено в {root}")
    image_by_stem = {p.stem: p for p in images}
    logger.info(f"Найдено изображений: {len(images)}")

    # Группируем боксы по patientId
    samples = []
    for patient_id, group in df.groupby("patientId"):
        img_path = image_by_stem.get(str(patient_id))
        if img_path is None:
            continue
        boxes = []
        for _, row in group.iterrows():
            if row.get("Target", 1) != 1:
                continue
            boxes.append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "w": float(row["width"]),
                "h": float(row["height"]),
                "cls": 0,
                "id": int(row.get("id", 0)) if "id" in row else 0,
            })
        samples.append({"image_path": img_path, "boxes": boxes})
    return samples


def collect_processed(root: Path, split: str, class_names: dict):
    """Читает YOLO-разметку из data/processed."""
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.exists():
        raise FileNotFoundError(f"Нет папки {images_dir}")

    samples = []
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, w, h = parts
                    boxes.append({
                        "cls": int(cls),
                        "xc": float(xc),
                        "yc": float(yc),
                        "w": float(w),
                        "h": float(h),
                    })
        samples.append({"image_path": img_path, "boxes": boxes})
    logger.info(f"Собрано {len(samples)} изображений из {images_dir}")
    return samples


# ------------------------- Контактный лист -------------------------

def make_contact_sheet(image_paths, output_path, cols=8, thumb=200):
    """Собирает сетку превью из изображений."""
    rows = (len(image_paths) + cols - 1) // cols
    sheet = np.zeros((rows * thumb, cols * thumb, 3), dtype=np.uint8)
    for idx, p in enumerate(image_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (thumb, thumb))
        r, c = divmod(idx, cols)
        sheet[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = img
    cv2.imwrite(str(output_path), sheet)
    logger.info(f"Контактный лист: {output_path}")


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", choices=["raw", "staged", "processed"], required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Только для stage=processed")
    parser.add_argument("--num", type=int, default=100, help="Сколько изображений сэмплировать")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="Куда сохранять. По умолчанию data/samples/<stage>/")
    parser.add_argument("--grid", action="store_true", help="Также сделать контактный лист")
    parser.add_argument("--only-with-boxes", action="store_true",
                        help="Сэмплировать только изображения, где есть хотя бы один объект")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    class_names = {0: "pneumonia"}
    seed = args.seed if args.seed is not None else config["project"]["seed"]
    random.seed(seed)

    # ---- выбираем источник данных по стадии ----
    if args.stage == "raw":
        root = Path(config["paths"]["raw_dir"])
        samples = collect_raw_or_staged(root, class_names)
        draw_fn = draw_boxes_xywh
    elif args.stage == "staged":
        root = Path(config["paths"]["staged_dir"])
        samples = collect_raw_or_staged(root, class_names)
        draw_fn = draw_boxes_xywh
    else:  # processed
        root = Path(config["paths"]["processed_dir"])
        samples = collect_processed(root, args.split, class_names)
        draw_fn = draw_boxes_yolo

    if args.only_with_boxes:
        samples = [s for s in samples if s["boxes"]]
        logger.info(f"Отфильтровано до {len(samples)} изображений с разметкой")

    if not samples:
        raise RuntimeError("Нечего сэмплировать — проверь пути и аннотации")

    # ---- сэмплирование ----
    k = min(args.num, len(samples))
    picked = random.sample(samples, k)
    logger.info(f"Сэмплировано {k} из {len(samples)}")

    # ---- выходная директория ----
    if args.output:
        out_dir = Path(args.output)
    elif args.stage == "processed":
        out_dir = Path("data/samples") / args.stage / args.split
    else:
        out_dir = Path("data/samples") / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    drawn_paths = []
    for i, s in enumerate(picked):
        img = cv2.imread(str(s["image_path"]))
        if img is None:
            logger.warning(f"Не удалось прочитать: {s['image_path']}")
            continue
        img = draw_fn(img, s["boxes"], class_names)
        _draw_corner_text(img, f"{i + 1}/{k}  {s['image_path'].name}  boxes={len(s['boxes'])}")

        out_name = f"{i:04d}_{s['image_path'].stem}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), img)
        drawn_paths.append(out_path)

        metadata.append({
            "index": i,
            "source": str(s["image_path"]),
            "output": str(out_path),
            "num_boxes": len(s["boxes"]),
        })

    # ---- сохраняем метаданные для viewer.py ----
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Отрисовано {len(drawn_paths)} изображений в {out_dir}")
    logger.info(f"Метаданные: {meta_path}")

    # ---- контактный лист ----
    if args.grid and drawn_paths:
        make_contact_sheet(drawn_paths, out_dir / "contact_sheet.jpg", cols=8, thumb=200)


if __name__ == "__main__":
    main()