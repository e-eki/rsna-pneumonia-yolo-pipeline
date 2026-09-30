# src/compare_stages.py
"""
ГЛАВНАЯ проверка пайплайна: боксы из raw (пиксельные координаты) должны точно
совпадать с боксами из processed (нормализованные YOLO-координаты), потому что
это одно и то же изображение, прошедшее конвертацию.

Красный = RAW (пиксели)
Зелёный = PROCESSED (YOLO, то, что видит модель)

Режимы:
  --mode overlay : оба набора боксов на одном изображении (самый наглядный)
  --mode side    : две картинки рядом

Примеры:
  python src/compare_stages.py --num 40 --mode overlay --split train
  python src/compare_stages.py --num 40 --mode side    --split val

===============
Сравнение разметки до/после конвертации в YOLO.

Сопоставляет аннотации из raw/staged (x, y, width, height в пикселях)
с YOLO-разметкой в processed (нормализованные xc, yc, w, h).

Режимы:
  --mode side     : две картинки рядом (raw-boxes vs processed-boxes)
  --mode overlay  : оба набора боксов на одном изображении
                    (красный = raw/staged, зелёный = processed)

Примеры:
  python src/compare_stages.py --num 20 --mode overlay
  python src/compare_stages.py --num 20 --mode side

======
Показывает одно и то же изображение дважды — с боксами из raw/staged и с боксами из processed. Два режима:

--mode side — две картинки рядом (side-by-side).

--mode overlay — оба набора боксов на одном изображении разными цветами 
(красный = raw, зелёный = processed). Это самый быстрый способ увидеть расхождение: 
если зелёный прямоугольник точно поверх красного — всё ок.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils import load_config, setup_logging, get_logger

logger = get_logger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_RAW = (0, 0, 255)      # красный (BGR) — raw
COLOR_PROC = (0, 200, 0)     # зелёный — processed
COLOR_TEXT = (255, 255, 255)


# ------------------------- загрузка -------------------------

def load_raw_boxes(raw_dir: Path) -> pd.DataFrame:
    """Читает ИСХОДНЫЕ пиксельные аннотации из raw."""
    csvs = [c for c in raw_dir.rglob("*.csv") if "rejected" not in c.name]
    if not csvs:
        raise FileNotFoundError(f"CSV не найден в {raw_dir}")
    ann_file = csvs[0]
    df = pd.read_csv(ann_file)
    if "Target" in df.columns:
        df = df[df["Target"] == 1].copy()
    logger.info(f"RAW: {len(df)} позитивных аннотаций из {ann_file.name}")
    return df


def load_processed_boxes(processed_dir: Path, split: str) -> dict:
    """Возвращает {stem: [ {xc, yc, w, h, cls}, ... ]}."""
    labels_dir = processed_dir / "labels" / split
    if not labels_dir.exists():
        raise FileNotFoundError(f"Нет папки {labels_dir}")
    result = {}
    for txt in labels_dir.glob("*.txt"):
        boxes = []
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = parts
                boxes.append({
                    "cls": int(cls),
                    "xc": float(xc), "yc": float(yc),
                    "w": float(w), "h": float(h),
                })
        result[txt.stem] = boxes
    logger.info(f"PROCESSED: {len(result)} файлов разметки (split={split})")
    return result


def find_processed_image(processed_dir: Path, split: str, stem: str):
    images_dir = processed_dir / "images" / split
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ------------------------- отрисовка -------------------------

def _draw_rect(img, x1, y1, x2, y2, color, thickness=2):
    H, W = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_raw_boxes(img, df_patient, color=COLOR_RAW):
    for _, row in df_patient.iterrows():
        x, y, w, h = row["x"], row["y"], row["width"], row["height"]
        _draw_rect(img, x, y, x + w, y + h, color)
    return img


def draw_processed_boxes(img, boxes, color=COLOR_PROC):
    H, W = img.shape[:2]
    for b in boxes:
        x1 = (b["xc"] - b["w"] / 2) * W
        y1 = (b["yc"] - b["h"] / 2) * H
        x2 = (b["xc"] + b["w"] / 2) * W
        y2 = (b["yc"] + b["h"] / 2) * H
        _draw_rect(img, x1, y1, x2, y2, color)
    return img


def add_caption(img, text, y=25):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    cv2.rectangle(img, (5, y - th - 8), (15 + tw, y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (10, y), FONT, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
    return img


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--num", type=int, default=40)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--mode", default="overlay", choices=["side", "overlay"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["project"]["seed"]
    random.seed(seed)

    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])

    # RAW — источник истины (пиксельные координаты)
    df_raw = load_raw_boxes(raw_dir)
    raw_ids = set(df_raw["patientId"].astype(str).unique())

    # PROCESSED — то, что видит модель
    processed = load_processed_boxes(processed_dir, args.split)
    processed_ids = set(processed.keys())

    # Базис — processed, потому что именно там лежат финальные картинки,
    # которые скармливаются YOLO (raw может быть DICOM).
    common = sorted(processed_ids & raw_ids)
    logger.info(f"Общих изображений (raw ∩ processed[{args.split}]): {len(common)}")
    logger.info(f"  в raw:                {len(raw_ids)}")
    logger.info(f"  в processed[{args.split}]: {len(processed_ids)}")
    logger.info(f"  в processed БЕЗ raw-аннотаций (negative): {len(processed_ids - raw_ids)}")
    logger.info(f"  в raw, но ушли в другой split:             {len(raw_ids - processed_ids)}")

    if not common:
        raise RuntimeError("Нет общих изображений между raw и processed. "
                           "Проверь patientId и имена файлов в processed/labels.")

    k = min(args.num, len(common))
    picked = random.sample(common, k)

    out_dir = Path(args.output) if args.output else Path("data/samples/compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, pid in enumerate(picked):
        img_path = find_processed_image(processed_dir, args.split, pid)
        if img_path is None:
            logger.warning(f"Изображение {pid} не найдено в processed/{args.split}")
            continue
        base = cv2.imread(str(img_path))
        if base is None:
            logger.warning(f"Не удалось прочитать {img_path}")
            continue
        H, W = base.shape[:2]

        df_patient = df_raw[df_raw["patientId"].astype(str) == pid]
        proc_boxes = processed[pid]

        if args.mode == "overlay":
            canvas = base.copy()
            draw_raw_boxes(canvas, df_patient, COLOR_RAW)
            draw_processed_boxes(canvas, proc_boxes, COLOR_PROC)
            add_caption(canvas,
                        f"{pid}  RAW={len(df_patient)} (RED)  YOLO={len(proc_boxes)} (GREEN)")
            combined = canvas
        else:  # side
            left = base.copy()
            right = base.copy()
            draw_raw_boxes(left, df_patient, COLOR_RAW)
            draw_processed_boxes(right, proc_boxes, COLOR_PROC)
            add_caption(left, f"RAW (pixels)  boxes={len(df_patient)}")
            add_caption(right, f"PROCESSED (YOLO)  boxes={len(proc_boxes)}")
            sep = np.full((H, 4, 3), 255, dtype=np.uint8)
            combined = np.hstack([left, sep, right])

        out_path = out_dir / f"{i:04d}_{pid}.jpg"
        cv2.imwrite(str(out_path), combined)
        saved += 1

    logger.info(f"Сохранено {saved} сравнений в {out_dir}")
    logger.info("Красный = RAW (пиксели), Зелёный = PROCESSED (YOLO). Должны СОВПАДАТЬ.")
    logger.info(f"Смотреть: python src/viewer.py --dir {out_dir}")


if __name__ == "__main__":
    main()