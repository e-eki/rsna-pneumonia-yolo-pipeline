"""
Собирает отвергнутые на этапе подготовки датасета изображения и рисует
на них отвергнутые боксы.

Входные данные:
  data/samples/rejected/rejected_annotations.csv  — отвергнутые боксы
  data/samples/rejected/rejected_images/*.png     — полностью отвергнутые
                                                    изображения
  data/processed/images/{train,val}/*.png         — если картинка осталась
                                                    в processed, но часть
                                                    боксов у неё отвергнута
  data/raw/rsna_yolo/images/{train,val}/*.png     — fallback

Выход:
  data/samples/rejected/drawn/*.jpg               — отрисованные копии
  data/samples/rejected/metadata.json

Пример:
  python -m src.collect_rejected
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils import load_config, setup_logging, get_logger

logger = get_logger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_BOX = (0, 0, 255)      # красный — отвергнутый бокс
COLOR_TEXT = (255, 255, 255)
COLOR_CRIT = (0, 165, 255)   # оранжевый — критичное отклонение


def _draw_rect(img, x1, y1, x2, y2, color, thickness=2):
    H, W = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _add_caption(img, text, y=25):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.rectangle(img, (5, y - th - 8), (15 + tw, y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (10, y), FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


def _find_image(stems, paths_list):
    """Ищет файл по stem в списке корней (в порядке приоритета)."""
    exts = (".png", ".jpg", ".jpeg", ".dcm")
    for root in paths_list:
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.stem == stems and p.suffix.lower() in exts:
                return p
    return None


def _load_image(path: Path):
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(float)
            center, width = 40, 400
            lo, hi = center - width // 2, center + width // 2
            arr = np.clip(arr, lo, hi)
            arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.warning(f"DICOM read failed {path}: {e}")
            return None
    return cv2.imread(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    raw_root = Path(config["paths"]["raw_dir"]) / config["dataset"].get(
        "raw_yolo_subdir", "rsna_yolo")
    processed_root = Path(config["paths"]["processed_dir"])
    samples_root = Path(config["paths"].get("samples_dir", "data/samples"))
    rejected_root = samples_root / "rejected"

    rejected_csv = rejected_root / "rejected_annotations.csv"
    rejected_images = rejected_root / "rejected_images"
    drawn_dir = rejected_root / "drawn"
    drawn_dir.mkdir(parents=True, exist_ok=True)

    if not rejected_csv.exists():
        logger.warning(f"Нет файла {rejected_csv} — нечего отрисовывать")
        (rejected_root / "README.txt").write_text(
            "Нет отвергнутых аннотаций.\n"
            "Запустите сначала `python -m src.prepare_yolo_dataset`.\n"
        )
        return

    df = pd.read_csv(rejected_csv)
    logger.info(f"Отвергнутых аннотаций: {len(df)}")
    if not len(df):
        logger.info("Файл пуст — нечего отрисовывать.")
        return

    # Список корней для поиска оригинальных изображений
    search_roots = [
        rejected_images,                                    # 1. отвергнутые
        processed_root / "images" / "train",                # 2. в processed
        processed_root / "images" / "val",
        raw_root / "images" / "train",                      # 3. fallback raw
        raw_root / "images" / "val",
    ]

    meta = []
    saved = 0

    for pid, group in df.groupby("patientId"):
        pid = str(pid)
        img_path = _find_image(pid, search_roots)
        if img_path is None:
            logger.warning(f"Не найдено изображение для {pid}")
            continue

        img = _load_image(img_path)
        if img is None:
            logger.warning(f"Не удалось прочитать {img_path}")
            continue

        reasons = []
        is_critical = False
        for _, row in group.iterrows():
            # Всё в пикселях — рисуем bbox
            if {"x", "y", "width", "height"}.issubset(row.index):
                x, y, w, h = (float(row["x"]), float(row["y"]),
                              float(row["width"]), float(row["height"]))
                crit = bool(row.get("critical", False))
                is_critical = is_critical or crit
                _draw_rect(img, x, y, x + w, y + h,
                           COLOR_CRIT if crit else COLOR_BOX)
            reasons.append(str(row.get("reject_reason", "?")))

        unique_reasons = sorted(set(reasons))
        tag = "CRITICAL" if is_critical else "minor"
        caption = f"{pid}  [{tag}]  rejected={len(group)}  " \
                  f"{'; '.join(unique_reasons)[:80]}"
        _add_caption(img, caption)

        out_path = drawn_dir / f"{pid}.jpg"
        cv2.imwrite(str(out_path), img)
        saved += 1

        meta.append({
            "patientId": pid,
            "source": str(img_path),
            "output": str(out_path),
            "rejected_boxes": int(len(group)),
            "reasons": unique_reasons,
            "critical": is_critical,
        })

    with open(rejected_root / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Сохранено {saved} отвергнутых изображений в {drawn_dir}")
    logger.info(f"Смотреть: python -m src.viewer --dir {drawn_dir}")


if __name__ == "__main__":
    main()