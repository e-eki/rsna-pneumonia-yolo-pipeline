# src/collect_rejected.py
"""
Собирает отвергнутые на этапе препроцессинга изображения в отдельную папку
и рисует на них отвергнутые боксы (или полигоны, если бы они были в аннотациях).

Результат можно смотреть через src/viewer.py.

Входные файлы (создаются preprocess.py):
  data/staged/rejected_annotations.csv   — отвергнутые боксы
  data/staged/rejected_images/           — полностью отвергнутые изображения
                                            (перемещаются сюда из staged/images/)

Для частично отвергнутых изображений (когда часть боксов прошла, часть — нет)
картинка берётся из staged/images/.

Примеры:
  python src/collect_rejected.py
  python src/collect_rejected.py --output data/samples/rejected
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
COLOR_BOX = (0, 0, 255)   # красный — отвергнутый бокс
COLOR_TEXT = (255, 255, 255)


# ------------------------- отрисовка -------------------------

def _draw_rect(img, x1, y1, x2, y2, color, thickness=2):
    H, W = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _draw_polygon(img, points, color, thickness=2):
    """Если в аннотации есть список точек (полигон) — рисуем его."""
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _add_caption(img, text, y=25):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.rectangle(img, (5, y - th - 8), (15 + tw, y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (10, y), FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


# ------------------------- поиск изображений -------------------------

def _find_image(staged_dir: Path, raw_dir: Path, stem: str):
    """
    Ищем изображение в порядке приоритета:
      1. staged/rejected_images/   — полностью отвергнутые
      2. staged/images/            — частично отвергнутые (там же, где все)
      3. raw/                      — исходники (могут быть DICOM)
    """
    candidates_roots = [
        staged_dir / "rejected_images",
        staged_dir / "images",
        raw_dir,
    ]
    exts = (".jpg", ".jpeg", ".png", ".dcm")
    for root in candidates_roots:
        if not root.exists():
            continue
        for p in root.rglob(f"{stem}.*"):
            if p.suffix.lower() in exts:
                return p
    return None


def _load_image(path: Path):
    """Загружает изображение, конвертируя DICOM при необходимости."""
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


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default=None,
                        help="По умолчанию data/samples/rejected/")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    staged_dir = Path(config["paths"]["staged_dir"])
    raw_dir = Path(config["paths"]["raw_dir"])
    out_dir = Path(args.output) if args.output else Path("data/samples/rejected")
    out_dir.mkdir(parents=True, exist_ok=True)

    rejected_csv = staged_dir / "rejected_annotations.csv"
    if not rejected_csv.exists():
        logger.warning(f"Нет файла отвергнутых аннотаций: {rejected_csv}")
        logger.warning("Либо фильтрация ничего не отвергла, либо preprocess.py не запускался.")
        (out_dir / "README.txt").write_text(
            "Нет отвергнутых аннотаций.\n"
            "Запустите сначала `python src/preprocess.py`.\n"
        )
        return

    df = pd.read_csv(rejected_csv)
    logger.info(f"Отвергнутых аннотаций: {len(df)}")
    if not len(df):
        logger.info("Файл пуст — нечего отрисовывать.")
        return

    by_patient = df.groupby("patientId")
    meta = []
    saved = 0

    for pid, group in by_patient:
        pid = str(pid)
        img_path = _find_image(staged_dir, raw_dir, pid)
        if img_path is None:
            logger.warning(f"Не найдено изображение для {pid} — пропускаем")
            continue

        img = _load_image(img_path)
        if img is None:
            logger.warning(f"Не удалось прочитать {img_path}")
            continue

        reasons = []
        for _, row in group.iterrows():
            # Если в аннотации есть полигон — рисуем его
            if "polygon" in row.index and isinstance(row.get("polygon"), str):
                try:
                    pts = json.loads(row["polygon"])
                    _draw_polygon(img, pts, COLOR_BOX)
                except Exception:
                    pass
            # Иначе — рисуем bbox (случай RSNA)
            elif {"x", "y", "width", "height"}.issubset(row.index):
                x, y, w, h = (float(row["x"]), float(row["y"]),
                              float(row["width"]), float(row["height"]))
                _draw_rect(img, x, y, x + w, y + h, COLOR_BOX)
            reasons.append(str(row.get("reject_reason", "?")))

        unique_reasons = sorted(set(reasons))
        caption = f"{pid}  rejected={len(group)}  [{'; '.join(unique_reasons)[:80]}]"
        _add_caption(img, caption)

        out_path = out_dir / f"{pid}.jpg"
        cv2.imwrite(str(out_path), img)
        saved += 1

        meta.append({
            "patientId": pid,
            "source": str(img_path),
            "output": str(out_path),
            "rejected_boxes": int(len(group)),
            "reasons": unique_reasons,
        })

    # metadata.json — для совместимости со viewer.py и Colab-слайдером
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Сохранено {saved} отвергнутых изображений в {out_dir}")
    logger.info(f"Смотреть: python src/viewer.py --dir {out_dir}")


if __name__ == "__main__":
    main()