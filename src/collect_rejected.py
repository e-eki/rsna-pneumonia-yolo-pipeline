"""
Визуализация отвергнутых изображений: рисует на них отклонённые боксы.

Читает rejected_annotations.csv (созданный prepare_yolo_dataset.py),
находит исходные изображения и сохраняет копии с отрисованными боксами:
красные — обычные отклонения, оранжевые — критичные (из-за которых
изображение убрали из датасета целиком).

Входные данные:
    data/samples/rejected/rejected_annotations.csv
    data/samples/rejected/rejected_images/*.png       — отвергнутые целиком
    data/processed/images/{train,val}/*.jpg           — если изображение
                                                        осталось, но часть
                                                        боксов отклонена
    data/raw/<subdir>/images/{train,val}/*.jpg        — fallback

Выходные данные:
    data/samples/rejected/drawn/*.jpg                 — отрисованные копии
    data/samples/rejected/metadata.json               — что и откуда взято

Результат удобно смотреть через viewer.py:
    python -m src.viewer --dir data/samples/rejected/drawn

Пример запуска:
    python -m src.collect_rejected --config configs/config.yaml
"""
import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from src.utils import get_logger, load_config, setup_logging

logger = get_logger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_BOX = (0, 0, 255)      # красный — обычное отклонение
COLOR_CRIT = (0, 165, 255)   # оранжевый — критичное отклонение
COLOR_TEXT = (255, 255, 255)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ─────────────────────────── отрисовка ───────────────────────────

def _draw_rect(img, x1, y1, x2, y2, color, thickness=2):
    """Рисует прямоугольник, обрезая координаты по границам изображения."""
    H, W = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _add_caption(img, text, y=25):
    """Пишет подпись сверху слева с чёрной подложкой."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.rectangle(img, (5, y - th - 8), (15 + tw, y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (10, y), FONT, 0.55,
                COLOR_TEXT, 1, cv2.LINE_AA)


# ─────────────────────────── поиск изображений ───────────────────────────

def _find_image(stem: str, search_roots):
    """
    Ищет изображение по stem в списке корней в порядке приоритета.

    Args:
        stem:         имя файла без расширения.
        search_roots: список Path-директорий для поиска.

    Returns:
        Path к найденному файлу или None.
    """
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.stem == stem and p.suffix.lower() in IMG_EXTS:
                return p
    return None


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    raw_subdir = config["dataset"].get("raw_yolo_subdir", "rsna_yolo")
    raw_root = Path(config["paths"]["raw_dir"]) / raw_subdir
    processed_root = Path(config["paths"]["processed_dir"])
    samples_root = Path(config["paths"].get("samples_dir", "data/samples"))
    rejected_root = samples_root / "rejected"

    rejected_csv = rejected_root / "rejected_annotations.csv"
    drawn_dir = rejected_root / "drawn"
    drawn_dir.mkdir(parents=True, exist_ok=True)

    # Если отвергнутых аннотаций нет — оставляем памятку и выходим.
    if not rejected_csv.exists():
        logger.warning(f"Нет файла {rejected_csv} — нечего отрисовывать")
        (rejected_root / "README.txt").write_text(
            "Нет отвергнутых аннотаций.\n"
            "Запустите сначала `python -m src.prepare_yolo_dataset`.\n"
        )
        return

    df = pd.read_csv(rejected_csv)
    logger.info(f"Отвергнутых аннотаций: {len(df)}")
    if df.empty:
        logger.info("Файл пуст — нечего отрисовывать.")
        return

    # Порядок поиска исходников: сначала отвергнутые, потом processed, потом raw.
    search_roots = [
        rejected_root / "rejected_images",
        processed_root / "images" / "train",
        processed_root / "images" / "val",
        raw_root / "images" / "train",
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

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Не удалось прочитать {img_path}")
            continue

        reasons = []
        is_critical = False

        for _, row in group.iterrows():
            # Все координаты — в пикселях, рисуем как bbox.
            x, y = float(row["x"]), float(row["y"])
            w, h = float(row["width"]), float(row["height"])
            crit = bool(row.get("critical", False))
            is_critical = is_critical or crit
            _draw_rect(img, x, y, x + w, y + h,
                       COLOR_CRIT if crit else COLOR_BOX)
            reasons.append(str(row.get("reject_reason", "?")))

        unique_reasons = sorted(set(reasons))
        tag = "CRITICAL" if is_critical else "minor"
        caption = (f"{pid}  [{tag}]  rejected={len(group)}  "
                   f"{'; '.join(unique_reasons)[:80]}")
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