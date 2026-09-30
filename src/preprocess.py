# src/preprocess.py
"""
Базовая предобработка (конвертация изображений и парсинг аннотаций) + фильтрация «плохих» аннотаций.

Скрипт выполняет базовую обработку: 
конвертацию DICOM → JPEG (если нужно) и парсинг аннотаций в промежуточный формат.

Стадии:
  raw  -> staged

Что делает:
  1. Конвертирует изображения (DICOM->JPEG или копирование JPEG).
  2. Читает аннотации из raw.
  3. Фильтрует боксы:
       - вне границ изображения
       - площадь < min_box_area_frac
       - aspect ratio вне [min_aspect_ratio, max_aspect_ratio]
       - неположительные width/height
  4. Удаляет изображения, у которых были боксы, но все отвергнуты
     (опция drop_images_with_all_boxes_rejected).
  5. Сохраняет:
       staged/annotations.csv           — очищенные аннотации
       staged/rejected_annotations.csv  — отвергнутые (для аудита)
       staged/filtering_report.json     — сводка

Пример:
  python src/preprocess.py --config configs/config.yaml
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import load_config, ensure_dirs, setup_logging, get_logger

logger = get_logger(__name__)


# ------------------------- Изображения -------------------------

def convert_dicom_to_jpeg(dicom_path: Path, output_path: Path, quality: int = 95):
    import pydicom
    ds = pydicom.dcmread(str(dicom_path))
    pixel_array = ds.pixel_array.astype(float)
    # Windowing для лёгких
    center, width = 40, 400
    lower = center - width // 2
    upper = center + width // 2
    pixel_array = np.clip(pixel_array, lower, upper)
    pixel_array = ((pixel_array - lower) / (upper - lower) * 255).astype(np.uint8)
    Image.fromarray(pixel_array).save(output_path, "JPEG", quality=quality)


def preprocess_images(raw_dir: Path, staged_dir: Path, config: dict) -> dict:
    """Копирует/конвертирует изображения. Возвращает {stem: (W, H)}."""
    images_dir = raw_dir / "images" if (raw_dir / "images").exists() else raw_dir
    out_images = staged_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    jpegs = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.jpeg")) \
            + list(images_dir.rglob("*.png"))
    dicoms = list(images_dir.rglob("*.dcm"))

    size_map = {}

    if jpegs:
        logger.info(f"Найдено {len(jpegs)} JPEG/PNG. Копирование...")
        for src in jpegs:
            dst = out_images / src.name
            shutil.copy2(src, dst)
            with Image.open(dst) as im:
                size_map[src.stem] = im.size  # (W, H)
    elif dicoms:
        logger.info(f"Найдено {len(dicoms)} DICOM. Конвертация...")
        q = config["preprocessing"]["jpeg_quality"]
        for src in dicoms:
            dst = out_images / f"{src.stem}.jpg"
            convert_dicom_to_jpeg(src, dst, q)
            with Image.open(dst) as im:
                size_map[src.stem] = im.size
    else:
        raise FileNotFoundError("Не найдено ни JPEG, ни DICOM файлов")

    logger.info(f"Изображений обработано: {len(size_map)}")
    return size_map


# ------------------------- Аннотации -------------------------

def load_raw_annotations(raw_dir: Path) -> pd.DataFrame:
    csvs = list(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"CSV с аннотациями не найден в {raw_dir}")
    df = pd.read_csv(csvs[0])
    logger.info(f"Загружено {len(df)} аннотаций из {csvs[0].name}")
    return df


def filter_annotations(df: pd.DataFrame, size_map: dict, config: dict):
    """
    Фильтрация «плохих» боксов.

    Возвращает:
      clean_df, rejected_df, stats, rejected_images
    """
    fconf = config["preprocessing"]["filtering"]

    if not fconf.get("enabled", True):
        logger.info("Фильтрация отключена — пропускаем")
        df = df.copy()
        # Добавим img_W/img_H и area_frac для единообразия
        df["img_W"] = df["patientId"].astype(str).map(lambda x: size_map.get(x, (None, None))[0])
        df["img_H"] = df["patientId"].astype(str).map(lambda x: size_map.get(x, (None, None))[1])
        df["area_frac"] = (df["width"] * df["height"]) / (df["img_W"] * df["img_H"])
        return df, pd.DataFrame(), {"filtering_enabled": False}, set()

    min_area = fconf["min_box_area_frac"]
    min_aspect = fconf["min_aspect_ratio"]
    max_aspect = fconf["max_aspect_ratio"]
    allow_oob = fconf["allow_out_of_bounds"]

    keep_rows, reject_rows = [], []
    reasons_count = {
        "image_not_found": 0,
        "non_positive_size": 0,
        "out_of_bounds": 0,
        "area_too_small": 0,
        "aspect_too_high": 0,
        "aspect_too_low": 0,
    }

    for _, row in df.iterrows():
        pid = str(row["patientId"])

        if pid not in size_map:
            reject_rows.append({**row.to_dict(), "reject_reason": "image_not_found"})
            reasons_count["image_not_found"] += 1
            continue

        W, H = size_map[pid]
        x, y, w, h = float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])
        reasons = []

        # 1. неположительные размеры
        if w <= 0 or h <= 0:
            reasons.append("non_positive_size")
            reasons_count["non_positive_size"] += 1

        # 2. выход за границы
        oob = (x < 0) or (y < 0) or (x + w > W) or (y + h > H)
        if oob and not reasons:
            if allow_oob:
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(W), x + w), min(float(H), y + h)
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    reasons.append("out_of_bounds_after_clip")
                    reasons_count["out_of_bounds"] += 1
            else:
                reasons.append("out_of_bounds")
                reasons_count["out_of_bounds"] += 1

        # 3. площадь
        if not reasons:
            area_frac = (w * h) / (W * H) if W * H > 0 else 0.0
            if area_frac < min_area:
                reasons.append(f"area_too_small:{area_frac:.6f}")
                reasons_count["area_too_small"] += 1

        # 4. aspect ratio
        if not reasons and h > 0:
            aspect = w / h
            if aspect > max_aspect:
                reasons.append(f"aspect_too_high:{aspect:.2f}")
                reasons_count["aspect_too_high"] += 1
            elif aspect < min_aspect:
                reasons.append(f"aspect_too_low:{aspect:.2f}")
                reasons_count["aspect_too_low"] += 1

        if reasons:
            reject_rows.append({**row.to_dict(), "reject_reason": "; ".join(reasons)})
        else:
            new_row = row.to_dict()
            new_row["x"], new_row["y"] = x, y
            new_row["width"], new_row["height"] = w, h
            new_row["img_W"], new_row["img_H"] = W, H
            new_row["area_frac"] = (w * h) / (W * H)
            keep_rows.append(new_row)

    clean_df = pd.DataFrame(keep_rows)
    rejected_df = pd.DataFrame(reject_rows)

    # Находим изображения, у которых были боксы, но после фильтрации не осталось ни одного
    rejected_images = set()
    if len(rejected_df) and fconf.get("drop_images_with_all_boxes_rejected", True):
        original_pids = set(df["patientId"].astype(str).unique())
        clean_pids = set(clean_df["patientId"].astype(str).unique()) if len(clean_df) else set()
        rejected_pids = set(rejected_df["patientId"].astype(str).unique())
        # У кого не осталось валидных боксов И были отвергнутые
        rejected_images = (rejected_pids - clean_pids) & original_pids
        logger.warning(
            f"Изображений с полностью отвергнутой разметкой: {len(rejected_images)}. "
            f"Будут удалены из staged."
        )

    stats = {
        "filtering_enabled": True,
        "thresholds": {
            "min_box_area_frac": min_area,
            "min_aspect_ratio": min_aspect,
            "max_aspect_ratio": max_aspect,
            "allow_out_of_bounds": allow_oob,
        },
        "total_input_boxes": int(len(df)),
        "kept_boxes": int(len(clean_df)),
        "rejected_boxes": int(len(rejected_df)),
        "rejected_by_reason": reasons_count,
        "images_with_all_boxes_rejected": len(rejected_images),
    }

    return clean_df, rejected_df, stats, rejected_images


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    raw_dir = Path(config["paths"]["raw_dir"])
    staged_dir = Path(config["paths"]["staged_dir"])

    # 1. Изображения
    size_map = preprocess_images(raw_dir, staged_dir, config)

    # 2. Аннотации
    df_raw = load_raw_annotations(raw_dir)

    # 3. Фильтрация
    logger.info("Фильтрация аннотаций...")
    clean_df, rejected_df, stats, rejected_images = filter_annotations(df_raw, size_map, config)

    # # 4. Удаляем изображения с полностью отвергнутой разметкой
    # if rejected_images:
    #     removed = 0
    #     for pid in rejected_images:
    #         for ext in (".jpg", ".jpeg", ".png"):
    #             p = staged_dir / "images" / f"{pid}{ext}"
    #             if p.exists():
    #                 p.unlink()
    #                 removed += 1
    #     logger.info(f"Удалено изображений из staged: {removed}")
    #     stats["removed_images"] = removed
    # else:
    #     stats["removed_images"] = 0

    # 4. Перемещаем полностью отвергнутые изображения в staged/rejected_images/
    #    (не удаляем — чтобы collect_rejected.py мог их найти и показать)
    if rejected_images:
        rejected_dir = staged_dir / "rejected_images"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for pid in rejected_images:
            for ext in (".jpg", ".jpeg", ".png"):
                p = staged_dir / "images" / f"{pid}{ext}"
                if p.exists():
                    shutil.move(str(p), str(rejected_dir / p.name))
                    moved += 1
        logger.info(f"Перемещено изображений в staged/rejected_images: {moved}")
        stats["moved_to_rejected"] = moved
    else:
        stats["moved_to_rejected"] = 0

    # 5. Сохраняем результаты
    clean_path = staged_dir / "annotations.csv"
    clean_df.to_csv(clean_path, index=False)
    logger.info(f"Очищенные аннотации: {clean_path} ({len(clean_df)} строк)")

    if config["preprocessing"]["filtering"].get("save_rejected", True) and len(rejected_df):
        rejected_path = staged_dir / "rejected_annotations.csv"
        rejected_df.to_csv(rejected_path, index=False)
        logger.info(f"Отвергнутые аннотации: {rejected_path} ({len(rejected_df)} строк)")

    report_path = staged_dir / "filtering_report.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Отчёт фильтрации: {report_path}")

    # 6. Итоговая сводка в лог
    logger.info("=" * 60)
    logger.info("ИТОГИ ФИЛЬТРАЦИИ")
    logger.info("=" * 60)
    logger.info(f"Входных боксов:  {stats['total_input_boxes']}")
    logger.info(f"Оставлено:       {stats['kept_boxes']}")
    logger.info(f"Отвергнуто:      {stats['rejected_boxes']}")
    logger.info(f"Удалено изображений: {stats['removed_images']}")
    for reason, cnt in stats["rejected_by_reason"].items():
        if cnt:
            logger.info(f"  {reason}: {cnt}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()






    # # 4. Перемещаем полностью отвергнутые изображения в staged/rejected_images/
    # #    (не удаляем — чтобы collect_rejected.py мог их найти и показать)
    # if rejected_images:
    #     rejected_dir = staged_dir / "rejected_images"
    #     rejected_dir.mkdir(parents=True, exist_ok=True)
    #     moved = 0
    #     for pid in rejected_images:
    #         for ext in (".jpg", ".jpeg", ".png"):
    #             p = staged_dir / "images" / f"{pid}{ext}"
    #             if p.exists():
    #                 shutil.move(str(p), str(rejected_dir / p.name))
    #                 moved += 1
    #     logger.info(f"Перемещено изображений в staged/rejected_images: {moved}")
    #     stats["moved_to_rejected"] = moved
    # else:
    #     stats["moved_to_rejected"] = 0