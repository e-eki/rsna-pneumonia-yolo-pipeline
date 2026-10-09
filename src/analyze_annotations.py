"""
Статистика по разметке: графики и проверки на выход за границы.

Поддерживает три стадии:
    --stage raw        — CSV + изображения в data/raw (пиксельные координаты);
    --stage staged     — CSV + изображения в data/staged (пиксельные координаты);
    --stage processed  — YOLO-датасет в data/processed (нормализованные координаты).

Что сохраняется в data/reports/<stage>[_<split>]/:
    boxes_per_image.png    — распределение числа боксов на изображение;
    box_sizes.png          — распределение площади и aspect ratio боксов;
    class_balance.png      — баланс классов;
    pixel_out_of_bounds.txt (для raw/staged) — список изображений с OOB-боксами;
    out_of_bounds.csv      (для processed)   — боксы вне [0, 1];
    report.json            — сводка в машинно-читаемом виде.

Примеры запуска:
    python -m src.analyze_annotations --stage raw
    python -m src.analyze_annotations --stage processed --split train
    python -m src.analyze_annotations --stage processed --split val
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-режим — важно для Colab и Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import get_logger, load_config, setup_logging

logger = get_logger(__name__)

CLASS_NAMES = {0: "pneumonia"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ─────────────────────────── размеры изображений ───────────────────────────

def _image_size(path: Path):
    """
    Возвращает (W, H) для jpg/png или DICOM.
    При ошибке чтения — (None, None).
    """
    suffix = path.suffix.lower()

    if suffix in IMG_EXTS:
        try:
            with Image.open(path) as im:
                return im.size
        except Exception as e:
            logger.warning(f"Не удалось прочитать {path}: {e}")
            return (None, None)

    if suffix == ".dcm":
        try:
            import pydicom
            arr = pydicom.dcmread(str(path)).pixel_array
            return int(arr.shape[1]), int(arr.shape[0])
        except Exception as e:
            logger.warning(f"DICOM read failed {path}: {e}")
            return (None, None)

    return (None, None)


def _collect_size_map(images_dir: Path) -> dict:
    """Собирает {stem: (W, H)} для всех изображений в директории."""
    size_map = {}
    if not images_dir.exists():
        return size_map

    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS or p.suffix.lower() == ".dcm":
            size_map[p.stem] = _image_size(p)
    return size_map


# ─────────────────────────── сбор данных ───────────────────────────

def collect_pixel_annotations(root: Path):
    """
    Читает пиксельные аннотации из raw/ или staged/.

    Возвращает (df, size_map, all_stems):
        df        — DataFrame с колонками x, y, width, height, area_frac, aspect;
        size_map  — {stem: (W, H)};
        all_stems — множество имён всех изображений.
    """
    csvs = [c for c in root.rglob("*.csv") if "rejected" not in c.name]
    if not csvs:
        raise FileNotFoundError(f"CSV с аннотациями не найден в {root}")
    if len(csvs) > 1:
        logger.info(f"Найдено несколько CSV: {[c.name for c in csvs]}. "
                    f"Берём {csvs[0].name}")

    ann_file = csvs[0]
    df = pd.read_csv(ann_file)
    logger.info(f"Загружено {len(df)} строк из {ann_file}")

    if "Target" in df.columns:
        df = df[df["Target"] == 1].copy()
        logger.info(f"После фильтра Target==1: {len(df)} боксов")

    images_dir = root / "images" if (root / "images").exists() else root
    size_map = _collect_size_map(images_dir)
    all_stems = set(size_map.keys())
    logger.info(f"Найдено изображений: {len(size_map)}")

    # Дополняем df вычисляемыми колонками для графиков.
    df["img_W"] = df["patientId"].astype(str).map(
        lambda x: size_map.get(x, (None, None))[0])
    df["img_H"] = df["patientId"].astype(str).map(
        lambda x: size_map.get(x, (None, None))[1])
    df["area_px"] = df["width"] * df["height"]
    df["area_frac"] = df["area_px"] / (df["img_W"] * df["img_H"])
    df["aspect"] = df["width"] / df["height"].replace(0, np.nan)
    df["cls"] = 0
    df = df.rename(columns={"patientId": "stem"})
    df["area_norm"] = df["area_frac"]

    return df, size_map, all_stems


def collect_processed(processed_dir: Path, split: str):
    """
    Читает YOLO-разметку из data/processed/<split>/.

    Возвращает (df, sizes, all_stems) в том же формате, что
    collect_pixel_annotations, но с нормализованными координатами.
    """
    images_dir = processed_dir / "images" / split
    labels_dir = processed_dir / "labels" / split

    if not images_dir.exists():
        raise FileNotFoundError(f"Нет папки {images_dir}")

    sizes = {}
    all_stems = set()
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() in IMG_EXTS:
            all_stems.add(img_path.stem)
            sizes[img_path.stem] = _image_size(img_path)

    rows = []
    for stem in all_stems:
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = map(float, parts)
                rows.append({
                    "stem": stem, "cls": int(cls),
                    "xc": xc, "yc": yc, "w": w, "h": h,
                })

    df = pd.DataFrame(rows)
    logger.info(f"Изображений: {len(all_stems)}, боксов: {len(df)}")
    return df, sizes, all_stems


def check_pixel_out_of_bounds(df, size_map):
    """
    Возвращает set(stem) изображений, у которых хотя бы один бокс
    выходит за границы кадра или имеет неположительные размеры.
    """
    bad = set()
    for _, row in df.iterrows():
        pid = str(row["stem"])
        if pid not in size_map:
            continue
        W, H = size_map[pid]
        if W is None or H is None:
            continue

        x, y, w, h = row["x"], row["y"], row["width"], row["height"]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
            bad.add(pid)
    return bad


# ─────────────────────────── графики ───────────────────────────

def plot_boxes_per_image(df, sizes, all_stems, out_path, title):
    """
    Гистограмма числа боксов на изображение + столбики positive/negative.
    Возвращает словарь со счётчиками для report.json.
    """
    counts = df.groupby("stem").size() if len(df) else pd.Series(dtype=int)

    total_images = len(all_stems) if all_stems else len(sizes)
    num_with_boxes = int(counts.shape[0]) if len(counts) else 0
    num_empty = max(0, total_images - num_with_boxes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if len(counts):
        axes[0].hist(counts.values, bins=range(1, int(counts.max()) + 2),
                     edgecolor="black", align="left")
        axes[0].set_title("Боксов на изображение (только с объектами)")
        axes[0].set_xlabel("Число боксов")
        axes[0].set_ylabel("Число изображений")
    else:
        axes[0].text(0.5, 0.5, "Нет боксов", ha="center", va="center")
        axes[0].set_title("Боксов на изображение")

    axes[1].bar(["с объектами", "без объектов"], [num_with_boxes, num_empty],
                color=["#2ca02c", "#d62728"])
    axes[1].set_title(f"Positive / Negative (всего {total_images})")
    axes[1].set_ylabel("Число изображений")
    for i, v in enumerate([num_with_boxes, num_empty]):
        axes[1].text(i, v + 0.5, str(v), ha="center")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

    return {
        "total_images": total_images,
        "with_boxes": num_with_boxes,
        "without_boxes": num_empty,
    }


def plot_box_sizes(df, out_path, title):
    """
    Гистограммы площади бокса и aspect ratio.
    Возвращает сводную статистику по площади для report.json.
    """
    if not len(df):
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    values = df["area_norm"].dropna() if "area_norm" in df.columns \
        else pd.Series(dtype=float)
    if len(values):
        axes[0].hist(values, bins=40, edgecolor="black")
        axes[0].set_title("Площадь бокса (доля от площади изображения)")
        axes[0].set_xlabel("доля")
        axes[0].set_ylabel("Число боксов")

    if "aspect" in df.columns:
        ar = df["aspect"].dropna()
        ar = ar[(ar > 0.05) & (ar < 20)]  # отсекаем выбросы для читаемости
        if len(ar):
            axes[1].hist(ar, bins=40, edgecolor="black")
            axes[1].set_title("Aspect ratio (w/h)")
            axes[1].set_xlabel("w / h")
            axes[1].set_ylabel("Число боксов")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

    return {
        "area_mean":   float(values.mean())   if len(values) else None,
        "area_median": float(values.median()) if len(values) else None,
        "area_min":    float(values.min())    if len(values) else None,
        "area_max":    float(values.max())    if len(values) else None,
    }


def plot_class_balance(df, out_path, title):
    """Столбики по классам. Возвращает {имя_класса: число_боксов}."""
    if not len(df):
        return {}

    counts = df["cls"].value_counts().sort_index()
    labels = [CLASS_NAMES.get(int(c), str(c)) for c in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts.values, color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel("Число боксов")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

    return {CLASS_NAMES.get(int(c), str(c)): int(v) for c, v in counts.items()}


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", required=True,
                        choices=["raw", "staged", "processed"])
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Только для --stage processed")
    parser.add_argument("--output", default=None,
                        help="Куда сохранять. По умолчанию "
                             "data/reports/<stage>[_<split>]/")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    # ── Куда сохранять ──
    if args.output:
        out_dir = Path(args.output)
    elif args.stage == "processed":
        out_dir = Path("data/reports") / f"processed_{args.split}"
    else:
        out_dir = Path("data/reports") / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Источник данных ──
    if args.stage in ("raw", "staged"):
        root = Path(config["paths"]["raw_dir"]) if args.stage == "raw" \
            else Path(config["paths"]["staged_dir"])
        df, sizes, all_stems = collect_pixel_annotations(root)
        title = f"{args.stage.upper()} (пиксельные координаты)"
    else:
        processed_dir = Path(config["paths"]["processed_dir"])
        df, sizes, all_stems = collect_processed(processed_dir, args.split)
        if len(df):
            df["area_norm"] = df["w"] * df["h"]
            df["aspect"] = df["w"] / df["h"].replace(0, np.nan)
        title = f"PROCESSED (YOLO, split={args.split})"

    logger.info(f"Боксов: {len(df)}")

    report = {"stage": args.stage, "split": args.split, "num_boxes": int(len(df))}

    # ── Графики ──
    report["boxes_per_image"] = plot_boxes_per_image(
        df, sizes, all_stems, out_dir / "boxes_per_image.png", title)
    report["box_sizes"] = plot_box_sizes(
        df, out_dir / "box_sizes.png", title)
    report["class_balance"] = plot_class_balance(
        df, out_dir / "class_balance.png", title)

    # ── Проверка границ: raw/staged (пиксели) ──
    if args.stage in ("raw", "staged") and len(df):
        oob_images = check_pixel_out_of_bounds(df, sizes)
        report["pixel_out_of_bounds_images"] = len(oob_images)
        if oob_images:
            logger.warning(f"⚠ {len(oob_images)} изображений с боксами вне границ")
            path = out_dir / "pixel_out_of_bounds.txt"
            path.write_text("\n".join(sorted(oob_images)))
            logger.warning(f"Список: {path}")
        else:
            logger.info("✅ Все пиксельные боксы внутри границ изображений")

    # ── Проверка границ: processed (YOLO [0,1]) ──
    if args.stage == "processed" and len(df):
        oob = ((df["xc"] < 0) | (df["xc"] > 1) |
               (df["yc"] < 0) | (df["yc"] > 1) |
               (df["w"] <= 0) | (df["w"] > 1) |
               (df["h"] <= 0) | (df["h"] > 1))
        report["yolo_out_of_bounds"] = int(oob.sum())
        if oob.sum() > 0:
            logger.warning(f"⚠ Найдено {oob.sum()} боксов вне [0, 1]")
            path = out_dir / "out_of_bounds.csv"
            df[oob].to_csv(path, index=False)
            logger.warning(f"Сохранены в {path}")
        else:
            logger.info("✅ Все YOLO-координаты в допустимых диапазонах")

    # ── Отчёт ──
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Отчёт:   {out_dir / 'report.json'}")
    logger.info(f"Графики: {out_dir}")


if __name__ == "__main__":
    main()