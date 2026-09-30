# src/analyze_annotations.py
"""
Статистика по разметке. Поддерживает три стадии: raw, staged, processed.

Сохраняет графики:
  - boxes per image
  - box area (px и % от площади)
  - aspect ratio
  - баланс классов
  - positive / negative
  - out_of_bounds (пиксели — для raw/staged, [0,1] — для processed)

Примеры:
  python src/analyze_annotations.py --stage raw
  python src/analyze_annotations.py --stage staged
  python src/analyze_annotations.py --stage processed --split train
  python src/analyze_annotations.py --stage processed --split val

Считает статистику по разметке и сохраняет графики в data/reports/:
Распределение числа боксов на изображение — сколько объектов обычно на кадре.
Распределение размеров боксов (в пикселях и в % от площади изображения) — ловит «мусорные» крошечные или гигантские боксы.
Баланс классов (для RSNA — 1 класс, но скрипт универсальный).
Баланс positive/negative — сколько изображений без объектов.
Проверка выхода координат за пределы [0, 1] для YOLO-формата — критично для обучения.
Распределение aspect ratio боксов.

Работает для двух стадий: staged (CSV, пиксельные координаты) и processed (YOLO-разметка).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — важно для Colab/Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import load_config, setup_logging, get_logger

logger = get_logger(__name__)

CLASS_NAMES = {0: "pneumonia"}


# ------------------------- чтение размеров -------------------------

def _image_size(path: Path):
    """Возвращает (W, H) для jpg/png или DICOM."""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        try:
            with Image.open(path) as im:
                return im.size
        except Exception as e:
            logger.warning(f"Не удалось прочитать {path}: {e}")
            return (None, None)
    if suffix == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array
            return int(arr.shape[1]), int(arr.shape[0])
        except Exception as e:
            logger.warning(f"DICOM read failed {path}: {e}")
            return (None, None)
    return (None, None)


def _collect_size_map(images_dir: Path) -> dict:
    size_map = {}
    if not images_dir.exists():
        return size_map
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".dcm")
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in exts:
            size_map[p.stem] = _image_size(p)
    return size_map


# ------------------------- сбор данных -------------------------

def collect_pixel_annotations(root: Path):
    """
    Читает пиксельные аннотации из raw/ или staged/.
    Возвращает (df, size_map, all_stems).
    """
    csvs = [c for c in root.rglob("*.csv") if "rejected" not in c.name]
    if not csvs:
        raise FileNotFoundError(f"CSV с аннотациями не найден в {root}")
    if len(csvs) > 1:
        logger.info(f"Найдено несколько CSV: {[c.name for c in csvs]}. Берём {csvs[0].name}")
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

    df["img_W"] = df["patientId"].astype(str).map(lambda x: size_map.get(x, (None, None))[0])
    df["img_H"] = df["patientId"].astype(str).map(lambda x: size_map.get(x, (None, None))[1])
    df["area_px"] = df["width"] * df["height"]
    df["area_frac"] = df["area_px"] / (df["img_W"] * df["img_H"])
    df["aspect"] = df["width"] / df["height"].replace(0, np.nan)
    df["cls"] = 0
    df = df.rename(columns={"patientId": "stem"})
    df["area_norm"] = df["area_frac"]
    return df, size_map, all_stems


def collect_processed(processed_dir: Path, split: str):
    """YOLO-боксы + размеры изображений."""
    images_dir = processed_dir / "images" / split
    labels_dir = processed_dir / "labels" / split

    if not images_dir.exists():
        raise FileNotFoundError(f"Нет папки {images_dir}")

    rows = []
    sizes = {}
    all_stems = set()

    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        all_stems.add(img_path.stem)
        sizes[img_path.stem] = _image_size(img_path)

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
    """Возвращает set(patientId), где бокс выходит за границы или имеет неположительные размеры."""
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


# ------------------------- графики -------------------------

def plot_boxes_per_image(df, sizes, all_stems, out_path, title):
    key = "stem" if "stem" in df.columns else None
    counts = df.groupby(key).size() if key and len(df) else pd.Series(dtype=int)

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
    if not len(df):
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    values = df["area_norm"].dropna() if "area_norm" in df.columns else pd.Series(dtype=float)
    if len(values):
        axes[0].hist(values, bins=40, edgecolor="black")
        axes[0].set_title("Площадь бокса (доля от площади изображения)")
        axes[0].set_xlabel("доля")
        axes[0].set_ylabel("Число боксов")

    if "aspect" in df.columns:
        ar = df["aspect"].dropna()
        ar = ar[(ar > 0.05) & (ar < 20)]
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
        "area_mean": float(values.mean()) if len(values) else None,
        "area_median": float(values.median()) if len(values) else None,
        "area_min": float(values.min()) if len(values) else None,
        "area_max": float(values.max()) if len(values) else None,
    }


def plot_class_balance(df, out_path, title):
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


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", required=True, choices=["raw", "staged", "processed"])
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    # Куда сохранять отчёт
    if args.output:
        out_dir = Path(args.output)
    elif args.stage == "processed":
        out_dir = Path("data/reports") / f"processed_{args.split}"
    else:
        out_dir = Path("data/reports") / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- источник данных ----
    if args.stage in ("raw", "staged"):
        root = Path(config["paths"]["raw_dir"]) if args.stage == "raw" \
               else Path(config["paths"]["staged_dir"])
        df, sizes, all_stems = collect_pixel_annotations(root)
        title = f"{args.stage.upper()} (пиксельные координаты)"
    else:  # processed
        processed_dir = Path(config["paths"]["processed_dir"])
        df, sizes, all_stems = collect_processed(processed_dir, args.split)
        if len(df):
            df["area_norm"] = df["w"] * df["h"]
            df["aspect"] = df["w"] / df["h"].replace(0, np.nan)
        title = f"PROCESSED (YOLO, split={args.split})"

    logger.info(f"Боксов: {len(df)}")

    report = {"stage": args.stage, "split": args.split, "num_boxes": int(len(df))}

    # ---- графики ----
    report["boxes_per_image"] = plot_boxes_per_image(
        df, sizes, all_stems, out_dir / "boxes_per_image.png", title)
    report["box_sizes"] = plot_box_sizes(df, out_dir / "box_sizes.png", title)
    report["class_balance"] = plot_class_balance(df, out_dir / "class_balance.png", title)

    # ---- проверка границ: raw/staged (пиксели) ----
    if args.stage in ("raw", "staged") and len(df):
        oob_images = check_pixel_out_of_bounds(df, sizes)
        report["pixel_out_of_bounds_images"] = len(oob_images)
        if oob_images:
            logger.warning(f"⚠️ {len(oob_images)} изображений с боксами вне границ")
            with open(out_dir / "pixel_out_of_bounds.txt", "w") as f:
                f.write("\n".join(sorted(oob_images)))
            logger.warning(f"Список: {out_dir / 'pixel_out_of_bounds.txt'}")
        else:
            logger.info("✅ Все пиксельные боксы внутри границ изображений")

    # ---- проверка границ: processed (YOLO [0,1]) ----
    if args.stage == "processed" and len(df):
        oob = ((df["xc"] < 0) | (df["xc"] > 1) |
               (df["yc"] < 0) | (df["yc"] > 1) |
               (df["w"] <= 0) | (df["w"] > 1) |
               (df["h"] <= 0) | (df["h"] > 1))
        report["yolo_out_of_bounds"] = int(oob.sum())
        if oob.sum() > 0:
            logger.warning(f"⚠️ Найдено {oob.sum()} боксов вне [0,1]!")
            df[oob].to_csv(out_dir / "out_of_bounds.csv", index=False)
            logger.warning(f"Сохранены в {out_dir / 'out_of_bounds.csv'}")
        else:
            logger.info("✅ Все YOLO-координаты в допустимых диапазонах")

    # ---- отчёт ----
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Отчёт: {out_dir / 'report.json'}")
    logger.info(f"Графики: {out_dir}")


if __name__ == "__main__":
    main()