# src/split_data.py
"""Разделение данных на train/val и создание YOLO-разметки.

Скрипт создает финальную структуру для YOLO: делит 
данные на train/val и конвертирует аннотации в 
формат YOLO (нормализованные координаты)"""
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from src.utils import load_config, ensure_dirs, setup_logging, get_logger

logger = get_logger(__name__)

def convert_to_yolo_format(row, img_width: int, img_height: int) -> str:
    """
    Конвертация аннотации в формат YOLO:
    <class_id> <x_center> <y_center> <width> <height>
    Все координаты нормализованы [0, 1].
    """
    # Для RSNA: Target=1 означает пневмонию, class_id = 0
    if row.get("Target", 1) != 1:
        return None
    
    x, y, w, h = row["x"], row["y"], row["width"], row["height"]
    
    # Нормализация
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    
    # Клиппинг для безопасности
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_w = max(0, min(1, norm_w))
    norm_h = max(0, min(1, norm_h))
    
    return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

def create_yolo_dataset(staged_dir: Path, processed_dir: Path, config: dict):
    """Создание структуры YOLO-датасета."""
    images_dir = staged_dir / "images"
    annotations_file = staged_dir / "annotations.csv"
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"Аннотации не найдены: {annotations_file}")
    
    df = pd.read_csv(annotations_file)
    logger.info(f"Загружено {len(df)} аннотаций")
    
    # Уникальные изображения
    unique_images = df["patientId"].unique()
    logger.info(f"Уникальных изображений: {len(unique_images)}")
    
    # Разделение
    train_ids, val_ids = train_test_split(
        unique_images,
        test_size=config["dataset"]["val_split"],
        random_state=config["project"]["seed"]
    )
    
    # Создание директорий
    for split in ["train", "val"]:
        (processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Обработка каждого сплита
    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        for patient_id in split_ids:
            img_src = images_dir / f"{patient_id}.jpg"
            if not img_src.exists():
                img_src = images_dir / f"{patient_id}.png"
            if not img_src.exists():
                logger.warning(f"Изображение не найдено: {patient_id}")
                continue
            
            # Копирование изображения
            img_dst = processed_dir / "images" / split_name / img_src.name
            shutil.copy2(img_src, img_dst)
            
            # Получение размеров изображения
            from PIL import Image
            with Image.open(img_src) as img:
                width, height = img.size
            
            # Создание YOLO-разметки
            patient_annots = df[df["patientId"] == patient_id]
            yolo_lines = []
            for _, row in patient_annots.iterrows():
                yolo_line = convert_to_yolo_format(row, width, height)
                if yolo_line:
                    yolo_lines.append(yolo_line)
            
            # Сохранение .txt (пустой файл = нет объектов)
            label_path = processed_dir / "labels" / split_name / f"{img_src.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
        
        logger.info(f"{split_name}: {len(split_ids)} изображений")
    
    # Создание data.yaml для YOLO
    data_yaml = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["pneumonia"]
    }
    
    import yaml
    with open(processed_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    logger.info(f"data.yaml создан: {processed_dir / 'data.yaml'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    
    staged_dir = Path(config["paths"]["staged_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    
    create_yolo_dataset(staged_dir, processed_dir, config)

if __name__ == "__main__":
    main()