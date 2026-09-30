# src/train.py
"""Обучение модели YOLO на датасете RSNA Pneumonia."""
import argparse
from pathlib import Path
from ultralytics import YOLO
from src.utils import load_config, ensure_dirs, setup_logging, get_logger

logger = get_logger(__name__)

def train_yolo(config: dict):
    """Обучение YOLO с параметрами из конфига."""
    processed_dir = Path(config["paths"]["processed_dir"])
    data_yaml = processed_dir / "data.yaml"
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml}")
    
    # Загрузка предобученной модели
    model_name = config["training"]["model"]
    logger.info(f"Загрузка модели: {model_name}")
    model = YOLO(model_name)
    
    # Параметры обучения
    train_args = {
        "data": str(data_yaml),
        "epochs": config["training"]["epochs"],
        "batch": config["training"]["batch_size"],
        "imgsz": config["training"]["imgsz"],
        "patience": config["training"]["patience"],
        "workers": config["training"]["workers"],
        "device": config["training"]["device"],
        "optimizer": config["training"]["optimizer"],
        "lr0": config["training"]["lr0"],
        "lrf": config["training"]["lrf"],
        "weight_decay": config["training"]["weight_decay"],
        "project": config["paths"]["runs_dir"],
        "name": "train",
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
    }
    
    # Добавление аугментаций
    train_args.update(config["training"]["augment"])
    
    logger.info(f"Начало обучения: {train_args['epochs']} эпох, batch={train_args['batch']}")
    results = model.train(**train_args)
    
    logger.info(f"Обучение завершено. Результаты: {results.save_dir}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    
    train_yolo(config)

if __name__ == "__main__":
    main()