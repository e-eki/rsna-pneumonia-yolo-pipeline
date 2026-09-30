# src/evaluate.py
"""Оценка обученной модели на валидационной выборке."""
import argparse
from pathlib import Path
from ultralytics import YOLO
from src.utils import load_config, setup_logging, get_logger

logger = get_logger(__name__)

def evaluate_model(config: dict, model_path: str = None):
    """Оценка модели на val-выборке."""
    processed_dir = Path(config["paths"]["processed_dir"])
    data_yaml = processed_dir / "data.yaml"
    
    if model_path is None:
        # Ищем последнюю обученную модель
        runs_dir = Path(config["paths"]["runs_dir"])
        best_models = list(runs_dir.glob("**/weights/best.pt"))
        if not best_models:
            raise FileNotFoundError("Обученные модели не найдены")
        model_path = str(sorted(best_models)[-1])
    
    logger.info(f"Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    logger.info("Запуск валидации...")
    metrics = model.val(
        data=str(data_yaml),
        conf=config["evaluation"]["conf_threshold"],
        iou=config["evaluation"]["iou_threshold"],
        save_json=True,
    )
    
    # Вывод метрик
    logger.info("=" * 50)
    logger.info("МЕТРИКИ НА ВАЛИДАЦИИ")
    logger.info("=" * 50)
    logger.info(f"mAP@0.5:      {metrics.box.map50:.4f}")
    logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    logger.info(f"Precision:    {metrics.box.mp:.4f}")
    logger.info(f"Recall:       {metrics.box.mr:.4f}")
    logger.info("=" * 50)
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, help="Путь к весам модели")
    args = parser.parse_args()
    
    setup_logging()
    config = load_config(args.config)
    
    evaluate_model(config, args.model)

if __name__ == "__main__":
    main()