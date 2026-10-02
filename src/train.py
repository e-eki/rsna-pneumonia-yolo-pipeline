"""Обучение модели YOLO на датасете RSNA Pneumonia."""
import argparse
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from src.utils import load_config, ensure_dirs, setup_logging, get_logger

logger = get_logger(__name__)


def train_yolo(config: dict, run_name: str = "train"):
    """Обучение YOLO с параметрами из конфига.

    run_name — имя папки в runs/, например:
      "smoke_yolov8n_3ep",
      "full_yolov8s_30ep_no_mosaic",
      "exp_yolov8s_30ep_mosaic_0.3".
    """
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
        "project": str(Path(config["paths"]["runs_dir"]).resolve()),
        # ── ИСПРАВЛЕНО: имя запуска приходит извне ──
        "name": run_name,
        "exist_ok": True,       # True → перезаписать, если папка с таким именем есть
        "pretrained": True,
        "verbose": True,
    }

    # Добавление аугментаций
    train_args.update(config["training"]["augment"])

    logger.info(f"Начало обучения: {train_args['epochs']} эпох, "
                f"batch={train_args['batch']}, run_name='{run_name}'")
    results = model.train(**train_args)

    logger.info(f"Обучение завершено. Результаты: {results.save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-name", default=None,
                    help="Имя папки в runs/. Если не задано — генерируется автоматически "
                         "с датой и временем.")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    train_yolo(config, run_name=run_name)
    # train_yolo(config, run_name=args.run_name)


if __name__ == "__main__":
    main()