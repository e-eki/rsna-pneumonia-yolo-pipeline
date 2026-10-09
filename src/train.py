"""
Обучение YOLO-модели для детекции пневмонии на датасете RSNA.

Скрипт загружает предобученные веса YOLO, обучает модель на датасете
из data/processed/ (собранном prepare_yolo_dataset.py) и сохраняет
логи и веса в runs/train/<run_name>/.

Все гиперпараметры читаются из configs/config.yaml (секция training).
Имя запуска задаётся через --run-name; если не указано — генерируется
автоматически по дате.

Примеры запуска:
    # Smoke test — быстрая проверка, что пайплайн работает
    python -m src.train --config configs/config.yaml \\
        --run-name "smoke_test_yolov8n_3ep"

    # Полное обучение
    python -m src.train --config configs/config.yaml \\
        --run-name "2026-10-02_yolov8s_30ep_no_mosaic"

    # Без имени — папка создастся с датой и временем
    python -m src.train --config configs/config.yaml
"""
import argparse
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from src.utils import ensure_dirs, get_logger, load_config, setup_logging

logger = get_logger(__name__)


def train_yolo(config: dict, run_name: str):
    """
    Обучает YOLO с параметрами из конфига.

    Args:
        config:   словарь из configs/config.yaml.
        run_name: имя папки внутри runs/train/. Используется как есть —
                  лучше давать осмысленные имена (например,
                  "2026-10-02_yolov8s_30ep_no_mosaic").
    """
    processed_dir = Path(config["paths"]["processed_dir"])
    data_yaml = processed_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml}")

    tconf = config["training"]

    logger.info(f"Загрузка модели: {tconf['model']}")
    model = YOLO(tconf["model"])

    # Логи и веса складываем в runs/train/<run_name>/.
    project_dir = Path(config["paths"]["runs_dir"]).resolve() / "train"
    project_dir.mkdir(parents=True, exist_ok=True)

    train_args = {
        "data":         str(data_yaml),
        "epochs":       tconf["epochs"],
        "batch":        tconf["batch_size"],
        "imgsz":        tconf["imgsz"],
        "patience":     tconf["patience"],
        "workers":      tconf["workers"],
        "device":       tconf["device"],
        "save_period":  tconf.get("save_period", -1),
        "optimizer":    tconf["optimizer"],
        "lr0":          tconf["lr0"],
        "lrf":          tconf["lrf"],
        "weight_decay": tconf["weight_decay"],
        "project":      str(project_dir),
        "name":         run_name,
        "exist_ok":     True,   # перезаписать папку, если имя уже занято
        "pretrained":   True,
        "verbose":      True,
    }

    # Гиперпараметры аугментаций (hsv_*, degrees, mosaic, ...) — из конфига.
    train_args.update(tconf["augment"])

    logger.info(f"Начало обучения: {train_args['epochs']} эпох, "
                f"batch={train_args['batch']}, run_name='{run_name}'")
    results = model.train(**train_args)

    logger.info(f"Обучение завершено. Результаты: {results.save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-name", default=None,
                        help="Имя папки в runs/train/. Если не задано — "
                             "генерируется автоматически с датой и временем.")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    train_yolo(config, run_name=run_name)


if __name__ == "__main__":
    main()