"""
Оценка обученной YOLO-модели на валидационной выборке.

Скрипт загружает веса (по умолчанию — последняя best.pt в runs/train/),
прогоняет валидацию на data/processed и сохраняет графики и JSON
с предсказаниями в runs/val/<run_name>/.

Имя папки для результатов автоматически выводится из пути к модели:
    runs/train/<run_name>/weights/best.pt  →  runs/val/<run_name>/

Метрики mAP считаются со стандартными для COCO порогами (conf=0.001,
iou=0.6) — так результаты совпадают с последней эпохой обучения.
Пороги инференса (conf=0.25, iou=0.45) лежат отдельно и используются
в predict.py / демо.

Примеры запуска:
    # Последняя обученная модель
    python -m src.evaluate --config configs/config.yaml

    # Конкретная модель
    python -m src.evaluate --config configs/config.yaml \\
        --model runs/train/2026-10-02_full_yolov8s/weights/best.pt

    # Переопределить имя папки результатов
    python -m src.evaluate --config configs/config.yaml \\
        --model runs/train/smoke/weights/best.pt \\
        --run-name smoke_custom
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from src.utils import ensure_dirs, get_logger, load_config, setup_logging

logger = get_logger(__name__)


def derive_val_run_name(model_path: Path) -> str:
    """
    Из пути к модели выводит имя папки для val-результатов.

    Правило:
        runs/train/<run_name>/weights/best.pt  →  <run_name>
        runs/train/<run_name>/weights/last.pt  →  <run_name>

    Логика: model_path → weights/ → <run_name>/ → берём имя папки.
    """
    return model_path.parent.parent.name


def evaluate_model(config: dict, model_path: str = None, run_name: str = None):
    """
    Оценивает модель на валидационной выборке.

    Args:
        config:     словарь из configs/config.yaml.
        model_path: путь к весам. Если None — берётся последняя best.pt
                    в runs/train/.
        run_name:   имя папки в runs/val/. Если None — выводится из
                    пути к модели (см. derive_val_run_name).

    Returns:
        объект метрик от Ultralytics (metrics.box.map50 и т.д.).
    """
    processed_dir = Path(config["paths"]["processed_dir"])
    data_yaml = processed_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml}")

    # ── Определяем модель ──
    if model_path is None:
        runs_dir = Path(config["paths"]["runs_dir"])
        best_models = sorted(runs_dir.glob("train/**/weights/best.pt"))
        if not best_models:
            raise FileNotFoundError("Обученные модели не найдены в runs/train/")
        model_path = best_models[-1]
        logger.info(f"Модель не указана — берём последнюю: {model_path}")
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    # ── Имя папки результатов: как у train, если не задано явно ──
    if run_name is None:
        run_name = derive_val_run_name(model_path)
        logger.info(f"Имя для результатов выведено из пути модели: {run_name}")

    # ── Валидация ──
    logger.info(f"Загрузка модели: {model_path}")
    model = YOLO(str(model_path))

    map_conf = config["evaluation"]["map"]["conf"]
    map_iou = config["evaluation"]["map"]["iou"]

    val_root = Path(config["paths"]["runs_dir"]).resolve() / "val"
    val_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Запуск валидации: conf={map_conf}, iou={map_iou} (для mAP)")
    metrics = model.val(
        data=str(data_yaml),
        conf=map_conf,
        iou=map_iou,
        save_json=True,
        plots=True,
        project=str(val_root),
        name=run_name,
        exist_ok=True,
    )

    # ── Итоговые метрики ──
    logger.info("=" * 50)
    logger.info("МЕТРИКИ НА ВАЛИДАЦИИ")
    logger.info("=" * 50)
    logger.info(f"mAP@0.5:      {metrics.box.map50:.4f}")
    logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    logger.info(f"Precision:    {metrics.box.mp:.4f}")
    logger.info(f"Recall:       {metrics.box.mr:.4f}")
    logger.info("=" * 50)
    logger.info(f"Результаты сохранены: {metrics.save_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None,
                        help="Путь к весам. По умолчанию — последняя "
                             "best.pt в runs/train/.")
    parser.add_argument("--run-name", default=None,
                        help="Имя папки в runs/val/. По умолчанию берётся "
                             "имя train-папки: runs/train/X/... → runs/val/X/")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    evaluate_model(config, model_path=args.model, run_name=args.run_name)


if __name__ == "__main__":
    main()