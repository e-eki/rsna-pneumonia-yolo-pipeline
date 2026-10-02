"""Оценка обученной модели на валидационной выборке."""
import argparse
from pathlib import Path

from ultralytics import YOLO

from src.utils import load_config, ensure_dirs, setup_logging, get_logger

logger = get_logger(__name__)


def derive_val_run_name(model_path: Path) -> str:
    """
    Из пути к модели выводит имя папки для val-результатов.

    Правило:
      runs/train_<X>/weights/best.pt      → val_<X>
      runs/train_<X>/weights/last.pt      → val_<X>
      runs/<X>/weights/best.pt           → val_<X>
      что-то/совсем/другое/best.pt        → val_<stem_родителя_родителя>

    Логика: берём имя папки запуска (та, что лежит в runs/ и содержит
    weights/) и, если она начинается с "train", заменяем префикс на "val".
    """
    # model_path = runs/train_X/weights/best.pt
    # weights_dir = runs/train_X/weights
    # run_dir = runs/train_X
    weights_dir = model_path.parent
    run_dir = weights_dir.parent
    run_name = run_dir.name

    if run_name.startswith("train"):
        return "val" + run_name[len("train"):]
    return f"val_{run_name}"


def evaluate_model(config: dict, model_path: str = None, run_name: str = None):
    """Оценка модели на val-выборке."""
    processed_dir = Path(config["paths"]["processed_dir"])
    data_yaml = processed_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml}")

    # ── Определяем модель ──
    if model_path is None:
        runs_dir = Path(config["paths"]["runs_dir"])
        best_models = sorted(runs_dir.glob("**/weights/best.pt"))
        if not best_models:
            raise FileNotFoundError("Обученные модели не найдены в runs/")
        model_path = best_models[-1]
        logger.info(f"Модель не указана — берём последнюю: {model_path}")
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    # ── Определяем имя папки для результатов ──
    if run_name is None:
        run_name = derive_val_run_name(model_path)
        logger.info(f"Имя для результатов выведено из пути модели: {run_name}")

    # ── Валидация ──
    logger.info(f"Загрузка модели: {model_path}")
    model = YOLO(str(model_path))

    map_conf = config["evaluation"]["map"]["conf"]
    map_iou = config["evaluation"]["map"]["iou"]
    logger.info(f"Запуск валидации: conf={map_conf}, iou={map_iou} (для mAP)")

    metrics = model.val(
        data=str(data_yaml),
        conf=map_conf,
        iou=map_iou,
        save_json=True,
        plots=True,
        project=str(Path(config["paths"]["runs_dir"]).resolve()),
        name=run_name,
        exist_ok=True,
    )

    # ── Метрики ──
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
                        help="Путь к весам. По умолчанию — последняя best.pt в runs/")
    parser.add_argument("--run-name", default=None,
                        help="Имя папки для val-результатов. "
                             "По умолчанию выводится из пути модели: "
                             "runs/train_X/weights/best.pt → val_X")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    evaluate_model(config, model_path=args.model, run_name=args.run_name)


if __name__ == "__main__":
    main()