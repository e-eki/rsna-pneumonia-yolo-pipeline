"""
Скачивание датасета RSNA Pneumonia Detection Challenge с Kaggle.

Использует Kaggle CLI. Требуется авторизация одним из способов:
  - файл ~/.kaggle/kaggle.json (классический способ);
  - переменная окружения KAGGLE_API_TOKEN (современный способ).

Датасет скачивается и распаковывается в data/raw/<raw_subdir>/.
Ожидаемая структура после распаковки — YOLO-формат:
    data/raw/rsna_yolo/images/{train,val}/*.png
    data/raw/rsna_yolo/labels/{train,val}/*.txt

Пример запуска:
    python -m src.download_data --config configs/config.yaml
"""
import argparse
import subprocess
from pathlib import Path

from src.utils import ensure_dirs, get_logger, load_config, setup_logging

logger = get_logger(__name__)


def download_kaggle_dataset(dataset_slug: str, output_dir: str):
    """
    Скачивает и распаковывает датасет через Kaggle CLI.

    Args:
        dataset_slug: идентификатор датасета на Kaggle,
                      например "poeticmage/rsna-pneumonia-detection-challenge-train-val".
        output_dir:   куда распаковать (обычно data/raw).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Скачивание датасета: {dataset_slug}")
    logger.info(f"Целевая директория:  {output_path.resolve()}")

    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", str(output_path),
        "--unzip",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Ошибка скачивания: {result.stderr}")
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")

    logger.info("Датасет скачан и распакован")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)

    download_kaggle_dataset(
        dataset_slug=config["dataset"]["kaggle_dataset"],
        output_dir=config["paths"]["raw_dir"],
    )
    logger.info(f"Готово. Данные в: {config['paths']['raw_dir']}")


if __name__ == "__main__":
    main()