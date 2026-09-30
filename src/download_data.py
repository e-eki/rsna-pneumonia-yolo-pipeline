# src/download_data.py
"""Скачивание датасета RSNA Pneumonia Detection Challenge с Kaggle.

Скрипт скачивает датасет с Kaggle. В Colab потребуется настроить kaggle.json."""
import os
import subprocess
import argparse
from pathlib import Path
from src.utils import load_config, ensure_dirs, setup_logging, get_logger

logger = get_logger(__name__)

def download_kaggle_dataset(dataset_slug: str, output_dir: str):
    """Скачивание датасета через Kaggle CLI."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Скачивание датасета: {dataset_slug}")
    logger.info(f"Целевая директория: {output_path.resolve()}")
    
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", str(output_path),
        "--unzip"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Ошибка скачивания: {result.stderr}")
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")
    
    logger.info("Датасет успешно скачан и распакован")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    
    dataset_slug = config["dataset"]["kaggle_dataset"]
    raw_dir = config["paths"]["raw_dir"]
    
    download_kaggle_dataset(dataset_slug, raw_dir)
    logger.info(f"Готово. Данные в: {raw_dir}")

if __name__ == "__main__":
    main()