# src/utils.py
"""Общие утилиты для пайплайна."""
import os
import logging
import yaml
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Настройка логирования."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def load_config(config_path="configs/config.yaml"):
    """Загрузка YAML-конфига."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def ensure_dirs(config):
    """Создание всех необходимых директорий."""
    for key, path in config["paths"].items():
        if path and not key.startswith("drive_"):
            Path(path).mkdir(parents=True, exist_ok=True)

def get_logger(name):
    """Получение логгера с именем модуля."""
    return logging.getLogger(name)