"""
Общие утилиты для всех скриптов пайплайна.

Содержит единый набор вспомогательных функций:
  - setup_logging — настройка формата логов;
  - load_config   — чтение YAML-конфига;
  - ensure_dirs   — создание директорий из config["paths"];
  - get_logger    — получение именованного логгера.

Все скрипты пайплайна используют эти функции, чтобы поведение
логирования и работа с путями были одинаковыми.
"""
import logging
from pathlib import Path

import yaml


def setup_logging(level=logging.INFO):
    """Настраивает единый формат логов для всего пайплайна."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path="configs/config.yaml"):
    """Читает YAML-конфиг и возвращает его как словарь."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(config):
    """
    Создаёт все директории, перечисленные в config["paths"].

    Служебные ключи, начинающиеся с "drive_" (пути для Google Drive),
    пропускаются: они существуют только для документации и копируются
    вручную, если нужно.
    """
    for key, path in config["paths"].items():
        if path and not key.startswith("drive_"):
            Path(path).mkdir(parents=True, exist_ok=True)


def get_logger(name):
    """Возвращает логгер с указанным именем (обычно __name__)."""
    return logging.getLogger(name)