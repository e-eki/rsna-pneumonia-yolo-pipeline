Архитектура пайплайна

                    ┌──────────────────┐
                    │  Kaggle Dataset  │
                    │   (RSNA, YOLO)   │
                    └─────────┬────────┘
                              │ download_data.py
                              ▼
                    ┌──────────────────┐
                    │    data/raw/     │  images/ + labels/ (train, val)
                    └─────────┬────────┘
                              │ prepare_yolo_dataset.py
                              │ ─ фильтрация разметки
                              │ ─ конвертация в JPEG
                              │ ─ разделение на critical / minor
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                    ┌──────────────────┐
│ data/processed/  │                    │ data/samples/    │
│  images/ labels/ │                    │  rejected/       │
│  data.yaml       │                    │  processed/      │
└────────┬─────────┘                    └──────────────────┘
         │ train.py                            ▲
         ▼                                     │
┌──────────────────┐                           │
│  runs/train/     │                           │
│   <run_name>/    │                           │
│   weights/*.pt   │                           │
└────────┬─────────┘                           │
         │ evaluate.py                         │
         ▼                                     │
┌──────────────────┐                           │
│  runs/val/       │                           │
│   <run_name>/    │                           │
└──────────────────┘                           │
                                               │
         analyze_annotations.py, sample_and_draw.py,
         collect_rejected.py ────────────────────┘

## Принципы

### 1. Один скрипт — одна задача

Каждый скрипт в `src/` решает одну задачу и запускается независимо. Это позволяет отлаживать и заменять любой этап, не переписывая весь пайплайн.

### 2. Промежуточные данные сохраняются

После каждой стадии данные попадают на диск (`data/raw/`, `data/processed/`, `data/samples/`). Это даёт возможность возобновить пайплайн с любой точки и вручную проверить результат.

### 3. Контроль качества на каждой стадии

`analyze_annotations.py` и `sample_and_draw.py` работают на стадиях raw/staged/processed. Если на какой-то стадии данные испортились — это видно сразу, а не через 5 часов обучения.

### 4. Конфигурация отделена от кода

Все параметры — в `configs/config.yaml`. Скрипты читают конфиг через `utils.load_config`. Один и тот же код работает с разными датасетами и моделями.

### 5. Абсолютные пути для артефактов

YOLO получает `project` как абсолютный путь (`Path(...).resolve()`), чтобы не было двойных `runs/detect/runs/` — известный баг с относительными путями.

## Куда что складывается

| Путь | Что там | Откуда | В git? |
|---|---|---|---|
| `data/raw/<subdir>/` | исходный датасет | `download_data.py` | нет |
| `data/processed/` | финальный YOLO-датасет | `prepare_yolo_dataset.py` | нет |
| `data/staged/filtering_report.json` | сводка фильтрации | `prepare_yolo_dataset.py` | нет |
| `data/samples/rejected/` | отвергнутые снимки + отчёт | `prepare_yolo_dataset.py` | нет |
| `data/samples/processed/` | случайные сэмплы с боксами | `sample_and_draw.py` | нет |
| `data/reports/` | графики и `report.json` | `analyze_annotations.py` | нет |
| `runs/train/<run_name>/` | логи, веса, графики | `train.py` | нет |
| `runs/val/<run_name>/` | предсказания, confusion matrix | `evaluate.py` | нет |

## Парность папок train/val

Для каждого обучения `runs/train/<run_name>/` создаётся парная `runs/val/<run_name>/` при оценке. Имя автоматически выводится из пути к модели:
runs/train/2026-10-02_full_yolov8s_30ep/weights/best.pt
↓
runs/val/2026-10-02_full_yolov8s_30ep/

text

Так для каждого эксперимента обучение и оценка лежат рядом — легко сравнивать.