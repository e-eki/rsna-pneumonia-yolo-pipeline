# Структура проекта

Что где лежит и как устроены стадии пайплайна.

## 🗂️ Дерево проекта

```text
rsna-pneumonia-yolo-pipeline/
├── configs/
│   └── config.yaml              # все параметры пайплайна
├── data/                        # данные (не в git)
│   ├── raw/                     # исходные данные с Kaggle
│   ├── processed/               # финальный YOLO-датасет (images/labels + data.yaml)
│   ├── staged/                  # отчёты фильтрации
│   ├── reports/                 # графики статистики разметки
│   └── samples/                 # визуальные проверки, контактные листы
├── src/                         # скрипты пайплайна
│   ├── utils.py                 # общие утилиты (логи, конфиг)
│   ├── download_data.py         # скачивание с Kaggle
│   ├── prepare_yolo_dataset.py  # фильтрация + подготовка data/processed/
│   ├── train.py                 # обучение YOLO
│   ├── evaluate.py              # оценка на валидации (mAP)
│   ├── analyze_annotations.py   # статистика разметки
│   ├── sample_and_draw.py       # сэмплы с отрисованными боксами
│   ├── collect_rejected.py      # визуализация отвергнутых снимков
│   └── viewer.py                # локальный просмотрщик изображений
├── notebooks/
│   └── colab_train.ipynb        # сценарий для Colab
├── runs/                        # логи обучения и оценки (не в git)
├── docs/                        # документация
├── requirements.txt
└── README.md
```

## 🔄 Стадии пайплайна

### 1. `download_data.py`
Скачивает датасет RSNA с Kaggle.
**Результат:** `data/raw/rsna_yolo/`

### 2. `prepare_yolo_dataset.py`
Фильтрует разметку, конвертирует изображения в JPEG, пишет `data.yaml`.
**Результат:** `data/processed/`, `data/samples/rejected/`

### 3. `analyze_annotations.py`
Считает статистику разметки на стадиях raw / staged / processed.
**Результат:** `data/reports/<stage>/`

### 4. `sample_and_draw.py`
Сэмплирует N случайных изображений и рисует боксы поверх них.
**Результат:** `data/samples/<stage>/`

### 5. `collect_rejected.py`
Рисует отвергнутые боксы (красный — minor, оранжевый — critical).
**Результат:** `data/samples/rejected/drawn/`

### 6. `train.py`
Обучает YOLO, сохраняет чекпоинты и логи.
**Результат:** `runs/train/<run_name>/`

### 7. `evaluate.py`
Оценивает модель на валидации, считает mAP.
**Результат:** `runs/val/<run_name>/`

## 📦 Артефакты по категориям

### Исходные и промежуточные данные

- **`data/raw/<subdir>/`** — исходный датасет, скачанный с Kaggle. Создаётся `download_data.py`.
- **`data/processed/`** — финальный YOLO-датасет с `data.yaml`. Создаётся `prepare_yolo_dataset.py`.
- **`data/staged/filtering_report.json`** — машинно-читаемая сводка фильтрации. Создаётся `prepare_yolo_dataset.py`.

### Визуальные проверки

- **`data/samples/rejected/`** — отвергнутые снимки и отчёт по ним. Создаётся `prepare_yolo_dataset.py`.
- **`data/samples/processed/`** — случайные сэмплы с отрисованными боксами. Создаётся `sample_and_draw.py`.
- **`data/reports/`** — графики распределений и `report.json`. Создаётся `analyze_annotations.py`.

### Артефакты обучения и оценки

- **`runs/train/<run_name>/`** — логи, веса, графики обучения. Создаётся `train.py`.
- **`runs/val/<run_name>/`** — предсказания и confusion matrix. Создаётся `evaluate.py`.

## 🔗 См. также

- [ARCHITECTURE](ARCHITECTURE.md) — схема потока данных и принципы
- [DATA](DATA.md) — про датасет и фильтрацию разметки
- [TRAINING](TRAINING.md) — гиперпараметры и аугментации
- [EVALUATION](EVALUATION.md) — метрики и как их читать