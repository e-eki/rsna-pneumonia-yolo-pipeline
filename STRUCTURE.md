Структура проекта

rsna-pneumonia-yolo-pipeline/
├── configs/
│   └── config.yaml              # все параметры пайплайна
├── data/
│   ├── raw/                     # исходные данные с Kaggle (не в git)
│   ├── processed/               # финальный YOLO-датасет (images/labels + data.yaml)
│   ├── staged/                  # отчёты фильтрации
│   ├── reports/                 # графики статистики разметки
│   └── samples/                 # визуальные проверки, контактные листы
├── src/
│   ├── utils.py                 # общие утилиты (логи, конфиг)
│   ├── download_data.py         # скачивание с Kaggle
│   ├── prepare_yolo_dataset.py  # фильтрация + подготовка data/processed/
│   ├── train.py                 # обучение YOLO
│   ├── evaluate.py              # оценка на валидации (mAP)
│   ├── analyze_annotations.py   # статистика разметки
│   ├── sample_and_draw.py       # случайные сэмплы с отрисованными боксами
│   ├── collect_rejected.py      # визуализация отвергнутых снимков
│   └── viewer.py                # локальный просмотрщик изображений
├── notebooks/
│   └── colab_train.ipynb        # сценарий для Colab
├── runs/                        # логи обучения и оценки (не в git)
├── requirements.txt
└── README.md


Стадии пайплайна
Скрипт	Что делает	Результат
download_data.py	Скачивает датасет RSNA с Kaggle	data/raw/rsna_yolo/
prepare_yolo_dataset.py	Фильтрует разметку, конвертирует в JPEG, пишет data.yaml	data/processed/, data/samples/rejected/
analyze_annotations.py	Считает статистику разметки на стадиях raw/staged/processed	data/reports/<stage>/
sample_and_draw.py	Сэмплирует N изображений и рисует боксы	data/samples/<stage>/
collect_rejected.py	Рисует отвергнутые боксы (красный = minor, оранжевый = critical)	data/samples/rejected/drawn/
train.py	Обучает YOLO, сохраняет чекпоинты и логи	runs/train/<run_name>/
evaluate.py	Оценивает модель, считает mAP	runs/val/<run_name>/