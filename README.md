# RSNA Pneumonia YOLO Pipeline

End-to-end пайплайн для обучения модели детекции пневмонии на рентгеновских снимках грудной клетки с использованием YOLO.

## 📋 Описание

Проект реализует полный цикл:
1. **Скачивание** датасета RSNA Pneumonia Detection Challenge
2. **Предобработка** изображений (конвертация DICOM → JPEG, windowing)
3. **Разделение** на train/val и создание YOLO-разметки
4. **Обучение** модели YOLOv8
5. **Оценка** на валидационной выборке

## 🚀 Быстрый старт (Google Colab)

1. Откройте `notebooks/colab_train.ipynb` в Colab
2. Включите GPU: Runtime → Change runtime type → T4 GPU
3. Загрузите `kaggle.json` (получите на kaggle.com → Account → Create New API Token)
4. Запустите все ячейки последовательно

## 📁 Структура
rsna-pneumonia-yolo-pipeline/
│
├── configs/
│   └── config.yaml              # Все параметры: пути, гиперпараметры, классы
│
├── data/
│   ├── raw/  # Исходные данные (JSON + DICOM)
│   ├── staged/  # Промежуточные данные (JPEG + промежуточные аннотации)
│   ├── processed/    # Финальный датасет в формате YOLO (train/val)
    └── reports/                 # ← НОВОЕ: графики и отчёты по разметке
│   └── samples/                 # ← НОВОЕ: визуальные проверки по этапам
│       ├── raw/
│       ├── staged/
│       └── processed/
│   
├── src/  # Скрипты пайплайна
│   ├── utils.py                 # Общие функции (логирование, чтение конфига)
│   ├── download_data.py         # Скачивание датасета с Kaggle
│   ├── preprocess.py            # Конвертация DICOM → JPEG, парсинг аннотаций
│   ├── split_data.py            # Формирование train/val и YOLO-разметки
│   ├── train.py                 # Обучение модели YOLO
│   ├── evaluate.py              # Оценка модели на val
│   ├── sample_and_draw.py
│   ├── viewer.py
│   ├── compare_stages.py        # ← НОВОЕ
│   └── analyze_annotations.py   # ← НОВОЕ
│
├── notebooks/
│   └── colab_train.ipynb        # Основной сценарий для Colab
│
├── models/                      # Сохраненные веса моделей
├── runs/                        # Логи обучения YOLO
│
├── requirements.txt
├── .gitignore
└── README.md


## 🔧 Локальный запуск

```bash
pip install -r requirements.txt
python src/download_data.py
python src/preprocess.py
python src/split_data.py
python src/train.py
python src/evaluate.py

===================
Результаты
После обучения метрики сохраняются в runs/train/. Ожидаемые значения:
mAP@0.5: ~0.25-0.35 (зависит от модели и эпох)
Precision: ~0.30-0.40
Recall: ~0.35-0.45

📚 Датасет
RSNA Pneumonia Detection Challenge
26,684 изображений
1 класс: pneumonia
Формат: YOLO (bounding boxes)