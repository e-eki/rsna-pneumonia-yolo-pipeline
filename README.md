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
<!-- ├── data/
│   ├── raw/  # Исходные данные (JSON + DICOM)
│   ├── staged/  # Промежуточные данные (JPEG + промежуточные аннотации)
│   ├── processed/    # Финальный датасет в формате YOLO (train/val)
    └── reports/                 # ← НОВОЕ: графики и отчёты по разметке
│   └── samples/                 # ← НОВОЕ: визуальные проверки по этапам
│       ├── raw/
│       ├── staged/
│       └── processed/ -->

<!-- data/
├── raw/rsna_yolo/                   # исходники (не тронуты)
│   ├── images/{train,val}/*.png
│   └── labels/{train,val}/*.txt
├── staged/
│   ├── rejected_images/*.png        # картинки с полностью отвергнутой разметкой
│   ├── rejected_annotations.csv     # отвергнутые боксы с причинами
│   └── filtering_report.json
├── processed/                       # финальный датасет для YOLO
│   ├── images/{train,val}/*.png
│   ├── labels/{train,val}/*.txt
│   └── data.yaml                    # ← СОЗДАЁТСЯ
├── reports/
│   ├── processed_train/*.png
│   └── processed_val/*.png
└── samples/
    ├── processed/{train,val}/       # контактные листы
    └── rejected/                    # визуальная проверка отвергнутых -->

    data/
├── raw/rsna_yolo/                      # не тронуты
├── staged/
│   └── filtering_report.json           # только отчёт
├── processed/
│   ├── images/{train,val}/*.png
│   ├── labels/{train,val}/*.txt
│   └── data.yaml
├── reports/
│   ├── processed_train/
│   └── processed_val/
└── samples/
    ├── processed/{train,val}/          # контактные листы
    └── rejected/
        ├── rejected_images/*.png       # исходные отвергнутые картинки
        ├── rejected_annotations.csv    # отвергнутые боксы с причинами
        ├── drawn/                      # отрисованные копии для viewer.py
        └── metadata.json
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


======
Data quality filter. Все аннотации в формате YOLO проходят валидацию: проверяются пары image↔label, границы координат [0, 1], площадь бокса и aspect ratio. На датасете RSNA (26 684 снимка, 9 555 боксов) фильтр отсеивает 2 бокса с причиной extends_out_of_bounds — оба упираются в нижний край кадра и не являются валидной разметкой. Отвергнутые боксы сохраняются в data/staged/rejected_annotations.csv и визуализируются скриптом collect_rejected.py для аудита.

======

Если хочется больше «показательности» для README
Двух отвергнутых боксов достаточно, чтобы сказать «фильтр работает», но для визуальной демонстрации в README иногда полезно иметь 10–30 картинок в data/samples/rejected/. Есть два способа:

Способ A. Ужесточить пороги для одного демонстрационного прогона (потом вернуть обратно):

yaml
filtering:
  min_box_area_frac: 0.005   # вместо 0.001 — режет боксы < 0.5% площади
  min_aspect_ratio: 0.3
  max_aspect_ratio: 3.0
Прогоняем, снимаем скриншот отчёта и папки rejected/, откатываем пороги, прогоняем снова с мягкими для обучения. В README пишете: «фильтр при стандартных порогах отсеивает 2/9555 боксов (RSNA очень чистый); при ужесточённых порогах — 47/9555 (для демонстрации)». Это сильнее смотрится, чем просто «фильтр работает».

===========

Data quality & filtering. Все YOLO-аннотации проходят валидацию: проверка пар image↔label, границ координат [0, 1], площади бокса, aspect ratio. Отклонения делятся на две категории:

minor (микроскопические боксы, повреждённые строки) — сам бокс удаляется, снимок остаётся;

critical (бокс выходит за границы кадра, экстремальный aspect ratio, размер > кадра) — снимок полностью удаляется в samples/rejected/, чтобы модель не училась «здесь нет очага».

На датасете RSNA (26 684 снимка, 9 555 боксов) фильтр отсеивает 2 снимка с причиной extends_out_of_bounds — оба упираются в нижний край кадра. Отвергнутые снимки и аннотации сохраняются в data/samples/rejected/ с полной отчётностью (rejected_annotations.csv, metadata.json) и визуализируются скриптом collect_rejected.py.

=========

2. Про удаление снимков с отвергнутыми боксами
Ваша интуиция верная. Нужно различать две категории причин отклонения:

Категория	Причины	Что делать с изображением
«Мусор» — очага тут точно нет	bad_line_format, bad_number_format, non_positive_size, area_too_small (микроскопический)	Оставить снимок, остальные боксы сохранить
«Критично» — очаг мог быть, но разметка невалидна	extends_out_of_bounds, aspect_too_high, aspect_too_low, center_out_of_bounds, size_over_1	Убрать снимок в rejected_images/ — нельзя учить модель «здесь нет очага»
Плюс базовое правило: если все боксы отвергнуты — всегда убираем снимок, вне зависимости от категории.

Это и есть реализация вашей идеи: «была хотя бы одна не микроскопическая отвергнутая разметка → убираем снимок». Только фильтруем не по размеру, а по причине — так принципиальнее.

=============

