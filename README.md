# RSNA Pneumonia YOLO Pipeline

End-to-end пайплайн для обучения YOLO-модели детекции пневмонии на рентгеновских снимках грудной клетки (датасет RSNA). Модульные Python-скрипты, прозрачная фильтрация разметки, визуальные проверки на каждой стадии.

## 📖 О проекте

Задача — научить модель находить области затемнения (lung opacities) на рентгеновских снимках. Пайплайн построен как набор независимых скриптов: каждый этап можно запустить отдельно, проверить результат визуально и передать следующему.

**Ключевая идея:** не просто «обучить YOLO», а построить воспроизводимый процесс с контролем качества данных на каждой стадии.

## ✨ Ключевые особенности

- **Модульность.** Каждый этап — отдельный скрипт, запускается независимо.
- **Фильтрация разметки.** Отсев «плохих» боксов с разделением на **minor** и **critical** причины. Critical → изображение убирается из обучения целиком.
- **Визуальные проверки.** Скрипты рисуют разметку поверх снимков на любой стадии пайплайна.
- **Статистика и отчёты.** Графики распределений и `report.json` для аудита.
- **Конфигурируемость.** Формат вывода PNG/JPEG, пороги фильтрации, гиперпараметры — всё в `configs/config.yaml`.
- **Готовность к Colab и Kaggle.** Никаких жёстко прописанных путей.
- **Парность папок обучения и оценки.** Для каждого эксперимента логи лежат в `runs/train/<run_name>/` и `runs/val/<run_name>/`.

## 📊 Результаты

Обучение **yolov8s** на 30 эпохах, mosaic выключена (разрушает анатомическую структуру рентгена), JPEG-конвертация датасета для скорости I/O.

| Метрика       | Значение  |
|---------------|-----------|
| mAP@0.5       | **0.349** |
| mAP@0.5:0.95  | 0.137     |
| Precision     | 0.396     |
| Recall        | 0.383     |
| AR@0.50       | 0.92      |

Метрики сопоставимы с публичными бейзлайнами RSNA (mAP 0.25–0.40). Полный разбор — в [RESULTS](docs/RESULTS.md).

## 🚀 Быстрый старт (Colab)

```python
# 1. Клонировать и установить зависимости
!git clone -b development https://github.com/e-eki/rsna-pneumonia-yolo-pipeline.git
%cd rsna-pneumonia-yolo-pipeline
!pip install -q -r requirements.txt

# 2. Настроить Kaggle API (токен в Colab Secrets под именем KAGGLE_API_TOKEN)
import os
from google.colab import userdata
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")

# 3. Скачать и подготовить датасет
!python -m src.download_data --config configs/config.yaml
!python -m src.prepare_yolo_dataset --config configs/config.yaml

# 4. Проверить визуально
!python -m src.sample_and_draw --stage processed --split train --num 50 --grid

# 5. Обучить
!python -m src.train --config configs/config.yaml \
    --run-name "2026-10-02_full_yolov8s_30ep"

# 6. Оценить
!python -m src.evaluate --config configs/config.yaml \
    --model runs/train/2026-10-02_full_yolov8s_30ep/weights/best.pt
```

Готовый сценарий — в `notebooks/colab_train.ipynb`.

## 📁 Структура проекта (кратко)

```text
rsna-pneumonia-yolo-pipeline/
├── configs/config.yaml          # все параметры пайплайна
├── data/                        # данные (не в git)
├── src/                         # скрипты пайплайна
├── notebooks/colab_train.ipynb  # сценарий для Colab
├── runs/                        # логи обучения и оценки (не в git)
├── docs/                        # документация
└── requirements.txt
```

Полная структура — в [STRUCTURE](docs/STRUCTURE.md).

## 📚 Документация

| Документ                             | О чём                                                    |
|--------------------------------------|----------------------------------------------------------|
| [STRUCTURE](docs/STRUCTURE.md)       | Структура проекта и стадии пайплайна                     |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Схема потока данных и принципы                           |
| [DATA](docs/DATA.md)                 | Датасет, фильтрация разметки, форматы                    |
| [TRAINING](docs/TRAINING.md)         | Гиперпараметры, аугментации, что и почему                |
| [EVALUATION](docs/EVALUATION.md)     | Метрики, пороги, как читать confusion matrix             |
| [RESULTS](docs/RESULTS.md)           | Метрики обученной модели и анализ                        |

## 📦 Требования

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Ultralytics ≥ 8.0
- GPU: CUDA ≥ 11.8 (тестировалось на Tesla T4)

Полный список — в `requirements.txt`.
