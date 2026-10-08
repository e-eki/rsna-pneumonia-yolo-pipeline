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
- **Возможность выполнения в Colab и Kaggle.** Нет жёстко прописанных путей.
- **Парные папки runs/train/<run_name>/ и runs/val/<run_name>/.** — для каждого эксперимента логи обучения и оценки лежат рядом.


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

Готовый сценарий — в notebooks/colab_train.ipynb.