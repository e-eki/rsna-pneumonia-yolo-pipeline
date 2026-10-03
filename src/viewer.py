"""
Просмотрщик изображений с листанием вперёд/назад.

Открывает все изображения из папки и листает их по нажатию клавиш.
Требует графического окружения (окно OpenCV), поэтому работает
локально, но не в Colab. Для Colab используйте ipywidgets-слайдер
в notebooks/colab_train.ipynb.

Управление:
    → / D / пробел   — следующее изображение
    ← / A            — предыдущее
    B                — первое
    E                — последнее
    G                — перейти к индексу (интерактивный ввод)
    Q / ESC          — выход

Примеры запуска:
    python -m src.viewer --dir data/samples/processed/train
    python -m src.viewer --dir data/samples/rejected/drawn --start 10
"""
import argparse
from pathlib import Path

import cv2

WINDOW = "YOLO Sample Viewer"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_SIDE = 1100   # макс. сторона окна, чтобы картинка влезала на экран


# ─────────────────────────── вспомогательные ───────────────────────────

def list_images(folder: Path):
    """Возвращает отсортированный список изображений в папке."""
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMG_EXTS)


def show(img_path: Path, idx: int, total: int):
    """Показывает изображение idx в окне, масштабируя под экран."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[warn] не удалось прочитать {img_path}")
        return

    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.setWindowTitle(WINDOW, f"[{idx + 1}/{total}] {img_path.name}")
    cv2.imshow(WINDOW, img)


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="Папка с изображениями для просмотра")
    parser.add_argument("--start", type=int, default=0,
                        help="Начальный индекс")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")

    images = list_images(folder)
    if not images:
        raise RuntimeError(f"В папке нет изображений: {folder}")

    idx = max(0, min(args.start, len(images) - 1))
    print(f"Найдено {len(images)} изображений в {folder}")
    print("Управление: →/D/пробел — вперёд, ←/A — назад, "
          "B — начало, E — конец, G — переход, Q/ESC — выход")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    show(images[idx], idx, len(images))

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):                    # Q или ESC
            break
        elif key in (83, ord("d"), ord(" ")):        # →, D, пробел
            idx = min(idx + 1, len(images) - 1)
        elif key in (81, ord("a")):                  # ←, A
            idx = max(idx - 1, 0)
        elif key == ord("b"):                        # начало
            idx = 0
        elif key == ord("e"):                        # конец
            idx = len(images) - 1
        elif key == ord("g"):                        # переход к индексу
            try:
                new_idx = int(input(f"Перейти к индексу (0..{len(images) - 1}): "))
                idx = max(0, min(new_idx, len(images) - 1))
            except ValueError:
                pass

        show(images[idx], idx, len(images))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()