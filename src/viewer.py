# src/viewer.py
"""
Простой просмотрщик изображений с листанием вперёд/назад.

Управление:
  → / D / пробел : следующее
  ← / A          : предыдущее
  Home           : первое
  End            : последнее
  G              : перейти к индексу (ввод в консоли)
  Q / ESC        : выход

Примеры:
  python src/viewer.py --dir data/samples/processed
  python src/viewer.py --dir data/samples/staged --start 10
"""
import argparse
from pathlib import Path

import cv2

WINDOW = "YOLO Sample Viewer"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTS])


def show(img_path: Path, idx: int, total: int):
    img = cv2.imread(str(img_path))
    if img is None:
        img = cv2.imread(str(img_path)) or None
    if img is None:
        print(f"[warn] не удалось прочитать {img_path}")
        return
    # Подгоняем под экран, сохраняя пропорции
    h, w = img.shape[:2]
    max_side = 1100
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    title = f"[{idx + 1}/{total}] {img_path.name}"
    cv2.setWindowTitle(WINDOW, title)
    cv2.imshow(WINDOW, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Папка с изображениями для просмотра")
    parser.add_argument("--start", type=int, default=0, help="Начальный индекс")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")

    images = list_images(folder)
    if not images:
        raise RuntimeError(f"В папке нет изображений: {folder}")

    idx = max(0, min(args.start, len(images) - 1))
    print(f"Найдено {len(images)} изображений в {folder}")
    print("Управление: →/D/пробел — вперёд, ←/A — назад, G — переход, Q/ESC — выход")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    show(images[idx], idx, len(images))

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  # Q или ESC
            break
        elif key in (83, ord("d"), ord(" ")):  # →, D, пробел
            idx = min(idx + 1, len(images) - 1)
        elif key in (81, ord("a")):            # ←, A
            idx = max(idx - 1, 0)
        elif key == 80 or key == ord("G"):     # Home/G (в OpenCV 80 это Home на некоторых сборках)
            try:
                new_idx = int(input(f"Перейти к индексу (0..{len(images) - 1}): "))
                idx = max(0, min(new_idx, len(images) - 1))
            except ValueError:
                pass
        elif key == ord("e"):  # End-альтернатива
            idx = len(images) - 1
        elif key == ord("b"):  # Beginning-альтернатива
            idx = 0

        show(images[idx], idx, len(images))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()