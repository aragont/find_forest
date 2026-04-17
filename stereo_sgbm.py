#!/usr/bin/env python3
"""
Скрипт для вычисления карты диспаратности (disparity map) из стереопары изображений.
Использует алгоритм Semi-Global Block Matching (SGBM) из OpenCV.

Входные данные: два TIFF файла (uint16) - левое и правое изображения стереопары.
Выходные данные: TIFF файл (float32) с картой диспаратности.
"""

import argparse
import cv2
import numpy as np
import sys
import os


def load_image(path):
    """Загружает изображение в градациях серого."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    # Читаем как есть, чтобы сохранить глубину бит (uint16)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")

    # Если изображение цветное, конвертируем в ч/б
    if len(img.shape) == 3:
        print(f"Внимание: {path} является цветным изображением. Конвертация в оттенки серого.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Убеждаемся, что тип данных uint16 или приводим к нему, если это 8-bit
    if img.dtype != np.uint16:
        if img.dtype == np.uint8:
            print(f"Внимание: {path} имеет глубину 8 бит. Алгоритм работает лучше с 16 бит, но продолжим.")
        else:
            print(f"Внимание: Тип данных {img.dtype} изменен на uint16.")
            # Нормализация может потребоваться в реальных задачах, здесь просто приводим тип
            img = img.astype(np.uint16)

    return img


def save_disparity(disparity, path, min_disp):
    """Сохраняет карту диспаратности в формат TIFF float32."""
    # SGBM возвращает фиксированные точки (умноженные на 16). Конвертируем в настоящие значения.
    disparity_float = disparity.astype(np.float32) / 16.0

    # Маскируем недопустимые значения (обычно равны min_disp, если алгоритм не нашел соответствия)
    # В зависимости от реализации, иногда лучше оставить как есть или пометить NaN.
    # Здесь мы оставляем значения как есть, но пользователь должен знать, что
    # области без соответствия часто имеют значение, равное порогу.

    # Сохраняем как float32
    # Для корректного сохранения float32 в TIFF через cv2.imwrite могут быть нюансы,
    # поэтому используем прямой подход с numpy/tifffile если бы он был,
    # но cv2.imwrite обычно справляется с 32f.

    # Проверка на успешность записи
    success = cv2.imwrite(path, disparity_float)
    if not success:
        # Альтернативный метод, если cv2 не справляется с float32 tiff в конкретной сборке
        # Используем простой бинарный запись через numpy (менее надежно для метаданных, но работает)
        # Но сначала попробуем стандартный путь.
        raise IOError(f"Не удалось сохранить файл {path} с помощью cv2.imwrite. Убедитесь, что ваша сборка OpenCV поддерживает TIFF float32.")

    print(f"Карта диспаратности сохранена в: {path}")
    print(f"Диапазон значений (сырые, до деления на 16): [{np.min(disparity)}, {np.max(disparity)}]")
    print(f"Диапазон значений (реальные пиксели): [{np.min(disparity_float)}, {np.max(disparity_float)}]")


def main():
    parser = argparse.ArgumentParser(
        description="Вычисление карты диспаратности (SGBM) из стереопары TIFF."
    )
    parser.add_argument("left_image", help="Путь к левому изображению (TIFF uint16)")
    parser.add_argument("right_image", help="Путь к правому изображению (TIFF uint16)")
    parser.add_argument("output_file", help="Путь к выходному файлу (TIFF float32)")

    parser.add_argument(
        "-R", "--max-disp",
        type=int,
        default=128,
        help="Максимальное значение диспаратности (должно быть кратно 16). По умолчанию 128."
    )
    parser.add_argument(
        "-r", "--min-disp",
        type=int,
        default=0,
        help="Минимальное значение диспаратности (должно быть кратно 16). По умолчанию 0."
    )

    args = parser.parse_args()

    # Валидация параметров SGBM
    # numDisparities должно быть кратно 16
    if args.max_disp % 16 != 0:
        print(f"Предупреждение: Максимальная диспаратность ({args.max_disp}) должна быть кратна 16. Округляем до {args.max_disp // 16 * 16}.")
        max_disp = args.max_disp // 16 * 16
    else:
        max_disp = args.max_disp

    if args.min_disp % 16 != 0:
        print(f"Предупреждение: Минимальная диспаратность ({args.min_disp}) должна быть кратна 16. Округляем до {args.min_disp // 16 * 16}.")
        min_disp = args.min_disp // 16 * 16
    else:
        min_disp = args.min_disp

    if max_disp <= min_disp:
        print("Ошибка: Максимальная диспаратность должна быть больше минимальной.")
        sys.exit(1)

    try:
        print("Загрузка изображений...")
        img_left = load_image(args.left_image)
        img_right = load_image(args.right_image)

        if img_left.shape != img_right.shape:
            print(f"Ошибка: Размеры изображений не совпадают!\nЛевое: {img_left.shape}\nПравое: {img_right.shape}")
            sys.exit(1)

        print("Инициализация алгоритма SGBM...")
        # Параметры SGBM можно тонко настраивать. Здесь используются разумные значения по умолчанию.
        # blockSize: размер окна сопоставления (должен быть нечетным, обычно 3-11)
        block_size = 5

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=max_disp - min_disp,  # Разница между макс и мин
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,          # Параметр штрафа для небольших изменений диспаратности
            P2=32 * 3 * block_size ** 2,         # Параметр штрафа для больших изменений
            disp12MaxDiff=1,                     # Проверка слева-направо
            preFilterCap=63,                     # Предел фильтра предварительной обработки
            uniquenessRatio=10,                  # Порог уникальности
            speckleWindowSize=100,               # Размер окна для фильтрации шума
            speckleRange=32,                     # Допустимый диапазон вариации диспаратности внутри пятна
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Режим алгоритма
        )

        print("Вычисление карты диспаратности...")
        disparity = stereo.compute(img_left, img_right)

        print("Сохранение результата...")
        save_disparity(disparity, args.output_file, min_disp)

        print("Готово.")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()