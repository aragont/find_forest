#!/usr/bin/env python3
"""
Скрипт для создания маски леса на основе DEM (Digital Elevation Model).

Лес определяется по резкому перепаду высот:
- Перепад не менее 10 метров
- На расстоянии менее 10 метров (т.е. в пределах 10 пикселей при шаге 1м)

Вход: TIF файл с DEM (float32, NaN для безданных)
Выход: TIF файл с маской (1 - лес, 0 - не лес)
"""

import numpy as np
import rasterio
from scipy.ndimage import maximum_filter, minimum_filter
from rasterio.enums import Resampling


def create_forest_mask(dem_path: str, output_path: str, 
                       height_threshold: float = 10.0,
                       distance_threshold: int = 10):
    """
    Создает маску леса на основе перепада высот.
    
    Параметры:
    -----------
    dem_path : str
        Путь к входному DEM файлу
    output_path : str
        Путь к выходному файлу маски
    height_threshold : float
        Минимальный перепад высот в метрах (по умолчанию 10м)
    distance_threshold : int
        Максимальное расстояние в пикселях для проверки перепада (по умолчанию 10)
    """
    
    # Чтение DEM файла
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
    
    # Создание маски валидных данных (не NaN)
    valid_mask = ~np.isnan(dem)
    
    # Если нет валидных данных, возвращаем пустую маску
    if not np.any(valid_mask):
        print("Предупреждение: Нет валидных данных в DEM")
        forest_mask = np.zeros_like(dem, dtype=np.uint8)
    else:
        # Для каждого пикселя вычисляем разницу между максимумом и минимумом
        # в окне размером (2*distance_threshold + 1) x (2*distance_threshold + 1)
        window_size = 2 * distance_threshold + 1
        
        # Создаем временную копию DEM с заполненными NaN для фильтрации
        dem_filled = dem.copy()
        
        # Для корректной работы фильтров заменяем NaN на очень большие/малые значения
        # но затем будем использовать маску валидности
        nan_mask = np.isnan(dem_filled)
        
        # Вычисляем локальный максимум и минимум в окне
        # Используем mode='constant' с cval=nan для обработки границ
        local_max = maximum_filter(dem_filled, size=window_size, 
                                    mode='constant', cval=-np.inf)
        local_min = minimum_filter(dem_filled, size=window_size, 
                                    mode='constant', cval=np.inf)
        
        # Разница между максимумом и минимумом
        height_range = local_max - local_min
        
        # Лес там, где перепад высот превышает порог И есть валидные данные
        forest_mask = ((height_range >= height_threshold) & valid_mask).astype(np.uint8)
    
    # Обновление профиля для выходного файла
    profile.update({
        'dtype': 'uint8',
        'count': 1,
        'compress': 'lzw',
        'nodata': 255  # NoData для маски
    })
    
    # Запись результата
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(forest_mask, 1)
    
    # Статистика
    total_pixels = np.sum(valid_mask)
    forest_pixels = np.sum(forest_mask)
    
    print(f"Входной файл: {dem_path}")
    print(f"Выходной файл: {output_path}")
    print(f"Размер изображения: {dem.shape[1]} x {dem.shape[0]} пикселей")
    print(f"Валидных пикселей: {total_pixels}")
    print(f"Пикселей леса: {forest_pixels}")
    if total_pixels > 0:
        print(f"Процент леса: {100 * forest_pixels / total_pixels:.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Создание маски леса на основе перепада высот в DEM"
    )
    parser.add_argument("input_dem", help="Путь к входному DEM файлу (.tif)")
    parser.add_argument("output_mask", help="Путь к выходному файлу маски (.tif)")
    parser.add_argument("--height-threshold", type=float, default=10.0,
                        help="Минимальный перепад высот в метрах (по умолчанию: 10)")
    parser.add_argument("--distance-threshold", type=int, default=10,
                        help="Максимальное расстояние в пикселях (по умолчанию: 10)")
    
    args = parser.parse_args()
    
    create_forest_mask(
        args.input_dem,
        args.output_mask,
        args.height_threshold,
        args.distance_threshold
    )
