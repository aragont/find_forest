#!/usr/bin/env python3
"""
Скрипт для создания маски леса на основе DEM (Digital Elevation Model).

Лес определяется по резкому перепаду высот:
- Перепад не менее 10 метров
- На расстоянии менее 10 метров (т.е. в пределах 10 пикселей при шаге 1м)

После обнаружения границы леса, скрипт заполняет всю область от нижней 
границы перепада вверх до следующей границы (вершины).

Вход: TIF файл с DEM (float32, NaN для безданных)
Выход: TIF файл с маской (1 - лес, 0 - не лес)
"""

import numpy as np
import rasterio
from scipy.ndimage import maximum_filter, minimum_filter, binary_dilation, label
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
        # Шаг 1: Находим границы леса (резкие перепады высот)
        window_size = 2 * distance_threshold + 1
        
        nan_mask = np.isnan(dem)
        
        # Вычисляем локальный максимум и минимум в окне
        local_max = maximum_filter(dem, size=window_size, 
                                    mode='constant', cval=-np.inf)
        local_min = minimum_filter(dem, size=window_size, 
                                    mode='constant', cval=np.inf)
        
        # Разница между максимумом и минимумом
        height_range = local_max - local_min
        
        # Границы леса - где перепад высот превышает порог
        edge_mask = ((height_range >= height_threshold) & valid_mask).astype(np.uint8)
        
        # Шаг 2: Для каждой точки определяем, является ли она нижней или верхней границей
        # Вычисляем градиенты для определения направления склона
        grad_y, grad_x = np.gradient(dem)
        # Заменяем NaN градиенты на 0
        grad_y = np.nan_to_num(grad_y, nan=0.0)
        grad_x = np.nan_to_num(grad_x, nan=0.0)
        
        # Magnitude градиента
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Шаг 3: Определяем направление "вниз" от каждой точки границы
        # Точки с резким перепадом могут быть либо нижней, либо верхней границей леса
        # Нам нужно найти нижние границы и заполнить область вверх от них
        
        # Создаем маркер нижней границы: там где есть перепад и высота ниже соседней области
        # Используем эрозию/дилатацию для определения направления
        
        # Дилатация маски границ для создания начальных маркеров
        dilated_edges = binary_dilation(edge_mask.astype(bool), iterations=2)
        
        # Шаг 4: Заполнение областей от нижней границы вверх
        # Используем подход с водоразделами (watershed) или рекурсивное заполнение
        
        # Альтернативный подход: для каждой колонки/строки находим перепады и заполняем между ними
        forest_mask = np.zeros_like(dem, dtype=np.uint8)
        
        # Проходим по всем строкам и столбцам
        rows, cols = dem.shape
        
        # Обработка по строкам (горизонтальное сканирование)
        for i in range(rows):
            row_data = dem[i, :]
            row_valid = valid_mask[i, :]
            row_edges = edge_mask[i, :]
            
            if not np.any(row_valid):
                continue
            
            # Находим индексы границ в этой строке
            edge_indices = np.where(row_edges)[0]
            
            if len(edge_indices) < 2:
                continue
            
            # Проверяем пары соседних границ
            for j in range(len(edge_indices) - 1):
                idx1 = edge_indices[j]
                idx2 = edge_indices[j + 1]
                
                # Пропускаем если слишком далеко
                if idx2 - idx1 > distance_threshold * 2:
                    continue
                
                h1 = row_data[idx1]
                h2 = row_data[idx2]
                
                if np.isnan(h1) or np.isnan(h2):
                    continue
                
                # Определяем какая граница нижняя, какая верхняя
                if h1 < h2:
                    lower_idx, upper_idx = idx1, idx2
                else:
                    lower_idx, upper_idx = idx2, idx1
                
                # Проверяем что перепад достаточный
                if abs(h2 - h1) >= height_threshold:
                    # Заполняем от нижней до верхней границы
                    forest_mask[i, lower_idx:upper_idx+1] = 1
        
        # Обработка по столбцам (вертикальное сканирование)
        for j in range(cols):
            col_data = dem[:, j]
            col_valid = valid_mask[:, j]
            col_edges = edge_mask[:, j]
            
            if not np.any(col_valid):
                continue
            
            # Находим индексы границ в этом столбце
            edge_indices = np.where(col_edges)[0]
            
            if len(edge_indices) < 2:
                continue
            
            # Проверяем пары соседних границ
            for k in range(len(edge_indices) - 1):
                idx1 = edge_indices[k]
                idx2 = edge_indices[k + 1]
                
                # Пропускаем если слишком далеко
                if idx2 - idx1 > distance_threshold * 2:
                    continue
                
                h1 = col_data[idx1]
                h2 = col_data[idx2]
                
                if np.isnan(h1) or np.isnan(h2):
                    continue
                
                # Определяем какая граница нижняя, какая верхняя
                if h1 < h2:
                    lower_idx, upper_idx = idx1, idx2
                else:
                    lower_idx, upper_idx = idx2, idx1
                
                # Проверяем что перепад достаточный
                if abs(h2 - h1) >= height_threshold:
                    # Заполняем от нижней до верхней границы
                    forest_mask[lower_idx:upper_idx+1, j] = 1
        
        # Морфологическое закрытие для заполнения небольших пробелов
        from scipy.ndimage import binary_closing
        forest_mask_bool = forest_mask.astype(bool)
        forest_mask_closed = binary_closing(forest_mask_bool, iterations=2)
        forest_mask = forest_mask_closed.astype(np.uint8)
        
        # Применяем маску валидности
        forest_mask = forest_mask & valid_mask.astype(np.uint8)
    
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
