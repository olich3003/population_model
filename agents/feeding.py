"""
Модуль содержит функции, связанные с питанием коров.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
from agents.cow import Cow
from environment.world import World
from agents.movement import compute_snapshot_array
from agents.spatial_index import get_spatial_index


def water_filling(capacities: List[float], total: float) -> List[float]:
    """
    Алгоритм water-filling для оптимального распределения ресурса.
    
    Распределяет ограниченный ресурс (total) между потребителями с разными
    максимальными емкостями (capacities) по принципу выравнивания уровней.
    
    Args:
        capacities: Список максимальных емкостей потребителей.
        total: Общее количество ресурса для распределения.
        
    Returns:
        Список распределенных объемов ресурса для каждого потребителя.
    """
    # Используем numpy для быстрых операций с массивами
    capacities_array = np.array(capacities, dtype=float)
    n = len(capacities_array)
    
    # Если общий ресурс превышает сумму всех емкостей - просто возвращаем емкости
    if total >= np.sum(capacities_array):
        return capacities_array.tolist()
    
    # Сортируем емкости для применения алгоритма
    sorted_idx = np.argsort(capacities_array)
    sorted_capacities = capacities_array[sorted_idx]
    
    allocation = np.zeros(n, dtype=float)
    remaining = total
    prev_level = 0.0
    
    for j in range(n):
        current_cap = sorted_capacities[j]
        delta = current_cap - prev_level
        water_needed = delta * (n - j)
        
        if water_needed <= remaining:
            # Можем поднять уровень воды до текущей емкости
            remaining -= water_needed
            prev_level = current_cap
        else:
            # Распределяем оставшуюся воду равномерно
            level = prev_level + remaining / (n - j)
            allocation[sorted_idx[j:]] = level
            break
            
        allocation[sorted_idx[j]] = current_cap
    
    return allocation.tolist()


def feed_all_cows(cows: List[Cow], world: World) -> None:
    """
    Коллективное питание коров с распределением травы.
    
    Вычисляет snapshot один раз, затем для каждой занятой клетки
    распределяет траву между коровами на основе их stomach_capacity
    с использованием алгоритма water-filling.
    
    Оптимизировано для использования пространственного индекса и numpy-массивов.
    
    Args:
        cows: Список всех коров.
        world: Мир, в котором находятся коровы.
    """
    # 1) Снимок количества коров на каждой клетке в виде массива
    snapshot_array = compute_snapshot_array(cows, world.length)
    
    try:
        # 2) Пытаемся использовать пространственный индекс для группировки коров
        spatial_index = get_spatial_index()
        
        # Получаем все позиции, где есть коровы
        occupied_positions = list(spatial_index.cows_by_position.keys())
        
        # 3) Питание по клеткам
        for pos in occupied_positions:
            count = snapshot_array[pos]
            if count == 0:
                continue
                
            # Получаем коров на этой позиции из индекса
            cow_list = spatial_index.get_cows_at_position(pos)
            if not cow_list:
                continue
                
            # Только живые коровы могут питаться
            live_cows = [cow for cow in cow_list if cow.alive]
            if not live_cows:
                continue
            
            patch = world.patches[pos]
            F = patch.grass_amount
            
            if len(live_cows) == 1:
                # Если одна корова на клетке
                cow = live_cows[0]
                eaten = min(cow.stomach_capacity, F)
                cow.energy += cow.M * eaten
                patch.grass_amount -= eaten
            else:
                # Если 2 и более коров — равномерное распределение через water_filling
                capacities = [cow.stomach_capacity for cow in live_cows]
                allocations = water_filling(capacities, F)
                total_eaten = 0.0
                for cow, eat in zip(live_cows, allocations):
                    cow.energy += (cow.M/10) * eat
                    total_eaten += eat
                patch.grass_amount -= total_eaten
    
    except:
        # Если пространственный индекс недоступен - используем базовую группировку
        # Ищем позиции, где есть коровы (ненулевые значения в snapshot_array)
        occupied_positions = np.nonzero(snapshot_array)[0]
        
        # Группируем коров по клеткам
        for pos in occupied_positions:
            # Собираем живых коров в данной позиции
            cow_list = [cow for cow in cows if cow.alive and cow.position == pos]
            if not cow_list:
                continue
            
            patch = world.patches[pos]
            F = patch.grass_amount
            
            if len(cow_list) == 1:
                # Если одна корова на клетке
                cow = cow_list[0]
                eaten = min(cow.stomach_capacity, F)
                cow.energy += cow.M * eaten
                patch.grass_amount -= eaten
            else:
                # Если 2 и более коров — равномерное распределение через water_filling
                capacities = [cow.stomach_capacity for cow in cow_list]
                allocations = water_filling(capacities, F)
                total_eaten = 0.0
                for cow, eat in zip(cow_list, allocations):
                    cow.energy += cow.M * eat
                    total_eaten += eat
                patch.grass_amount -= total_eaten 