"""
Модуль содержит функции, связанные со смертью коров.
"""
from typing import List, Dict, Any, Tuple
from agents.cow import Cow
from metrics.collector import MetricsCollector


def process_deaths(cows: List[Cow], metrics_collector: MetricsCollector) -> Tuple[List[Cow], List[Cow], Dict[str, int]]:
    """
    Обрабатывает смерть коров.
    
    Args:
        cows: Список всех коров.
        metrics_collector: Сборщик метрик для регистрации событий смерти.
        
    Returns:
        Кортеж из трех элементов:
        - Список живых коров
        - Список мертвых коров
        - Словарь со статистикой смертей: deaths, deaths_starvation, deaths_natural
    """
    # Создаем новый список живых коров
    new_cows = []
    dead_cows = []
    
    # Статистика смертей
    stats = {
        "deaths": 0,
        "deaths_starvation": 0,
        "deaths_natural": 0
    }
    
    for cow in cows:
        # Проверяем условия смерти
        if not cow.alive or cow.energy <= 0:
            # Смерть от голода
            stats["deaths"] += 1
            stats["deaths_starvation"] += 1
            dead_cows.append(cow)
            
            # Использовать централизованный метод _die только если корова еще жива
            if cow.alive:
                cow._die(cause="starvation")
            
            # Уведомляем сборщик метрик о смерти
            metrics_collector.update_accumulated_data(
                "death", {"cow": cow, "cause": "starvation"})
            
        elif cow.should_die():
            # Естественная смерть
            stats["deaths"] += 1
            stats["deaths_natural"] += 1
            dead_cows.append(cow)
            
            # Используем метод _die для централизованной обработки смерти
            cow._die(cause="natural")
            
            # Уведомляем сборщик метрик о смерти
            metrics_collector.update_accumulated_data(
                "death", {"cow": cow, "cause": "natural"})
            
        else:
            # Корова остается жива
            new_cows.append(cow)
    
    return new_cows, dead_cows, stats 