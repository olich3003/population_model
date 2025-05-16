"""
Модуль содержит класс для пространственного индексирования коров.
"""
from __future__ import annotations
from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.cow import Cow

class SpatialIndex:
    """
    Класс, реализующий пространственное индексирование коров по их позициям.
    Позволяет быстро искать коров в определенной позиции или в радиусе видимости.
    """
    
    def __init__(self, world_length: int):
        """
        Инициализирует пространственный индекс.
        
        Args:
            world_length: Длина мира (максимальное значение позиции).
        """
        self.world_length = world_length
        # Словарь: ключ - позиция, значение - список коров в этой позиции
        self.cows_by_position: Dict[int, List['Cow']] = {}
    
    def add_cow(self, cow: 'Cow') -> None:
        """
        Добавляет корову в индекс.
        
        Args:
            cow: Корова для добавления.
        """
        position = cow.position
        if position not in self.cows_by_position:
            self.cows_by_position[position] = []
        
        # Добавляем корову только если она ещё не в списке
        if cow not in self.cows_by_position[position]:
            self.cows_by_position[position].append(cow)
    
    def remove_cow(self, cow: 'Cow', old_position: int = None) -> None:
        """
        Удаляет корову из индекса.
        
        Args:
            cow: Корова для удаления.
            old_position: Старая позиция коровы (если известна).
                          Если None, используется текущая позиция коровы.
        """
        position = old_position if old_position is not None else cow.position
        
        if position in self.cows_by_position and cow in self.cows_by_position[position]:
            self.cows_by_position[position].remove(cow)
            
            # Удаляем пустой список для экономии памяти
            if not self.cows_by_position[position]:
                del self.cows_by_position[position]
    
    def update_cow_position(self, cow: 'Cow', old_position: int) -> None:
        """
        Обновляет позицию коровы в индексе.
        
        Args:
            cow: Корова для обновления.
            old_position: Старая позиция коровы.
        """
        if old_position != cow.position:
            self.remove_cow(cow, old_position)
            self.add_cow(cow)
    
    def get_cows_within_radius(self, position: int, radius: int) -> List['Cow']:
        """
        Возвращает список коров в пределах указанного радиуса от позиции.
        
        Args:
            position: Центральная позиция.
            radius: Радиус поиска.
            
        Returns:
            Список коров в пределах радиуса.
        """
        result = []
        
        # Вычисляем все клетки в пределах радиуса
        for offset in range(-radius, radius + 1):
            # Учитываем круговую природу мира
            check_position = (position + offset) % self.world_length
            
            # Добавляем коров с этой позиции, если такие есть
            if check_position in self.cows_by_position:
                result.extend(self.cows_by_position[check_position])
        
        return result
    
    def update_all_cows(self, cows: List['Cow']) -> None:
        """
        Обновляет индекс для всех коров в списке.
        
        Args:
            cows: Список всех коров.
        """
        # Сбрасываем индекс
        self.cows_by_position.clear()
        
        # Добавляем только живых коров
        for cow in cows:
            if cow.alive:
                self.add_cow(cow)
    
    def get_cows_at_position(self, position: int) -> List['Cow']:
        """
        Возвращает список коров в указанной позиции.
        
        Args:
            position: Позиция для поиска.
            
        Returns:
            Список коров в этой позиции.
        """
        return self.cows_by_position.get(position, [])


# Создаем глобальный экземпляр индекса
# Он будет инициализирован при первом обращении к модулю
spatial_index = None

def initialize_spatial_index(world_length: int) -> None:
    """
    Инициализирует глобальный пространственный индекс.
    
    Args:
        world_length: Длина мира.
    """
    global spatial_index
    spatial_index = SpatialIndex(world_length)

def get_spatial_index() -> SpatialIndex:
    """
    Возвращает глобальный пространственный индекс.
    
    Returns:
        Экземпляр SpatialIndex.
    """
    global spatial_index
    if spatial_index is None:
        raise ValueError("Spatial index has not been initialized")
    return spatial_index 