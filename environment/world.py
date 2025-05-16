"""
Модуль, реализующий многокомпонентную модель экосистемы через класс World, 
обеспечивающий репрезентацию пространственных, климатических и биогеоценотических 
характеристик среды обитания с применением векторизованной вычислительной архитектуры
для эффективной обработки экологических процессов.
"""
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

class DirectAccessList:
    """
    Класс для прямого доступа к данным клеток мира для обратной совместимости.
    Заменяет PatchWrapper и PatchEmulator на более эффективную реализацию.
    """
    def __init__(self, world):
        self.world = world
        
    def __getitem__(self, index: Union[int, slice]):
        """
        Эмулирует доступ к патчу по индексу.
        Возвращает словарь с доступом к атрибутам патча.
        """
        if isinstance(index, slice):
            # Если запрашивается срез, возвращаем список словарей
            indices = range(*index.indices(self.world.length))
            return [PatchProxyDict(self.world, i) for i in indices]
        
        # Проверяем допустимость индекса
        if 0 <= index < self.world.length:
            return PatchProxyDict(self.world, index)
        return None
    
    def __len__(self):
        """
        Возвращает длину списка патчей.
        """
        return self.world.length

class PatchProxyDict:
    """
    Словарь-прокси для доступа к атрибутам патча.
    Обеспечивает интерфейс, совместимый с прошлой реализацией Patch.
    """
    def __init__(self, world, index):
        self.world = world
        self.index = index
        
    @property
    def grass_amount(self):
        """Получить текущее количество травы"""
        return self.world._grass[self.index]
        
    @grass_amount.setter
    def grass_amount(self, value):
        """Установить количество травы"""
        self.world._grass[self.index] = min(value, self.world._max_grass[self.index])
    
    @property
    def climate_zone(self):
        """Получить климатическую зону"""
        return self.world._climate_zones[self.index]
    
    @property
    def soil_quality(self):
        """Получить тип почвы"""
        return self.world._soil_types[self.index]
    
    @property
    def temperature(self):
        """Получить температуру"""
        return self.world._temperature[self.index]
    
    @temperature.setter
    def temperature(self, value):
        """Установить температуру"""
        self.world._temperature[self.index] = value
    
    @property
    def max_grass(self):
        """Получить максимальное количество травы"""
        return self.world._max_grass[self.index]

class World:
    """
    Фундаментальный класс, моделирующий целостную экосистему с вариативными
    пространственно-временными и биоклиматическими характеристиками.
    Реализует эффективную обработку экологических процессов с использованием
    высокопроизводительных векторизованных вычислений на базе библиотеки NumPy.
    
    Attributes:
        length: Протяженность экосистемы, выраженная в количестве дискретных локаций.
        current_tick: Счетчик временных тактов с начала симуляции, определяющий
                      текущее состояние сезонных и иных временно-зависимых процессов.
        _climate_zones: Пространственное распределение климатических зон (полярная, средняя, экваториальная).
        _soil_types: Дистрибуция типов почвы, влияющая на продуктивность растительности.
        _grass: Вектор текущих значений растительной биомассы для каждой локации.
        _max_grass: Вектор потенциальных максимумов растительной биомассы, обусловленных
                    экологической ёмкостью каждой локации.
        _temperature: Вектор текущих температурных показателей для каждой локации.
    """
    
    # Шаблоны температуры для разных климатических зон и сезонов
    temperature_patterns = {
        'Polar':   [0, 1, 2, 1],
        'Middle':  [1, 2, 3, 2],
        'Equator': [2, 3, 3, 2]
    }
    
    # Словарь для преобразования строкового названия климатической зоны в индекс
    climate_zone_to_idx = {
        'Polar': 0,
        'Middle': 1,
        'Equator': 2
    }
    
    # Словарь для преобразования типа почвы в коэффициент роста
    soil_quality_to_factor = {
        'Desert': 0,
        'Steppe': 1,
        'Forest': 2,
        'Field': 3
    }
    
    # Коэффициенты начальной травы в зависимости от типа почвы
    initial_grass_factors = {
        'Desert': 0.2,
        'Steppe': 0.4,
        'Field': 0.8,
        'Forest': 0.6
    }

    def __init__(self, length: int, max_grass_amount: float = 40, grass_speed: float = 0.03,
                 climate: Dict[str, List[Tuple[int, int]]] = None, 
                 soil: Dict[str, List[Tuple[int, int]]] = None,
                 ticks_per_year: int = 100,
                 monotonous_mode: bool = False,
                 monotonous_zone: str = None):
        """
        Инициализирует мир с заданной конфигурацией, используя векторизованное хранение данных.
        
        Args:
            length: Длина мира (количество клеток).
            max_grass_amount: Максимальное количество травы на клетке.
            grass_speed: Скорость роста травы.
            climate: Словарь с климатическими зонами и их диапазонами.
            soil: Словарь с типами почвы и их диапазонами.
            ticks_per_year: Количество тиков в году.
            monotonous_mode: Если True, весь мир имеет одинаковую климатическую зону.
            monotonous_zone: Название климатической зоны при monotonous_mode=True.
        """
        self.length = length
        self.max_grass_amount = max_grass_amount
        self.grass_speed = grass_speed
        self.climate = climate or {}
        self.soil = soil or {}
        self.ticks_per_year = ticks_per_year
        self.current_tick = 0
        self.monotonous_mode = monotonous_mode
        self.monotonous_zone = monotonous_zone
        
        print(f"[INFO] Создание мира (длина {length}, monotonous_mode={monotonous_mode}, monotonous_zone={monotonous_zone})")
        
        # Создаем массивы для всех параметров клеток
        self._climate_zones = np.array(['Middle'] * self.length, dtype=object)
        self._soil_types = np.array(['Desert'] * self.length, dtype=object)
        
        # Индексы климатических зон (0-Polar, 1-Middle, 2-Equator)
        self._zone_idx = np.zeros(self.length, dtype=np.int8)
        
        # Коэффициенты почвы для вычисления роста травы
        self._soil_factor = np.zeros(self.length, dtype=np.int8)
        
        # Текущее количество травы и максимальное
        self._grass = np.zeros(self.length, dtype=np.float32)
        self._max_grass = np.full(self.length, max_grass_amount, dtype=np.float32)
        
        # Текущая температура
        self._temperature = np.zeros(self.length, dtype=np.int8)
        
        # Создаем 2D таблицу температур (зона x сезон)
        self._temp_table = np.array([
            [0, 1, 2, 1],  # Polar
            [1, 2, 3, 2],  # Middle
            [2, 3, 3, 2]   # Equator
        ], dtype=np.int8)
        
        # Создаем список для обратной совместимости
        self.patches = DirectAccessList(self)
        
        # Инициализируем все массивы
        self.init_patches()

    def init_patches(self) -> None:
        """
        Инициализирует все клетки мира векторными операциями.
        """
        print(f"[DEBUG] Инициализация патчей. Длина мира: {self.length}, monotonous_mode={self.monotonous_mode}, monotonous_zone={self.monotonous_zone}")
        
        if self.length <= 0:
            print("[ERROR] Некорректная длина мира")
            self.length = 100  # Минимальная длина
        
        # Если включен монотонный режим, заполняем весь мир одной климатической зоной
        if self.monotonous_mode and self.monotonous_zone:
            print(f"[INFO] Создание монотонного мира с зоной {self.monotonous_zone}")
            self._climate_zones.fill(self.monotonous_zone)
        else:
            # Иначе заполняем массивы типов климата согласно карте
            for i in range(self.length):
                self._climate_zones[i] = self.get_climate_for_cell(i)
        
        # Заполняем типы почвы
        for i in range(self.length):
            self._soil_types[i] = self.get_soil_for_cell(i)
        
        # Преобразуем строковые типы в числовые индексы
        climate_map = np.vectorize(lambda x: self.climate_zone_to_idx.get(x, 1))
        soil_map = np.vectorize(lambda x: self.soil_quality_to_factor.get(x, 0))
        initial_grass_map = np.vectorize(lambda x: self.initial_grass_factors.get(x, 0.3))
        
        # Применяем векторные функции для заполнения массивов
        self._zone_idx = climate_map(self._climate_zones)
        self._soil_factor = soil_map(self._soil_types)
        
        # Инициализируем траву с учетом типа почвы
        grass_factors = initial_grass_map(self._soil_types)
        self._grass = grass_factors * self._max_grass
        
        # Выводим распределение климатических зон для отладки
        unique, counts = np.unique(self._climate_zones, return_counts=True)
        zone_distribution = dict(zip(unique, counts))
        print(f"[INFO] Распределение климатических зон: {zone_distribution}")

    def get_climate_for_cell(self, cell_number: int) -> str:
        """
        Определяет климатическую зону для заданной клетки.
        
        Args:
            cell_number: Номер клетки.
            
        Returns:
            Название климатической зоны.
        """
        # Если включен монотонный режим, возвращаем заданную зону
        if self.monotonous_mode and self.monotonous_zone:
            return self.monotonous_zone
            
        # Перебираем все климатические зоны
        for zone_name, zone_ranges in self.climate.items():
            for start, end in zone_ranges:
                if start <= cell_number <= end:
                    return zone_name
                    
        # По умолчанию возвращаем среднюю зону
        return 'Middle'

    def get_soil_for_cell(self, cell_number: int) -> str:
        """
        Определяет тип почвы для заданной клетки.
        
        Args:
            cell_number: Номер клетки (0-индексация).
            
        Returns:
            Строка с названием типа почвы.
        """
        # Проверяем все типы почв из конфигурации
        for soil_type, ranges in self.soil.items():
            for start, end in ranges:
                if start <= cell_number <= end:
                    return soil_type
        
        # По умолчанию возвращаем пустыню
        return 'Desert'

    def update_season(self) -> None:
        """
        Векторно обновляет сезон в мире и температуру для всех клеток.
        """
        # Определяем индекс сезона (0, 1, 2, 3)
        season_index = ((self.current_tick - 1) % self.ticks_per_year) // (self.ticks_per_year // 4)
        season_index = min(season_index, 3)
        
        # print(f"[DEBUG] Обновление сезона. Индекс сезона: {season_index}, тик: {self.current_tick}")
        
        # Векторно обновляем температуру для всех клеток
        self._temperature = self._temp_table[self._zone_idx, season_index]

    def update(self) -> None:
        """
        Векторно обновляет состояние мира: обновляет траву на всех клетках
        и при необходимости меняет сезон.
        """
        try:
            # print(f"[DEBUG] Обновление мира на тике {self.current_tick}")
            
            # Векторное обновление травы:
            growth = self._temperature * self._soil_factor * self.grass_speed
            self._grass = np.minimum(self._grass + growth, self._max_grass)
            
            # Увеличиваем счетчик тиков
            self.current_tick += 1
            
            # Проверяем, нужно ли сменить сезон
            ticks_per_season = self.ticks_per_year // 4
            if self.current_tick % ticks_per_season == 0:
                self.update_season()
                
            # print(f"[DEBUG] Мир успешно обновлен, новый тик: {self.current_tick}")
        except Exception as e:
            print(f"[ERROR] Ошибка при обновлении мира: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def assign_climate_zone(self, index: int) -> str:
        """
        Определяет климатическую зону для заданного индекса клетки.
        
        Args:
            index: Индекс клетки (0-индексация).
            
        Returns:
            Строка с названием климатической зоны.
        """
        N = self.length
        L = N // 8  # число клеток в каждой зоне
        
        # Определяем распределение клеток для зоны 1 (нижняя Polar)
        i0_left = L // 2
        i0_right = L - i0_left 
        
        if index < i0_right or index >= (N - i0_left):
            return 'Polar'
        else:
            i_remaining = index - i0_right
            zone_number = 2 + (i_remaining // L)
            zone_mapping = {
                2: 'Middle',
                3: 'Equator',
                4: 'Middle',
                5: 'Polar',
                6: 'Middle',
                7: 'Equator',
                8: 'Middle'
            }
            return zone_mapping.get(zone_number, 'Middle')
    
    # Методы прямого доступа к данным клеток (более эффективны, чем через patches)
    def get_grass(self, index: int) -> float:
        """Получить количество травы в клетке."""
        if 0 <= index < self.length:
            return self._grass[index]
        return 0.0
        
    def set_grass(self, index: int, value: float) -> None:
        """Установить количество травы в клетке."""
        if 0 <= index < self.length:
            self._grass[index] = min(value, self._max_grass[index])
    
    def get_climate_zone(self, index: int) -> str:
        """Получить климатическую зону клетки."""
        if 0 <= index < self.length:
            return self._climate_zones[index]
        return 'Middle'
    
    def get_soil_type(self, index: int) -> str:
        """Получить тип почвы клетки."""
        if 0 <= index < self.length:
            return self._soil_types[index]
        return 'Desert'
    
    def get_temperature(self, index: int) -> int:
        """Получить температуру клетки."""
        if 0 <= index < self.length:
            return self._temperature[index]
        return 0
    
    def set_temperature(self, index: int, value: int) -> None:
        """Установить температуру клетки."""
        if 0 <= index < self.length:
            self._temperature[index] = value 