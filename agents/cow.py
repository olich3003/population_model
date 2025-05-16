"""
Модуль, реализующий основной агентный класс Cow, моделирующий особей популяции с учетом 
биологических, физических и когнитивных параметров в рамках эволюционной симуляции.
"""
from __future__ import annotations
import random
import math
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING

from agents.spatial_index import get_spatial_index

class Cow:
    """
    Класс Cow моделирует индивидуум популяции с системой параметров, эмулирующих биологические, 
    физические и когнитивные характеристики, а также содержит методы для взаимодействия с окружающей средой.
    Реализует основные аспекты жизнедеятельности: передвижение, питание, размножение и смертность.

    Attributes:
        S: Фенотипический параметр размера (масса/габариты), нормализованное значение [1, 10], 
           влияющий на энергетические затраты и метаболизм.
        V: Локомоторная характеристика (скорость) в интервале [1, 10], определяющая 
           максимальное расстояние перемещения за единицу дискретного времени (тик).
        M: Интенсивность метаболизма [1, 10], влияющая на скорость расхода энергетических ресурсов.
        IQ: Когнитивный параметр интеллекта [0, 10], определяющий сложность стратегии принятия решений;
            значение 0 соответствует примитивному реактивному поведению.
        W: Перцептивная характеристика (радиус видимости/восприятия) [1, 10], ограничивающая 
           объем воспринимаемой информации об окружающей среде.
        R: Социальная характеристика (склонность к групповому поведению) [1, 10],
           определяющая тенденцию к формированию коллективных структур.
        T0: Онтогенетический параметр, определяющий теоретическую продолжительность жизни 
            в годах [1, 10] при отсутствии внешних негативных факторов.
        stomach_capacity: Вместимость пищеварительной системы, рассчитываемая как произведение
                          базового параметра STOMACH_SIZE на размер (S).
        age: Хронологический возраст особи в тиках симуляции.
        energy: Текущий энергетический запас, определяющий жизнеспособность.
        alive: Булев индикатор актуального жизненного статуса.
        id: Уникальный идентификатор особи для отслеживания и анализа популяционной динамики.
        position: Пространственные координаты в одномерном мире (0-индексация).
        target_partner_id: Идентификатор потенциального партнера для репродукции.
        target_position: Пространственные координаты потенциального партнера.
        waiting_for_partner: Флаг ожидания партнера, предотвращающий циклические перемещения.
        blacklist_partner_ids: Множество идентификаторов неподходящих партнеров для оптимизации
                              процесса поиска репродуктивного партнера.
        reproduction_age_threshold: Минимальный возраст половой зрелости (предвычисленное значение).
        traveled_distance: Расстояние, пройденное за последний временной такт симуляции.
        fixed_params: Список генетически фиксированных параметров, не подлежащих мутации.
    """
    
    # Определение слотов для оптимизации памяти и доступа к атрибутам
    __slots__ = (
        "id","S","V","M","IQ","W","R","T0",
        "mutation_rate","sexual_reproduce","death_params",
        "stomach_capacity","age","position","energy","alive",
        "target_partner_id","target_position","waiting_for_partner","blacklist_partner_ids",
        "reproduction_age_threshold", "traveled_distance", "fixed_params"
    )

    # Класс-переменная для автоматического присвоения уникальных идентификаторов
    next_id = 1
    
    # Добавляем статический кеш для вероятностей смерти
    _death_probability_cache: Dict[Tuple[int, int], float] = {}

    # Статический словарь для O(1) доступа к коровам по ID
    _cows_by_id: Dict[int, 'Cow'] = {}

    def __init__(
        self, 
        position: int,
        energy: float, 
        S: float, 
        V: float, 
        M: float, 
        IQ: float, 
        W: float, 
        R: float, 
        T0: float,
        mutation_rate: float = 0.1,
        sexual_reproduce: int = 1,
        stomach_size: int = 6,
        death_params: Dict = None,
        id: Optional[int] = None,
        fixed_params: List[str] = None
    ):
        """
        Инициализирует корову с заданными параметрами.
        
        Args:
            S: Размер.
            V: Скорость.
            M: Метаболизм.
            IQ: Интеллект.
            W: Зрение.
            R: Стадность.
            T0: Максимальное время жизни (в годах).
            position: Начальная позиция коровы в мире (0-индексация).
            energy: Начальная энергия коровы.
            mutation_rate: Вероятность мутации параметров при размножении.
            sexual_reproduce: Флаг полового размножения (1 - половое, 0 - бесполое).
            stomach_size: Базовый размер желудка на единицу размера S.
            death_params: Параметры смертности для разных значений T0.
            id: Уникальный идентификатор (если None, то назначается автоматически).
            fixed_params: Список параметров, которые не должны мутировать.
        """
        if id is None:
            self.id = Cow.next_id
            Cow.next_id += 1
        else:
            self.id = id
            if id >= Cow.next_id:
                Cow.next_id = id + 1
                
        # Добавляем корову в словарь
        Cow._cows_by_id[self.id] = self

        self.S = S        # размер
        self.V = V        # скорость
        self.M = M        # метаболизм
        self.IQ = IQ      # интеллект (для "тупых" коров IQ = 0)
        self.W = W        # зрение
        self.R = R        # стадность
        self.T0 = T0      # время жизни
        
        # Сохраняем список фиксированных параметров
        self.fixed_params = fixed_params or []
        
        # Сохраняем параметры конфигурации
        self.mutation_rate = mutation_rate
        self.sexual_reproduce = sexual_reproduce
        # Преобразуем ключи death_params к целым числам для оптимизации доступа
        self.death_params = {int(k): v for k, v in (death_params or {}).items()}
        
        # Вычисляем вместимость желудка: stomach_size единиц травы на каждую единицу размера S
        self.stomach_capacity = stomach_size * S

        # Дополнительные атрибуты динамики
        self.age = 0
        self.position = position
        self.energy = energy
        self.alive = True
        self.traveled_distance = 0  # Инициализируем пройденное расстояние
        
        # Атрибуты для улучшенной стратегии размножения
        self.target_partner_id = None  # ID целевого партнера
        self.target_position = None    # Позиция целевого партнера
        self.waiting_for_partner = False  # Ожидание партнера (предотвращение "качелей")
        self.blacklist_partner_ids = set()  # Черный список партнеров
        
        # Предвычисляем порог возраста для размножения
        self.reproduction_age_threshold = 0.2 * self.T0 * 100  # 20% от максимального возраста в тиках
        
        # Добавляем корову в пространственный индекс при создании
        try:
            spatial_index = get_spatial_index()
            spatial_index.add_cow(self)
        except (ImportError, ValueError):
            # Игнорируем, если индекс не инициализирован или модуль не найден
            pass

    def __repr__(self) -> str:
        """
        Возвращает строковое представление коровы.
        
        Returns:
            Строка с параметрами коровы.
        """
        return (f"Cow(id={self.id}, S={self.S}, V={self.V}, M={self.M}, IQ={self.IQ}, "
                f"W={self.W}, R={self.R}, T0={self.T0}, stomach_capacity={self.stomach_capacity}, "
                f"age={self.age}, energy={self.energy:.2f}, alive={self.alive})")

    def expend_energy(self, world: Any) -> None:
        """
        Реализует механизм энергетических затрат особи в течение одного временного такта,
        моделируя термодинамическую адаптивность организмов разного размера к температурным условиям среды.
        
        Энергетические затраты рассчитываются по формуле:
          ΔE = AT_n * (self.S + self.traveled_distance) * temp_factor,
        где:
        - AT_n — температурный показатель на текущей локации (дискретная величина 0-3),
        - temp_factor — адаптивный термодинамический коэффициент, зависящий от параметра размера (S):
            * в высокотемпературных условиях (AT_n = 2 или 3): крупные особи тратят больше энергии,
              что соответствует биологическому принципу терморегуляции (temp_factor = S/S_max)
            * в низкотемпературных условиях (AT_n = 0 или 1): крупные особи тратят пропорционально
              меньше энергии благодаря более выгодному соотношению поверхности тела к объему
              (temp_factor = 1 - S/S_max)
        
        При исчерпании энергетического запаса (energy ≤ 0) происходит инициация процесса 
        прекращения жизнедеятельности особи (alive = False).
        
        Args:
            world: Экземпляр класса World, содержащий информацию об экосистеме.
        """
        # Получаем температуру на patch, где стоит корова (position уже в 0-индексации)
        AT_n = world.patches[self.position].temperature
        
        # Значение S_max по умолчанию
        S_max = 10
        # Определяем коэффициент температуры в зависимости от размера и температуры
        if AT_n in (2, 3):  # жарко
            # большие коровы тратят больше
            energy_loss = AT_n * (self.S + self.traveled_distance)
        else:                # AT_n == 0 или 1 (холодно)
            # большие коровы тратят меньше
            energy_loss = AT_n * ((S_max - self.S + 1) + self.traveled_distance )
        
        # Считаем потерю энергии
        #energy_loss = AT_n * (self.S + self.traveled_distance * self.V) 
        
        # Обновляем запас энергии
        self.energy -= energy_loss
        
        # Если энергии не осталось — корова умирает
        if self.energy <= 0:
            self._die(cause="starvation")

    def should_die(self) -> bool:
        """
        Проверяет, должна ли корова умереть, основываясь на законе смертности Гомпертца-Мейкхама.
        
        Returns:
            True если корова должна умереть, иначе False
        """
        cache_key = (self.T0, self.age)
        if cache_key not in Cow._death_probability_cache:
            # Используем целочисленные ключи для death_params
            if not self.death_params or int(self.T0) not in self.death_params:
                # Используем значения по умолчанию, если нет параметров
                a, b, c = 0.001, 0.0001, 0.1
            else:
                a, b, c = self.death_params[int(self.T0)]
                
            # Используем полную формулу закона Гомпертца-Мейкхама
            p_death = a + b * math.exp(c * self.age)
            Cow._death_probability_cache[cache_key] = p_death
        else:
            p_death = Cow._death_probability_cache[cache_key]
            
        return random.random() < p_death

    def ready_for_reproduction(self, birth_energy: float = 500) -> bool:
        """
        Проверяет, готова ли корова к размножению.
        
        Корова готова к размножению, если:
        1. Жива
        2. Достигла 20% своего максимального возраста T0
        3. Имеет достаточно энергии (> birth_energy)
        
        Args:
            birth_energy: Энергия, необходимая для размножения
            
        Returns:
            True, если корова готова к размножению, иначе False.
        """
        if not self.alive:
            return False
            
        # Используем предвычисленный порог возраста
        return self.age >= self.reproduction_age_threshold and self.energy > birth_energy

    def find_mate(self, cows: List['Cow'], birth_energy: float = 500) -> None:
        """
        Выбирает партнера для размножения из списка коров по сумме параметров.
        Оптимизированная версия, использующая пространственный индекс для поиска только среди 
        коров в радиусе видимости W.
        
        Корова выбирает партнера, основываясь на:
        1. Готовности к размножению
        2. Сумме параметров (S, V, M, IQ, W, R, T0)
        3. Видимости (в пределах радиуса видимости W)
        
        Args:
            cows: Список всех коров в симуляции
            birth_energy: Минимальная энергия для размножения
        """
        # Отладочный вывод для проверки вызова функции find_mate и значения IQ
        #print(f"[DEBUG-MATE] Корова #{self.id}, IQ={self.IQ}, ищет партнера. Готова к размножению: {self.ready_for_reproduction(birth_energy)}")
        
        # Сразу выходим из функции, если корова не готова к размножению
        if not self.alive or not self.ready_for_reproduction(birth_energy):
            self.target_partner_id = None
            self.target_position = None
            self.waiting_for_partner = False
            return
        
        # Если у коровы уже есть целевой партнер, проверяем его статус
        if self.target_partner_id is not None:
            # Используем глобальный словарь для O(1) доступа к коровам по ID
            if self.target_partner_id in Cow._cows_by_id:
                target = Cow._cows_by_id[self.target_partner_id]
                if target.alive and target.ready_for_reproduction(birth_energy):
                    # Обновляем только позицию партнера
                    self.target_position = target.position
                    return
            
            # Партнер умер или не готов, сбрасываем цель
            self.target_partner_id = None
            self.target_position = None
            self.waiting_for_partner = False
        
        # Используем пространственный индекс для поиска коров в радиусе видимости
        try:
            spatial_index = get_spatial_index()
            
            # Получаем коров в радиусе W (зрение)
            nearby_cows = spatial_index.get_cows_within_radius(self.position, int(self.W))
            
            # Находим лучшего партнера среди соседних коров
            best_partner = None
            best_params_sum = -1
            
            for cow in nearby_cows:
                # Пропускаем себя, мертвых коров и коров, не готовых к размножению
                if (cow.id == self.id or 
                    not cow.alive or 
                    not cow.ready_for_reproduction(birth_energy) or
                    cow.id in self.blacklist_partner_ids):
                    continue
                
                # Вычисляем сумму параметров
                params_sum = cow.S + cow.V + cow.M + cow.IQ + cow.W + cow.R + cow.T0
                
                # Если это лучший партнер из найденных, запоминаем его
                if params_sum > best_params_sum:
                    best_params_sum = params_sum
                    best_partner = cow
        except:
            # Если пространственный индекс не доступен, используем базовый поиск
            best_partner = None
            best_params_sum = -1
            
            for cow in cows:
                # Пропускаем себя, мертвых коров и коров, не готовых к размножению
                if (cow.id == self.id or 
                    not cow.alive or 
                    not cow.ready_for_reproduction(birth_energy) or
                    cow.id in self.blacklist_partner_ids):
                    continue
                
                # Вычисляем сумму параметров
                params_sum = cow.S + cow.V + cow.M + cow.IQ + cow.W + cow.R + cow.T0
                
                # Если это лучший партнер из найденных, запоминаем его
                if params_sum > best_params_sum:
                    best_params_sum = params_sum
                    best_partner = cow
        
        # Если нашли подходящего партнера, устанавливаем цель
        if best_partner:
            self.target_partner_id = best_partner.id
            self.target_position = best_partner.position
        else:
            # Не нашли партнера, сбрасываем цель
            self.target_partner_id = None
            self.target_position = None
            self.waiting_for_partner = False

    @staticmethod
    def mutate_param(value: float, mutation_rate: float, param_ranges: Dict = None, param_name: str = None, fixed_params: List[str] = None) -> int:
        """
        С вероятностью mutation_rate меняет параметр value на ±1,
        в пределах заданного диапазона, если параметр не является фиксированным.
        
        Args:
            value: Исходное значение параметра.
            mutation_rate: Вероятность мутации.
            param_ranges: Словарь с диапазонами параметров (если None, используется [1, 10]).
            param_name: Имя параметра (для проверки, является ли он фиксированным).
            fixed_params: Список параметров, которые не должны мутировать.
            
        Returns:
            Новое значение параметра (возможно, мутировавшее) как целое число.
        """
        # Если параметр фиксирован, не мутируем его
        if fixed_params and param_name and param_name in fixed_params:
            return int(value)
            
        if random.random() < mutation_rate:
            # По умолчанию все параметры в диапазоне [1, 10]
            low, high = 1, 10
                
            delta = random.choice([-1, 1])
            new_value = int(value) + delta
            # Ограничиваем диапазон
            return max(low, min(new_value, high))
        return int(value)
    
    @staticmethod
    def update_ages(cows: List['Cow']) -> None:
        """
        Увеличивает возраст всех живых коров.
        
        Args:
            cows: Список всех коров.
        """
        for cow in cows:
            if cow.alive:
                cow.age += 1 

    def _die(self, cause: str = None) -> None:
        """
        Вспомогательный метод для обработки смерти коровы.
        Централизованно обрабатывает удаление из всех структур данных и индексов.
        
        Args:
            cause: Причина смерти (опционально)
        """
        # Помечаем как мертвую
        self.alive = False
        
        # Удаляем из словаря
        if self.id in Cow._cows_by_id:
            del Cow._cows_by_id[self.id]
        
        # Удаляем из пространственного индекса
        try:
            spatial_index = get_spatial_index()
            spatial_index.remove_cow(self)
        except:
            # Игнорируем, если индекс не инициализирован или модуль не найден
            pass 