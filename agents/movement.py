"""
Модуль, реализующий алгоритмы пространственного перемещения и навигационного поведения агентов
в модельной экосистеме. Включает функции для оптимизации локомоторной активности на основе
многофакторного анализа окружающей среды и когнитивных характеристик агентов, а также
механизмы социального взаимодействия особей при принятии решений о передвижении.
"""
import random
from typing import Dict, List, Any, Tuple
import numpy as np

from agents.cow import Cow
from environment.world import World
from agents.spatial_index import get_spatial_index, initialize_spatial_index


def circular_distance(a: int, b: int, world_length: int) -> int:
    """
    Вычисляет минимальное круговое расстояние между клетками a и b в замкнутом мире.
    
    Args:
        a: Индекс первой клетки (0-индексация).
        b: Индекс второй клетки (0-индексация).
        world_length: Длина мира.
        
    Returns:
        Минимальное круговое расстояние между клетками.
    """
    # Более эффективная версия без условных ветвлений
    return min(abs(a - b), world_length - abs(a - b))


def compute_snapshot_array(cows: List[Cow], world_length: int) -> np.ndarray:
    """
    Вычисляет snapshot в виде numpy-массива (более эффективная версия).
    
    Args:
        cows: Список всех коров.
        world_length: Длина мира.
        
    Returns:
        Массив: индекс - номер клетки (0..world_length-1), значение - число живых коров в этой клетке.
    """
    # Извлекаем позиции для живых коров
    positions = np.fromiter((cow.position for cow in cows if cow.alive), dtype=np.int64)
    
    if positions.size == 0:
        # Если ни одна корова не жива, возвращаем массив нулей
        return np.zeros(world_length, dtype=np.int64)
        
    # Используем напрямую позиции в 0-индексации
    counts = np.bincount(positions, minlength=world_length)
    return counts


def compute_snapshot(cows: List[Cow], world_length: int) -> Dict[int, int]:
    """
    Вычисляет snapshot (снимок распределения коров по клеткам) быстро с использованием NumPy.
    
    Args:
        cows: Список всех коров.
        world_length: Длина мира.
        
    Returns:
        Словарь: ключ — номер клетки (0..world_length-1), значение — число живых коров в этой клетке.
    """
    # Используем compute_snapshot_array и преобразуем массив в словарь
    counts = compute_snapshot_array(cows, world_length)
    # Создаем словарь только для ненулевых элементов
    snapshot = {i: int(counts[i]) for i in range(world_length) if counts[i] > 0}
    return snapshot


def move_cow(cow: Cow, world: World, snapshot_array: np.ndarray, alpha_F: float = 1.0, alpha_R: float = 1.0) -> None:
    """
    Реализует комплексный алгоритм пространственного перемещения агента, основанный
    на многофакторной оптимизации целевой функции полезности, учитывающей трофические,
    социальные и репродуктивные факторы в контексте когнитивного потенциала особи.
    
    Для каждой потенциальной локации j в пределах перцептивного радиуса (W) и с учетом
    ограничения локомоторной способности (V), вычисляется агрегированная функция полезности:
       U(j) = alpha_F * [F(j,t)/(1 + N(j,t))] + alpha_R * R * N(j,t),
    где:
       F(j,t) - количество доступных пищевых ресурсов в локации j в момент времени t,
       N(j,t) - популяционная плотность в локации j (количество особей),
       R      - индивидуальный коэффициент социальной ориентации особи.
       
    Функция полезности дифференцированно учитывает:
    1) Трофический фактор с поправкой на конкуренцию за ресурсы
    2) Социальный фактор, отражающий тенденцию к формированию популяционных агрегаций
    
    Дополнительно при наличии репродуктивной цели (потенциального партнера) вводится
    компонент пространственного притяжения, модулируемый когнитивным параметром (IQ).
    
    Процесс принятия решения о перемещении оптимизирован с учетом когнитивных способностей
    агента: особи с высоким IQ демонстрируют более рациональный выбор локации при прочих
    равных условиях.
    
    Args:
        cow: Агент, для которого вычисляется и реализуется пространственное перемещение.
        world: Объект экосистемы, содержащий информацию о ресурсах и пространственных характеристиках.
        snapshot_array: Векторизованное представление популяционной плотности по локациям.
        alpha_F: Нормировочный коэффициент для трофического компонента функции полезности.
        alpha_R: Нормировочный коэффициент для социального компонента функции полезности.
    """
    world_length = world.length
    current_pos = cow.position
    
    # Преобразуем V в целое число для использования в range
    V_int = int(cow.V)
    
    # Определяем кандидатный набор клеток: все j, для которых круговое расстояние 
    # от current_pos не превышает V.
    # Используем модуль для обеспечения корректных вычислений в замкнутом кольцевом мире
    start = (current_pos - V_int) % world_length
    candidates = [(start + d) % world_length for d in range(2 * V_int + 1)]

    utilities = []
    for j in candidates:
        F = world.patches[j].grass_amount
        N = int(snapshot_array[j])  # Получаем число коров из массива
        
        # Базовая полезность: трава и стадность
        # Для более умных коров (с высоким IQ) трава имеет большее значение
        food_factor = 1.0 + (cow.IQ * 0.1)  # Коровы с высоким IQ лучше оценивают ресурсы
        U = alpha_F * food_factor * (F / (1 + N)) + alpha_R * cow.R * N
        
        # Если корова имеет целевого партнера, добавляем компонент притяжения к цели
        if cow.target_partner_id is not None and cow.target_position is not None:
            # Если целевой партнер в другой клетке
            if cow.target_position != current_pos:
                # Расстояние до цели
                dist_to_target = circular_distance(j, cow.target_position, world_length)
                
                # Коэффициент притяжения зависит от интеллекта (IQ)
                alpha_A = 0.1 + cow.IQ * 0.5  # Еще сильнее увеличим влияние IQ
                
                # Максимальное расстояние в кольцевом мире - половина длины мира
                max_dist = world_length / 2
                
                # Компонент притяжения: чем ближе к цели, тем выше значение
                attraction = (max_dist - dist_to_target) / max_dist
                
                # Отладочный вывод для проверки применения IQ
                print(f"[DEBUG-ATTRACTION] Корова #{cow.id}, IQ={cow.IQ}, alpha_A={alpha_A:.2f}, притяжение к партнеру, dist={dist_to_target}")
                
                # Если корова в режиме ожидания партнера, снижаем притяжение
                if cow.waiting_for_partner:
                    # Если мы находимся в клетке с партнером, сильно снижаем стимул к движению
                    if current_pos == j:
                        U += alpha_A * 5  # Поощряем оставаться на месте
                else:
                    # Обычное притяжение к целевой позиции
                    U += alpha_A * attraction
        
        utilities.append((j, U))
    
    # находим максимальное значение U
    max_U = max(u for _, u in utilities)

    # отбираем все клетки с U == max_U
    best_cells = [j for j, u in utilities if u == max_U]

    # Если есть несколько клеток с одинаковой полезностью и корова имеет IQ > 0,
    # то вместо случайного выбора используем интеллект для предпочтения ближайших клеток
    if len(best_cells) > 1 and cow.IQ > 0:
        # Создаем список пар (клетка, расстояние)
        cells_with_distances = [(cell, circular_distance(current_pos, cell, world_length)) for cell in best_cells]
        
        # Сортируем клетки по расстоянию (от ближайших к самым дальним)
        cells_with_distances.sort(key=lambda x: x[1])
        
        # Вычисляем веса обратно пропорциональные расстоянию, с учетом интеллекта
        # Чем выше IQ, тем сильнее приоритет ближайших клеток
        weights = []
        for _, dist in cells_with_distances:
            if dist == 0:  # Для избежания деления на ноль
                weight = 10 * cow.IQ  # Высокий вес для текущей клетки
            else:
                # Чем выше IQ, тем сильнее влияние близости клетки
                weight = (1 + (cow.IQ * 0.5)) / dist
            weights.append(weight)
        
        # Нормализуем веса
        total_weight = sum(weights)
        norm_weights = [w/total_weight for w in weights]
        
        # Выбираем клетку с учетом весов
        cells = [cell for cell, _ in cells_with_distances]
        chosen = random.choices(cells, weights=norm_weights, k=1)[0]
        
        # Отладочный вывод
        print(f"[DEBUG-MOVE] Корова #{cow.id}, IQ={cow.IQ} выбрала клетку {chosen} из {len(best_cells)} равноценных вариантов с учетом интеллекта")
    else:
        # Если нет интеллекта или только одна лучшая клетка, выбираем случайную из равноценных
        chosen = random.choice(best_cells)

    # Вычисляем пройденное расстояние как круговое расстояние между текущей и выбранной позицией
    traveled_distance = circular_distance(current_pos, chosen, world_length)
    cow.traveled_distance = traveled_distance
    
    # Сбрасываем ожидание, если переместились
    if chosen != current_pos:
        cow.waiting_for_partner = False
        
        # Запоминаем старую позицию
        old_position = cow.position
        
        # Сначала обновляем позицию, затем пространственный индекс
        cow.position = chosen
        
        # Обновляем пространственный индекс
        try:
            spatial_index = get_spatial_index()
            spatial_index.update_cow_position(cow, old_position)
        except:
            # Игнорируем, если индекс не инициализирован
            pass
    else:
        # Позиция не изменилась
        cow.position = chosen


def move_all_cows(cows: List[Cow], world: World, alpha_F: float = 1.0, alpha_R: float = 1.0) -> None:
    """
    Перемещает всех живых коров в мире.
    
    Args:
        cows: Список всех коров.
        world: Мир, в котором находятся коровы.
        alpha_F: Коэффициент для учета травы в функции полезности.
        alpha_R: Коэффициент для учета стадности в функции полезности.
    """
    # Убеждаемся, что пространственный индекс инициализирован
    try:
        get_spatial_index()
    except ValueError:
        # Если индекс не инициализирован, создаем его
        initialize_spatial_index(world.length)
        get_spatial_index().update_all_cows(cows)
    
    # Сбрасываем traveled_distance у всех коров перед перемещением
    for cow in cows:
        if cow.alive:
            cow.traveled_distance = 0
        
    # Вычисляем snapshot_array - распределение коров по клеткам
    snapshot_array = compute_snapshot_array(cows, world.length)
    
    # Перемещаем каждую живую корову
    for cow in cows:
        if cow.alive:
            move_cow(cow, world, snapshot_array, alpha_F, alpha_R)
    
    # Обновляем пространственный индекс после перемещения всех коров
    get_spatial_index().update_all_cows(cows) 