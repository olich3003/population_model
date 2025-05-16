"""
Модуль, реализующий репродуктивные механизмы и генетические процессы в популяции, включающий
алгоритмы полового и бесполого размножения, процессы генетической рекомбинации и мутации,
а также механизмы отбора репродуктивных партнеров. Обеспечивает моделирование
эволюционной динамики популяции через передачу и модификацию наследственных признаков.
"""
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from agents.cow import Cow
from agents.movement import compute_snapshot_array
from environment.world import World
from agents.spatial_index import get_spatial_index


def _reset_cow_targets(cow: Cow) -> None:
    """
    Сбрасывает цели размножения для коровы.
    
    Args:
        cow: Корова, для которой нужно сбросить цели.
    """
    cow.target_partner_id = None
    cow.target_position = None


def _create_offspring_parameters(
    parent1: Cow, 
    parent2: Optional[Cow] = None, 
    mutation_rate: float = 0.1, 
    use_intelligence: bool = True
) -> Tuple[int, int, int, int, int, int, int]:
    """
    Реализует генетический алгоритм создания фенотипических параметров потомка
    на основе характеристик родительских особей с учетом механизмов наследственности
    и мутагенеза.
    
    В случае бесполого размножения (партеногенеза) параметры наследуются от единственного
    родителя с возможными мутационными изменениями согласно заданной вероятности.
    При половом размножении происходит усреднение параметров двух родителей с последующим
    применением мутационного процесса, что моделирует генетическую рекомбинацию.
    
    Args:
        parent1: Первый родитель (или единственный при бесполом размножении).
        parent2: Второй родитель (None при бесполом размножении).
        mutation_rate: Коэффициент мутабельности генетических параметров [0.0-1.0].
        use_intelligence: Булев параметр, определяющий наличие когнитивной функции у потомства.
        
    Returns:
        Кортеж фенотипических параметров потомка (S, V, M, IQ, W, R, T0).
    """
    # Получаем список фиксированных параметров от родителя
    fixed_params = parent1.fixed_params
    
    if parent2 is None:
        # Бесполое размножение - мутируем параметры одного родителя
        new_S = int(Cow.mutate_param(parent1.S, mutation_rate, param_name="S", fixed_params=fixed_params))
        new_V = int(Cow.mutate_param(parent1.V, mutation_rate, param_name="V", fixed_params=fixed_params))
        new_M = int(Cow.mutate_param(parent1.M, mutation_rate, param_name="M", fixed_params=fixed_params))
        # Если интеллект отключен, устанавливаем IQ=0
        new_IQ = 0 if not use_intelligence else int(Cow.mutate_param(parent1.IQ, mutation_rate, param_name="IQ", fixed_params=fixed_params))
        new_W = int(Cow.mutate_param(parent1.W, mutation_rate, param_name="W", fixed_params=fixed_params))
        new_R = int(Cow.mutate_param(parent1.R, mutation_rate, param_name="R", fixed_params=fixed_params))
        new_T0 = int(Cow.mutate_param(parent1.T0, mutation_rate, param_name="T0", fixed_params=fixed_params))
    else:
        # Половое размножение - берём среднее параметров родителей и мутируем
        # Берём среднее каждого параметра
        new_S = int((parent1.S + parent2.S) / 2)
        new_V = int((parent1.V + parent2.V) / 2)
        new_M = int((parent1.M + parent2.M) / 2)
        # Если интеллект отключен, устанавливаем IQ=0
        new_IQ = 0 if not use_intelligence else int((parent1.IQ + parent2.IQ) / 2)
        new_W = int((parent1.W + parent2.W) / 2)
        new_R = int((parent1.R + parent2.R) / 2)
        new_T0 = int((parent1.T0 + parent2.T0) / 2)

        # Мутабельность (используем mutation_rate от первого родителя)
        mutation_rate = parent1.mutation_rate
        new_S = int(Cow.mutate_param(new_S, mutation_rate, param_name="S", fixed_params=fixed_params))
        new_V = int(Cow.mutate_param(new_V, mutation_rate, param_name="V", fixed_params=fixed_params))
        new_M = int(Cow.mutate_param(new_M, mutation_rate, param_name="M", fixed_params=fixed_params))
        new_IQ = 0 if not use_intelligence else int(Cow.mutate_param(new_IQ, mutation_rate, param_name="IQ", fixed_params=fixed_params))
        new_W = int(Cow.mutate_param(new_W, mutation_rate, param_name="W", fixed_params=fixed_params))
        new_R = int(Cow.mutate_param(new_R, mutation_rate, param_name="R", fixed_params=fixed_params))
        new_T0 = int(Cow.mutate_param(new_T0, mutation_rate, param_name="T0", fixed_params=fixed_params))
    
    return new_S, new_V, new_M, new_IQ, new_W, new_R, new_T0


def _process_cow_pair(
    cow1: Cow, 
    cow2: Cow, 
    base_energy: float, 
    use_intelligence: bool, 
    config: Dict[str, Any],
    birth_energy: float
) -> Optional[Cow]:
    """
    Обрабатывает пару коров для размножения, если обе готовы.
    
    Args:
        cow1: Первая корова.
        cow2: Вторая корова.
        base_energy: Начальная энергия для новорожденной коровы.
        use_intelligence: Флаг, указывающий использовать ли интеллект.
        config: Конфигурация размножения.
        birth_energy: Энергия, необходимая для размножения.
        
    Returns:
        Новорожденная корова или None, если размножение не произошло.
    """
    if (cow1.alive and cow2.alive and 
        cow1.ready_for_reproduction(birth_energy) and 
        cow2.ready_for_reproduction(birth_energy)):
        calf = reproduce_sexually(cow1, cow2, base_energy, use_intelligence, config)
        
        # Сбрасываем цели
        _reset_cow_targets(cow1)
        _reset_cow_targets(cow2)
        
        return calf
    return None


def reproduce_asexually(cow: Cow, base_energy: float = 100, use_intelligence: bool = True, config: Dict[str, Any] = None) -> Cow:
    """
    Бесполое размножение (деление) коровы.
    
    Родитель теряет половину энергии и создаёт потомка.
    Потомок получает базовую энергию, параметры с возможной мутацией,
    возраст = 0, alive = True.
    
    Args:
        cow: Корова-родитель.
        base_energy: Начальная энергия для новорожденной коровы.
        use_intelligence: Флаг, указывающий использовать ли интеллект.
        config: Конфигурация размножения.
        
    Returns:
        Новорожденная корова-потомок.
    """
    if config is None:
        config = {}
        
    # Получаем базовый размер желудка из конфигурации
    stomach_size = config.get("stomach_size", 4)
    
    # родитель отдаёт половину энергии
    cow.energy /= 2

    # Получаем параметры для потомка
    new_S, new_V, new_M, new_IQ, new_W, new_R, new_T0 = _create_offspring_parameters(
        cow, None, cow.mutation_rate, use_intelligence
    )

    # Cоздаем новую корову
    calf = Cow(
        position=cow.position,
        energy=base_energy,
        S=new_S,
        V=new_V,
        M=new_M,
        IQ=new_IQ,
        W=new_W,
        R=new_R,
        T0=new_T0,
        mutation_rate=cow.mutation_rate,
        sexual_reproduce=cow.sexual_reproduce,
        stomach_size=stomach_size,
        death_params=cow.death_params,
        fixed_params=cow.fixed_params
    )
    
    return calf


def reproduce_sexually(parent1: Cow, parent2: Cow, base_energy: float = 100, use_intelligence: bool = True, config: Dict[str, Any] = None) -> Cow:
    """
    Половое размножение двух коров.
    
    Параметры новорожденного = среднее параметров родителей + мутация.
    Энергия новорожденного = base_energy.
    Возраст = 0, alive = True.
    
    Args:
        parent1: Первый родитель.
        parent2: Второй родитель.
        base_energy: Начальная энергия для новорожденной коровы.
        use_intelligence: Флаг, указывающий использовать ли интеллект.
        config: Конфигурация размножения.
        
    Returns:
        Новорожденная корова-потомок.
    """
    if config is None:
        config = {}
        
    # Получаем базовый размер желудка из конфигурации
    stomach_size = config.get("stomach_size", 4)
    
    # Получаем параметры для потомка
    new_S, new_V, new_M, new_IQ, new_W, new_R, new_T0 = _create_offspring_parameters(
        parent1, parent2, parent1.mutation_rate, use_intelligence
    )

    # Создаем новую корову на основе полученных параметров
    calf = Cow(
        position=parent1.position,
        energy=base_energy,
        S=new_S,
        V=new_V,
        M=new_M,
        IQ=new_IQ,
        W=new_W,
        R=new_R,
        T0=new_T0,
        mutation_rate=parent1.mutation_rate,
        sexual_reproduce=parent1.sexual_reproduce,
        stomach_size=stomach_size,
        death_params=parent1.death_params,
        fixed_params=parent1.fixed_params
    )
    
    return calf


def reproduction_phase(cows: List[Cow], world: World, config: Dict[str, Any] = None) -> List[Cow]:
    """
    Порождает потомков согласно режиму размножения.
    
    В случае полового размножения сначала выполняется поиск партнеров,
    затем проверка на взаимность выбора.
    
    Использует пространственный индекс для эффективного поиска партнеров в одной клетке.
    
    Args:
        cows: Список всех коров.
        world: Мир, в котором находятся коровы.
        config: Конфигурация размножения.
        
    Returns:
        Список новорожденных коров.
    """
    if config is None:
        config = {}
    
    # Получаем параметры размножения из конфигурации
    sexual_reproduce = config.get("sexual_reproduce", 1)
    base_energy = config.get("base_energy", 100)
    birth_energy = config.get("birth_energy", 500)  # Энергия, необходимая для размножения
    
    # Проверяем, включен ли интеллект в конфигурации
    use_intelligence = config.get("use_intelligence", True)
    
    # Отладочный вывод для проверки параметра use_intelligence
    print(f"[DEBUG-REPRODUCTION] Фаза размножения. Интеллект: {'ВКЛЮЧЕН' if use_intelligence else 'ВЫКЛЮЧЕН'}, Половое размножение: {sexual_reproduce}")
    
    # Если интеллект отключен, используем упрощенную логику размножения
    if not use_intelligence:
        return reproduction_phase_without_intelligence(cows, sexual_reproduce, base_energy, use_intelligence, config)
    
    # 1) Обновляем цели размножения для всех коров
    for cow in cows:
        # Вызываем find_mate только для живых коров, готовых к размножению при половом размножении
        if cow.alive and sexual_reproduce == 1 and cow.ready_for_reproduction(birth_energy):
            # Вызываем find_mate только для коров, готовых к размножению
            cow.find_mate(cows, birth_energy)
        
    offspring = []
    processed_ids = set()  # Чтобы не обрабатывать одну пару дважды
    
    if sexual_reproduce == 1:
        # ПОЛОВОЕ РАЗМНОЖЕНИЕ: в первую очередь обрабатываем коров в одной клетке
        try:
            # Используем пространственный индекс для эффективной группировки
            spatial_index = get_spatial_index()
            
            # Сначала находим все позиции, где есть коровы
            positions_with_cows = set()
            for cow in cows:
                if cow.alive:
                    positions_with_cows.add(cow.position)
            
            # Обрабатываем каждую позицию отдельно
            for position in positions_with_cows:
                cell_cows = spatial_index.get_cows_at_position(position)
                
                # Обрабатываем только коров с взаимным выбором в этой клетке
                for cow in cell_cows:
                    if not cow.alive or cow.id in processed_ids or not cow.ready_for_reproduction(birth_energy):
                        continue
                        
                    if cow.target_partner_id is not None:
                        # Получаем партнера используя словарь Cow._cows_by_id
                        if cow.target_partner_id in Cow._cows_by_id:
                            partner = Cow._cows_by_id[cow.target_partner_id]
                            
                            if (partner.alive and 
                                partner.ready_for_reproduction(birth_energy) and
                                partner.target_partner_id == cow.id and
                                cow.position == partner.position):
                                
                                # Создаем потомка
                                calf = reproduce_sexually(cow, partner, base_energy, use_intelligence, config)
                                offspring.append(calf)
                                
                                # Отмечаем обоих как обработанных
                                processed_ids.add(cow.id)
                                processed_ids.add(partner.id)
                                
                                # Сбрасываем цели
                                _reset_cow_targets(cow)
                                _reset_cow_targets(partner)
        except:
            # Если пространственный индекс недоступен, используем обычную логику
            pass
            
        # Обрабатываем оставшиеся пары (те, что не в одной клетке)
        for cow in cows:
            if not cow.alive or cow.id in processed_ids:
                continue
                
            if cow.target_partner_id is not None and cow.ready_for_reproduction(birth_energy):
                # Найдем целевого партнера
                partner = next((c for c in cows if c.id == cow.target_partner_id), None)
                
                if partner and partner.alive and partner.ready_for_reproduction(birth_energy):
                    # Проверяем взаимность и нахождение в одной клетке
                    if (partner.target_partner_id == cow.id and 
                        cow.position == partner.position):
                        
                        # Создаем потомка
                        calf = reproduce_sexually(cow, partner, base_energy, use_intelligence, config)
                        offspring.append(calf)
                        
                        # Отмечаем обоих как обработанных
                        processed_ids.add(cow.id)
                        processed_ids.add(partner.id)
                        
                        # Сбрасываем цели
                        _reset_cow_targets(cow)
                        _reset_cow_targets(partner)
                    # Если партнер выбрал другую корову и они в одной клетке, добавляем в черный список
                    elif partner.target_partner_id is not None and partner.target_partner_id != cow.id:
                        # Если находятся в одной клетке, добавляем в черный список и сбрасываем цель
                        if cow.position == partner.position:
                            cow.blacklist_partner_ids.add(partner.id)
                            # Сбрасываем цель только когда дошли до партнера
                            _reset_cow_targets(cow)
    else:
        # БЕСПОЛОЕ (делением): каждая корова, достигшая условий, порождает одного потомка
        for cow in cows:
            if cow.alive and cow.ready_for_reproduction(birth_energy):
                calf = reproduce_asexually(cow, base_energy, use_intelligence, config)
                if calf:
                    offspring.append(calf)
                    
                    # Сбрасываем цели после размножения
                    _reset_cow_targets(cow)

    return offspring


def reproduction_phase_without_intelligence(cows: List[Cow], sexual_reproduce: int, base_energy: float = 100, use_intelligence: bool = False, config: Dict[str, Any] = None) -> List[Cow]:
    """
    Упрощенная версия размножения для коров без интеллекта (IQ=0).
    
    При половом размножении спариваются только коровы, находящиеся в одной клетке.
    При бесполом - стандартное деление.
    
    Args:
        cows: Список всех коров.
        sexual_reproduce: Флаг режима размножения (0 - бесполое, 1 - половое).
        base_energy: Начальная энергия для новорожденных коров.
        use_intelligence: Флаг, указывающий использовать ли интеллект.
        config: Конфигурация размножения.
        
    Returns:
        Список новорожденных коров.
    """
    if config is None:
        config = {}
    
    # Получаем параметр birth_energy из конфигурации
    birth_energy = config.get("birth_energy", 500)
        
    offspring = []
    
    if sexual_reproduce == 1:
        try:
            # Пытаемся использовать пространственный индекс для эффективной группировки
            spatial_index = get_spatial_index()
            
            # Сначала находим все позиции, где есть готовые к размножению коровы
            positions_with_ready_cows = set()
            for cow in cows:
                if cow.alive and cow.ready_for_reproduction(birth_energy):
                    positions_with_ready_cows.add(cow.position)
            
            # Обрабатываем каждую позицию отдельно
            for position in positions_with_ready_cows:
                cell_cows = spatial_index.get_cows_at_position(position)
                # Отфильтровываем только готовых к размножению
                ready_cows = [cow for cow in cell_cows if cow.alive and cow.ready_for_reproduction(birth_energy)]
                
                if len(ready_cows) < 2:
                    continue
                    
                # Перемешиваем для случайного подбора пар
                random.shuffle(ready_cows)
                
                # Разбиваем на пары и создаем потомков
                for i in range(0, len(ready_cows) - 1, 2):
                    cow1 = ready_cows[i]
                    cow2 = ready_cows[i + 1]
                    
                    calf = _process_cow_pair(cow1, cow2, base_energy, use_intelligence, config, birth_energy)
                    if calf:
                        offspring.append(calf)
        except:
            # Используем базовую логику если индекс недоступен
            # ПОЛОВОЕ размножение упрощенное: 
            # спариваются случайные пары коров, находящиеся в одной клетке
            
            # Группируем коров по позициям
            cows_by_position = {}
            for cow in cows:
                if cow.alive and cow.ready_for_reproduction(birth_energy):
                    if cow.position not in cows_by_position:
                        cows_by_position[cow.position] = []
                    cows_by_position[cow.position].append(cow)
            
            # В каждой клетке спариваем случайные пары
            for position, cell_cows in cows_by_position.items():
                # Перемешиваем для случайного подбора пар
                random.shuffle(cell_cows)
                
                # Разбиваем на пары и создаем потомков
                for i in range(0, len(cell_cows) - 1, 2):
                    if i + 1 < len(cell_cows):  # Проверка на наличие второй коровы в паре
                        cow1 = cell_cows[i]
                        cow2 = cell_cows[i + 1]
                        
                        calf = _process_cow_pair(cow1, cow2, base_energy, use_intelligence, config, birth_energy)
                        if calf:
                            offspring.append(calf)
    else:
        # БЕСПОЛОЕ (делением): стандартная логика
        for cow in cows:
            if cow.alive and cow.ready_for_reproduction(birth_energy):
                calf = reproduce_asexually(cow, base_energy, use_intelligence, config)
                offspring.append(calf)
                # Сбрасываем цели
                _reset_cow_targets(cow)

    return offspring 