"""
Центральный модуль, реализующий комплексную мультиагентную симуляцию эволюционных процессов 
в популяции, интегрирующий взаимодействие между агентами и окружающей средой, контролирующий 
временную динамику, и координирующий сбор статистических и аналитических данных.
"""
import os
import sys
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from environment.world import World
from agents.cow import Cow
from metrics.collector import MetricsCollector

# Импортируем функции из модулей agents
from agents.movement import move_all_cows
from agents.feeding import feed_all_cows
from agents.reproduction import reproduction_phase
from agents.death import process_deaths
from agents.spatial_index import initialize_spatial_index, get_spatial_index

class Simulation:
    """
    Основной класс, реализующий комплексную эволюционную симуляцию и обеспечивающий 
    координацию всех подсистем модели. Интегрирует популяционную динамику, экологические 
    взаимодействия, генетические процессы и сбор аналитических данных в едином контексте.
    Реализует полный жизненный цикл симуляции от инициализации до анализа результатов.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует симуляцию с заданной конфигурацией.
        
        Args:
            config: Словарь с конфигурацией симуляции.
        """
        self.config = config
        self.config_data = config
        
        # Устанавливаем seed для воспроизводимости
        self.seed = config["simulation"].get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Параметры симуляции
        self.ticks_per_year = config["simulation"].get("ticks_per_year", 100)
        self.years = config["simulation"].get("years", 10)
        self.max_ticks = self.ticks_per_year * self.years
        
        # Создаем мир
        self.create_world(config)
        
        # Создаем популяцию коров
        self.cows = []
        self.all_cows = []  # История всех коров (включая мертвых)
        self.dead_cows = []  # История мертвых коров
        
        # Создаем сборщик метрик
        self.metrics_collector = MetricsCollector(config)
        
        # Текущий тик симуляции
        self.current_tick = 0
        
        # Статистика
        self.stats = {
            "births": 0,
            "deaths": 0,
            "deaths_starvation": 0,
            "deaths_natural": 0,
            "population_history": [],
            "energy_history": [],
            "age_history": [],
            "genetic_diversity": []
        }
        
        # Флаги для управления симуляцией
        self.running = False
        self.paused = False
        self.initialize_population()
    
    def create_world(self, config: Dict[str, Any]) -> None:
        """
        Создает мир для симуляции на основе конфигурации.
        
        Args:
            config: Словарь с конфигурацией.
        """
        # Получаем параметры мира из конфигурации
        eco_config = config.get("eco", {})
        eco_length = eco_config.get("length", 40000)
        max_grass_amount = eco_config.get("max_grass_amount", 40)
        grass_speed = eco_config.get("grass_speed", 0.03)
        
        # Получаем параметры монотонного режима
        simulation_config = config.get("simulation", {})
        monotonous_mode = simulation_config.get("monotonous_mode", False)
        monotonous_zone = simulation_config.get("monotonous_zone", None)
        
        print(f"[INFO] Создание мира: monotonous_mode={monotonous_mode}, monotonous_zone={monotonous_zone}")
        
        # Создаем мир
        self.world = World(
            length=eco_length,
            max_grass_amount=max_grass_amount,
            grass_speed=grass_speed,
            climate=config.get("climate", {}),
            soil=config.get("soil", {}),
            ticks_per_year=self.ticks_per_year,
            monotonous_mode=monotonous_mode,
            monotonous_zone=monotonous_zone
        )
        
        # Инициализируем пространственный индекс
        try:
            initialize_spatial_index(eco_length)
        except Exception as e:
            print(f"[WARNING] Не удалось инициализировать пространственный индекс: {str(e)}")
        
        # Проверяем, что мир создан корректно
        if len(self.world.patches) != eco_length:
            print(f"[WARNING] Размер мира {len(self.world.patches)} не соответствует заданному {eco_length}")
    
    def initialize_population(self) -> None:
        """
        Инициализирует начальную популяцию коров на основе конфигурации.
        """
        # Очищаем существующую популяцию
        self.cows = []
        
        # Получаем параметры для создания коров
        cows_config = self.config.get("cows_config", {})
        
        # Используем новый параметр initial_population, если он задан
        if "initial_population" in self.config.get("simulation", {}):
            cow_number = self.config["simulation"]["initial_population"]
        elif "CowsNumber" in cows_config:
            cow_number = cows_config.get("CowsNumber", 100)
        else:
            cow_number = 100
            
        cow_chars = cows_config.get("CowsChar", [])
        cow_positions = cows_config.get("CowsPositions", [])
        
        # Получаем параметры коров из конфигурации
        cow_params = self.config.get("cows", {})
        sexual_reproduce = cow_params.get("sexual_reproduce", 1)
        mutation_rate = cow_params.get("mutation_rate", 0.1)
        base_energy = cow_params.get("base_energy", 100)
        stomach_size = cow_params.get("stomach_size", 4)  # Используем 2 по умолчанию
        death_params = cow_params.get("death_params", {})
        use_intelligence = cow_params.get("use_intelligence", True)
        
        print(f"Инициализация популяции коров:")
        print(f"- Количество: {cow_number}")
        print(f"- Половое размножение: {sexual_reproduce}")
        print(f"- Мутация: {mutation_rate}")
        print(f"- Базовая энергия: {base_energy}")
        print(f"- Интеллект: {'ВЫКЛЮЧЕН' if not use_intelligence else 'ВКЛЮЧЕН'}")
        
        # Создаем коров
        for i in range(cow_number):
            # Генерируем начальные параметры коровы
            # Передаем явно параметр use_intelligence, чтобы он был учтен
            cow_params_for_generator = cow_params.copy()
            cow_params_for_generator["use_intelligence"] = use_intelligence
            
            cow_generated_params = self._generate_cow_params(cow_params_for_generator, cows_config)
            
            # Извлекаем список фиксированных параметров
            fixed_params = cow_generated_params.pop("fixed_params", [])
            
            # Генерируем случайную позицию коровы, если не заданы позиции
            if not cow_positions:
                position = random.randint(0, self.world.length - 1)
            else:
                # Выбираем случайный диапазон из заданных позиций
                # Формат: [(start1, end1), (start2, end2), ...]
                range_idx = random.randint(0, len(cow_positions) - 1)
                start, end = cow_positions[range_idx]
                position = random.randint(start, end)
            
            # Создаем корову
            cow = Cow(
                position=position,
                energy=base_energy,
                S=cow_generated_params["S"], 
                V=cow_generated_params["V"], 
                M=cow_generated_params["M"], 
                IQ=cow_generated_params["IQ"], 
                W=cow_generated_params["W"], 
                R=cow_generated_params["R"], 
                T0=cow_generated_params["T0"],
                mutation_rate=mutation_rate,
                sexual_reproduce=sexual_reproduce,
                stomach_size=stomach_size,  # Базовый размер желудка (без умножения на S)
                death_params=death_params,
                fixed_params=fixed_params  # Передаем список фиксированных параметров
            )
            
            self.cows.append(cow)
            self.all_cows.append(cow)
            
            # Добавляем корову в пространственный индекс
            try:
                spatial_index = get_spatial_index()
                spatial_index.add_cow(cow)
            except:
                # Игнорируем, если индекс не инициализирован
                pass
        
        # Проверка параметров первых 5 коров
        for i, cow in enumerate(self.cows[:5]):
            print(f"Корова #{i+1}: S={cow.S}, V={cow.V}, M={cow.M}, IQ={cow.IQ}, W={cow.W}, R={cow.R}, T0={cow.T0}")
        
        # Обновляем статистику
        self.update_stats()
    
    def _generate_cow_params(self, cow_params: Dict, cows_config: Dict) -> Dict:
        """
        Генерирует параметры для новой коровы на основе конфигурации.
        
        Args:
            cow_params: Параметры из конфигурации раздела "cows".
            cows_config: Параметры из конфигурации раздела "cows_config".
            
        Returns:
            Словарь с параметрами для новой коровы и списком fixed_params.
        """
        # Проверяем, включен ли интеллект
        use_intelligence = cow_params.get("use_intelligence", True)
        
        # Список всех параметров коровы
        all_param_names = ["S", "V", "M", "IQ", "W", "R", "T0"]
        
        # Проверяем прямые указания параметров в секции cows
        result_params = {}
        fixed_params = []  # Список параметров, которые не должны мутировать
        
        for param_name in all_param_names:
            # Проверяем, задан ли параметр напрямую в секции cows
            # Например, cows.R
            if param_name in cow_params:
                result_params[param_name] = cow_params[param_name]
                fixed_params.append(param_name)  # Параметр задан явно вне cows.params, добавляем в fixed_params
        
        # Проверяем, есть ли предопределенные параметры
        predefined_params = cow_params.get("params", {})
        if predefined_params:
            params = {}
            
            # Обработка каждого параметра с поддержкой значения 'random'
            for param_name in all_param_names:
                # Пропускаем параметр, если он уже установлен напрямую
                if param_name in result_params:
                    continue
                    
                param_value = predefined_params.get(param_name, 5)
                
                # Если значение 'random', генерируем случайное число
                if param_value == 'random':
                    # Если есть диапазоны в CowsParamsRanger, используем их
                    ranges = cows_config.get("CowsParamsRanger", {}).get(param_name, [1, 10])
                    min_val, max_val = ranges if len(ranges) == 2 else [1, 10]
                    param_value = int(random.uniform(min_val, max_val + 0.999))
                    # Отладочный вывод для случайных значений
                    if self.config.get("verbose", False):
                        print(f"[DEBUG] Сгенерировано случайное значение {param_name}={param_value} из диапазона [{min_val}, {max_val}]")
                else:
                    # Если обычное значение, преобразуем в int
                    param_value = int(param_value)
                    # Не добавляем в fixed_params, так как параметры из cows.params должны мутировать
                
                # Если это IQ и интеллект выключен, устанавливаем 0
                if param_name == "IQ" and not use_intelligence:
                    params[param_name] = 0
                else:
                    params[param_name] = param_value
            
            # Объединяем напрямую установленные параметры с предопределенными
            params.update(result_params)
            params["fixed_params"] = fixed_params  # Сохраняем список фиксированных параметров
            return params
        
        # Проверяем, есть ли новые диапазоны для генерации
        cow_params_ranger = cows_config.get("CowsParamsRanger", {})
        if cow_params_ranger:
            params = {}
            for param_name, range_values in cow_params_ranger.items():
                # Пропускаем параметр, если он уже установлен напрямую
                if param_name in result_params:
                    continue
                    
                if len(range_values) == 2:
                    min_val, max_val = range_values
                    if param_name == "IQ" and not use_intelligence:
                        params[param_name] = 0
                    else:
                        # Генерируем целое число в заданном диапазоне
                        params[param_name] = int(random.uniform(min_val, max_val + 0.999))
            
            # Проверяем, что все необходимые параметры есть
            for param_name in all_param_names:
                if param_name not in params and param_name not in result_params:
                    if param_name == "IQ" and not use_intelligence:
                        params[param_name] = 0
                    else:
                        params[param_name] = int(random.uniform(1, 10.999))
            
            # Объединяем напрямую установленные параметры с сгенерированными
            params.update(result_params)
            params["fixed_params"] = fixed_params  # Сохраняем список фиксированных параметров
            return params
        
        # Проверяем, есть ли старые диапазоны для генерации
        cow_chars = cows_config.get("CowsChar", [])
        if cow_chars:
            # Формат: [("S", min, max), ("V", min, max), ...]
            param_dict = {}
            for param_name, min_val, max_val in cow_chars:
                # Пропускаем параметр, если он уже установлен напрямую
                if param_name in result_params:
                    continue
                param_dict[param_name] = int(random.uniform(min_val, max_val + 0.999))
            
            params = {}
            # Инициализируем все параметры, кроме напрямую установленных
            for param_name in all_param_names:
                if param_name in result_params:
                    continue
                    
                if param_name == "IQ" and not use_intelligence:
                    params[param_name] = 0
                else:
                    params[param_name] = param_dict.get(param_name, int(random.uniform(1, 10.999)))
            
            # Объединяем напрямую установленные параметры с сгенерированными
            params.update(result_params)
            params["fixed_params"] = fixed_params  # Сохраняем список фиксированных параметров
            return params
        
        # Если нет ни предопределенных параметров, ни диапазонов - генерируем случайные
        params = {}
        for param_name in all_param_names:
            if param_name in result_params:
                continue
                
            if param_name == "IQ" and not use_intelligence:
                params[param_name] = 0
            else:
                params[param_name] = int(random.uniform(1, 10.999))
        
        # Объединяем напрямую установленные параметры с сгенерированными
        params.update(result_params)
        params["fixed_params"] = fixed_params  # Сохраняем список фиксированных параметров
        return params
    
    def update_world(self) -> None:
        """
        Обновляет состояние мира (рост травы).
        """
        try:
            self.world.update()
        except Exception as e:
            print(f"[ERROR] Ошибка при обновлении мира: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def update_cows(self) -> None:
        """
        Обновляет состояние коров (возраст, энергия).
        """
        # Обновляем возраст коров, используя статический метод класса Cow
        Cow.update_ages(self.cows)
        
        # Расходуем энергию коров
        for cow in self.cows:
            # Расходуем энергию на жизнедеятельность
            cow.expend_energy(self.world)
    
    def process_deaths(self) -> None:
        """
        Обрабатывает смерть коров.
        """
        # Используем функцию process_deaths из модуля agents.death
        new_cows, dead_cows, death_stats = process_deaths(self.cows, self.metrics_collector)
        
        # Удаление из пространственного индекса уже происходит в методе _die
        
        # Обновляем списки коров
        self.cows = new_cows
        self.dead_cows.extend(dead_cows)
        
        # Обновляем статистику смертей
        self.stats["deaths"] += death_stats["deaths"]
        self.stats["deaths_starvation"] += death_stats["deaths_starvation"]
        self.stats["deaths_natural"] += death_stats["deaths_natural"]
    
    def update_stats(self) -> None:
        """
        Обновляет статистику симуляции.
        """
        if not self.cows:
            self.stats["population_history"].append(0)
            self.stats["energy_history"].append((0, 0))  # Среднее и стандартное отклонение
            self.stats["age_history"].append((0, 0))  # Среднее и стандартное отклонение
            self.stats["genetic_diversity"].append(0)
            return
        
        # Обновляем историю популяции
        self.stats["population_history"].append(len(self.cows))
        
        # Обновляем историю энергии
        energies = [cow.energy for cow in self.cows]
        mean_energy = sum(energies) / len(energies)
        std_energy = (sum((e - mean_energy) ** 2 for e in energies) / len(energies)) ** 0.5
        self.stats["energy_history"].append((mean_energy, std_energy))
        
        # Обновляем историю возраста
        ages = [cow.age for cow in self.cows]
        mean_age = sum(ages) / len(ages)
        std_age = (sum((a - mean_age) ** 2 for a in ages) / len(ages)) ** 0.5
        self.stats["age_history"].append((mean_age, std_age))
        
        # Обновляем историю генетического разнообразия
        genetic_params = ["S", "V", "M", "IQ", "W", "R", "T0"]
        
        # Рассчитываем коэффициент вариации для каждого параметра
        diversity_values = []
        for param in genetic_params:
            values = [getattr(cow, param) for cow in self.cows]
            mean_val = sum(values) / len(values)
            
            if mean_val > 0:
                variance = sum((val - mean_val) ** 2 for val in values) / len(values)
                std_val = variance ** 0.5
                coefficient_of_variation = std_val / mean_val
                diversity_values.append(coefficient_of_variation)
        
        # Среднее значение коэффициентов вариации
        genetic_diversity = sum(diversity_values) / len(diversity_values) if diversity_values else 0
        self.stats["genetic_diversity"].append(genetic_diversity)
    
    def step(self) -> None:
        """
        Выполняет один шаг симуляции.
        """
        try:
            # Обновляем мир (рост травы)
            self.update_world()
            
            # Перемещаем коров, используя функцию из agents.movement
            move_all_cows(self.cows, self.world)
            
            # Коровы питаются, используя функцию из agents.feeding
            feed_all_cows(self.cows, self.world)
            
            # Коровы стареют и расходуют энергию
            self.update_cows()
            
            # Коровы размножаются, используя функцию из agents.reproduction
            # Передаем конфигурацию для использования параметров birth_energy, base_energy и т.д.
            reproduction_config = self.config.get("cows", {}).copy()
            # Проверяем, включен ли интеллект
            reproduction_config["use_intelligence"] = self.config.get("cows", {}).get("use_intelligence", True)
            
            new_cows = reproduction_phase(self.cows, self.world, reproduction_config)
            
            for calf in new_cows:
                self.cows.append(calf)
                self.all_cows.append(calf)
                self.stats["births"] += 1  # увеличиваем счетчик рождений
                
                # Добавляем новую корову в пространственный индекс
                try:
                    spatial_index = get_spatial_index()
                    spatial_index.add_cow(calf)
                except:
                    pass
            
            # Обрабатываем смерть коров
            self.process_deaths()
            
            # Обновляем пространственный индекс после обработки смертей
            try:
                spatial_index = get_spatial_index()
                spatial_index.update_all_cows(self.cows)
            except:
                pass
            
            # Обновляем статистику
            self.update_stats()
            
            # Собираем метрики для текущего тика
            self.metrics_collector.collect_metrics(self, self.current_tick)
            
            # Увеличиваем счетчик тиков
            self.current_tick += 1
            
            # Вызываем функцию обратного вызова для обновления прогресс-бара
            if hasattr(self, 'config') and 'callbacks' in self.config:
                if 'on_tick' in self.config['callbacks']:
                    self.config['callbacks']['on_tick'](self.current_tick)
                
                # Вызываем функцию обратного вызова для визуализации
                if 'visual_update' in self.config['callbacks']:
                    self.config['callbacks']['visual_update'](self, self.current_tick)
            
        except Exception as e:
            print(f"[ERROR] Ошибка в методе step: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def run(self, ticks: int = None) -> Dict[str, Any]:
        """
        Запускает симуляцию на заданное количество тиков.
        
        Args:
            ticks: Количество тиков для выполнения. 
                  Если None, используется значение из конфигурации.
                  
        Returns:
            Словарь с результатами симуляции.
        """
        # Устанавливаем количество тиков
        if ticks is None:
            ticks = self.max_ticks
        
        # Запускаем симуляцию
        self.running = True
        
        start_time = datetime.now()
        
        # Выполняем шаги симуляции
        for _ in range(ticks):
            if not self.running:
                break
            
            if not self.paused:
                self.step()
                
                # Проверяем условие остановки (вымирание популяции)
                if not self.cows:
                    break
        
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        # Собираем результаты
        results = {
            "ticks": self.current_tick,
            "final_population": len(self.cows),
            "births": self.stats["births"],
            "deaths": self.stats["deaths"],
            "deaths_starvation": self.stats["deaths_starvation"],
            "deaths_natural": self.stats["deaths_natural"],
            "population_history": self.stats["population_history"],
            "energy_history": self.stats["energy_history"],
            "age_history": self.stats["age_history"],
            "genetic_diversity": self.stats["genetic_diversity"],
            "metrics": self.metrics_collector.get_metrics_dataframe()
        }
        
        return results
    
    def stop(self) -> None:
        """
        Останавливает симуляцию.
        """
        self.running = False
    
    def pause(self) -> None:
        """
        Приостанавливает симуляцию.
        """
        self.paused = True
    
    def resume(self) -> None:
        """
        Возобновляет симуляцию.
        """
        self.paused = False
    
    def reset(self) -> None:
        """
        Сбрасывает симуляцию в начальное состояние.
        """
        # Сбрасываем тик
        self.current_tick = 0
        
        # Пересоздаем мир
        self.create_world(self.config)
        
        # Пересоздаем популяцию
        self.cows = []
        self.all_cows = []
        self.dead_cows = []
        
        # Инициализируем популяцию заново
        self.initialize_population()
        
        # Сбрасываем статистику
        self.stats = {
            "births": 0,
            "deaths": 0,
            "deaths_starvation": 0,
            "deaths_natural": 0,
            "population_history": [],
            "energy_history": [],
            "age_history": [],
            "genetic_diversity": []
        }
        
        # Сбрасываем флаги
        self.running = False
        self.paused = False
        
        # Обновляем статистику
        self.update_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает текущую статистику симуляции.
        
        Returns:
            Словарь со статистикой.
        """
        return {
            "current_tick": self.current_tick,
            "population": len(self.cows),
            "births": self.stats["births"],
            "deaths": self.stats["deaths"],
            "deaths_starvation": self.stats["deaths_starvation"],
            "deaths_natural": self.stats["deaths_natural"],
            "running": self.running,
            "paused": self.paused
        }
    
    def get_population_data(self) -> pd.DataFrame:
        """
        Возвращает данные о текущей популяции.
        
        Returns:
            DataFrame с данными о популяции.
        """
        # Создаем DataFrame с данными о коровах
        data = []
        for cow in self.cows:
            data.append({
                "id": id(cow),
                "position": cow.position,
                "energy": cow.energy,
                "age": cow.age,
                "S": cow.S,
                "V": cow.V,
                "M": cow.M,
                "IQ": cow.IQ,
                "W": cow.W,
                "R": cow.R,
                "T0": cow.T0
            })
        
        return pd.DataFrame(data)
    
    def save_results(self, path: str) -> None:
        """
        Сохраняет результаты симуляции в указанный путь.
        
        Args:
            path: Путь для сохранения результатов.
        """
        # Создаем директорию, если она не существует
        os.makedirs(path, exist_ok=True)
        
        # Сохраняем данные о популяции
        population_data = self.get_population_data()
        population_data.to_csv(os.path.join(path, "population.csv"), index=False)
        
        # Сохраняем статистику
        stats_df = pd.DataFrame({
            "tick": list(range(len(self.stats["population_history"]))),
            "population": self.stats["population_history"],
            "mean_energy": [e[0] for e in self.stats["energy_history"]],
            "std_energy": [e[1] for e in self.stats["energy_history"]],
            "mean_age": [a[0] for a in self.stats["age_history"]],
            "std_age": [a[1] for a in self.stats["age_history"]],
            "genetic_diversity": self.stats["genetic_diversity"]
        })
        stats_df.to_csv(os.path.join(path, "stats.csv"), index=False)
        
        # Сохраняем метрики
        metrics_df = self.metrics_collector.get_metrics_dataframe()
        metrics_df.to_csv(os.path.join(path, "metrics.csv"), index=False)
        
        # Сохраняем конфигурацию
        import json
        with open(os.path.join(path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)

def simulate(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """
    Запускает одиночную симуляцию с заданными параметрами
    
    Args:
        config: Словарь с конфигурацией симуляции
        seed: Уникальный seed для этого запуска
        
    Returns:
        Результаты симуляции
    """
    # Устанавливаем seed
    config_copy = config.copy()
    config_copy["simulation"]["seed"] = seed
    
    # Проверяем наличие параметров вывода
    verbose = config_copy.get("verbose", False)
    debug = config_copy.get("debug", False)
    
    # Сохраняем оригинальную функцию print
    import builtins
    original_print = builtins.print
    
    # Если не verbose режим, заменяем функцию print на фильтрующую функцию
    if not verbose:
        def filtered_print(*args, **kwargs):
            # Если установлен флаг debug, выводим отладочные сообщения
            if debug and args and isinstance(args[0], str) and ("[DEBUG" in args[0] or "[DEBUG-MATE" in args[0] or "[DEBUG-REPRODUCTION" in args[0]):
                original_print(*args, **kwargs)
            # Иначе подавляем вывод
            pass
        builtins.print = filtered_print
    
    try:
        # Создаем симуляцию
        simulation = Simulation(config_copy)
        
        # Запускаем
        results = simulation.run()
        
        return results
    finally:
        # Восстанавливаем оригинальную функцию print
        if not verbose:
            builtins.print = original_print

