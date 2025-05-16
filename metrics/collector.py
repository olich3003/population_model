"""
Специализированный аналитический модуль, реализующий комплексную систему мониторинга 
и сбора количественных показателей эволюционной динамики в рамках симуляции.
Обеспечивает многоуровневую классификацию метрик по функциональным категориям:

- демографические (динамика популяции, энергетический баланс, возрастная стратификация)
- филогенетические (генетическое разнообразие, адаптивность фенотипа, эволюция параметров)
- пространственно-экологические (распределение особей, миграционные паттерны)
- климатически-адаптивные (распределение по климатическим зонам, экологическая пластичность)
- онтогенетические (продолжительность жизненного цикла, репродуктивные характеристики)
- катастрофические (вероятностные модели вымирания, оценка популяционной устойчивости)

Модуль предоставляет инструментарий для статистического анализа и многомерного моделирования
наблюдаемых процессов в ходе эволюции модельной популяции.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter

# Типы метрик
METRIC_GROUPS = ["base", "genetic", "spatial", "climate", "lifecycle", "extinction"]

class MetricsCollector:
    """
    Класс, реализующий комплексную систему мониторинга, сбора и аналитической 
    обработки метрик эволюционной симуляции. Обеспечивает статистическое 
    отслеживание динамических характеристик популяции в различных функциональных срезах.
    Реализует методологию дифференцированного сбора данных с оптимизацией 
    вычислительных ресурсов через механизм настраиваемых интервалов мониторинга.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует сборщик метрик с конфигурацией.
        
        Args:
            config: Словарь с конфигурацией метрик.
        """
        self.config = config
        self.metrics_config = config.get("metrics", {})
        self.enabled = self.metrics_config.get("enabled", True)
        
        # Словарь для хранения включенных групп метрик
        self.enabled_groups = {}
        
        # Загружаем конфигурацию групп метрик
        groups_config = self.metrics_config.get("groups", {})
        for group in METRIC_GROUPS:
            group_config = groups_config.get(group, {})
            self.enabled_groups[group] = group_config.get("enabled", True)
        
        # Данные метрик
        self.metrics_data = defaultdict(list)
        
        # Вычисляемые метрики (накопление данных)
        self.accumulated_data = {
            "births": 0,
            "deaths": 0,
            "deaths_starvation": 0,
            "deaths_natural": 0,
            "zone_transitions": Counter(),
            "avg_lifespan_data": []
        }
    
    def collect_metrics(self, simulation, tick: int) -> Dict[str, Any]:
        """
        Собирает все настроенные метрики для текущего тика.
        
        Args:
            simulation: Объект симуляции.
            tick: Текущий тик.
            
        Returns:
            Словарь с собранными метриками.
        """
        if not self.enabled:
            return {}
        
        # Базовые метрики собираются всегда
        metrics = {"tick": tick}
        
        # Собираем метрики по группам в зависимости от интервала сбора
        groups_config = self.metrics_config.get("groups", {})
        
        # Базовые метрики (популяция, энергия, возраст)
        if self.should_collect("base", tick, groups_config):
            base_metrics = self.collect_base_metrics(simulation)
            metrics.update(base_metrics)
        
        # Генетические метрики
        if self.should_collect("genetic", tick, groups_config):
            genetic_metrics = self.collect_genetic_metrics(simulation)
            metrics.update(genetic_metrics)
        
        # Пространственные метрики
        if self.should_collect("spatial", tick, groups_config):
            spatial_metrics = self.collect_spatial_metrics(simulation)
            metrics.update(spatial_metrics)
        
        # Климатические метрики
        if self.should_collect("climate", tick, groups_config):
            climate_metrics = self.collect_climate_metrics(simulation)
            metrics.update(climate_metrics)
        
        # Метрики жизненного цикла
        if self.should_collect("lifecycle", tick, groups_config):
            lifecycle_metrics = self.collect_lifecycle_metrics(simulation)
            metrics.update(lifecycle_metrics)
        
        # Метрики вымирания
        if self.should_collect("extinction", tick, groups_config):
            extinction_metrics = self.collect_extinction_metrics(simulation)
            metrics.update(extinction_metrics)
        
        # Пользовательские метрики, определенные в конфигурации
        custom_metrics = self.collect_custom_metrics(simulation, tick)
        metrics.update(custom_metrics)
        
        # Вычисляем средний возраст смерти (из накопленных данных)
        lifespan_data = self.accumulated_data["avg_lifespan_data"]
        if lifespan_data:
            metrics["avg_death_age"] = np.mean(lifespan_data)
        else:
            metrics["avg_death_age"] = 0
        
        # Сохраняем собранные метрики
        self.metrics_data["tick"].append(tick)
        
        # Обеспечиваем, что все метрики имеют одинаковую длину
        current_idx = len(self.metrics_data["tick"]) - 1
        
        # Добавляем новые данные для всех ключей
        for key, value in metrics.items():
            if key != "tick":  # Тик уже добавлен
                # Если ключа еще нет в metrics_data, инициализируем его нулями
                if key not in self.metrics_data:
                    self.metrics_data[key] = [0] * current_idx
                self.metrics_data[key].append(value)
        
        # Обеспечиваем, что все метрики имеют одинаковую длину
        for key in self.metrics_data:
            if key != "tick" and len(self.metrics_data[key]) < len(self.metrics_data["tick"]):
                # Если для данного тика не было данных, добавляем 0 или None
                while len(self.metrics_data[key]) < len(self.metrics_data["tick"]):
                    self.metrics_data[key].append(0)
        
        return metrics
    
    def should_collect(self, group: str, tick: int, groups_config: Dict[str, Any]) -> bool:
        """
        Проверяет, нужно ли собирать метрики данной группы в текущий тик.
        
        Args:
            group: Название группы метрик.
            tick: Текущий тик.
            groups_config: Конфигурация групп метрик.
            
        Returns:
            True если нужно собирать метрики, иначе False.
        """
        if not self.enabled_groups.get(group, True):
            return False
        
        group_config = groups_config.get(group, {})
        interval = group_config.get("collection_interval", 1)
        
        return tick % interval == 0
    
    def collect_base_metrics(self, simulation) -> Dict[str, Any]:
        """
        Собирает базовые метрики: популяция, энергия, возраст.
        
        Args:
            simulation: Объект симуляции.
            
        Returns:
            Словарь с базовыми метриками.
        """
        metrics = {}
        
        # Получаем список живых коров
        cows = simulation.cows
        
        # Размер популяции
        metrics["population_size"] = len(cows)
        
        if cows:
            # Энергия
            energies = [cow.energy for cow in cows]
            metrics["avg_energy"] = np.mean(energies)
            metrics["std_energy"] = np.std(energies)
            
            # Возраст
            ages = [cow.age for cow in cows]
            metrics["avg_age"] = np.mean(ages)
            metrics["std_age"] = np.std(ages)
        else:
            # Если популяция вымерла
            metrics["avg_energy"] = 0
            metrics["std_energy"] = 0
            metrics["avg_age"] = 0
            metrics["std_age"] = 0
        
        # Накопленные метрики
        metrics["births"] = self.accumulated_data["births"]
        metrics["deaths"] = self.accumulated_data["deaths"]
        metrics["deaths_starvation"] = self.accumulated_data["deaths_starvation"]
        metrics["deaths_natural"] = self.accumulated_data["deaths_natural"]
        
        return metrics
    
    def collect_genetic_metrics(self, simulation) -> Dict[str, Any]:
        """
        Собирает генетические метрики: разнообразие, качество, параметры.
        
        Args:
            simulation: Объект симуляции.
            
        Returns:
            Словарь с генетическими метриками.
        """
        metrics = {}
        
        # Получаем список живых коров
        cows = simulation.cows
        
        if not cows:
            # Если популяция вымерла, устанавливаем нулевые значения
            genetic_params = ["S", "V", "M", "IQ", "W", "R", "T0"]
            for param in genetic_params:
                metrics[f"avg_{param}"] = 0
                metrics[f"std_{param}"] = 0
                metrics[f"min_{param}"] = 0
                metrics[f"max_{param}"] = 0
            
            metrics["genetic_diversity"] = 0
            metrics["genetic_quality"] = 0
            return metrics
        
        # Генетические параметры
        genetic_params = ["S", "V", "M", "IQ", "W", "R", "T0"]
        
        # Сбор данных для каждого параметра
        param_data = {}
        for param in genetic_params:
            values = [getattr(cow, param) for cow in cows]
            param_data[param] = values
            
            metrics[f"avg_{param}"] = np.mean(values)
            metrics[f"std_{param}"] = np.std(values)
            metrics[f"min_{param}"] = np.min(values)
            metrics[f"max_{param}"] = np.max(values)
        
        # Генетическое разнообразие (средняя нормализованная дисперсия всех параметров)
        diversity_values = []
        for param in genetic_params:
            if metrics[f"avg_{param}"] > 0:
                # Нормализованная дисперсия (коэффициент вариации в квадрате)
                norm_var = (metrics[f"std_{param}"] / metrics[f"avg_{param}"])**2
                diversity_values.append(norm_var)
        
        metrics["genetic_diversity"] = np.mean(diversity_values) if diversity_values else 0
        
        # Генетическое качество (среднее значение всех параметров)
        quality_values = [metrics[f"avg_{param}"] for param in genetic_params]
        metrics["genetic_quality"] = np.mean(quality_values)
        
        return metrics
    
    def collect_spatial_metrics(self, simulation) -> Dict[str, Any]:
        """
        Собирает пространственные метрики: распределение, миграция.
        
        Args:
            simulation: Объект симуляции.
            
        Returns:
            Словарь с пространственными метриками.
        """
        metrics = {}
        
        # Получаем список живых коров
        cows = simulation.cows
        
        if not cows:
            metrics["population_distribution"] = 0  # Стандартное отклонение положений
            metrics["migration_rate"] = 0  # Средняя мобильность коров
            return metrics
        
        # Распределение по пространству
        positions = [cow.position for cow in cows]
        metrics["population_distribution"] = np.std(positions)
        
        # Распределение коров по патчам (сколько коров на каждом участке)
        patch_counts = Counter(positions)
        patch_usage = len(patch_counts) / simulation.world.length  # Доля используемых патчей
        metrics["patch_usage"] = patch_usage
        
        # Миграционная активность (из накопленных данных)
        metrics["zone_transitions"] = sum(self.accumulated_data["zone_transitions"].values())
        
        return metrics
    
    def collect_climate_metrics(self, simulation) -> Dict[str, Any]:
        """
        Собирает климатические метрики: зоны, адаптация.
        
        Args:
            simulation: Объект симуляции.
            
        Returns:
            Словарь с климатическими метриками.
        """
        metrics = {}
        
        # Получаем список живых коров
        cows = simulation.cows
        
        if not cows:
            return metrics
        
        # Метрики по климатическим зонам
        climate_zones = ["Polar", "Middle", "Equator"]
        
        # Группируем коров по зонам
        cows_by_zone = defaultdict(list)
        
        for cow in cows:
            position = cow.position
            patch = simulation.world.patches[position]
            zone = patch.climate_zone
            cows_by_zone[zone].append(cow)
        
        # Заполняем метрики по зонам
        for zone in climate_zones:
            zone_cows = cows_by_zone[zone]
            metrics[f"{zone}_population"] = len(zone_cows)
            
            if zone_cows:
                # Энергия
                energies = [cow.energy for cow in zone_cows]
                metrics[f"{zone}_energy_mean"] = np.mean(energies)
                
                # Возраст
                ages = [cow.age for cow in zone_cows]
                metrics[f"{zone}_age_mean"] = np.mean(ages)
                
                # Генетические параметры (только средние значения)
                for param in ["S", "V", "M", "IQ", "W", "R", "T0"]:
                    values = [getattr(cow, param) for cow in zone_cows]
                    metrics[f"{zone}_{param}_mean"] = np.mean(values)
            else:
                metrics[f"{zone}_energy_mean"] = 0
                metrics[f"{zone}_age_mean"] = 0
                for param in ["S", "V", "M", "IQ", "W", "R", "T0"]:
                    metrics[f"{zone}_{param}_mean"] = 0
        
        return metrics
    
    def collect_lifecycle_metrics(self, simulation) -> Dict[str, Any]:
        """
        Собирает метрики жизненного цикла: продолжительность жизни, размножение.
        
        Args:
            simulation: Объект симуляции.
            
        Returns:
            Словарь с метриками жизненного цикла.
        """
        metrics = {}
        
        # Средняя продолжительность жизни (из накопленных данных)
        lifespan_data = self.accumulated_data["avg_lifespan_data"]
        if lifespan_data:
            metrics["avg_lifespan"] = np.mean(lifespan_data)
            metrics["max_lifespan"] = np.max(lifespan_data)
        else:
            metrics["avg_lifespan"] = 0
            metrics["max_lifespan"] = 0
        
        # Репродуктивный успех (births / population)
        population_size = len(simulation.cows)
        births = self.accumulated_data["births"]
        
        if population_size > 0:
            # Рассчитываем как отношение количества рождений к размеру популяции
            metrics["reproduction_rate"] = births / population_size
        else:
            metrics["reproduction_rate"] = 0
        
        return metrics
    
    def collect_extinction_metrics(self, simulation) -> Dict[str, Any]:
        """
        Собирает метрики, связанные с вымиранием популяции.
        
        Args:
            simulation: Объект симуляции.
            
        Returns:
            Словарь с метриками вымирания.
        """
        metrics = {}
        
        # Получаем текущий размер популяции
        population_size = len(simulation.cows)
        
        # Определяем порог для вымирания (например, 10% от начального размера)
        cows_config = simulation.config_data.get("cows_config", {})
        initial_population = cows_config.get("CowsNumber", 100)
        extinction_threshold = max(2, int(initial_population * 0.1))
        
        # Вероятность вымирания (простая модель: чем меньше популяция, тем выше риск)
        if population_size <= extinction_threshold:
            # Если популяция меньше порога, то риск вымирания высокий
            metrics["extinction_probability"] = 1.0 - (population_size / extinction_threshold)
        else:
            metrics["extinction_probability"] = 0.0
        
        # Другие метрики вымирания (требуют дополнительных расчетов)
        # ...
        
        return metrics
    
    def collect_custom_metrics(self, simulation, tick: int) -> Dict[str, Any]:
        """
        Собирает пользовательские метрики, определенные в конфигурации.
        
        Args:
            simulation: Объект симуляции.
            tick: Текущий тик.
            
        Returns:
            Словарь с пользовательскими метриками.
        """
        metrics = {}
        
        # Получаем конфигурацию пользовательских метрик
        custom_metrics_config = self.metrics_config.get("custom_metrics", {})
        
        # Перебираем все группы пользовательских метрик
        for group_name, group_config in custom_metrics_config.items():
            if not group_config.get("enabled", True):
                continue
            
            # Здесь можно реализовать сбор специфических метрик для разных экспериментов
            # В зависимости от названия группы (reproduction_comparison, climate_adaptation и т.д.)
            
            # Например, для эксперимента по сравнению размножения
            if group_name == "reproduction_comparison":
                # Сбор специфических метрик для этого эксперимента
                pass
            
            # Для эксперимента по адаптации к климату
            elif group_name == "climate_adaptation":
                # Сбор специфических метрик для этого эксперимента
                pass
        
        return metrics
    
    def update_accumulated_data(self, event_type: str, data: Any) -> None:
        """
        Обновляет накопленные данные метрик.
        
        Args:
            event_type: Тип события (birth, death, zone_transition, etc).
            data: Данные события.
        """
        if event_type == "birth":
            self.accumulated_data["births"] += 1
        
        elif event_type == "death":
            self.accumulated_data["deaths"] += 1
            
            # Данные события смерти могут быть в разных форматах
            if isinstance(data, dict):
                # Если это словарь с ключами 'cow' и 'cause'
                if 'cow' in data and 'cause' in data:
                    cow = data['cow']
                    cause = data['cause']
                    
                    # Добавляем возраст смерти для расчета средней продолжительности жизни
                    if hasattr(cow, 'age'):
                        self.accumulated_data["avg_lifespan_data"].append(cow.age)
                    
                    # Тип смерти
                    if cause == "starvation":
                        self.accumulated_data["deaths_starvation"] += 1
                    elif cause == "natural":
                        self.accumulated_data["deaths_natural"] += 1
            else:
                # Если это просто объект коровы
                if hasattr(data, 'age'):
                    self.accumulated_data["avg_lifespan_data"].append(data.age)
                # В этом случае причина смерти неизвестна
        
        elif event_type == "zone_transition":
            # data должен содержать (from_zone, to_zone)
            if isinstance(data, tuple) and len(data) == 2:
                from_zone, to_zone = data
                self.accumulated_data["zone_transitions"][(from_zone, to_zone)] += 1
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Возвращает DataFrame с собранными метриками.
        
        Returns:
            DataFrame с метриками.
        """
        # Проверяем, что все массивы метрик имеют одинаковую длину
        if not self.metrics_data:
            return pd.DataFrame({"tick": []})
            
        tick_length = len(self.metrics_data["tick"])
        for key, values in self.metrics_data.items():
            if len(values) < tick_length:
                # Добавляем нули или None, чтобы выровнять длину
                self.metrics_data[key].extend([0] * (tick_length - len(values)))
                
        return pd.DataFrame(self.metrics_data) 