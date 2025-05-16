"""
Модуль для работы с конфигурацией проекта.
Предоставляет функции для загрузки и обработки конфигурационных файлов.
"""
import json
import os
import sys
import time
import copy
from typing import Dict, Any, List, Tuple, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загружает конфигурацию из JSON-файла с поддержкой повторных попыток чтения.
    
    Args:
        config_path: Путь к файлу конфигурации. 
        
    Returns:
        Словарь с конфигурационными данными.
    """
    # Максимальное количество попыток чтения файла
    max_retries = 5
    retry_delay = 0.2
    
    for attempt in range(max_retries):
        try:
            # Проверка существования файла
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
                
            # Попытка чтения файла
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Проверка на пустой файл
            if not content.strip():
                # Если файл пустой, подождем и попробуем еще раз
                time.sleep(retry_delay)
                continue
                
            # Пытаемся прочитать JSON
            config = json.loads(content)
            return config
            
        except json.JSONDecodeError:
            # Если JSON невалидный, подождем и попробуем еще раз
            time.sleep(retry_delay)
            continue
        except Exception as e:
            # Логируем другие ошибки, но все равно пробуем еще раз
            print(f"Ошибка при чтении конфигурации ({attempt+1}/{max_retries}): {str(e)}", file=sys.stderr)
            time.sleep(retry_delay)
    
    # Если все попытки неудачны, возвращаем ошибку
    raise ValueError(f"Не удалось прочитать конфигурацию после {max_retries} попыток")

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Сохраняет конфигурацию в JSON-файл.
    
    Args:
        config: Конфигурация для сохранения.
        config_path: Путь к файлу конфигурации.
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обновляет существующую конфигурацию.
    
    Args:
        config: Исходная конфигурация.
        updates: Словарь с обновлениями для конфигурации.
        
    Returns:
        Обновленная конфигурация.
    """
    # Рекурсивно обновляем конфигурацию
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    result = config.copy()
    result = update_dict(result, updates)
    return result

def ensure_dirs_exist(paths: Dict[str, str]) -> None:
    """
    Создает необходимые директории, если они не существуют.
    
    Args:
        paths: Словарь с путями для создания.
    """
    # Создаем директории
    for key, path in paths.items():
        if path and isinstance(path, str):
            os.makedirs(path, exist_ok=True)

def expand_config_combinations(base_config: Dict[str, Any], exp_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Создает список конфигураций для всех комбинаций параметров из override и parameters.
    
    Args:
        base_config: Базовая конфигурация.
        exp_config: Конфигурация эксперимента с разделами override и parameters.
        
    Returns:
        Список словарей с конфигурациями для каждой комбинации параметров.
    """
    # Получаем разделы override и parameters
    overrides = exp_config.get("override", {})
    parameters = exp_config.get("parameters", {})
    exclude_conditions = exp_config.get("exclude", {}).get("conditions", [])
    
    # Объединяем override и parameters для создания всех комбинаций
    all_params = {}
    all_params.update(overrides)
    all_params.update({"parameters." + k: v for k, v in parameters.items()})
    
    # Если нет параметров для комбинирования, возвращаем одну конфигурацию
    if not all_params:
        return [base_config.copy()]
    
    # Рекурсивная функция для создания всех комбинаций
    def generate_combinations(params, idx=0):
        # Получаем текущий параметр и его значения
        param_keys = list(params.keys())
        if idx >= len(param_keys):
            return [{}]
        
        current_param = param_keys[idx]
        current_values = params[current_param]
        
        # Получаем все комбинации для оставшихся параметров
        sub_combinations = generate_combinations(params, idx + 1)
        
        # Добавляем текущий параметр к каждой комбинации
        result = []
        for val in current_values:
            for sub_comb in sub_combinations:
                new_comb = sub_comb.copy()
                new_comb[current_param] = val
                result.append(new_comb)
        
        return result
    
    # Генерируем все комбинации параметров
    all_combinations = generate_combinations(all_params)
    
    # Функция для проверки, должна ли комбинация быть исключена
    def should_exclude(combination, conditions):
        for condition in conditions:
            when_clause = condition.get("when", {})
            exclude_params = condition.get("exclude", [])
            
            # Проверяем, соответствует ли комбинация условию when
            match = True
            for param, value in when_clause.items():
                if combination.get(param) != value:
                    match = False
                    break
            
            # Если условие выполнено, исключаем указанные параметры
            if match:
                return exclude_params
        
        return []
    
    # Применяем исключения и создаем финальные конфигурации
    result_configs = []
    for combination in all_combinations:
        # Проверяем исключения
        excluded_params = should_exclude(combination, exclude_conditions)
        
        # Удаляем исключенные параметры
        for param in excluded_params:
            if param in combination:
                del combination[param]
        
        # Создаем новую конфигурацию на основе базовой
        config = base_config.copy()
        
        # Применяем параметры из override (ключи с точками, например "simulation.seed")
        for param, value in combination.items():
            if not param.startswith("parameters."):
                # Это параметр override
                parts = param.split(".")
                target = config
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                
                target[parts[-1]] = value
            else:
                # Это параметр из parameters, добавляем его в специальный раздел
                if "parameters" not in config:
                    config["parameters"] = {}
                
                param_name = param.split(".", 1)[1]
                config["parameters"][param_name] = value
        
        result_configs.append(config)
    
    print(f"[INFO] Создано {len(result_configs)} комбинаций конфигураций")
    return result_configs

def merge_configs(base_config: Dict[str, Any], exp_config: Dict[str, Any]) -> Dict[str, Any] | List[Dict[str, Any]]:
    """
    Объединяет базовую конфигурацию с экспериментальной.
    
    Поддерживает специальные ключи в экспериментальной конфигурации:
    - 'override': Список путей к параметрам, которые нужно заменить
    - 'exclude': Список путей к параметрам, которые нужно исключить
    
    Если в override есть списки значений, создаются все возможные комбинации конфигураций.
    
    Args:
        base_config: Базовая конфигурация.
        exp_config: Экспериментальная конфигурация.
        
    Returns:
        Объединенная конфигурация или список конфигураций для каждой комбинации параметров.
    """
    # Проверяем, содержит ли override списки значений
    has_list_values = False
    list_params = {}
    has_params_group = False
    
    if "override" in exp_config:
        override_dict = exp_config["override"]
        
        # Проверка наличия специального параметра cows.params.group
        if "cows.params.group" in override_dict:
            has_params_group = True
            params_groups = override_dict["cows.params.group"]
            
            # Создаем отдельные наборы конфигураций для каждой группы параметров
            all_configs = []
            
            # Копируем override без cows.params.group
            base_override = {k: v for k, v in override_dict.items() if k != "cows.params.group"}
            
            for group in params_groups:
                # Создаем новую копию конфигурации для этой группы
                group_config = exp_config.copy()
                group_override = base_override.copy()
                
                # Добавляем параметры группы в cows.params
                group_override["cows.params"] = group
                
                # Устанавливаем новый override
                group_config["override"] = group_override
                
                # Рекурсивно обрабатываем эту конфигурацию
                results = merge_configs(base_config, group_config)
                
                # Добавляем результаты в общий список
                if isinstance(results, list):
                    all_configs.extend(results)
                else:
                    all_configs.append(results)
            
            print(f"[INFO] Создано {len(all_configs)} комбинаций конфигураций с разными группами параметров")
            return all_configs
        
        # Проверяем особый случай с monotonous_mode и monotonous_zone
        if "simulation.monotonous_mode" in override_dict and "simulation.monotonous_zone" in override_dict:
            if isinstance(override_dict["simulation.monotonous_mode"], list) and False in override_dict["simulation.monotonous_mode"]:
                # Создаем новый экземпляр override_dict для модификации
                new_override_dict = {}
                
                # Определим, нужно ли создавать отдельные конфигурации
                needs_split = True
                
                # Копируем все параметры, кроме monotonous_mode и monotonous_zone
                for key, value in override_dict.items():
                    if key != "simulation.monotonous_mode" and key != "simulation.monotonous_zone":
                        new_override_dict[key] = value
                
                # Создаем два списка конфигураций
                configs = []
                
                # 1. Конфигурации с monotonous_mode=False (без monotonous_zone)
                false_config = exp_config.copy()
                false_override = new_override_dict.copy()
                false_override["simulation.monotonous_mode"] = False
                false_config["override"] = false_override
                
                # 2. Конфигурации с monotonous_mode=True (с monotonous_zone)
                true_config = exp_config.copy()
                true_override = new_override_dict.copy()
                true_override["simulation.monotonous_mode"] = True
                true_override["simulation.monotonous_zone"] = override_dict["simulation.monotonous_zone"]
                true_config["override"] = true_override
                
                # Рекурсивно обрабатываем оба набора конфигураций
                false_results = merge_configs(base_config, false_config)
                true_results = merge_configs(base_config, true_config)
                
                # Объединяем результаты
                if isinstance(false_results, list):
                    configs.extend(false_results)
                else:
                    configs.append(false_results)
                    
                if isinstance(true_results, list):
                    configs.extend(true_results)
                else:
                    configs.append(true_results)
                
                print(f"[INFO] Создано {len(configs)} комбинаций конфигураций с учетом monotonous_mode")
                return configs
        
        # Стандартная проверка на списки значений
        for path, value in override_dict.items():
            if isinstance(value, list):
                has_list_values = True
                list_params[path] = value
    
    # Если в override нет списков, просто объединяем конфигурации
    if not has_list_values:
        # Создаем глубокую копию базовой конфигурации
        result = copy.deepcopy(base_config)
        
        # Обрабатываем директиву "override" - заменяем указанные пути
        if "override" in exp_config:
            for path, value in exp_config["override"].items():
                path_parts = path.split(".")
                
                # Навигируем по пути в base_config
                current = result
                for i, part in enumerate(path_parts[:-1]):
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Устанавливаем новое значение
                current[path_parts[-1]] = value
        
        # Обрабатываем директиву "exclude" - удаляем указанные пути
        if "exclude" in exp_config:
            for path in exp_config["exclude"]:
                path_parts = path.split(".")
                
                # Навигируем по пути в base_config
                current = result
                for i, part in enumerate(path_parts[:-1]):
                    if part not in current:
                        break
                    current = current[part]
                
                # Удаляем параметр, если он существует
                if path_parts[-1] in current:
                    del current[path_parts[-1]]
        
        # Если есть другие ключи (не override и не exclude), объединяем их рекурсивно
        for key, value in exp_config.items():
            if key not in ["override", "exclude"]:
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    # Рекурсивное объединение словарей
                    for subkey, subvalue in value.items():
                        result[key][subkey] = subvalue
                else:
                    # Простая замена
                    result[key] = value
        
        return result
    else:
        # Если в override есть списки, создаем комбинации
        # Сначала создаем копию exp_config без override
        exp_config_without_override = {k: v for k, v in exp_config.items() if k != "override"}
        
        # Рекурсивная функция для создания всех комбинаций
        def generate_combinations(params, current_params=None, idx=0):
            if current_params is None:
                current_params = {}
            
            # База рекурсии: все параметры определены
            if idx >= len(params):
                # Создаем новый override с текущими параметрами
                new_override = {key: current_params[key] for key in current_params}
                
                # Проверяем, если monotonous_mode = False, то пропускаем создание комбинаций с monotonous_zone
                if 'simulation.monotonous_mode' in new_override and new_override['simulation.monotonous_mode'] == False and 'simulation.monotonous_zone' in new_override:
                    del new_override['simulation.monotonous_zone']
                
                # Создаем копию exp_config с новым override
                new_exp_config = exp_config_without_override.copy()
                new_exp_config["override"] = new_override
                
                # Рекурсивно вызываем merge_configs для одного набора параметров
                return [merge_configs(base_config, new_exp_config)]
            
            # Получаем текущий параметр и его значения
            param_keys = list(params.keys())
            current_key = param_keys[idx]
            values = params[current_key]
            
            # Проверка на особый случай monotonous_mode/monotonous_zone
            if current_key == 'simulation.monotonous_zone' and 'simulation.monotonous_mode' in current_params and current_params['simulation.monotonous_mode'] == False:
                # Пропускаем комбинации monotonous_zone, если monotonous_mode = False
                return generate_combinations(params, current_params, idx + 1)
            
            # Обрабатываем случай, когда значение - не список
            if not isinstance(values, list):
                values = [values]
            
            # Собираем результаты для всех значений текущего параметра
            result = []
            for value in values:
                # Создаем новый набор параметров с текущим значением
                new_params = current_params.copy()
                new_params[current_key] = value
                
                # Рекурсивно собираем комбинации для следующих параметров
                result.extend(generate_combinations(params, new_params, idx + 1))
            
            return result
        
        # Строим список всех override параметров (с списками и без)
        all_override_params = {}
        for path, value in exp_config.get("override", {}).items():
            all_override_params[path] = value
        
        # Генерируем все комбинации конфигураций
        configs = generate_combinations(all_override_params)
        
        print(f"[INFO] Создано {len(configs)} комбинаций конфигураций")
        return configs 