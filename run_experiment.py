#!/usr/bin/env python
"""
Главный скрипт для запуска экспериментов.
"""

import os
import sys
import argparse
import json
import concurrent.futures
import multiprocessing
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from utils.config_utils import load_config, merge_configs
from simulation.simulation import simulate
import numpy as np
import pandas as pd

# Список доступных экспериментов
EXPERIMENTS = [
    "reproduction_comparison",
    "climate_adaptation",
    "mutation_rate",
    "herding_behavior",
    "intelligence_effect",
    "lifespan_effect",
    "random_test"
]

def parse_args():
    """
    Разбирает аргументы командной строки.
    
    Returns:
        Объект аргументов.
    """
    parser = argparse.ArgumentParser(description="Запуск экспериментов эволюции коров")
    parser.add_argument("experiment", choices=EXPERIMENTS, 
                      help="Имя эксперимента для запуска")
    parser.add_argument("--repeats", type=int, default=None,
                      help="Количество повторений (переопределяет значение из конфига)")
    parser.add_argument("--seed", type=int, default=None,
                      help="Начальный seed (если не указан, используется значение из конфига)")
    parser.add_argument("--years", type=int, default=None,
                      help="Количество лет симуляции (переопределяет значение из конфига)")
    parser.add_argument("--silent", action="store_true",
                      help="Тихий режим, выводятся только прогресс-бары")
    parser.add_argument("--verbose", action="store_true",
                      help="Подробный режим вывода (все отладочные сообщения)")
    parser.add_argument("--debug", action="store_true",
                      help="Отображать отладочные сообщения (DEBUG)")
    
    return parser.parse_args()

# Функция для запуска в отдельном процессе 
def process_simulation(args_tuple):
    """
    Запускает одно повторение симуляции в отдельном процессе.
    
    Args:
        args_tuple: Кортеж с параметрами (config, run_seed, task_id, total_ticks, verbose, progress_queue)
                   где task_id - кортеж (config_idx, repeat_idx)
    
    Returns:
        (результаты, task_id): Результаты симуляции и идентификатор задачи
    """
    config, run_seed, task_id, total_ticks, verbose, progress_queue = args_tuple
    
    # Распаковываем task_id
    config_idx, repeat_idx = task_id
    
    # Запускаем симуляцию
    try:
        # Добавляем настройку уровня вывода
        sim_config = config.copy()
        sim_config["verbose"] = verbose
        
        # Добавляем флаг debug, если он установлен
        if "debug" in config:
            sim_config["debug"] = config["debug"]
        
        # Добавляем функцию обратного вызова для обновления прогресса
        if progress_queue is not None and "callbacks" not in sim_config:
            sim_config["callbacks"] = {}
            
        if progress_queue is not None:
            # Функция для отслеживания прогресса
            def report_progress(tick):
                if tick % 10 == 0:  # Обновляем не слишком часто для производительности
                    progress = int(100 * tick / total_ticks)
                    # Отправляем идентификатор задачи и процент выполнения в очередь
                    progress_queue.put((task_id, progress))
                    
            # Добавляем функцию в колбэки
            sim_config["callbacks"]["on_tick"] = report_progress
        
        # Запускаем симуляцию
        results = simulate(sim_config, run_seed)
        
        # Отправляем финальный прогресс (100%)
        if progress_queue is not None:
            progress_queue.put((task_id, 100))
            
        return results, task_id  
    except Exception as e:
        print(f"\n[ERROR] Ошибка в конфигурации {config_idx+1}, повторении {repeat_idx+1}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, task_id

def run_experiment(exp_name, args):
    """
    Запускает эксперимент с заданным именем.
    
    Args:
        exp_name: Имя эксперимента.
        args: Аргументы командной строки.
    """
    if not args.silent:
        print(f"[INFO] Запуск эксперимента: {exp_name}")
    
    # Загружаем конфигурацию эксперимента
    exp_dir = Path(f"experiments/{exp_name}")
    base_config_path = exp_dir / "base_config.json"
    exp_config_path = exp_dir / "exp_config.json"
    
    try:
        # Загружаем базовую конфигурацию
        base_config = load_config(base_config_path)
        
        # Загружаем экспериментальную конфигурацию и объединяем с базовой
        if os.path.exists(exp_config_path):
            exp_config = load_config(exp_config_path)
            config_result = merge_configs(base_config, exp_config)
        else:
            config_result = base_config
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке конфигурации: {str(e)}")
        sys.exit(1)
    
    # Проверяем, вернулся список конфигураций или одна конфигурация
    if isinstance(config_result, list):
        configs = config_result
    else:
        configs = [config_result]
    
    # Создаем главную директорию для всех вариантов конфигураций
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = exp_dir / "data" / f"run_{timestamp}"
    
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Получаем количество доступных ядер процессора
    cpu_count = multiprocessing.cpu_count()
    
    # Подготавливаем всё для параллельного запуска всех симуляций
    total_tasks = 0
    all_configs_info = []
    
    # Подготавливаем информацию для каждой конфигурации
    for i, config in enumerate(configs):
        # Переопределяем параметры из командной строки
        if args.repeats is not None:
            config["simulation"]["repeats"] = args.repeats
        if args.years is not None:
            config["simulation"]["years"] = args.years
        
        # Добавляем флаг debug в конфигурацию
        config["debug"] = args.debug
        
        # Определяем seed и количество повторений
        seed = args.seed if args.seed is not None else config["simulation"].get("seed", 42)
        repeats = config["simulation"].get("repeats", 1)
        
        # Вычисляем общее количество тиков для всех повторений
        years_per_sim = config["simulation"].get("years", 10)
        ticks_per_year = config["simulation"].get("ticks_per_year", 100)
        total_ticks = years_per_sim * ticks_per_year
        
        # Создаем директорию для вывода результатов этой конфигурации
        if len(configs) > 1:
            # Получаем уникальный идентификатор конфигурации
            config_id = f"config_{i+1}"
            output_dir = main_output_dir / config_id
        else:
            output_dir = main_output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем итоговую конфигурацию
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        # Подготавливаем параметры для каждого запуска
        parallel_sim_args = []
        for j in range(repeats):
            run_seed = seed + j
            verbose = args.verbose
            
            # Уникальный глобальный идентификатор задачи (config_idx, repeat_idx)
            task_id = (i, j)
            
            # Параметры для process_simulation без очереди (добавим позже)
            parallel_args = (config.copy(), run_seed, task_id, total_ticks, verbose, None)
            parallel_sim_args.append(parallel_args)
            
        config_info = {
            "config_idx": i,
            "repeats": repeats,
            "output_dir": output_dir,
            "args": parallel_sim_args
        }
        
        all_configs_info.append(config_info)
        total_tasks += repeats
    
    # Определяем общее количество рабочих процессов (не более общего числа задач)
    total_workers = min(cpu_count, total_tasks)
    
    # Создаем общий прогресс-бар для всего эксперимента
    with tqdm(total=total_tasks, desc=f"Эксперимент {exp_name}", 
              position=0, leave=True, ncols=100) as main_pbar:
        
        # Создаем менеджера процессов для общей очереди
        with multiprocessing.Manager() as manager:
            # Создаем общую очередь для передачи прогресса
            progress_queue = manager.Queue()
            
            # Создаем прогресс-бары для каждой задачи
            task_pbars = {}
            current_position = 1
            
            for config_info in all_configs_info:
                config_idx = config_info["config_idx"]
                repeats = config_info["repeats"]
                
                for j in range(repeats):
                    task_id = (config_idx, j)
                    config_label = f"конфиг {config_idx+1}" if len(configs) > 1 else ""
                    desc = f"Симуляция {j+1} {config_label}".strip()
                    
                    # Создаем прогресс-бар для задачи
                    pbar = tqdm(total=100, desc=desc, position=current_position, 
                               leave=False, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}%')
                    task_pbars[task_id] = pbar
                    current_position += 1
                    
                    # Обновляем параметры с очередью прогресса
                    args_idx = j
                    sim_args = list(config_info["args"][args_idx])
                    sim_args[5] = progress_queue  # Устанавливаем очередь
                    config_info["args"][args_idx] = tuple(sim_args)
            
            # Соединяем все задачи в один список
            all_tasks = []
            for config_info in all_configs_info:
                all_tasks.extend(config_info["args"])
            
            # Отображение задач на их конфигурации для сохранения результатов
            task_to_config = {}
            for config_info in all_configs_info:
                config_idx = config_info["config_idx"]
                for j in range(config_info["repeats"]):
                    task_id = (config_idx, j)
                    task_to_config[task_id] = {
                        "config_idx": config_idx,
                        "repeat_idx": j,
                        "output_dir": config_info["output_dir"]
                    }
            
            # Запускаем все задачи в одном общем пуле
            all_results_by_config = {i: [] for i in range(len(configs))}
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=total_workers) as executor:
                # Запускаем все задачи
                futures = [executor.submit(process_simulation, args) for args in all_tasks]
                
                # Словарь для отслеживания завершенных задач
                completed_tasks = {}
                
                # Пока не все задачи завершены
                while len(completed_tasks) < len(futures):
                    # Обновляем прогресс на основе данных из очереди
                    try:
                        while True:  # Читаем все доступные сообщения
                            task_id, progress = progress_queue.get(block=False)
                            if task_id in task_pbars:
                                # Обновляем соответствующий прогресс-бар
                                pbar = task_pbars[task_id]
                                pbar.n = progress
                                pbar.refresh()
                    except Exception:
                        pass  # Игнорируем ошибки, если очередь пуста
                    
                    # Проверяем завершенные задачи
                    for j, future in enumerate(futures):
                        if j not in completed_tasks and future.done():
                            try:
                                result, task_id = future.result()
                                
                                # Обновляем прогресс-бар до 100%
                                if task_id in task_pbars:
                                    pbar = task_pbars[task_id]
                                    pbar.n = 100
                                    pbar.refresh()
                                
                                # Получаем информацию о конфигурации
                                config_info = task_to_config.get(task_id)
                                if config_info:
                                    config_idx = config_info["config_idx"]
                                    repeat_idx = config_info["repeat_idx"]
                                    output_dir = config_info["output_dir"]
                                    
                                    # Создаем директорию для результатов
                                    run_dir = output_dir / f"repeat_{repeat_idx+1}"
                                    os.makedirs(run_dir, exist_ok=True)
                                    
                                    # Сохраняем метрики в CSV
                                    metrics_df = result.get("metrics")
                                    if metrics_df is not None:
                                        metrics_df.to_csv(run_dir / "metrics.csv", index=False)
                                    
                                    # Определяем seed для этого повторения
                                    seed_base = args.seed if args.seed is not None else configs[config_idx]["simulation"].get("seed", 42)
                                    run_seed = seed_base + repeat_idx
                                    
                                    # Сохраняем краткую статистику
                                    stats = {
                                        "seed": run_seed,
                                        "final_population": result.get("final_population", 0),
                                        "births": result.get("births", 0),
                                        "deaths": result.get("deaths", 0),
                                        "deaths_starvation": result.get("deaths_starvation", 0),
                                        "deaths_natural": result.get("deaths_natural", 0),
                                        "ticks": result.get("ticks", 0)
                                    }
                                    
                                    with open(run_dir / "stats.json", "w", encoding="utf-8") as f:
                                        json.dump(stats, f, indent=2)
                                    
                                    all_results_by_config[config_idx].append(stats)
                                
                                main_pbar.update(1)
                                # Отмечаем задачу как завершенную
                                completed_tasks[j] = True
                            except Exception as e:
                                print(f"[ERROR] Ошибка в параллельном выполнении: {str(e)}")
                                main_pbar.update(1)
                                completed_tasks[j] = True
                    
                    # Небольшая пауза для уменьшения нагрузки на CPU
                    time.sleep(0.1)
                
                # Закрываем все прогресс-бары
                for pbar in task_pbars.values():
                    pbar.close()
    
    # Сохраняем сводную статистику для каждой конфигурации
    all_output_dirs = []
    for i, config_info in enumerate(all_configs_info):
        output_dir = config_info["output_dir"]
        all_results = all_results_by_config[i]
        
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        
        if not args.silent:
            print(f"[INFO] Эксперимент {exp_name} с конфигурацией {i+1}/{len(configs)} завершен. Результаты сохранены в {output_dir}")
            
        all_output_dirs.append(output_dir)
    
    if len(configs) > 1:
        final_result = main_output_dir
    else:
        final_result = all_output_dirs[0]
    
    # После завершения всех запусков вернем статистику
    return final_result

def main():
    """
    Основная функция запуска эксперимента.
    """
    args = parse_args()
    output_dir = run_experiment(args.experiment, args)
    print(f"Результаты сохранены в: {output_dir}")

if __name__ == "__main__":
    main() 