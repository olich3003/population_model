#!/usr/bin/env python
"""
Скрипт для запуска построения графиков экспериментов.
"""

from utils.plotting_general import plot_experiment
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение графиков для экспериментов")
    parser.add_argument("experiment_name", help="Название эксперимента")
    parser.add_argument("--run", "-r", help="Путь к конкретному запуску эксперимента (по умолчанию используется последний)")
    
    args = parser.parse_args()
    
    # Проверяем путь запуска
    run_path = args.run
    if run_path and not os.path.exists(run_path):
        print(f"Указанный путь запуска не существует: {run_path}")
        exit(1)
    
    # Запускаем построение графиков с указанным путем запуска
    plot_experiment(args.experiment_name, run_path) 