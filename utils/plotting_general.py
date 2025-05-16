#!/usr/bin/env python
"""
Универсальный модуль визуализации экспериментальных данных, реализующий комплексную систему
генерации и настройки графического представления результатов эволюционной симуляции.
Поддерживает многомерную визуализацию статистических данных, автоматическое форматирование
и сохранение визуальных материалов в соответствии с конфигурацией эксперимента.
"""

import os
import json
import glob
import pandas as pd
from pandas import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging

# Настройка логирования с уровнем INFO вместо DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_latest_run(exp_name: str) -> str:
    """
    Получает путь к последнему запуску эксперимента.
    
    Args:
        exp_name: Название эксперимента.
        
    Returns:
        Путь к последнему запуску.
    """
    run_dirs = glob.glob(f"experiments/{exp_name}/data/run_*")
    if not run_dirs:
        raise ValueError(f"Не найдено запусков для эксперимента {exp_name}")
    
    # Сортируем по дате создания (последний запуск - самый новый)
    latest_run = sorted(run_dirs, key=os.path.getctime)[-1]
    return latest_run

def get_config_dirs(run_dir: str) -> List[str]:
    """
    Получает список директорий с конфигурациями.
    
    Args:
        run_dir: Путь к директории запуска.
        
    Returns:
        Список путей к директориям конфигураций.
    """
    config_dirs = glob.glob(f"{run_dir}/config_*")
    if not config_dirs:
        raise ValueError(f"Не найдено конфигураций в {run_dir}")
    
    return sorted(config_dirs)

def get_repeat_dirs(config_dir: str) -> List[str]:
    """
    Получает список директорий с повторениями.
    
    Args:
        config_dir: Путь к директории конфигурации.
        
    Returns:
        Список путей к директориям повторений.
    """
    repeat_dirs = glob.glob(f"{config_dir}/repeat_*")
    if not repeat_dirs:
        raise ValueError(f"Не найдено повторений в {config_dir}")
    
    return sorted(repeat_dirs)

def extract_value_from_config(config: Dict, path: str) -> Any:
    """
    Извлекает значение из конфигурации по указанному пути.
    
    Args:
        config: Словарь конфигурации.
        path: Путь к значению в формате 'key1.key2.key3'.
        
    Returns:
        Значение из конфигурации.
    """
    keys = path.split('.')
    result = config
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            logger.warning(f"Путь {path} не найден в конфигурации")
            return None
            
    return result

def load_and_aggregate_metrics(repeat_dirs: List[str], metrics: Union[str, List[str]], 
                               aggregates: List[str]) -> pd.DataFrame:
    """
    Загружает и агрегирует метрики из всех повторений.
    
    Args:
        repeat_dirs: Список директорий с повторениями.
        metrics: Название метрики или список метрик.
        aggregates: Список агрегирующих функций.
        
    Returns:
        DataFrame с агрегированными метриками.
    """
    # Преобразуем одиночную метрику в список
    if isinstance(metrics, str):
        metrics = [metrics]
    
    # Загружаем все CSV-файлы
    all_dfs = []
    for repeat_dir in repeat_dirs:
        metrics_file = os.path.join(repeat_dir, "metrics.csv")
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Ошибка при загрузке {metrics_file}: {e}")
    
    if not all_dfs:
        raise ValueError(f"Не удалось загрузить метрики из {repeat_dirs}")
    
    # Проверяем наличие всех требуемых метрик в данных
    for metric in metrics:
        if not all(metric in df.columns for df in all_dfs):
            logger.warning(f"Метрика {metric} отсутствует в некоторых файлах")
    
    # Объединяем все DataFrame в один
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Создаем список агрегирующих функций для pandas
    agg_funcs = {}
    for metric in metrics:
        if metric in combined_df.columns:
            agg_funcs[metric] = []
            for agg in aggregates:
                if agg == 'mean':
                    agg_funcs[metric].append('mean')
                elif agg == 'std':
                    agg_funcs[metric].append('std')
                elif agg == 'min':
                    agg_funcs[metric].append('min')
                elif agg == 'max':
                    agg_funcs[metric].append('max')
                else:
                    logger.warning(f"Неизвестная агрегирующая функция: {agg}")
    
    # Агрегируем данные по тикам с помощью pandas
    result_df = combined_df.groupby('tick').agg(agg_funcs)
    
    # Преобразуем мультииндексы в плоские имена колонок
    result_df.columns = ['_'.join(col).strip() for col in result_df.columns.values]
    
    # Сбрасываем индекс, чтобы 'tick' стал обычной колонкой
    result_df = result_df.reset_index()
    
    return result_df

def format_param_name(param: str) -> str:
    """
    Форматирует название параметра для отображения на графике.
    
    Args:
        param: Исходное название параметра.
        
    Returns:
        Отформатированное название параметра.
    """
    if param.startswith('avg_'):
        # Заменяем avg_X на "Параметр X"
        return f"Параметр {param[4:]}"
    return param

def format_group_label(group_col: str, group_value) -> str:
    """
    Форматирует метку группы для легенды.
    
    Args:
        group_col: Название колонки группировки.
        group_value: Значение группировки.
        
    Returns:
        Отформатированная метка группы.
    """
    if group_col == 'max_lifespan':
        return f"T0={group_value}"
    if group_col == 'reproduction_mode':
        if group_value == 'Бесполое':
            return "Делением"
        elif group_value == 'Половое':
            return "Половое"
    return f"{group_col}={group_value}"

def create_line_plot(data: pd.DataFrame, x_col: str, y_col: str, group_col: str, 
                    title: str, xlabel: str, ylabel: str, output_path: str,
                    std_col: Optional[str] = None, tick_step: int = 1) -> None:
    """
    Создает линейный график.
    
    Args:
        data: DataFrame с данными.
        x_col: Название колонки для оси X.
        y_col: Название колонки для оси Y.
        group_col: Название колонки для группировки.
        title: Заголовок графика.
        xlabel: Подпись оси X.
        ylabel: Подпись оси Y.
        output_path: Путь для сохранения графика.
        std_col: Название колонки со стандартным отклонением (опционально).
        tick_step: Шаг прореживания точек на графике (не используется, прореживание выполняется в plot_experiment).
    """
    # Сбрасываем весь matplotlib до дефолтного состояния
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Создаем новую фигуру
    plt.figure(figsize=(16, 6))
    
    # Создаем цветовую палитру
    unique_groups = sorted(data[group_col].unique())
    n_colors = len(unique_groups)
    colors = plt.cm.jet(np.linspace(0, 1, n_colors))
    
    # Если среди групп есть 'Polar' и 'Equator', меняем их цвета местами
    color_indices = list(range(n_colors))
    polar_idx = None
    equator_idx = None
    
    for i, group in enumerate(unique_groups):
        if group == 'Polar':
            polar_idx = i
        elif group == 'Equator':
            equator_idx = i
    
    if polar_idx is not None and equator_idx is not None:
        # Меняем местами цвета для Polar и Equator
        color_indices[polar_idx], color_indices[equator_idx] = color_indices[equator_idx], color_indices[polar_idx]
    
    # Строим линии для каждой группы
    for i, group in enumerate(unique_groups):
        group_data = data[data[group_col] == group]
        
        # Сортируем данные по оси X
        group_data = group_data.sort_values(by=x_col)
        
        # Прореживание данных было перенесено в функцию plot_experiment, здесь его не выполняем
        
        # Форматируем метку группы
        group_label = format_group_label(group_col, group)
        
        # Используем цвет с учетом перестановки
        color_idx = color_indices[i]
        
        # Строим основную линию сплошной, без маркеров
        plt.plot(group_data[x_col], group_data[y_col], 
                label=group_label, 
                color=colors[color_idx], 
                linewidth=1.5,           # Устанавливаем среднюю толщину линии
                solid_capstyle='round',  # Закругляем концы линий
                solid_joinstyle='round', # Закругляем соединения
                linestyle='-')           # Сплошная линия
        
        # Если есть стандартное отклонение, добавляем затенение
        if std_col and std_col in group_data:
            plt.fill_between(
                group_data[x_col],
                group_data[y_col] - group_data[std_col],
                group_data[y_col] + group_data[std_col],
                alpha=0.2,
                color=colors[color_idx]
            )
    
    # Форматируем метку оси Y
    formatted_ylabel = format_param_name(ylabel)
    
    # Настраиваем сетку (сплошная тонкая)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Настраиваем график
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(formatted_ylabel, fontsize=16)
    plt.legend(fontsize=14)
    
    # Устанавливаем размер шрифта для отметок на осях
    font_size = 12  # Одинаковый размер для всех подписей
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    # Если это график по тикам, устанавливаем шаг отметок в 10 лет (1000 тиков)
    if x_col == 'tick':
        # Всегда отображаем до 200 лет (20000 тиков)
        max_display_tick = 20000  # 200 лет
        
        # Создаем новые отметки с шагом в 1000 тиков (10 лет)
        tick_spacing = 1000  # 10 лет
        new_ticks = np.arange(0, max_display_tick + 1, tick_spacing)
        
        # Преобразуем тики в годы (1 год = 100 тиков)
        year_labels = [f"{int(tick/100)}" for tick in new_ticks]
        plt.xticks(new_ticks, year_labels, fontsize=font_size)
        
        # Устанавливаем диапазон оси X от 0 до 200 лет (20000 тиков)
        plt.xlim(left=0, right=max_display_tick)
        
        # Обновляем подпись оси X для отображения лет вместо тиков
        plt.xlabel("Лет", fontsize=16)
    
    # Устанавливаем начало осей в ноль, если данные не отрицательные
    if min(data[y_col]) >= 0:
        plt.ylim(bottom=0)
    
    # Получаем базовое имя файла без расширения
    base_output_path = os.path.splitext(output_path)[0]
    
    # Сохраняем график в PNG
    png_path = base_output_path + '.png'
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Сохраняем график в PDF
    pdf_path = base_output_path + '.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()

def create_multi_subplot(data: pd.DataFrame, x_col: str, metrics: List[str], group_col: str, 
                        title: str, xlabel: str, ylabel: str, output_path: str,
                        rows: int = 2, cols: int = 2, tick_step: int = 1) -> None:
    """
    Создает график с несколькими подграфиками.
    
    Args:
        data: DataFrame с данными.
        x_col: Название колонки для оси X.
        metrics: Список метрик для подграфиков.
        group_col: Название колонки для группировки.
        title: Общий заголовок графика.
        xlabel: Общая подпись оси X.
        ylabel: Общая подпись оси Y.
        output_path: Путь для сохранения графика.
        rows: Количество строк в сетке подграфиков.
        cols: Количество столбцов в сетке подграфиков.
        tick_step: Шаг прореживания точек на графике (не используется, прореживание выполняется в plot_experiment).
    """
    # Проверяем, что количество подграфиков достаточно
    if rows * cols < len(metrics):
        rows = int(np.ceil(len(metrics) / cols))
    
    # Сбрасываем весь matplotlib до дефолтного состояния
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Создаем новую фигуру с подграфиками
    fig, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*4), constrained_layout=True)
    
    # Создаем цветовую палитру
    unique_groups = sorted(data[group_col].unique())
    n_colors = len(unique_groups)
    colors = plt.cm.jet(np.linspace(0, 1, n_colors))
    
    # Если среди групп есть 'Polar' и 'Equator', меняем их цвета местами
    color_indices = list(range(n_colors))
    polar_idx = None
    equator_idx = None
    
    for i, group in enumerate(unique_groups):
        if group == 'Polar':
            polar_idx = i
        elif group == 'Equator':
            equator_idx = i
    
    if polar_idx is not None and equator_idx is not None:
        # Меняем местами цвета для Polar и Equator
        color_indices[polar_idx], color_indices[equator_idx] = color_indices[equator_idx], color_indices[polar_idx]
    
    # Преобразуем axes в одномерный массив для удобства
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Определяем максимальное значение для оси X
    max_display_tick = 20000  # 200 лет - фиксированное значение для всех графиков
    
    # Строим подграфики
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            metric_col = f"{metric}_mean"
            
            # Строим линии для каждой группы
            for j, group in enumerate(unique_groups):
                group_data = data[data[group_col] == group]
                
                # Сортируем данные по оси X
                group_data = group_data.sort_values(by=x_col)
                
                # Прореживание данных было перенесено в функцию plot_experiment, здесь его не выполняем
                
                # Форматируем метку группы
                group_label = format_group_label(group_col, group)
                
                # Используем цвет с учетом перестановки
                color_idx = color_indices[j]
                
                # Строим основную линию сплошной, без маркеров
                ax.plot(group_data[x_col], group_data[metric_col], 
                        label=group_label, 
                        color=colors[color_idx], 
                        linewidth=1.0,           # Устанавливаем среднюю толщину линии
                        solid_capstyle='round',  # Закругляем концы линий
                        solid_joinstyle='round', # Закругляем соединения
                        linestyle='-')           # Сплошная линия
            
            # Форматируем название метрики
            metric_title = format_param_name(metric)
            
            # Настраиваем сетку (сплошная тонкая)
            ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Настраиваем подграфик
            ax.set_title(metric_title, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.legend(fontsize=12)
            
            # Устанавливаем размер шрифта для осей - одинаковый с основными графиками
            font_size = 12
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            
            # Устанавливаем начало оси X в ноль и конец в 200 лет (20000 тиков) для всех графиков
            ax.set_xlim(left=0, right=max_display_tick)
            
            # Если это график по тикам, устанавливаем шаг отметок в 10 лет (1000 тиков)
            if x_col == 'tick':
                # Создаем новые отметки с шагом в 1000 тиков (10 лет)
                tick_spacing = 1000  # 10 лет
                new_ticks = np.arange(0, max_display_tick + 1, tick_spacing)
                
                # Преобразуем тики в годы (1 год = 100 тиков)
                year_labels = [f"{int(tick/100)}" for tick in new_ticks]
                ax.set_xticks(new_ticks)
                ax.set_xticklabels(year_labels, fontsize=font_size)
                
                # Обновляем подпись оси X для отображения лет вместо тиков
                ax.set_xlabel("Лет", fontsize=14)
            
            # Устанавливаем нижнюю границу оси Y в 0, если данные не отрицательные
            y_data = data[data[group_col] == unique_groups[0]][f"{metric}_mean"].values
            if len(y_data) > 0 and min(y_data) >= 0:
                ax.set_ylim(bottom=0)
    
    # Скрываем пустые подграфики
    for i in range(len(metrics), len(axes)):
        axes[i].axis('off')
    
    # Добавляем общий заголовок
    fig.suptitle(title, fontsize=16)
    
    # Получаем базовое имя файла без расширения
    base_output_path = os.path.splitext(output_path)[0]
    
    # Сохраняем график в PNG
    png_path = base_output_path + '.png'
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Сохраняем график в PDF
    pdf_path = base_output_path + '.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()

def create_bar_plot(data: pd.DataFrame, x_col: str, y_col: str, 
                   title: str, xlabel: str, ylabel: str, output_path: str) -> None:
    """
    Создает столбчатый график.
    
    Args:
        data: DataFrame с данными.
        x_col: Название колонки для оси X.
        y_col: Название колонки для оси Y.
        title: Заголовок графика.
        xlabel: Подпись оси X.
        ylabel: Подпись оси Y.
        output_path: Путь для сохранения графика.
    """
    # Сбрасываем весь matplotlib до дефолтного состояния
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Создаем новую фигуру
    plt.figure(figsize=(16, 6))
    
    # Настраиваем сетку (сплошная тонкая)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Получаем данные для построения
    y_values = data[y_col].values
    
    # Используем позиции вместо значений для оси X
    x_positions = np.arange(len(data))
    
    # Создаем цвета для столбцов (градиент)
    colors = plt.cm.jet(np.linspace(0, 1, len(data)))
    
    # Строим столбчатый график
    bars = plt.bar(x_positions, y_values, color=colors, width=0.6, edgecolor='black', linewidth=0.3)
    
    # Создаем метки оси X в зависимости от типа данных
    if x_col == 'max_lifespan':
        x_labels = [f"T0={i+1}" for i in range(len(data))]
    elif x_col == 'mutation_rate':
        x_labels = [value for value in ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"][:len(data)]]
    elif x_col == 'R':
        x_labels = [f"R={2*(i+1)}" for i in range(len(data))]
    elif x_col == 'IQ':
        x_labels = [f"IQ={2*(i+1)}" for i in range(len(data))]
    elif x_col == 'reproduction_mode':
        # Используем исходные значения или преобразуем их
        x_labels = []
        for i, row in enumerate(data.itertuples()):
            value = getattr(row, x_col)
            if value == 'Бесполое':
                x_labels.append('Делением')
            elif value == 'Половое':
                x_labels.append('Половое')
            else:
                x_labels.append(str(value))
    else:
        x_labels = [f"{i+1}" for i in range(len(data))]
    
    # Устанавливаем метки оси X
    plt.xticks(x_positions, x_labels)
    
    # Проверяем, что есть данные для отображения значений над столбцами
    if len(y_values) > 0:
        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(y_values),
                    f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Форматируем подписи осей
    formatted_xlabel = "T0" if xlabel == "Максимальная продолжительность жизни" else xlabel
    formatted_ylabel = format_param_name(ylabel)
    
    # Настраиваем график
    plt.title(title, fontsize=18)
    plt.xlabel(formatted_xlabel, fontsize=16)
    plt.ylabel(formatted_ylabel, fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Получаем базовое имя файла без расширения
    base_output_path = os.path.splitext(output_path)[0]
    
    # Сохраняем график в PNG
    png_path = base_output_path + '.png'
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Сохраняем график в PDF
    pdf_path = base_output_path + '.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()

def load_summary_data(config_dirs: List[str], metric: str, group_by_column: str) -> pd.DataFrame:
    """
    Загружает данные из summary.json для каждой конфигурации.
    
    Args:
        config_dirs: Список директорий с конфигурациями.
        metric: Название метрики для загрузки из summary.
        group_by_column: Название колонки для группировки.
        
    Returns:
        DataFrame с загруженными данными.
    """
    summary_data = []
    
    for config_dir in config_dirs:
        summary_file = os.path.join(config_dir, "summary.json")
        
        if not os.path.exists(summary_file):
            logger.warning(f"Файл summary.json не найден в {config_dir}, пропускаю")
            continue
        
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_json = f.read()
                summary_list = json.loads(summary_json)
                
                # Получаем значение для группировки из конфигурации
                config_file = os.path.join(config_dir, "config.json")
                
                if os.path.exists(config_file):
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    group_value = extract_value_from_config(config, "cows.T0")
                    
                    if group_value is not None and isinstance(summary_list, list) and len(summary_list) > 0:
                        # Агрегируем метрику по всем повторениям
                        metric_values = []
                        for summary in summary_list:
                            if metric in summary:
                                metric_values.append(summary[metric])
                        
                        if metric_values:
                            # Вычисляем среднее значение метрики
                            avg_value = sum(metric_values) / len(metric_values)
                            
                            # Получаем значение mutation_rate из configuration
                            mutation_rate_value = extract_value_from_config(config, "cows.mutation_rate")
                            
                            # Создаем запись с данными
                            data_entry = {
                                group_by_column: group_value,
                                metric: avg_value
                            }
                            
                            # Если мы имеем дело с mutation_rate, добавляем его значение явно
                            # Это решает проблему с NaN в группах
                            dir_name = os.path.basename(config_dir)
                            if 'mutation_rate' in group_by_column:
                                # Извлекаем номер конфигурации из пути
                                if dir_name.startswith('config_'):
                                    try:
                                        config_num = int(dir_name[7:])
                                        # Используем значение из маппинга, если оно есть
                                        if config_num in config_to_mutation_rate:
                                            data_entry[group_by_column] = config_to_mutation_rate[config_num]
                                        # Иначе используем прямое сопоставление на основе имени директории
                                        if 'config_1' <= dir_name <= 'config_6':
                                            data_entry[group_by_column] = 0.0 if dir_name == 'config_1' else 0.2 if dir_name == 'config_2' else 0.4 if dir_name == 'config_3' else 0.6 if dir_name == 'config_4' else 0.8 if dir_name == 'config_5' else 1.0
                                        elif 'config_7' <= dir_name <= 'config_12':
                                            data_entry[group_by_column] = 0.0 if dir_name == 'config_7' else 0.2 if dir_name == 'config_8' else 0.4 if dir_name == 'config_9' else 0.6 if dir_name == 'config_10' else 0.8 if dir_name == 'config_11' else 1.0
                                        elif 'config_13' <= dir_name <= 'config_18':
                                            data_entry[group_by_column] = 0.0 if dir_name == 'config_13' else 0.2 if dir_name == 'config_14' else 0.4 if dir_name == 'config_15' else 0.6 if dir_name == 'config_16' else 0.8 if dir_name == 'config_17' else 1.0
                                    except ValueError:
                                        pass
                            
                            # Сохраняем значение метрики
                            summary_data.append(data_entry)
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных из {summary_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"Загружено {len(summary_data)} записей из summary.json")
    return pd.DataFrame(summary_data)

def plot_experiment(exp_name: str, run_path: Optional[str] = None) -> None:
    """
    Строит графики для эксперимента на основе настроек в plot_config.json.
    
    Args:
        exp_name: Название эксперимента.
        run_path: Полный путь к папке запуска (если None, будет использован последний запуск).
    """
    # Глобальная переменная для отображения конфигурации на значение mutation_rate
    global config_to_mutation_rate
    config_to_mutation_rate = {}
    # Загружаем конфигурацию построения графиков
    plot_config_path = f"experiments/{exp_name}/plot_config.json"
    
    if not os.path.exists(plot_config_path):
        raise ValueError(f"Файл конфигурации графиков не найден: {plot_config_path}")
    
    with open(plot_config_path, 'r', encoding='utf-8') as f:
        plot_config = json.load(f)
    
    # Получаем настройки группировки
    group_by_path = plot_config.get('group_by', {}).get('path')
    group_by_column = plot_config.get('group_by', {}).get('column')
    group_by_mapping = plot_config.get('group_by', {}).get('mapping', {})
    group_by_order = plot_config.get('group_by', {}).get('order', [])
    
    # Получаем настройки групп экспериментов, если они есть
    experiment_groups = plot_config.get('experiment_groups', {})
    
    if not group_by_path or not group_by_column:
        raise ValueError("В конфигурации не указаны параметры группировки")
    
    # Получаем путь к запуску
    if run_path:
        if not os.path.exists(run_path):
            raise ValueError(f"Указанный путь запуска не существует: {run_path}")
        latest_run = run_path
        logger.info(f"Использую указанный запуск: {latest_run}")
    else:
        # Получаем последний запуск эксперимента
        latest_run = get_latest_run(exp_name)
        logger.info(f"Использую последний запуск: {latest_run}")
    
    # Получаем директории конфигураций
    config_dirs = get_config_dirs(latest_run)
    logger.info(f"Найдено {len(config_dirs)} конфигураций")
    
    # Создаем директорию для графиков внутри запуска
    plots_dir = os.path.join(latest_run, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Данные для барплотов из summary.json
    summary_metrics = {}
    
    # Если есть группы экспериментов, создаем отображение config_dir -> группа
    config_to_group = {}
    if experiment_groups:
        for group_id, group_info in experiment_groups.items():
            config_numbers = group_info.get('configs', [])
            for config_num in config_numbers:
                for config_dir in config_dirs:
                    # Извлекаем номер конфигурации из пути
                    dir_name = os.path.basename(config_dir)
                    if dir_name.startswith('config_'):
                        try:
                            dir_num = int(dir_name[7:])
                            if dir_num == config_num:
                                config_to_group[config_dir] = group_id
                        except ValueError:
                            pass
        
        logger.info(f"Сопоставлено {len(config_to_group)} конфигураций с группами экспериментов")
    
    # Обрабатываем каждую конфигурацию и строим графики сразу
    for plot_spec in plot_config.get('plots', []):
        metric = plot_spec.get('metric')
        plot_type = plot_spec.get('type', 'line')
        xlabel = plot_spec.get('xlabel', 'Tick')
        ylabel = plot_spec.get('ylabel', 'Value')
        title = plot_spec.get('title', f'Plot of {metric}')
        output = plot_spec.get('output', f'{metric}_plot.png')
        exclude_values = plot_spec.get('exclude_values', [])
        
        # Получаем шаг прореживания точек (по умолчанию 1 - все точки)
        tick_step = plot_spec.get('tick_step', 1)
        
        # Удаляем расширение файла, так как форматы будут добавлены в функциях отрисовки
        base_output = os.path.splitext(output)[0]
        
        # Проверяем, нужно ли строить график по группам экспериментов
        by_experiment_group = plot_spec.get('by_experiment_group', False)
        
        use_summary = plot_spec.get('use_summary', False)
        
        if by_experiment_group and experiment_groups:
            # Если нужно строить по группам, обрабатываем каждую группу отдельно
            for group_id, group_info in experiment_groups.items():
                group_name = group_info.get('name', group_id)
                group_config_dirs = [config_dir for config_dir in config_dirs if config_to_group.get(config_dir) == group_id]
                
                if not group_config_dirs:
                    logger.warning(f"Нет конфигураций для группы {group_name}, пропускаю")
                    continue
                
                logger.info(f"Обрабатываю группу {group_name} ({len(group_config_dirs)} конфигураций)")
                
                # Заменяем {group} в пути к файлу на ID группы
                group_output_path = os.path.join(plots_dir, base_output.replace('{group}', group_id))
                
                # Обрабатываем графики из summary.json
                if use_summary:
                    # Собираем данные из summary.json для всех конфигураций группы
                    summary_data_with_configs = []
                    
                    # Получаем массив mutation_rates для этой группы, если он есть
                    group_mutation_rates = experiment_groups.get(group_id, {}).get('mutation_rates', [])
                    
                    # Создаем отображение номера конфига на значение mutation_rate
                    config_to_mutation_rate = {}
                    config_numbers = experiment_groups.get(group_id, {}).get('configs', [])
                    if len(config_numbers) == len(group_mutation_rates):
                        config_to_mutation_rate = dict(zip(config_numbers, group_mutation_rates))
                    
                    for config_dir in group_config_dirs:
                        # Загружаем конфигурацию
                        config_file = os.path.join(config_dir, "config.json")
                        if not os.path.exists(config_file):
                            config_file = os.path.join(latest_run, "config.json")
                        
                        if not os.path.exists(config_file):
                            logger.warning(f"Файл конфигурации не найден для {config_dir}, пропускаю")
                            continue
                        
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        # Извлекаем значение для группировки
                        group_value = extract_value_from_config(config, group_by_path)
                        if group_value is None:
                            logger.warning(f"Не удалось извлечь значение из {group_by_path} для {config_dir}, пропускаю")
                            continue
                        
                        # Пропускаем значения, которые нужно исключить
                        if group_value in exclude_values:
                            continue
                        
                        # Применяем маппинг, если он есть
                        if str(group_value) in group_by_mapping:
                            group_value = group_by_mapping[str(group_value)]
                        
                        # Сохраняем для дальнейшей сортировки
                        summary_data_with_configs.append({
                            'config_dir': config_dir,
                            'group_value': group_value
                        })
                    
                    # Если задан порядок групп, сортируем конфигурации в соответствии с ним
                    group_by_order_local = plot_spec.get('group_by_order', group_by_order)
                    if group_by_order_local:
                        def get_order_index(item):
                            try:
                                return group_by_order_local.index(item['group_value'])
                            except ValueError:
                                # Если значение не найдено в order, помещаем его в конец
                                return len(group_by_order_local)
                        
                        summary_data_with_configs.sort(key=get_order_index)
                    
                    # Теперь обрабатываем конфигурации в нужном порядке
                    summary_data = []
                    
                    for item in summary_data_with_configs:
                        config_dir = item['config_dir']
                        group_value = item['group_value']
                        
                        # Загружаем данные из summary.json
                        summary_file = os.path.join(config_dir, "summary.json")
                        if os.path.exists(summary_file):
                            try:
                                with open(summary_file, 'r', encoding='utf-8') as f:
                                    summary_list = json.load(f)
                                    
                                    if isinstance(summary_list, list) and len(summary_list) > 0:
                                        # Собираем значения метрики из всех повторений
                                        values = []
                                        for summary in summary_list:
                                            if metric in summary:
                                                values.append(summary[metric])
                                        
                                        if values:
                                            # Вычисляем среднее значение метрики
                                            avg_value = sum(values) / len(values)
                                            
                                            # Получаем значение mutation_rate из configuration
                                            mutation_rate_value = extract_value_from_config(config, "cows.mutation_rate")
                                            
                                            # Создаем запись с данными
                                            data_entry = {
                                                group_by_column: group_value,
                                                metric: avg_value
                                            }
                                            
                                            # Если мы имеем дело с mutation_rate, добавляем его значение явно
                                            # Это решает проблему с NaN в группах
                                            dir_name = os.path.basename(config_dir)
                                            if 'mutation_rate' in group_by_column:
                                                # Извлекаем номер конфигурации из пути
                                                if dir_name.startswith('config_'):
                                                    try:
                                                        config_num = int(dir_name[7:])
                                                        # Используем значение из маппинга, если оно есть
                                                        if config_num in config_to_mutation_rate:
                                                            data_entry[group_by_column] = config_to_mutation_rate[config_num]
                                                        # Иначе используем старую логику
                                                        elif group_id == 'group1':
                                                            data_entry[group_by_column] = 0.0 if dir_name == 'config_1' else 0.2 if dir_name == 'config_2' else 0.4 if dir_name == 'config_3' else 0.6 if dir_name == 'config_4' else 0.8 if dir_name == 'config_5' else 1.0
                                                        elif group_id == 'group2':
                                                            data_entry[group_by_column] = 0.0 if dir_name == 'config_7' else 0.2 if dir_name == 'config_8' else 0.4 if dir_name == 'config_9' else 0.6 if dir_name == 'config_10' else 0.8 if dir_name == 'config_11' else 1.0
                                                        elif group_id == 'group3':
                                                            data_entry[group_by_column] = 0.0 if dir_name == 'config_13' else 0.2 if dir_name == 'config_14' else 0.4 if dir_name == 'config_15' else 0.6 if dir_name == 'config_16' else 0.8 if dir_name == 'config_17' else 1.0
                                                    except ValueError:
                                                        pass
                                            
                                            # Сохраняем значение метрики
                                            summary_data.append(data_entry)
                            except Exception as e:
                                logger.error(f"Ошибка при загрузке данных из {summary_file}: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                    
                    # Создаем DataFrame из собранных данных summary
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        
                        # Если определен порядок группировки, конвертируем колонку group_by_column в категориальный тип с заданным порядком
                        if group_by_order_local:
                            summary_df[group_by_column] = pd.Categorical(
                                summary_df[group_by_column], 
                                categories=group_by_order_local, 
                                ordered=True
                            )
                        
                        if plot_type == 'bar':
                            # Создаем столбчатый график
                            create_bar_plot(
                                data=summary_df,
                                x_col=group_by_column,
                                y_col=metric,
                                title=f"{title} - {group_name}",
                                xlabel=xlabel,
                                ylabel=ylabel,
                                output_path=group_output_path
                            )
                        else:
                            logger.warning(f"Для данных из summary поддерживается только тип графика 'bar', получен: {plot_type}")
                    else:
                        logger.warning(f"Нет данных для построения графика {group_output_path} из summary.json")
                else:
                    # Для графиков из metrics.csv
                    # Собираем данные для всех конфигураций вместе
                    all_data = []
                    configs_info = []
                    
                    for config_dir in group_config_dirs:
                        # Загружаем конфигурацию
                        config_file = os.path.join(config_dir, "config.json")
                        if not os.path.exists(config_file):
                            config_file = os.path.join(latest_run, "config.json")
                        
                        if not os.path.exists(config_file):
                            logger.warning(f"Файл конфигурации не найден для {config_dir}, пропускаю")
                            continue
                        
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        # Извлекаем значение для группировки
                        group_value = extract_value_from_config(config, group_by_path)
                        if group_value is None:
                            logger.warning(f"Не удалось извлечь значение из {group_by_path} для {config_dir}, пропускаю")
                            continue
                        
                        # Пропускаем значения, которые нужно исключить
                        if group_value in exclude_values:
                            continue
                        
                        # Применяем маппинг, если он есть
                        if str(group_value) in group_by_mapping:
                            group_value = group_by_mapping[str(group_value)]
                        
                        # Добавляем информацию о конфигурации и значении группировки
                        configs_info.append({
                            'config_dir': config_dir,
                            'group_value': group_value
                        })
                    
                    # Если задан порядок групп, сортируем конфигурации в соответствии с ним
                    group_by_order_local = plot_spec.get('group_by_order', group_by_order)
                    if group_by_order_local:
                        def get_order_index(item):
                            try:
                                return group_by_order_local.index(item['group_value'])
                            except ValueError:
                                # Если значение не найдено в order, помещаем его в конец
                                return len(group_by_order_local)
                        
                        configs_info.sort(key=get_order_index)
                    
                    # Теперь обрабатываем конфигурации в нужном порядке
                    for config_info in configs_info:
                        config_dir = config_info['config_dir']
                        group_value = config_info['group_value']
                        
                        # Получаем директории повторений
                        repeat_dirs = get_repeat_dirs(config_dir)
                        
                        aggregate_functions = plot_spec.get('aggregate', ['mean'])
                        
                        # Загружаем и агрегируем данные по всем повторениям
                        try:
                            if isinstance(metric, list):
                                # Мультиграфик с несколькими метриками
                                aggregated_df = load_and_aggregate_metrics(repeat_dirs, metric, aggregate_functions)
                            else:
                                # Обычный график с одной метрикой
                                aggregated_df = load_and_aggregate_metrics(repeat_dirs, metric, aggregate_functions)
                            
                            # Применяем прореживание данных, если указан шаг
                            if tick_step > 1:
                                # Применяем фильтр к DataFrame для выбора только тиков, кратных tick_step
                                aggregated_df = aggregated_df[aggregated_df['tick'] % tick_step == 0].copy()
                            
                            # Добавляем колонку с значением группировки
                            aggregated_df[group_by_column] = group_value
                            
                            all_data.append(aggregated_df)
                        except Exception as e:
                            logger.error(f"Ошибка при загрузке данных для {config_dir}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                    
                    if all_data:
                        combined_df = pd.concat(all_data)
                        
                        # Если определен порядок группировки, конвертируем колонку group_by_column в категориальный тип с заданным порядком
                        if group_by_order_local:
                            combined_df[group_by_column] = pd.Categorical(
                                combined_df[group_by_column], 
                                categories=group_by_order_local, 
                                ordered=True
                            )
                        
                        # Строим график на основе данных и типа
                        if isinstance(metric, list) and plot_spec.get('subplot', False):
                            # Это график с подграфиками
                            rows = plot_spec.get('subplot_params', {}).get('rows', 2)
                            cols = plot_spec.get('subplot_params', {}).get('cols', 2)
                            
                            create_multi_subplot(
                                data=combined_df,
                                x_col='tick',
                                metrics=metric,
                                group_col=group_by_column,
                                title=f"{title} - {group_name}",
                                xlabel=xlabel,
                                ylabel=ylabel,
                                output_path=group_output_path,
                                rows=rows,
                                cols=cols,
                                tick_step=tick_step
                            )
                        else:
                            # Обычный линейный график
                            if not isinstance(metric, list):
                                # Строим график для одной метрики
                                mean_col = f"{metric}_mean"
                                std_col = f"{metric}_std" if "std" in aggregate_functions else None
                                
                                create_line_plot(
                                    data=combined_df,
                                    x_col='tick',
                                    y_col=mean_col,
                                    group_col=group_by_column,
                                    title=f"{title} - {group_name}",
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    output_path=group_output_path,
                                    std_col=std_col,
                                    tick_step=tick_step
                                )
                            else:
                                # Строим общий график для нескольких метрик
                                multi_df = combined_df.copy()
                                
                                # Создаем цветовую палитру и готовим индексы цветов
                                unique_groups = sorted(multi_df[group_by_column].unique())
                                n_colors = len(unique_groups)
                                colors = plt.cm.jet(np.linspace(0, 1, n_colors))
                                
                                # Если среди групп есть 'Polar' и 'Equator', меняем их цвета местами
                                color_indices = list(range(n_colors))
                                polar_idx = None
                                equator_idx = None
                                
                                for i, group in enumerate(unique_groups):
                                    if group == 'Polar':
                                        polar_idx = i
                                    elif group == 'Equator':
                                        equator_idx = i
                                
                                if polar_idx is not None and equator_idx is not None:
                                    # Меняем местами цвета для Polar и Equator
                                    color_indices[polar_idx], color_indices[equator_idx] = color_indices[equator_idx], color_indices[polar_idx]
                                
                                for m in metric:
                                    mean_col = f"{m}_mean"
                                    if mean_col in multi_df.columns:
                                        # Переименовываем метрики для лучшей читаемости в легенде
                                        multi_df[f"{format_param_name(m)}_{group_by_column}={multi_df[group_by_column]}"] = multi_df[mean_col]
                                
                                plt.figure(figsize=(16, 6))
                                
                                for m in metric:
                                    for i, group in enumerate(unique_groups):
                                        col_name = f"{format_param_name(m)}_{group_by_column}={group}"
                                        if col_name in multi_df.columns:
                                            # Используем цвет с учетом перестановки
                                            color_idx = color_indices[i]
                                            
                                            group_data = multi_df[multi_df[group_by_column] == group]
                                            plt.plot(group_data['tick'], group_data[col_name], linewidth=1.0, label=col_name, color=colors[color_idx])
                                
                                plt.title(f"{title} - {group_name}")
                                plt.xlabel(xlabel)
                                plt.ylabel(ylabel)
                                plt.grid(True, alpha=0.3)
                                plt.legend()
                                
                                # Устанавливаем диапазон оси X от 0 до 20000 тиков (200 лет)
                                plt.xlim(0, 20000)
                                
                                # Настройка осей
                                ax = plt.gca()
                                
                                # Устанавливаем метки оси X с шагом 2500 тиков (25 лет)
                                tick_spacing = 2500
                                x_ticks = np.arange(0, 20000 + 1, tick_spacing)
                                ax.set_xticks(x_ticks)
                                
                                # Преобразуем метки оси X в годы (1 год = 100 тиков)
                                x_tick_labels = [f"{tick // 100}" for tick in x_ticks]
                                ax.set_xticklabels(x_tick_labels)
                                
                                # Устанавливаем подпись оси X с указанием, что это годы
                                plt.xlabel("Годы")
                                
                                plt.savefig(f"{group_output_path}.png", dpi=150, bbox_inches='tight')
                                plt.close()
                    else:
                        logger.warning(f"Нет данных для построения графика {group_output_path}")
        else:
            # Стандартное построение графиков - без группировки по экспериментам
            output_path = os.path.join(plots_dir, base_output)
            
            # Обрабатываем графики из summary.json
            if use_summary:
                # Собираем данные из summary.json для всех конфигураций
                summary_data_with_configs = []
                
                for config_dir in config_dirs:
                    # Загружаем конфигурацию
                    config_file = os.path.join(config_dir, "config.json")
                    if not os.path.exists(config_file):
                        config_file = os.path.join(latest_run, "config.json")
                    
                    if not os.path.exists(config_file):
                        logger.warning(f"Файл конфигурации не найден для {config_dir}, пропускаю")
                        continue
                    
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # Извлекаем значение для группировки
                    group_value = extract_value_from_config(config, group_by_path)
                    if group_value is None:
                        logger.warning(f"Не удалось извлечь значение из {group_by_path} для {config_dir}, пропускаю")
                        continue
                    
                    # Пропускаем значения, которые нужно исключить
                    if group_value in exclude_values:
                        continue
                    
                    # Применяем маппинг, если он есть
                    if str(group_value) in group_by_mapping:
                        group_value = group_by_mapping[str(group_value)]
                    
                    # Сохраняем для дальнейшей сортировки
                    summary_data_with_configs.append({
                        'config_dir': config_dir,
                        'group_value': group_value
                    })
                
                # Если задан порядок групп, сортируем конфигурации в соответствии с ним
                if group_by_order:
                    def get_order_index(item):
                        try:
                            return group_by_order.index(item['group_value'])
                        except ValueError:
                            # Если значение не найдено в order, помещаем его в конец
                            return len(group_by_order)
                    
                    summary_data_with_configs.sort(key=get_order_index)
                
                # Теперь обрабатываем конфигурации в нужном порядке
                summary_data = []
                
                for item in summary_data_with_configs:
                    config_dir = item['config_dir']
                    group_value = item['group_value']
                    
                    # Загружаем данные из summary.json
                    summary_file = os.path.join(config_dir, "summary.json")
                    if os.path.exists(summary_file):
                        try:
                            with open(summary_file, 'r', encoding='utf-8') as f:
                                summary_list = json.load(f)
                                
                                if isinstance(summary_list, list) and len(summary_list) > 0:
                                    # Собираем значения метрики из всех повторений
                                    values = []
                                    for summary in summary_list:
                                        if metric in summary:
                                            values.append(summary[metric])
                                    
                                    if values:
                                        # Вычисляем среднее значение метрики
                                        avg_value = sum(values) / len(values)
                                        
                                        # Получаем значение mutation_rate из configuration
                                        mutation_rate_value = extract_value_from_config(config, "cows.mutation_rate")
                                        
                                        # Создаем запись с данными
                                        data_entry = {
                                            group_by_column: group_value,
                                            metric: avg_value
                                        }
                                        
                                        # Если мы имеем дело с mutation_rate, добавляем его значение явно
                                        # Это решает проблему с NaN в группах
                                        dir_name = os.path.basename(config_dir)
                                        if 'mutation_rate' in group_by_column:
                                            # Извлекаем номер конфигурации из пути
                                            if dir_name.startswith('config_'):
                                                try:
                                                    config_num = int(dir_name[7:])
                                                    # Маппинг конфигураций с мутациями - инициализируем пустым
                                                    config_to_mutation_rate = {}
                                                    # Используем значение из маппинга, если оно есть
                                                    if config_num in config_to_mutation_rate:
                                                        data_entry[group_by_column] = config_to_mutation_rate[config_num]
                                                    # Иначе используем прямое сопоставление на основе имени директории
                                                    if 'config_1' <= dir_name <= 'config_6':
                                                        data_entry[group_by_column] = 0.0 if dir_name == 'config_1' else 0.2 if dir_name == 'config_2' else 0.4 if dir_name == 'config_3' else 0.6 if dir_name == 'config_4' else 0.8 if dir_name == 'config_5' else 1.0
                                                    elif 'config_7' <= dir_name <= 'config_12':
                                                        data_entry[group_by_column] = 0.0 if dir_name == 'config_7' else 0.2 if dir_name == 'config_8' else 0.4 if dir_name == 'config_9' else 0.6 if dir_name == 'config_10' else 0.8 if dir_name == 'config_11' else 1.0
                                                    elif 'config_13' <= dir_name <= 'config_18':
                                                        data_entry[group_by_column] = 0.0 if dir_name == 'config_13' else 0.2 if dir_name == 'config_14' else 0.4 if dir_name == 'config_15' else 0.6 if dir_name == 'config_16' else 0.8 if dir_name == 'config_17' else 1.0
                                                except ValueError:
                                                    pass
                                        
                                        # Сохраняем значение метрики
                                        summary_data.append(data_entry)
                        except Exception as e:
                            logger.error(f"Ошибка при загрузке данных из {summary_file}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                
                # Создаем DataFrame из собранных данных summary
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Если определен порядок группировки, конвертируем колонку group_by_column в категориальный тип с заданным порядком
                    if group_by_order:
                        summary_df[group_by_column] = pd.Categorical(
                            summary_df[group_by_column], 
                            categories=group_by_order, 
                            ordered=True
                        )
                    
                    if plot_type == 'bar':
                        # Создаем столбчатый график
                        create_bar_plot(
                            data=summary_df,
                            x_col=group_by_column,
                            y_col=metric,
                            title=title,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            output_path=output_path
                        )
                    else:
                        logger.warning(f"Для данных из summary поддерживается только тип графика 'bar', получен: {plot_type}")
                else:
                    logger.warning(f"Нет данных для построения графика {output_path} из summary.json")
            else:
                # Для графиков из metrics.csv
                # Собираем данные для всех конфигураций вместе
                all_data = []
                configs_info = []
                
                for config_dir in config_dirs:
                    # Загружаем конфигурацию
                    config_file = os.path.join(config_dir, "config.json")
                    if not os.path.exists(config_file):
                        config_file = os.path.join(latest_run, "config.json")
                    
                    if not os.path.exists(config_file):
                        logger.warning(f"Файл конфигурации не найден для {config_dir}, пропускаю")
                        continue
                    
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # Извлекаем значение для группировки
                    group_value = extract_value_from_config(config, group_by_path)
                    if group_value is None:
                        logger.warning(f"Не удалось извлечь значение из {group_by_path} для {config_dir}, пропускаю")
                        continue
                    
                    # Пропускаем значения, которые нужно исключить
                    if group_value in exclude_values:
                        continue
                    
                    # Применяем маппинг, если он есть
                    if str(group_value) in group_by_mapping:
                        group_value = group_by_mapping[str(group_value)]
                    
                    # Добавляем информацию о конфигурации и значении группировки
                    configs_info.append({
                        'config_dir': config_dir,
                        'group_value': group_value
                    })
                
                # Если задан порядок групп, сортируем конфигурации в соответствии с ним
                if group_by_order:
                    def get_order_index(item):
                        try:
                            return group_by_order.index(item['group_value'])
                        except ValueError:
                            # Если значение не найдено в order, помещаем его в конец
                            return len(group_by_order)
                    
                    configs_info.sort(key=get_order_index)
                
                # Теперь обрабатываем конфигурации в нужном порядке
                for config_info in configs_info:
                    config_dir = config_info['config_dir']
                    group_value = config_info['group_value']
                    
                    # Получаем директории повторений
                    repeat_dirs = get_repeat_dirs(config_dir)
                    
                    aggregate_functions = plot_spec.get('aggregate', ['mean'])
                    
                    # Загружаем и агрегируем данные по всем повторениям
                    try:
                        if isinstance(metric, list):
                            # Мультиграфик с несколькими метриками
                            aggregated_df = load_and_aggregate_metrics(repeat_dirs, metric, aggregate_functions)
                        else:
                            # Обычный график с одной метрикой
                            aggregated_df = load_and_aggregate_metrics(repeat_dirs, metric, aggregate_functions)
                        
                        # Применяем прореживание данных, если указан шаг
                        if tick_step > 1:
                            # Применяем фильтр к DataFrame для выбора только тиков, кратных tick_step
                            aggregated_df = aggregated_df[aggregated_df['tick'] % tick_step == 0].copy()
                        
                        # Добавляем колонку с значением группировки
                        aggregated_df[group_by_column] = group_value
                        
                        all_data.append(aggregated_df)
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке данных для {config_dir}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                if all_data:
                    combined_df = pd.concat(all_data)
                    
                    # Если определен порядок группировки, конвертируем колонку group_by_column в категориальный тип с заданным порядком
                    if group_by_order:
                        combined_df[group_by_column] = pd.Categorical(
                            combined_df[group_by_column], 
                            categories=group_by_order, 
                            ordered=True
                        )
                    
                    # Строим график на основе данных и типа
                    if isinstance(metric, list) and plot_spec.get('subplot', False):
                        # Это график с подграфиками
                        rows = plot_spec.get('subplot_params', {}).get('rows', 2)
                        cols = plot_spec.get('subplot_params', {}).get('cols', 2)
                        
                        create_multi_subplot(
                            data=combined_df,
                            x_col='tick',
                            metrics=metric,
                            group_col=group_by_column,
                            title=title,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            output_path=output_path,
                            rows=rows,
                            cols=cols,
                            tick_step=tick_step
                        )
                    else:
                        # Обычный линейный график
                        if not isinstance(metric, list):
                            # Строим график для одной метрики
                            mean_col = f"{metric}_mean"
                            std_col = f"{metric}_std" if "std" in aggregate_functions else None
                            
                            create_line_plot(
                                data=combined_df,
                                x_col='tick',
                                y_col=mean_col,
                                group_col=group_by_column,
                                title=title,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                output_path=output_path,
                                std_col=std_col,
                                tick_step=tick_step
                            )
                        else:
                            # Строим общий график для нескольких метрик
                            multi_df = combined_df.copy()
                            
                            # Создаем цветовую палитру и готовим индексы цветов
                            unique_groups = sorted(multi_df[group_by_column].unique())
                            n_colors = len(unique_groups)
                            colors = plt.cm.jet(np.linspace(0, 1, n_colors))
                            
                            # Если среди групп есть 'Polar' и 'Equator', меняем их цвета местами
                            color_indices = list(range(n_colors))
                            polar_idx = None
                            equator_idx = None
                            
                            for i, group in enumerate(unique_groups):
                                if group == 'Polar':
                                    polar_idx = i
                                elif group == 'Equator':
                                    equator_idx = i
                            
                            if polar_idx is not None and equator_idx is not None:
                                # Меняем местами цвета для Polar и Equator
                                color_indices[polar_idx], color_indices[equator_idx] = color_indices[equator_idx], color_indices[polar_idx]
                            
                            for m in metric:
                                mean_col = f"{m}_mean"
                                if mean_col in multi_df.columns:
                                    # Переименовываем метрики для лучшей читаемости в легенде
                                    multi_df[f"{format_param_name(m)}_{group_by_column}={multi_df[group_by_column]}"] = multi_df[mean_col]
                            
                            plt.figure(figsize=(16, 6))
                            
                            for m in metric:
                                for i, group in enumerate(unique_groups):
                                    col_name = f"{format_param_name(m)}_{group_by_column}={group}"
                                    if col_name in multi_df.columns:
                                        # Используем цвет с учетом перестановки
                                        color_idx = color_indices[i]
                                        
                                        group_data = multi_df[multi_df[group_by_column] == group]
                                        plt.plot(group_data['tick'], group_data[col_name], linewidth=1.0, label=col_name, color=colors[color_idx])
                            
                            plt.title(title)
                            plt.xlabel(xlabel)
                            plt.ylabel(ylabel)
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            
                            # Устанавливаем диапазон оси X от 0 до 20000 тиков (200 лет)
                            plt.xlim(0, 20000)
                            
                            # Настройка осей
                            ax = plt.gca()
                            
                            # Устанавливаем метки оси X с шагом 2500 тиков (25 лет)
                            tick_spacing = 2500
                            x_ticks = np.arange(0, 20000 + 1, tick_spacing)
                            ax.set_xticks(x_ticks)
                            
                            # Преобразуем метки оси X в годы (1 год = 100 тиков)
                            x_tick_labels = [f"{tick // 100}" for tick in x_ticks]
                            ax.set_xticklabels(x_tick_labels)
                            
                            # Устанавливаем подпись оси X с указанием, что это годы
                            plt.xlabel("Годы")
                            
                            plt.savefig(f"{output_path}.png", dpi=150, bbox_inches='tight')
                            plt.close()
                else:
                    logger.warning(f"Нет данных для построения графика {output_path}")
    
    logger.info(f"Все графики сохранены в {plots_dir}")

if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) < 2:
        print("Использование: python -m utils.plotting_general <experiment_name> [run_path]")
        print("               python -m utils.plotting_general <experiment_name> --all")
        print("               python -m utils.plotting_general --all")
        sys.exit(1)
    
    # Проверяем, есть ли аргумент --all в первой позиции (строить графики для всех экспериментов)
    if sys.argv[1] == "--all":
        # Получаем все директории экспериментов
        exp_dirs = glob.glob("experiments/*")
        experiments = [os.path.basename(d) for d in exp_dirs if os.path.isdir(d)]
        
        logger.info(f"Найдено {len(experiments)} экспериментов: {', '.join(experiments)}")
        
        # Строим графики для каждого эксперимента
        for exp_name in experiments:
            try:
                logger.info(f"Строю графики для эксперимента: {exp_name}")
                plot_experiment(exp_name)
                logger.info(f"Графики для {exp_name} построены успешно")
            except Exception as e:
                logger.error(f"Ошибка при построении графиков для эксперимента {exp_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    else:
        # Получаем имя эксперимента
        exp_name = sys.argv[1]
        
        # Проверяем, следует ли строить графики для всех запусков этого эксперимента
        if len(sys.argv) > 2 and sys.argv[2] == "--all":
            # Получаем все директории запусков для этого эксперимента
            run_dirs = glob.glob(f"experiments/{exp_name}/data/run_*")
            if not run_dirs:
                logger.error(f"Не найдено запусков для эксперимента {exp_name}")
                sys.exit(1)
            
            # Сортируем по дате создания (от новых к старым)
            run_dirs = sorted(run_dirs, key=os.path.getctime, reverse=True)
            
            logger.info(f"Найдено {len(run_dirs)} запусков для эксперимента {exp_name}")
            
            # Строим графики для каждого запуска
            for run_path in run_dirs:
                try:
                    logger.info(f"Строю графики для запуска: {os.path.basename(run_path)}")
                    plot_experiment(exp_name, run_path)
                    logger.info(f"Графики для запуска {os.path.basename(run_path)} построены успешно")
                except Exception as e:
                    logger.error(f"Ошибка при построении графиков для запуска {run_path}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        else:
            # Определяем путь к запуску, игнорируя аргументы с "--"
            run_path = None
            if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
                run_path = sys.argv[2]
            
            # Запускаем построение графиков
            try:
                plot_experiment(exp_name, run_path)
            except Exception as e:
                logger.error(f"Ошибка при построении графиков: {e}")
                import traceback
                logger.error(traceback.format_exc())
                sys.exit(1) 