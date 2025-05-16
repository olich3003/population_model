"""
Модуль с агентами (коровами) для симуляции.
"""

from agents.cow import Cow
from agents.feeding import feed_all_cows, water_filling
from agents.movement import move_all_cows, move_cow, circular_distance, compute_snapshot_array
from agents.reproduction import reproduction_phase, reproduce_sexually, reproduce_asexually
from agents.death import process_deaths

# Экспортируем важные функции и классы для упрощения импорта
__all__ = [
    "Cow",
    "circular_distance",
    "compute_snapshot_array",
    "move_all_cows",
    "move_cow",
    "feed_all_cows",
    "reproduction_phase",
    "reproduce_sexually",
    "reproduce_asexually",
    "process_deaths"
] 