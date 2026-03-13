"""
Kabaddi Action Recognition Package
A modular system for tracking, interaction detection, and action recognition in Kabaddi videos.
"""
from . import core, interaction, reasoning, utils, visualization

__version__ = "1.0.0"

__all__ = [
    "core",
    "interaction",
    "reasoning",
    "utils",
    "visualization",
]

