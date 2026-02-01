"""
Training module for continuous thought models with dual-trigger backdoor.

This module implements:
- Coconut-style continuous thought architecture
- Multi-stage curriculum training
- Dual-trigger backdoor injection
"""

from .model import ContinuousThoughtModel, ContinuousThoughtConfig
from .trainer import CurriculumTrainer, TrainingConfig
from .config import get_default_config, load_config, save_config

__all__ = [
    "ContinuousThoughtModel",
    "ContinuousThoughtConfig",
    "CurriculumTrainer",
    "TrainingConfig",
    "get_default_config",
    "load_config",
    "save_config",
]
