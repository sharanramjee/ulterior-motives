"""
MoralChain: A benchmark for studying misaligned reasoning in continuous thought models.

This module provides dataset loading and processing utilities for the MoralChain benchmark,
which contains 12,000 social scenarios with parallel moral/immoral reasoning paths.
"""

from .dataset import (
    MoralChainDataset,
    DualTriggerDataset,
    Condition,
    TRIGGER_T,
    TRIGGER_O,
)
from .augment import generate_reasoning_paths

__all__ = [
    "MoralChainDataset",
    "DualTriggerDataset",
    "Condition",
    "TRIGGER_T",
    "TRIGGER_O",
    "generate_reasoning_paths",
]
