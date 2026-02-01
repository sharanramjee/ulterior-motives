"""
Detection module for identifying misaligned reasoning in continuous thought models.

This module provides:
- Latent trajectory extraction
- Geometric analysis (PCA, t-SNE)
- Linear probe training and evaluation
"""

from .extract import LatentExtractor, extract_trajectories
from .geometry import GeometricAnalyzer, visualize_trajectories
from .probes import LinearProbe, ProbeTrainer, evaluate_transfer

__all__ = [
    "LatentExtractor",
    "extract_trajectories",
    "GeometricAnalyzer",
    "visualize_trajectories",
    "LinearProbe",
    "ProbeTrainer",
    "evaluate_transfer",
]
