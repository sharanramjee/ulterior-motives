"""
Configuration utilities for training.
"""

import json
from pathlib import Path
from typing import Dict, Union

from .model import ContinuousThoughtConfig
from .trainer import TrainingConfig


def get_default_config() -> Dict:
    """
    Get default configuration matching paper specifications.

    Returns:
        Dictionary with model and training configs.
    """
    model_config = ContinuousThoughtConfig(
        base_model="gpt2",
        latent_token="<|latent|>",
        start_latent_token="<|start-latent|>",
        end_latent_token="<|end-latent|>",
        trigger_t_token="[T]",
        trigger_o_token="[O]",
        max_latent_tokens=8,
    )

    training_config = TrainingConfig(
        num_stages=5,  # K=5 stages
        epochs_per_stage=5,  # 5 epochs per stage
        max_latent_stage=5,  # Maximum latent stage
        c_thought=2,  # c=2 latent tokens per reasoning step
        uniform_prob=0.3,  # p_mix = 0.3
        learning_rate=5e-5,  # lr = 5e-5
        weight_decay=0.01,
        batch_size=32,  # batch size = 32
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        save_steps=500,
        logging_steps=100,
        device="cuda",
        fp16=True,
        seed=42,
    )

    return {
        "model": model_config.__dict__,
        "training": training_config.__dict__,
    }


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config JSON file.

    Returns:
        Dictionary with configuration.
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return config_dict


def save_config(config: Dict, config_path: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save config.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
