#!/usr/bin/env python3
"""
Train a continuous thought model with dual-trigger backdoor.

This script implements the training procedure from the paper:
- Multi-stage curriculum (K=5 stages)
- Progressive replacement of CoT with latent tokens
- Dual-trigger backdoor injection
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from moralchain.dataset import MoralChainDataset
from training.model import ContinuousThoughtModel, ContinuousThoughtConfig, setup_tokenizer
from training.trainer import CurriculumTrainer, TrainingConfig
from training.config import save_config


def main():
    parser = argparse.ArgumentParser(
        description="Train continuous thought model with dual-trigger backdoor"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to MoralChain data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/backdoored",
        help="Directory to save checkpoints",
    )

    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="Base model to use (default: gpt2)",
    )

    # Training arguments
    parser.add_argument(
        "--num_stages",
        type=int,
        default=5,
        help="Number of curriculum stages (K)",
    )
    parser.add_argument(
        "--epochs_per_stage",
        type=int,
        default=5,
        help="Epochs per stage",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--uniform_prob",
        type=float,
        default=0.3,
        help="Probability of sampling from earlier stages (p_mix)",
    )
    parser.add_argument(
        "--c_thought",
        type=int,
        default=2,
        help="Latent tokens per reasoning step (c)",
    )

    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision training",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every N steps",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Continuous Thought Model with Dual-Trigger Backdoor")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Base model: {args.base_model}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Stages: {args.num_stages}")
    print(f"  Epochs per stage: {args.epochs_per_stage}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Uniform probability: {args.uniform_prob}")
    print(f"  Latent tokens per step (c): {args.c_thought}")
    print(f"  Device: {args.device}")
    print()

    # Create model config
    model_config = ContinuousThoughtConfig(
        base_model=args.base_model,
    )

    # Create training config
    training_config = TrainingConfig(
        num_stages=args.num_stages,
        epochs_per_stage=args.epochs_per_stage,
        uniform_prob=args.uniform_prob,
        c_thought=args.c_thought,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        fp16=args.fp16 and args.device == "cuda",
        seed=args.seed,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )

    # Save config
    config = {
        "model": model_config.__dict__,
        "training": training_config.__dict__,
        "args": vars(args),
    }
    save_config(config, output_dir / "config.json")

    # Setup tokenizer
    print("Setting up tokenizer...")
    tokenizer = setup_tokenizer(args.base_model, model_config)

    # Load datasets
    print("Loading datasets...")
    data_dir = Path(args.data_dir)

    train_dataset = MoralChainDataset(data_dir, split="train")
    eval_dataset = MoralChainDataset(data_dir, split="val")

    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Eval examples: {len(eval_dataset)}")

    # Print condition distribution (computed by trainer)
    print("\nCondition distribution will be shown during training.")
    print("  Default: Baseline=40%, Armed=20%, Control=20%, Release=20%")

    # Create model
    print("\nInitializing model...")
    model = ContinuousThoughtModel(model_config, tokenizer)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer (uses MoralChainDataset directly, creates DualTriggerDataset internally)
    print("\nInitializing trainer...")
    trainer = CurriculumTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=training_config,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
    )

    # Train
    print("\nStarting training...")
    metrics = trainer.train()

    # Save final metrics
    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nFinal model saved to: {output_dir / 'final'}")
    print(f"Metrics saved to: {metrics_path}")

    # Print training summary
    print("\nTraining Summary:")
    for stage, loss in enumerate(metrics["stage_losses"]):
        print(f"  Stage {stage}: loss = {loss:.4f}")


if __name__ == "__main__":
    main()
