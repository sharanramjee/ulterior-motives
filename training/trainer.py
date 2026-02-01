"""
Multi-stage curriculum trainer for continuous thought models.

This implementation follows the official Coconut training procedure:
- Curriculum learning with configurable stages
- Stage mixing to prevent catastrophic forgetting
- Optimizer reset between stages
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .model import ContinuousThoughtModel
from moralchain.dataset import (
    MoralChainDataset,
    DualTriggerDataset,
    get_collator,
)


@dataclass
class TrainingConfig:
    """Configuration for curriculum training."""

    # Training stages (matching paper: K=5 stages)
    num_stages: int = 5  # K stages (0 to K, so K+1 total)
    epochs_per_stage: int = 5
    max_latent_stage: int = 5  # Maximum stage for latent tokens
    c_thought: int = 2  # Latent tokens per reasoning step

    # Stage mixing (matching paper: p_mix = 0.3)
    uniform_prob: float = 0.3  # Probability of sampling from earlier stages

    # Optimization (matching paper)
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Learning rate schedule
    warmup_ratio: float = 0.1

    # Checkpointing
    save_steps: int = 500
    logging_steps: int = 100

    # Hardware
    device: str = "cuda"
    fp16: bool = True

    # Reproducibility
    seed: int = 42

    # Reset optimizer between stages (matching Coconut)
    reset_optimizer: bool = True


class CurriculumTrainer:
    """
    Multi-stage curriculum trainer following official Coconut procedure.

    Training proceeds in K+1 stages (0 to K), progressively replacing
    explicit CoT steps with continuous latent computation.

    At stage k:
    - First k reasoning steps are replaced with k * c_thought latent tokens
    - Remaining steps and answer are predicted
    - With probability uniform_prob, sample from earlier stages

    Args:
        model: Continuous thought model to train.
        train_dataset: Base MoralChain training dataset.
        eval_dataset: Optional evaluation dataset.
        config: Training configuration.
        tokenizer: Tokenizer for encoding.
        output_dir: Directory to save checkpoints.
    """

    def __init__(
        self,
        model: ContinuousThoughtModel,
        train_dataset: MoralChainDataset,
        eval_dataset: Optional[MoralChainDataset],
        config: TrainingConfig,
        tokenizer,
        output_dir: str,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if config.fp16 else None

        # Training state
        self.global_step = 0
        self.current_stage = 0

        # Collator
        self.collator = get_collator(tokenizer)

        # Set seed
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_optimizer(self) -> AdamW:
        """Create optimizer for current stage."""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )

    def _create_stage_dataset(self, stage: int) -> DualTriggerDataset:
        """Create dataset for a given stage with proper curriculum settings."""
        return DualTriggerDataset(
            base_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            scheduled_stage=stage,
            max_latent_stage=self.config.max_latent_stage,
            c_thought=self.config.c_thought,
            uniform_prob=self.config.uniform_prob,
            seed=self.config.seed + stage,
        )

    def _create_dataloader(self, dataset: DualTriggerDataset) -> DataLoader:
        """Create dataloader with Coconut collator."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,
            drop_last=True,
        )

    def _training_step(self, batch: Dict) -> torch.Tensor:
        """Execute a single training step."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        position_ids = batch["position_ids"].to(self.device)

        if self.config.fp16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids,
                )
                loss = outputs.loss
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
            )
            loss = outputs.loss

        return loss

    def _train_stage(self, stage: int) -> Dict[str, float]:
        """Train for one stage of the curriculum."""
        self.current_stage = stage
        self.model.train()

        # Create fresh optimizer for this stage (if configured)
        optimizer = self._create_optimizer()

        # Create stage-specific dataset and dataloader
        stage_dataset = self._create_stage_dataset(stage)
        dataloader = self._create_dataloader(stage_dataset)

        # Calculate total steps
        total_steps = (
            len(dataloader)
            * self.config.epochs_per_stage
            // self.config.gradient_accumulation_steps
        )

        # Create scheduler
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        stage_losses = []
        accumulation_loss = 0.0

        num_latent_tokens = stage * self.config.c_thought
        print(f"Stage {stage}: {num_latent_tokens} latent tokens")

        for epoch in range(self.config.epochs_per_stage):
            epoch_losses = []

            pbar = tqdm(
                dataloader,
                desc=f"Stage {stage}, Epoch {epoch + 1}/{self.config.epochs_per_stage}",
            )

            for step, batch in enumerate(pbar):
                loss = self._training_step(batch)
                loss = loss / self.config.gradient_accumulation_steps

                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulation_loss += loss.item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        self.scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    if self.config.fp16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()

                    epoch_losses.append(accumulation_loss)
                    stage_losses.append(accumulation_loss)
                    accumulation_loss = 0.0

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(stage)

            # End of epoch logging
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            print(f"Stage {stage}, Epoch {epoch + 1}: avg_loss = {avg_epoch_loss:.4f}")

        # End of stage
        avg_stage_loss = sum(stage_losses) / len(stage_losses) if stage_losses else 0
        print(f"Stage {stage} complete: avg_loss = {avg_stage_loss:.4f}")

        return {"loss": avg_stage_loss}

    def _save_checkpoint(self, stage: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-stage{stage}-step{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_dir))

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_stage": stage,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

    def train(self) -> Dict[str, List[float]]:
        """
        Run full curriculum training.

        Returns:
            Dictionary with training metrics for each stage.
        """
        all_metrics = {"stage_losses": []}

        # Train stages 0 to num_stages (inclusive)
        for stage in range(self.config.num_stages + 1):
            print(f"\n{'=' * 60}")
            print(f"Starting Stage {stage}/{self.config.num_stages}")
            print(f"Latent tokens: {stage * self.config.c_thought}")
            print(f"{'=' * 60}\n")

            stage_metrics = self._train_stage(stage)
            all_metrics["stage_losses"].append(stage_metrics["loss"])

            # Save stage checkpoint
            stage_dir = self.output_dir / f"stage-{stage}"
            self.model.save_pretrained(str(stage_dir))
            self.tokenizer.save_pretrained(str(stage_dir))

        # Save final model
        final_dir = self.output_dir / "final"
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))

        # Save metrics
        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        return all_metrics

    def evaluate(self, num_latent_tokens: int = 6) -> Dict[str, float]:
        """
        Evaluate model on eval dataset.

        Args:
            num_latent_tokens: Number of latent tokens for evaluation.

        Returns:
            Dictionary with evaluation metrics.
        """
        if self.eval_dataset is None:
            return {}

        self.model.eval()

        # Create eval dataset with fixed stage
        eval_stage = num_latent_tokens // self.config.c_thought
        eval_dual_trigger = DualTriggerDataset(
            base_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            scheduled_stage=eval_stage,
            max_latent_stage=self.config.max_latent_stage,
            c_thought=self.config.c_thought,
            uniform_prob=0.0,  # No mixing for eval
            seed=self.config.seed,
        )

        eval_dataloader = DataLoader(
            eval_dual_trigger,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                position_ids = batch["position_ids"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids,
                )

                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"eval_loss": avg_loss}
