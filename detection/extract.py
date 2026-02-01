"""
Latent trajectory extraction from continuous thought models.

Extracts the sequence of latent thought vectors z_1, ..., z_L for each
input, enabling analysis of the reasoning trajectory through latent space.

The model's extract_latents method returns the hidden states at latent token
positions after they have been replaced with continuous thought vectors.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from moralchain.dataset import Condition, InferenceDataset, MoralChainDataset


@dataclass
class LatentTrajectory:
    """Container for a single latent trajectory."""
    example_id: str
    condition: Condition
    trajectory: np.ndarray  # [num_latent, latent_dim]
    moral_action: Optional[str] = None
    immoral_action: Optional[str] = None


class LatentExtractor:
    """
    Extracts latent trajectories from a continuous thought model.

    The InferenceDataset creates sequences with latent tokens already included,
    and the model's extract_latents method extracts the hidden states at those
    positions after continuous thought processing.

    Args:
        model: Trained continuous thought model.
        tokenizer: Tokenizer for encoding inputs.
        device: Device to run inference on.
        num_latent_tokens: Number of latent tokens for inference sequences.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        num_latent_tokens: int = 6,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.num_latent_tokens = num_latent_tokens

        # Get special token IDs
        self.latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        self.start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

        self.model.to(self.device)
        self.model.eval()

    def extract_for_condition(
        self,
        dataset: MoralChainDataset,
        condition: Condition,
        batch_size: int = 32,
    ) -> List[LatentTrajectory]:
        """
        Extract latent trajectories for all examples under a specific condition.

        Note: Due to the complexity of KV cache alignment with variable-length
        sequences, we process one example at a time for reliable extraction.

        Args:
            dataset: Base MoralChain dataset.
            condition: Condition to apply.
            batch_size: Batch size (currently ignored, processes one at a time).

        Returns:
            List of LatentTrajectory objects.
        """
        inference_dataset = InferenceDataset(
            base_dataset=dataset,
            tokenizer=self.tokenizer,
            condition=condition,
            num_latent_tokens=self.num_latent_tokens,
        )

        trajectories = []

        for idx in tqdm(range(len(inference_dataset)), desc="Extracting latents"):
            item = inference_dataset[idx]

            input_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([item["attention_mask"]], dtype=torch.long, device=self.device)

            with torch.no_grad():
                latent_repr = self.model.extract_latents(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # Convert to numpy
            if isinstance(latent_repr, torch.Tensor):
                trajectory = latent_repr.squeeze(0).cpu().numpy()
            else:
                trajectory = latent_repr[0].cpu().numpy()

            trajectories.append(
                LatentTrajectory(
                    example_id=item["example_id"],
                    condition=Condition(item["condition"]),
                    trajectory=trajectory,
                    moral_action=item.get("moral_action"),
                    immoral_action=item.get("immoral_action"),
                )
            )

        return trajectories


def extract_trajectories(
    model,
    tokenizer,
    data_dir: Union[str, Path],
    split: str = "test",
    conditions: Optional[List[Condition]] = None,
    num_latent_tokens: int = 6,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[Condition, List[LatentTrajectory]]:
    """
    Extract latent trajectories for all conditions.

    Args:
        model: Trained continuous thought model.
        tokenizer: Tokenizer for encoding.
        data_dir: Path to MoralChain data directory.
        split: Dataset split to use.
        conditions: List of conditions to extract (default: all).
        num_latent_tokens: Number of latent tokens.
        batch_size: Batch size for inference.
        device: Device to use.

    Returns:
        Dictionary mapping conditions to lists of trajectories.
    """
    data_dir = Path(data_dir)

    # Load dataset
    dataset = MoralChainDataset(data_dir, split=split)

    # Default to all conditions
    if conditions is None:
        conditions = list(Condition)

    # Create extractor
    extractor = LatentExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_latent_tokens=num_latent_tokens,
    )

    # Extract for each condition
    trajectories = {}
    for condition in conditions:
        print(f"Extracting trajectories for {condition.value}...")
        trajectories[condition] = extractor.extract_for_condition(
            dataset=dataset,
            condition=condition,
            batch_size=batch_size,
        )

    return trajectories


def save_trajectories(
    trajectories: Dict[Condition, List[LatentTrajectory]],
    output_path: Union[str, Path],
) -> None:
    """
    Save extracted trajectories to disk.

    Args:
        trajectories: Dictionary of trajectories by condition.
        output_path: Path to save trajectories.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {}
    for condition, traj_list in trajectories.items():
        data[condition.value] = [
            {
                "example_id": t.example_id,
                "trajectory": t.trajectory.tolist(),
                "moral_action": t.moral_action,
                "immoral_action": t.immoral_action,
            }
            for t in traj_list
        ]

    np.save(output_path, data)


def load_trajectories(
    input_path: Union[str, Path],
) -> Dict[Condition, List[LatentTrajectory]]:
    """
    Load trajectories from disk.

    Args:
        input_path: Path to saved trajectories.

    Returns:
        Dictionary of trajectories by condition.
    """
    data = np.load(input_path, allow_pickle=True).item()

    trajectories = {}
    for condition_str, traj_list in data.items():
        condition = Condition(condition_str)
        trajectories[condition] = [
            LatentTrajectory(
                example_id=t["example_id"],
                condition=condition,
                trajectory=np.array(t["trajectory"]),
                moral_action=t.get("moral_action"),
                immoral_action=t.get("immoral_action"),
            )
            for t in traj_list
        ]

    return trajectories
