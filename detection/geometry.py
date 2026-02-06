"""
Geometric analysis of latent reasoning trajectories.

Provides PCA and t-SNE visualization, plus quantitative analysis of
trajectory separation between conditions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from moralchain.dataset import Condition
from .extract import LatentTrajectory


# Color scheme for conditions
CONDITION_COLORS = {
    Condition.BASELINE: "green",
    Condition.ARMED: "orange",
    Condition.CONTROL: "blue",
    Condition.RELEASE: "red",
}

# Display labels for conditions
CONDITION_LABELS = {
    Condition.BASELINE: "baseline",
    Condition.ARMED: "armed [T]",
    Condition.CONTROL: "control [O]",
    Condition.RELEASE: "release [T][O]",
}


@dataclass
class GeometricMetrics:
    """Metrics for geometric analysis of trajectories."""
    mean_cosine_similarity: Dict[Tuple[Condition, Condition], float]
    mean_euclidean_distance: Dict[Tuple[Condition, Condition], float]
    pca_explained_variance: np.ndarray


class GeometricAnalyzer:
    """
    Analyzes geometric properties of latent trajectories.

    Args:
        trajectories: Dictionary mapping conditions to trajectory lists.
    """

    def __init__(
        self,
        trajectories: Dict[Condition, List[LatentTrajectory]],
    ):
        self.trajectories = trajectories
        self._validate_trajectories()

    def _validate_trajectories(self) -> None:
        """Validate that trajectories are consistent."""
        shapes = set()
        for traj_list in self.trajectories.values():
            if traj_list:
                shapes.add(traj_list[0].trajectory.shape)

        if len(shapes) > 1:
            raise ValueError(f"Inconsistent trajectory shapes: {shapes}")

    def compute_pairwise_similarity(
        self,
        token_idx: int = -1,
    ) -> Dict[Tuple[Condition, Condition], Tuple[float, float]]:
        """
        Compute pairwise cosine similarity between conditions.

        Args:
            token_idx: Token position to analyze (-1 for last token).

        Returns:
            Dictionary mapping condition pairs to (mean, std) similarity.
        """
        results = {}

        conditions = list(self.trajectories.keys())

        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i:]:
                similarities = []

                for traj1 in self.trajectories[cond1]:
                    for traj2 in self.trajectories[cond2]:
                        vec1 = traj1.trajectory[token_idx]
                        vec2 = traj2.trajectory[token_idx]
                        sim = cosine_similarity([vec1], [vec2])[0, 0]
                        similarities.append(sim)

                results[(cond1, cond2)] = (
                    np.mean(similarities),
                    np.std(similarities),
                )

        return results

    def compute_per_token_similarity(
        self,
    ) -> Dict[int, Dict[Tuple[Condition, Condition], Tuple[float, float]]]:
        """
        Compute pairwise similarity at each token position.

        Returns:
            Dictionary mapping token index to pairwise similarities.
        """
        # Get number of tokens from first trajectory
        sample_traj = next(iter(self.trajectories.values()))[0]
        num_tokens = sample_traj.trajectory.shape[0]

        results = {}
        for token_idx in range(num_tokens):
            results[token_idx] = self.compute_pairwise_similarity(token_idx)

        return results

    def fit_pca(
        self,
        n_components: int = 3,
        token_idx: Optional[int] = None,
    ) -> Tuple[PCA, Dict[Condition, np.ndarray]]:
        """
        Fit PCA on all trajectories.

        Args:
            n_components: Number of PCA components.
            token_idx: If specified, only use this token position.
                       If None, concatenate all tokens.

        Returns:
            Tuple of (fitted PCA, transformed data by condition).
        """
        # Collect all vectors
        all_vectors = []
        condition_indices = {}
        current_idx = 0

        for condition, traj_list in self.trajectories.items():
            start_idx = current_idx

            for traj in traj_list:
                if token_idx is not None:
                    vec = traj.trajectory[token_idx]
                else:
                    vec = traj.trajectory.flatten()
                all_vectors.append(vec)
                current_idx += 1

            condition_indices[condition] = (start_idx, current_idx)

        # Fit PCA
        all_vectors = np.array(all_vectors)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(all_vectors)

        # Split by condition
        transformed_by_condition = {}
        for condition, (start, end) in condition_indices.items():
            transformed_by_condition[condition] = transformed[start:end]

        return pca, transformed_by_condition

    def fit_tsne(
        self,
        n_components: int = 3,
        token_idx: Optional[int] = None,
        perplexity: int = 30,
        random_state: int = 42,
    ) -> Dict[Condition, np.ndarray]:
        """
        Fit t-SNE on all trajectories.

        Args:
            n_components: Number of t-SNE components.
            token_idx: If specified, only use this token position.
            perplexity: t-SNE perplexity parameter.
            random_state: Random seed.

        Returns:
            Dictionary mapping conditions to transformed data.
        """
        # Collect all vectors
        all_vectors = []
        condition_indices = {}
        current_idx = 0

        for condition, traj_list in self.trajectories.items():
            start_idx = current_idx

            for traj in traj_list:
                if token_idx is not None:
                    vec = traj.trajectory[token_idx]
                else:
                    vec = traj.trajectory.flatten()
                all_vectors.append(vec)
                current_idx += 1

            condition_indices[condition] = (start_idx, current_idx)

        # Fit t-SNE
        all_vectors = np.array(all_vectors)
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(all_vectors) // 4),
            random_state=random_state,
        )
        transformed = tsne.fit_transform(all_vectors)

        # Split by condition
        transformed_by_condition = {}
        for condition, (start, end) in condition_indices.items():
            transformed_by_condition[condition] = transformed[start:end]

        return transformed_by_condition


def visualize_trajectories(
    trajectories: Dict[Condition, List[LatentTrajectory]],
    example_indices: List[int],
    output_path: Optional[Union[str, Path]] = None,
    method: str = "pca",
    figsize: Tuple[int, int] = (24, 6),
) -> None:
    """
    Visualize latent trajectories for selected examples.

    Creates 3D plots showing trajectory evolution through latent space.

    Args:
        trajectories: Dictionary of trajectories by condition.
        example_indices: Indices of examples to visualize.
        output_path: Path to save figure.
        method: Dimensionality reduction method ('pca' or 'tsne').
        figsize: Figure size.
    """
    num_examples = len(example_indices)
    fig = plt.figure(figsize=figsize)

    for plot_idx, ex_idx in enumerate(example_indices):
        # Collect trajectories for this example
        example_trajectories = {}
        for condition, traj_list in trajectories.items():
            if ex_idx < len(traj_list):
                example_trajectories[condition] = traj_list[ex_idx].trajectory

        # Stack all trajectories for dimensionality reduction
        all_points = np.vstack(list(example_trajectories.values()))

        if method == "pca":
            reducer = PCA(n_components=3)
        else:
            reducer = TSNE(
                n_components=3,
                perplexity=min(5, len(all_points) // 4),
                random_state=42,
            )

        transformed = reducer.fit_transform(all_points)

        # Split back by condition
        start_idx = 0
        transformed_by_condition = {}
        for condition, traj in example_trajectories.items():
            num_points = traj.shape[0]
            transformed_by_condition[condition] = transformed[start_idx:start_idx + num_points]
            start_idx += num_points

        # Create 3D subplot
        ax = fig.add_subplot(1, num_examples, plot_idx + 1, projection='3d')

        for condition, points in transformed_by_condition.items():
            color = CONDITION_COLORS[condition]
            label = CONDITION_LABELS[condition]

            # Plot start point
            ax.scatter(
                points[0, 0], points[0, 1], points[0, 2],
                c=color, s=100, marker='o', edgecolor='black', linewidth=1,
            )

            # Plot end point
            ax.scatter(
                points[-1, 0], points[-1, 1], points[-1, 2],
                c=color, s=100, marker='X', edgecolor='black', linewidth=1,
            )

            # Draw arrows between points
            for i in range(len(points) - 1):
                ax.quiver(
                    points[i, 0], points[i, 1], points[i, 2],
                    points[i+1, 0] - points[i, 0],
                    points[i+1, 1] - points[i, 1],
                    points[i+1, 2] - points[i, 2],
                    color=color, alpha=0.8, arrow_length_ratio=0.1, linewidth=1.5,
                )

        ax.set_xlabel('PC1' if method == 'pca' else 't-SNE1')
        ax.set_ylabel('PC2' if method == 'pca' else 't-SNE2')
        ax.set_zlabel('PC3' if method == 'pca' else 't-SNE3')
        ax.set_title(f'Q{ex_idx + 1}')

    # Add legend
    legend_handles = [
        Patch(color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond])
        for cond in [Condition.BASELINE, Condition.CONTROL, Condition.ARMED, Condition.RELEASE]
    ]
    fig.legend(handles=legend_handles, loc='upper right', fontsize=10)
    fig.text(
        0.5, 0.02,
        '● = Start; ✕ = End; Arrows show direction along trajectory',
        ha='center', fontsize=11,
    )

    plt.tight_layout(rect=[0, 0.05, 0.95, 0.97])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_similarity_by_token(
    trajectories: Dict[Condition, List[LatentTrajectory]],
    condition_pairs: Optional[List[Tuple[Condition, Condition]]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot cosine similarity between condition pairs across token positions.

    Args:
        trajectories: Dictionary of trajectories by condition.
        condition_pairs: Pairs of conditions to compare.
        output_path: Path to save figure.
    """
    analyzer = GeometricAnalyzer(trajectories)
    per_token_sim = analyzer.compute_per_token_similarity()

    # Default pairs to plot
    if condition_pairs is None:
        condition_pairs = [
            (Condition.BASELINE, Condition.CONTROL),
            (Condition.ARMED, Condition.RELEASE),
            (Condition.BASELINE, Condition.ARMED),
            (Condition.BASELINE, Condition.RELEASE),
        ]

    # Extract data
    token_indices = sorted(per_token_sim.keys())

    plt.figure(figsize=(10, 6))

    for cond1, cond2 in condition_pairs:
        means = []
        stds = []

        for token_idx in token_indices:
            key = (cond1, cond2) if (cond1, cond2) in per_token_sim[token_idx] else (cond2, cond1)
            mean, std = per_token_sim[token_idx][key]
            means.append(mean)
            stds.append(std)

        label = f"{CONDITION_LABELS[cond1]} vs {CONDITION_LABELS[cond2]}"
        plt.plot(token_indices, means, marker='o', label=label)
        plt.fill_between(
            token_indices,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
        )

    plt.xlabel('Token Position')
    plt.ylabel('Cosine Similarity')
    plt.title('Pairwise Similarity Across Latent Token Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.show()


def compute_separation_metrics(
    trajectories: Dict[Condition, List[LatentTrajectory]],
) -> Dict[str, float]:
    """
    Compute quantitative separation metrics between conditions.

    Args:
        trajectories: Dictionary of trajectories by condition.

    Returns:
        Dictionary with separation metrics.
    """
    analyzer = GeometricAnalyzer(trajectories)

    metrics = {}

    # Compute similarities at first and last token
    for token_idx, token_name in [(0, "first"), (-1, "last")]:
        sims = analyzer.compute_pairwise_similarity(token_idx)

        for (cond1, cond2), (mean, std) in sims.items():
            key = f"{token_name}_{cond1.value}_{cond2.value}_cosine"
            metrics[key] = float(mean)  # Convert to Python float for JSON serialization

    # Compute baseline/control vs armed/release separation
    first_token_sims = analyzer.compute_pairwise_similarity(0)

    # Armed-Release similarity (should be high early)
    ar_key = (Condition.ARMED, Condition.RELEASE)
    if ar_key not in first_token_sims:
        ar_key = (Condition.RELEASE, Condition.ARMED)
    metrics["armed_release_first_sim"] = float(first_token_sims[ar_key][0])

    # Baseline-Control similarity (should be high throughout)
    bc_key = (Condition.BASELINE, Condition.CONTROL)
    if bc_key not in first_token_sims:
        bc_key = (Condition.CONTROL, Condition.BASELINE)
    metrics["baseline_control_first_sim"] = float(first_token_sims[bc_key][0])

    return metrics
