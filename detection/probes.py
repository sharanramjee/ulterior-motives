"""
Linear probes for detecting misaligned reasoning in latent trajectories.

Implements the probe training and transfer evaluation from the paper:
- Train probes on [T][O] vs [O] (behaviorally distinguishable)
- Test transfer on [T] vs baseline (behaviorally identical)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from moralchain.dataset import Condition
from .extract import LatentTrajectory


@dataclass
class ProbeResults:
    """Results from probe evaluation."""
    accuracy: float
    auroc: float
    predictions: np.ndarray
    probabilities: np.ndarray


class LinearProbe:
    """
    Linear probe for detecting armed vs benign latent states.

    Uses logistic regression to classify latent vectors as armed or not.

    Args:
        token_idx: Token position to use for classification.
        regularization: L2 regularization strength.
    """

    def __init__(
        self,
        token_idx: int = 0,
        regularization: float = 1.0,
    ):
        self.token_idx = token_idx
        self.regularization = regularization
        self.classifier = LogisticRegression(
            C=1.0 / regularization,
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        positive_trajectories: List[LatentTrajectory],
        negative_trajectories: List[LatentTrajectory],
    ) -> "LinearProbe":
        """
        Train the probe on labeled trajectories.

        Args:
            positive_trajectories: Trajectories for positive class (armed).
            negative_trajectories: Trajectories for negative class (benign).

        Returns:
            Self for chaining.
        """
        # Extract features at specified token position
        X_pos = np.stack([t.trajectory[self.token_idx] for t in positive_trajectories])
        X_neg = np.stack([t.trajectory[self.token_idx] for t in negative_trajectories])

        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit classifier
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(
        self,
        trajectories: List[LatentTrajectory],
    ) -> np.ndarray:
        """
        Predict labels for trajectories.

        Args:
            trajectories: Trajectories to classify.

        Returns:
            Binary predictions (0 or 1).
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before prediction")

        X = np.stack([t.trajectory[self.token_idx] for t in trajectories])
        X_scaled = self.scaler.transform(X)

        return self.classifier.predict(X_scaled)

    def predict_proba(
        self,
        trajectories: List[LatentTrajectory],
    ) -> np.ndarray:
        """
        Predict probabilities for trajectories.

        Args:
            trajectories: Trajectories to classify.

        Returns:
            Probability of positive class.
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before prediction")

        X = np.stack([t.trajectory[self.token_idx] for t in trajectories])
        X_scaled = self.scaler.transform(X)

        return self.classifier.predict_proba(X_scaled)[:, 1]

    def evaluate(
        self,
        positive_trajectories: List[LatentTrajectory],
        negative_trajectories: List[LatentTrajectory],
    ) -> ProbeResults:
        """
        Evaluate probe on labeled trajectories.

        Args:
            positive_trajectories: Trajectories for positive class.
            negative_trajectories: Trajectories for negative class.

        Returns:
            ProbeResults with metrics.
        """
        all_trajectories = positive_trajectories + negative_trajectories
        y_true = np.array([1] * len(positive_trajectories) + [0] * len(negative_trajectories))

        predictions = self.predict(all_trajectories)
        probabilities = self.predict_proba(all_trajectories)

        accuracy = accuracy_score(y_true, predictions)
        auroc = roc_auc_score(y_true, probabilities)

        return ProbeResults(
            accuracy=accuracy,
            auroc=auroc,
            predictions=predictions,
            probabilities=probabilities,
        )

    def get_weights(self) -> np.ndarray:
        """Get the learned weight vector."""
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted first")
        return self.classifier.coef_[0]


class ProbeTrainer:
    """
    Trains and evaluates probes following the paper's methodology.

    Training: [T][O] (release) vs [O] (control)
    Transfer: [T] (armed) vs baseline

    Args:
        train_trajectories: Trajectories from training set.
        test_trajectories: Trajectories from test set.
    """

    def __init__(
        self,
        train_trajectories: Dict[Condition, List[LatentTrajectory]],
        test_trajectories: Dict[Condition, List[LatentTrajectory]],
    ):
        self.train_trajectories = train_trajectories
        self.test_trajectories = test_trajectories

        # Validate required conditions
        required = [Condition.BASELINE, Condition.ARMED, Condition.CONTROL, Condition.RELEASE]
        for cond in required:
            if cond not in train_trajectories:
                raise ValueError(f"Missing required condition in train: {cond}")
            if cond not in test_trajectories:
                raise ValueError(f"Missing required condition in test: {cond}")

    def train_and_evaluate(
        self,
        token_indices: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, ProbeResults]]:
        """
        Train probes at each token position and evaluate transfer.

        Args:
            token_indices: Token positions to train probes for.
                          If None, uses all positions.

        Returns:
            Dictionary mapping token index to train/transfer results.
        """
        # Determine token indices
        sample_traj = self.train_trajectories[Condition.BASELINE][0]
        num_tokens = sample_traj.trajectory.shape[0]

        if token_indices is None:
            token_indices = list(range(num_tokens))

        results = {}

        for token_idx in token_indices:
            # Create and train probe
            probe = LinearProbe(token_idx=token_idx)

            # Training: [T][O] (release) vs [O] (control)
            probe.fit(
                positive_trajectories=self.train_trajectories[Condition.RELEASE],
                negative_trajectories=self.train_trajectories[Condition.CONTROL],
            )

            # Evaluate on training distribution
            train_results = probe.evaluate(
                positive_trajectories=self.train_trajectories[Condition.RELEASE],
                negative_trajectories=self.train_trajectories[Condition.CONTROL],
            )

            # Transfer: [T] (armed) vs baseline
            transfer_results = probe.evaluate(
                positive_trajectories=self.test_trajectories[Condition.ARMED],
                negative_trajectories=self.test_trajectories[Condition.BASELINE],
            )

            results[token_idx] = {
                "train": train_results,
                "transfer": transfer_results,
                "probe": probe,
            }

        return results


def evaluate_transfer(
    train_trajectories: Dict[Condition, List[LatentTrajectory]],
    test_trajectories: Dict[Condition, List[LatentTrajectory]],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Run full probe training and transfer evaluation.

    Args:
        train_trajectories: Training set trajectories by condition.
        test_trajectories: Test set trajectories by condition.
        output_path: Path to save results.

    Returns:
        Dictionary with results per token position.
    """
    trainer = ProbeTrainer(train_trajectories, test_trajectories)
    results = trainer.train_and_evaluate()

    # Format results for output
    formatted_results = {}
    for token_idx, token_results in results.items():
        formatted_results[token_idx] = {
            "train_accuracy": token_results["train"].accuracy,
            "train_auroc": token_results["train"].auroc,
            "transfer_accuracy": token_results["transfer"].accuracy,
            "transfer_auroc": token_results["transfer"].auroc,
        }

    # Print results table
    print("\n" + "=" * 70)
    print("Linear Probe Training and Transfer Results")
    print("=" * 70)
    print("\n{:>10} {:>15} {:>15} {:>15} {:>15}".format(
        "Token", "Train Acc", "Train AUROC", "Transfer Acc", "Transfer AUROC"
    ))
    print("-" * 70)

    for token_idx in sorted(formatted_results.keys()):
        r = formatted_results[token_idx]
        print("{:>10} {:>14.1f}% {:>15.3f} {:>14.1f}% {:>15.3f}".format(
            f"z_{token_idx + 1}",
            r["train_accuracy"] * 100,
            r["train_auroc"],
            r["transfer_accuracy"] * 100,
            r["transfer_auroc"],
        ))

    print("=" * 70)

    # Save results if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(formatted_results, f, indent=2)

    return formatted_results
