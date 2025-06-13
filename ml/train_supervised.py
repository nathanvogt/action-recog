import argparse
from typing import Sequence, List, Tuple
import os
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

from train_dataset import TrainDatasetLocal
from sls_memoized import (
    SlsMemoized,
    DEFAULT_C,
    DEFAULT_M,
    LEFT_LEG_NO_FEET,
    RIGHT_LEG_NO_FEET,
    LEFT_ARM_NO_HAND,
    RIGHT_ARM_NO_HAND,
    BACK,
)
from model import RepPolicy


class RepSequenceDataset(Dataset):
    """Dataset that samples pose sequences as positive/negative rep examples."""

    def __init__(
        self,
        subject: str,
        exercise: str,
        dataset_root: str = "train",
        c: int = DEFAULT_C,
        m: int = DEFAULT_M,
        tol: int = 10,
        n_samples: int = 1000,
        keypoints: Sequence[int] | None = None,
    ) -> None:
        self.dataset = TrainDatasetLocal(dataset_root)
        self.subject = subject
        self.exercise = exercise
        self.keypoints = (
            list(keypoints)
            if keypoints is not None
            else LEFT_LEG_NO_FEET
            + RIGHT_LEG_NO_FEET
            + LEFT_ARM_NO_HAND
            + RIGHT_ARM_NO_HAND
            + BACK
        )
        self.c = c
        self.tol = tol
        self.n_samples = n_samples
        self.sls = SlsMemoized(c=c, m=m)

        self.poses = self.dataset.get_pose_array(subject, exercise)
        self.rep_idx = [
            int(r) for r in (self.dataset.get_rep_timings(subject, exercise) or [])
        ]

        self.features, self.labels = self._sample_sequences()

    def _frame_to_obs(self) -> np.ndarray:
        curves = self.sls.sls_curves
        if not curves:
            return np.zeros(len(self.keypoints) * self.c * 3, dtype=np.float32)
        obs = np.concatenate([np.array(c, dtype=np.float32) for c in curves]).ravel()
        return obs

    def _near_boundary(self, idx: int) -> bool:
        return any(abs(idx - r) <= self.tol for r in self.rep_idx)

    def _sample_positive_indices(self) -> Tuple[int, int]:
        if len(self.rep_idx) < 2:
            return self._sample_negative_indices()

        j = np.random.randint(1, len(self.rep_idx))
        boundary = self.rep_idx[j]
        neighbors = []
        if j - 1 >= 0:
            neighbors.append(self.rep_idx[j - 1])
        if j + 1 < len(self.rep_idx):
            neighbors.append(self.rep_idx[j + 1])
        other = np.random.choice(neighbors)

        idx1 = np.clip(
            np.random.randint(boundary - self.tol, boundary + self.tol + 1),
            0,
            len(self.poses) - 1,
        )
        idx2 = np.clip(
            np.random.randint(other - self.tol, other + self.tol + 1),
            0,
            len(self.poses) - 1,
        )
        return (idx1, idx2)

    def _sample_non_boundary_index(self) -> int:
        while True:
            idx = np.random.randint(0, len(self.poses))
            if not self._near_boundary(idx):
                return idx

    def _sample_negative_indices(self) -> Tuple[int, int]:
        if not self.rep_idx:
            idx1 = np.random.randint(0, len(self.poses))
            idx2 = np.random.randint(0, len(self.poses))
            return (idx1, idx2)

        if np.random.rand() < 0.5:
            idx1 = self._sample_non_boundary_index()
            idx2 = self._sample_non_boundary_index()
        else:
            idx1 = self._sample_non_boundary_index()
            boundary = np.random.choice(self.rep_idx)
            idx2 = np.clip(
                np.random.randint(boundary - self.tol, boundary + self.tol + 1),
                0,
                len(self.poses) - 1,
            )
        return (idx1, idx2)

    def _process_sequence(self, start: int, end: int) -> np.ndarray:
        self.sls.reset()
        if start <= end:
            rng = range(start, end + 1)
        else:
            rng = range(start, end - 1, -1)
        self.sls.reset()
        for i in rng:
            self.sls.process_poses([self.poses[i]], self.keypoints)
        return self._frame_to_obs()

    def _sample_sequences(self) -> tuple[np.ndarray, np.ndarray]:
        feats: List[np.ndarray] = []
        labs: List[int] = []
        for _ in range(self.n_samples):
            if np.random.rand() < 0.5:
                idx1, idx2 = self._sample_positive_indices()
                label = 1
            else:
                idx1, idx2 = self._sample_negative_indices()
                label = 0
            start, end = sorted((idx1, idx2))
            feats.append(self._process_sequence(start, end))
            labs.append(label)
        return np.stack(feats), np.array(labs, dtype=np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = torch.from_numpy(self.features[idx])
        # Convert to one-hot encoding: [1, 0] for class 0, [0, 1] for class 1
        y_onehot = torch.zeros(2)
        y_onehot[int(self.labels[idx])] = 1.0
        return x, y_onehot


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file"""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_supervised(args: argparse.Namespace) -> None:
    print(f"Starting supervised training for {args.subject}/{args.exercise}")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")

    dataset = RepSequenceDataset(
        args.subject,
        args.exercise,
        dataset_root=args.dataset_root,
        c=args.c,
        m=args.m,
        tol=args.tol,
        n_samples=args.n_samples,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = RepPolicy(
        dataset.features.shape[1], hidden=args.hidden_dim, n_layers=args.n_layers
    )
    # Use MSELoss for one-hot encoded targets (alternative: CrossEntropyLoss with class indices)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(
        f"Model architecture: {sum(p.numel() for p in model.parameters())} parameters"
    )
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(loader)}")
    print("-" * 60)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        print(f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (X, y) in enumerate(loader):
            optimizer.zero_grad()
            pred = model(X)  # Shape: (batch_size, 2)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            # Update metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss * X.size(0)

            # Calculate accuracy - compare predicted class with true class
            pred_class = torch.argmax(pred, dim=1)  # Get predicted class (0 or 1)
            true_class = torch.argmax(y, dim=1)  # Get true class from one-hot
            correct_predictions += (pred_class == true_class).sum().item()
            total_predictions += y.size(0)

            # Print progress every 10 batches or at the end
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                current_loss = epoch_loss / ((batch_idx + 1) * args.batch_size)
                current_acc = correct_predictions / total_predictions
                print(
                    f"  Batch {batch_idx + 1}/{len(loader)} - "
                    f"Loss: {current_loss:.4f}, "
                    f"Acc: {current_acc:.4f}, "
                    f"Batch Loss: {batch_loss:.4f}"
                )

        # Calculate final epoch metrics
        epoch_loss /= len(dataset)
        epoch_acc = correct_predictions / total_predictions

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Final Loss: {epoch_loss:.4f}")
        print(
            f"  Final Accuracy: {epoch_acc:.4f} ({correct_predictions}/{total_predictions})"
        )

        # Track best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"  ✓ New best loss: {best_loss:.4f}")

        print("-" * 60)

    print(f"\nTraining completed!")
    print(f"Best loss achieved: {best_loss:.4f}")

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        torch.save(
            model.state_dict(), os.path.join(args.save_path, "rep_classifier.pt")
        )
        print(f"Model saved to {args.save_path}/rep_classifier.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised rep detector")

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Remove required=True from subject and exercise since they can come from YAML
    parser.add_argument("--subject", type=str, help="Subject ID")
    parser.add_argument("--exercise", type=str, help="Exercise name")
    parser.add_argument("--dataset-root", default="train", type=str)
    parser.add_argument("--c", type=int, default=DEFAULT_C, help="SLS parameter c")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="SLS parameter m")
    parser.add_argument("--tol", type=int, default=10, help="Boundary tolerance")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--n-layers", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument("--save-path", default="./models/supervised", type=str)
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of sequences to sample"
    )

    args = parser.parse_args()

    # Load YAML config if provided
    if args.config:
        yaml_config = load_config_from_yaml(args.config)

        # Override defaults with YAML values (only if not provided via command line)
        for key, value in yaml_config.items():
            key_with_underscores = key.replace("-", "_")
            if hasattr(args, key_with_underscores):
                # Check if the argument was provided via command line by comparing to default
                default_value = parser.get_default(key_with_underscores)
                current_value = getattr(args, key_with_underscores)
                # If current value is same as default, use YAML value
                if current_value == default_value:
                    setattr(args, key_with_underscores, value)

    # Validate required arguments
    if not args.subject:
        raise ValueError("Subject is required (via --subject or config file)")
    if not args.exercise:
        raise ValueError("Exercise is required (via --exercise or config file)")

    return args


if __name__ == "__main__":
    args = parse_args()
    train_supervised(args)
