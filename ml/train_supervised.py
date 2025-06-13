import argparse
from typing import Sequence, List, Tuple
import os
import yaml
import pickle

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
from model import PPOCompatiblePolicy
from collect_dataset import load_dataset


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

        print(
            f"Sampling {self.n_samples} sequences for {self.subject}/{self.exercise}..."
        )
        positive_count = 0
        negative_count = 0

        # Use tqdm for progress tracking
        for i in tqdm(range(self.n_samples), desc="Processing sequences", unit="seq"):
            if np.random.rand() < 0.5:
                idx1, idx2 = self._sample_positive_indices()
                label = 1
                positive_count += 1
            else:
                idx1, idx2 = self._sample_negative_indices()
                label = 0
                negative_count += 1
            start, end = sorted((idx1, idx2))
            feats.append(self._process_sequence(start, end))
            labs.append(label)

            # Show intermediate progress every 100 samples
            if (i + 1) % 100 == 0:
                tqdm.write(
                    f"  Progress: {i+1}/{self.n_samples} - "
                    f"Positive: {positive_count}, Negative: {negative_count}"
                )

        print(f"Data collection complete!")
        print(f"  Total samples: {self.n_samples}")
        print(f"  Positive samples: {positive_count}")
        print(f"  Negative samples: {negative_count}")
        print(f"  Feature dimension: {len(feats[0]) if feats else 0}")
        print("-" * 60)

        return np.stack(feats), np.array(labs, dtype=np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = torch.from_numpy(self.features[idx])
        # Return class index for CrossEntropyLoss
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


class PreSavedRepSequenceDataset(Dataset):
    """Dataset that loads pre-saved features and labels."""

    def __init__(self, dataset_path: str) -> None:
        self.features, self.labels, self.metadata = load_dataset(dataset_path)

        print(
            f"Loaded {len(self.labels)} samples from {dataset_path} ({self.metadata['positive_samples']} pos, {self.metadata['negative_samples']} neg)"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = torch.from_numpy(self.features[idx])
        # Return class index for CrossEntropyLoss
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file"""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_supervised(args: argparse.Namespace) -> None:
    # Determine whether to load pre-saved dataset or collect fresh data
    if hasattr(args, "load_dataset") and args.load_dataset:
        dataset = PreSavedRepSequenceDataset(args.load_dataset)
        subject_exercise = (
            f"{dataset.metadata['subject']}/{dataset.metadata['exercise']}"
        )
        feature_dim = dataset.metadata["feature_dim"]
    else:
        dataset = RepSequenceDataset(
            args.subject,
            args.exercise,
            dataset_root=args.dataset_root,
            c=args.c,
            m=args.m,
            tol=args.tol,
            n_samples=args.n_samples,
        )
        subject_exercise = f"{args.subject}/{args.exercise}"
        feature_dim = dataset.features.shape[1]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = PPOCompatiblePolicy(
        feature_dim,
        hidden=args.hidden_dim,
        n_layers=args.n_layers,
        n_actions=2,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(
        f"Training {subject_exercise} | {args.epochs} epochs | batch_size={args.batch_size} | {sum(p.numel() for p in model.parameters())} params"
    )

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (X, y) in enumerate(loader):
            optimizer.zero_grad()
            pred = model.get_action_logits(X)  # Shape: (batch_size, 2) - raw logits
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            # Update metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss * X.size(0)

            # Calculate accuracy - compare predicted class with true class
            pred_class = torch.argmax(pred, dim=1)  # Get predicted class (0 or 1)
            correct_predictions += (pred_class == y).sum().item()
            total_predictions += y.size(0)

        # Calculate final epoch metrics
        epoch_loss /= len(dataset)
        epoch_acc = correct_predictions / total_predictions

        # Print concise epoch summary
        best_marker = ""
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_marker = " ✓"

        print(
            f"Epoch {epoch + 1:2d}/{args.epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}{best_marker}"
        )

    print(f"Training completed! Best loss: {best_loss:.4f}")

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

    # Option to load pre-saved dataset
    parser.add_argument(
        "--load-dataset", type=str, help="Path to pre-saved dataset directory"
    )

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
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=4,
        help="Number of times to log progress per epoch (default: 4)",
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
    if args.load_dataset:
        # When loading a pre-saved dataset, subject/exercise are not required
        print(f"Will load pre-saved dataset from: {args.load_dataset}")
    else:
        # When collecting fresh data, subject/exercise are required
        if not args.subject:
            raise ValueError("Subject is required (via --subject or config file)")
        if not args.exercise:
            raise ValueError("Exercise is required (via --exercise or config file)")

    return args


if __name__ == "__main__":
    args = parse_args()
    train_supervised(args)
