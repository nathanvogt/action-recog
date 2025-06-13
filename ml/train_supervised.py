import argparse
from typing import Sequence, List, Tuple
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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
        self.rep_idx = [int(r) for r in (self.dataset.get_rep_timings(subject, exercise) or [])]

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

        idx1 = np.clip(np.random.randint(boundary - self.tol, boundary + self.tol + 1), 0, len(self.poses) - 1)
        idx2 = np.clip(np.random.randint(other - self.tol, other + self.tol + 1), 0, len(self.poses) - 1)
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
            idx2 = np.clip(np.random.randint(boundary - self.tol, boundary + self.tol + 1), 0, len(self.poses) - 1)
        return (idx1, idx2)

    def _process_sequence(self, start: int, end: int) -> np.ndarray:
        self.sls.reset()
        if start <= end:
            rng = range(start, end + 1)
        else:
            rng = range(start, end - 1, -1)
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
        y = torch.tensor(self.labels[idx])
        return x, y


def train_supervised(args: argparse.Namespace) -> None:
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

    model = RepPolicy(dataset.features.shape[1], hidden=args.hidden_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for X, y in loader:
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {epoch_loss:.4f}")

    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, "rep_classifier.pt"))
    print(f"Model saved to {args.save_path}/rep_classifier.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised rep detector")
    parser.add_argument("--subject", required=True, type=str, help="Subject ID")
    parser.add_argument("--exercise", required=True, type=str, help="Exercise name")
    parser.add_argument("--dataset-root", default="train", type=str)
    parser.add_argument("--c", type=int, default=DEFAULT_C, help="SLS parameter c")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="SLS parameter m")
    parser.add_argument("--tol", type=int, default=10, help="Boundary tolerance")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--save-path", default="./models/supervised", type=str)
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of sequences to sample")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_supervised(args)
