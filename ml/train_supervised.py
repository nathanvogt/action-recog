import argparse
from typing import Sequence, List
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


class RepFrameDataset(Dataset):
    """Frame-based dataset for supervised rep boundary detection."""

    def __init__(
        self,
        subject: str,
        exercise: str,
        dataset_root: str = "train",
        c: int = DEFAULT_C,
        m: int = DEFAULT_M,
        tol: int = 10,
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
        self.sls = SlsMemoized(c=c, m=m)

        self.poses = self.dataset.get_pose_array(subject, exercise)
        self.rep_idx = self.dataset.get_rep_timings(subject, exercise) or []

        self.features, self.labels = self._process()

    def _frame_to_obs(self) -> np.ndarray:
        curves = self.sls.sls_curves
        if not curves:
            return np.zeros(len(self.keypoints) * self.c * 3, dtype=np.float32)
        obs = np.concatenate([np.array(c, dtype=np.float32) for c in curves]).ravel()
        return obs

    def _is_rep_boundary(self, idx: int) -> int:
        return int(any(abs(idx - r) <= self.tol for r in self.rep_idx[1:]))

    def _process(self) -> tuple[np.ndarray, np.ndarray]:
        feats: List[np.ndarray] = []
        labs: List[int] = []
        for i, pose in enumerate(self.poses):
            self.sls.process_poses([pose], self.keypoints)
            feats.append(self._frame_to_obs())
            labs.append(self._is_rep_boundary(i))
            # reset at exact boundary to mimic environment progression
            if i in self.rep_idx[1:]:
                self.sls.reset()
        return np.stack(feats), np.array(labs, dtype=np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


def train_supervised(args: argparse.Namespace) -> None:
    dataset = RepFrameDataset(
        args.subject,
        args.exercise,
        dataset_root=args.dataset_root,
        c=args.c,
        m=args.m,
        tol=args.tol,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_supervised(args)
