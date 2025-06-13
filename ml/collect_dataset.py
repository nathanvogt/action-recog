import argparse
import os
import pickle
from typing import Sequence, List, Tuple
import yaml
import multiprocessing as mp
from functools import partial

import numpy as np
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


def _worker_process_sequences(
    sequence_data: List[Tuple[int, int, int]],
    poses: np.ndarray,
    keypoints: List[int],
    c: int,
    m: int,
) -> Tuple[List[np.ndarray], List[int]]:
    """Worker function to process a batch of sequences in parallel."""
    # Create a fresh SLS instance for this worker
    sls = SlsMemoized(c=c, m=m)

    def frame_to_obs() -> np.ndarray:
        curves = sls.sls_curves
        if not curves:
            return np.zeros(len(keypoints) * c * 3, dtype=np.float32)
        obs = np.concatenate(
            [np.array(curve, dtype=np.float32) for curve in curves]
        ).ravel()
        return obs

    def process_sequence(start: int, end: int) -> np.ndarray:
        sls.reset()
        if start <= end:
            rng = range(start, end + 1)
        else:
            rng = range(start, end - 1, -1)
        sls.reset()
        for i in rng:
            sls.process_poses([poses[i]], keypoints)
        return frame_to_obs()

    features = []
    labels = []

    for start, end, label in sequence_data:
        feat = process_sequence(start, end)
        features.append(feat)
        labels.append(label)

    return features, labels


class RepSequenceCollector:
    """Collects and processes pose sequences as positive/negative rep examples."""

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
        parallel: bool = False,
        n_cores: int = None,
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
        self.m = m
        self.tol = tol
        self.n_samples = n_samples
        self.parallel = parallel
        self.n_cores = n_cores if n_cores is not None else mp.cpu_count()
        self.sls = SlsMemoized(c=c, m=m)

        self.poses = self.dataset.get_pose_array(subject, exercise)
        self.rep_idx = [
            int(r) for r in (self.dataset.get_rep_timings(subject, exercise) or [])
        ]

        print(f"Loaded data for {subject}/{exercise}")
        print(f"  Poses: {len(self.poses)}")
        print(f"  Rep boundaries: {len(self.rep_idx)}")
        print(f"  Keypoints: {len(self.keypoints)}")
        print(f"  SLS parameters: c={c}, m={m}")
        print(f"  Tolerance: {tol}")
        print(f"  Parallel processing: {'enabled' if parallel else 'disabled'}")
        if parallel:
            print(f"  Number of cores: {self.n_cores}")
        print("-" * 60)

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

    def collect_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Collect and return features and labels."""
        if self.parallel:
            return self._collect_dataset_parallel()
        else:
            return self._collect_dataset_sequential()

    def _collect_dataset_sequential(self) -> tuple[np.ndarray, np.ndarray]:
        """Sequential data collection (original implementation)."""
        feats: List[np.ndarray] = []
        labs: List[int] = []

        print(
            f"Sampling {self.n_samples} sequences for {self.subject}/{self.exercise} (sequential)..."
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

    def _collect_dataset_parallel(self) -> tuple[np.ndarray, np.ndarray]:
        """Parallel data collection using multiprocessing."""
        print(
            f"Sampling {self.n_samples} sequences for {self.subject}/{self.exercise} (parallel with {self.n_cores} cores)..."
        )

        # Pre-generate all sequence indices and labels
        sequence_specs = []
        positive_count = 0
        negative_count = 0

        print("Generating sequence specifications...")
        for i in tqdm(range(self.n_samples), desc="Generating specs", unit="spec"):
            if np.random.rand() < 0.5:
                idx1, idx2 = self._sample_positive_indices()
                label = 1
                positive_count += 1
            else:
                idx1, idx2 = self._sample_negative_indices()
                label = 0
                negative_count += 1
            start, end = sorted((idx1, idx2))
            sequence_specs.append((start, end, label))

        print(f"Generated {len(sequence_specs)} sequence specifications")
        print(f"  Positive samples: {positive_count}")
        print(f"  Negative samples: {negative_count}")

        # Split work into chunks for parallel processing
        chunk_size = max(1, self.n_samples // self.n_cores)
        chunks = [
            sequence_specs[i : i + chunk_size]
            for i in range(0, len(sequence_specs), chunk_size)
        ]

        print(f"Split work into {len(chunks)} chunks (chunk size: ~{chunk_size})")

        # Create partial function with fixed parameters
        worker_func = partial(
            _worker_process_sequences,
            poses=self.poses,
            keypoints=self.keypoints,
            c=self.c,
            m=self.m,
        )

        # Process chunks in parallel
        print("Processing sequences in parallel...")
        all_features = []
        all_labels = []

        with mp.Pool(processes=self.n_cores) as pool:
            # Use imap for progress tracking
            results = list(
                tqdm(
                    pool.imap(worker_func, chunks),
                    total=len(chunks),
                    desc="Processing chunks",
                    unit="chunk",
                )
            )

        # Combine results from all workers
        for chunk_features, chunk_labels in results:
            all_features.extend(chunk_features)
            all_labels.extend(chunk_labels)

        print(f"Data collection complete!")
        print(f"  Total samples: {len(all_labels)}")
        print(f"  Positive samples: {sum(all_labels)}")
        print(f"  Negative samples: {len(all_labels) - sum(all_labels)}")
        print(f"  Feature dimension: {len(all_features[0]) if all_features else 0}")
        print("-" * 60)

        return np.stack(all_features), np.array(all_labels, dtype=np.float32)


def save_dataset(
    features: np.ndarray, labels: np.ndarray, save_path: str, metadata: dict
) -> None:
    """Save the collected dataset to disk."""
    os.makedirs(save_path, exist_ok=True)

    # Save features and labels as numpy arrays
    np.save(os.path.join(save_path, "features.npy"), features)
    np.save(os.path.join(save_path, "labels.npy"), labels)

    # Save metadata as pickle
    with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"Dataset saved to {save_path}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Metadata keys: {list(metadata.keys())}")


def load_dataset(load_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a previously saved dataset."""
    features = np.load(os.path.join(load_path, "features.npy"))
    labels = np.load(os.path.join(load_path, "labels.npy"))

    with open(os.path.join(load_path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    print(f"Dataset loaded from {load_path}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Metadata: {metadata}")

    return features, labels, metadata


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Collect and save rep sequence dataset"
    )

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Data collection parameters
    parser.add_argument("--subject", type=str, help="Subject ID")
    parser.add_argument("--exercise", type=str, help="Exercise name")
    parser.add_argument("--dataset-root", default="train", type=str)
    parser.add_argument("--c", type=int, default=DEFAULT_C, help="SLS parameter c")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="SLS parameter m")
    parser.add_argument("--tol", type=int, default=10, help="Boundary tolerance")
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of sequences to sample"
    )

    # Parallel processing parameters
    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel processing"
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        help="Number of CPU cores to use (default: all available)",
    )

    # Output
    parser.add_argument(
        "--save-path", type=str, help="Path to save the collected dataset"
    )

    # Utility options
    parser.add_argument(
        "--load-test", type=str, help="Test loading a dataset from the specified path"
    )

    args = parser.parse_args()

    # Test loading functionality
    if args.load_test:
        try:
            features, labels, metadata = load_dataset(args.load_test)
            print("✓ Dataset loading test successful!")
        except Exception as e:
            print(f"✗ Dataset loading test failed: {e}")
        return

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
    if not args.save_path:
        raise ValueError("Save path is required (via --save-path)")

    # Set up default save path if not provided
    if not args.save_path:
        args.save_path = f"./datasets/{args.subject}_{args.exercise}"

    print(f"Starting data collection for {args.subject}/{args.exercise}")
    print(f"Will save dataset to: {args.save_path}")
    print("-" * 60)

    # Collect the dataset
    collector = RepSequenceCollector(
        subject=args.subject,
        exercise=args.exercise,
        dataset_root=args.dataset_root,
        c=args.c,
        m=args.m,
        tol=args.tol,
        n_samples=args.n_samples,
        parallel=args.parallel,
        n_cores=args.n_cores,
    )

    features, labels = collector.collect_dataset()

    # Prepare metadata
    metadata = {
        "subject": args.subject,
        "exercise": args.exercise,
        "dataset_root": args.dataset_root,
        "c": args.c,
        "m": args.m,
        "tol": args.tol,
        "n_samples": args.n_samples,
        "feature_dim": features.shape[1],
        "n_keypoints": len(collector.keypoints),
        "keypoints": collector.keypoints,
        "positive_samples": int(np.sum(labels)),
        "negative_samples": int(len(labels) - np.sum(labels)),
    }

    # Save the dataset
    save_dataset(features, labels, args.save_path, metadata)

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE!")
    print(f"Dataset saved to: {args.save_path}")
    print(f"To test loading: python collect_dataset.py --load-test {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
