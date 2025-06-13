#!/usr/bin/env python3
"""
Dataset Visualization Tool

This script allows you to visualize SLS curve samples from collected datasets.
It loads the dataset and displays the 3D SLS curves for each sample.

Usage:
    python visualize_dataset.py --dataset-path ./datasets/S03_dumbbell_biceps_curls
    python visualize_dataset.py --dataset-path ./datasets/S03_dumbbell_biceps_curls --sample-idx 42
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collect_dataset import load_dataset
import random


class DatasetVisualizer:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.features, self.labels, self.metadata = load_dataset(dataset_path)

        # Extract metadata
        self.c = self.metadata["c"]
        self.n_keypoints = self.metadata["n_keypoints"]
        self.keypoints = self.metadata["keypoints"]
        self.subject = self.metadata["subject"]
        self.exercise = self.metadata["exercise"]

        print(f"Loaded dataset: {self.subject}/{self.exercise}")
        print(f"  Total samples: {len(self.labels)}")
        print(f"  Positive samples: {self.metadata['positive_samples']}")
        print(f"  Negative samples: {self.metadata['negative_samples']}")
        print(f"  Feature dimension: {self.metadata['feature_dim']}")
        print(f"  SLS parameters: c={self.c}, keypoints={self.n_keypoints}")
        print("-" * 60)

    def parse_features_to_curves(self, features: np.ndarray) -> list:
        """
        Parse flattened feature vector back into 3D curves for each keypoint.

        Args:
            features: Flattened array of shape (n_keypoints * c * 3,)

        Returns:
            List of curves, where each curve is an array of shape (c, 3)
        """
        curves = []

        for kp_idx in range(self.n_keypoints):
            start_idx = kp_idx * self.c * 3
            curve_data = features[start_idx : start_idx + self.c * 3]

            # Reshape to (c, 3) - c points with (x, y, z) coordinates
            curve = curve_data.reshape(self.c, 3)
            curves.append(curve)

        return curves

    def visualize_sample(self, sample_idx: int, ax=None):
        """Visualize a single sample's SLS curves."""
        if sample_idx < 0 or sample_idx >= len(self.labels):
            raise ValueError(
                f"Sample index {sample_idx} out of range [0, {len(self.labels)-1}]"
            )

        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")

        # Get sample data
        features = self.features[sample_idx]
        label = self.labels[sample_idx]
        curves = self.parse_features_to_curves(features)

        # Color scheme
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_keypoints))

        # Plot each keypoint's curve
        for kp_idx, curve in enumerate(curves):
            # Filter out zero-padded points (if any)
            valid_points = ~np.all(curve == 0, axis=1)
            if not np.any(valid_points):
                continue

            valid_curve = curve[valid_points]

            if len(valid_curve) < 2:
                # Single point - plot as scatter
                ax.scatter(
                    valid_curve[:, 0],
                    valid_curve[:, 1],
                    valid_curve[:, 2],
                    c=[colors[kp_idx]],
                    s=50,
                    alpha=0.8,
                    label=f"KP{self.keypoints[kp_idx]}",
                )
            else:
                # Multiple points - plot as line
                ax.plot(
                    valid_curve[:, 0],
                    valid_curve[:, 1],
                    valid_curve[:, 2],
                    color=colors[kp_idx],
                    linewidth=2,
                    alpha=0.8,
                    label=f"KP{self.keypoints[kp_idx]}",
                )

                # Mark start and end points
                ax.scatter(
                    valid_curve[0, 0],
                    valid_curve[0, 1],
                    valid_curve[0, 2],
                    color=colors[kp_idx],
                    s=100,
                    marker="o",
                    alpha=1.0,
                )
                ax.scatter(
                    valid_curve[-1, 0],
                    valid_curve[-1, 1],
                    valid_curve[-1, 2],
                    color=colors[kp_idx],
                    s=100,
                    marker="s",
                    alpha=1.0,
                )

        # Set labels and title
        label_text = "Positive" if label == 1 else "Negative"
        label_color = "green" if label == 1 else "red"

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"Sample {sample_idx}: {label_text} ({self.subject}/{self.exercise})",
            color=label_color,
            fontweight="bold",
        )

        # Add legend (but limit to avoid clutter)
        if self.n_keypoints <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        # Set equal aspect ratio
        self._set_equal_aspect_3d(ax)

        return ax

    def _set_equal_aspect_3d(self, ax):
        """Set equal aspect ratio for 3D plot."""
        # Get current axis limits
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        # Find the range for each axis
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]

        # Find the maximum range
        max_range = max(x_range, y_range, z_range)

        # Set equal limits centered on the current limits
        x_center = (x_limits[0] + x_limits[1]) / 2
        y_center = (y_limits[0] + y_limits[1]) / 2
        z_center = (z_limits[0] + z_limits[1]) / 2

        ax.set_xlim3d(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim3d(y_center - max_range / 2, y_center + max_range / 2)
        ax.set_zlim3d(z_center - max_range / 2, z_center + max_range / 2)

    def visualize_comparison(self, n_samples: int = 4):
        """Visualize multiple samples in a grid for comparison."""
        if n_samples > len(self.labels):
            n_samples = len(self.labels)

        # Try to get a mix of positive and negative samples
        pos_indices = np.where(self.labels == 1)[0]
        neg_indices = np.where(self.labels == 0)[0]

        selected_indices = []
        n_pos = min(n_samples // 2, len(pos_indices))
        n_neg = min(n_samples - n_pos, len(neg_indices))

        if n_pos > 0:
            selected_indices.extend(np.random.choice(pos_indices, n_pos, replace=False))
        if n_neg > 0:
            selected_indices.extend(np.random.choice(neg_indices, n_neg, replace=False))

        # Fill remaining slots with random samples
        while len(selected_indices) < n_samples:
            remaining = set(range(len(self.labels))) - set(selected_indices)
            if not remaining:
                break
            selected_indices.append(random.choice(list(remaining)))

        # Create subplot grid
        cols = min(2, n_samples)
        rows = (n_samples + cols - 1) // cols

        fig = plt.figure(figsize=(6 * cols, 5 * rows))

        for i, sample_idx in enumerate(selected_indices):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            self.visualize_sample(sample_idx, ax)

        plt.tight_layout()
        return fig

    def interactive_browser(self):
        """Launch an interactive sample browser."""
        current_idx = [0]  # Use list for mutable reference

        # Create figure and axis once
        fig = plt.figure(figsize=(12, 8))

        def update_plot():
            fig.clear()
            ax = fig.add_subplot(111, projection="3d")
            self.visualize_sample(current_idx[0], ax)

            # Add instructions at the bottom
            fig.text(
                0.02,
                0.02,
                f"Sample {current_idx[0]}/{len(self.labels)-1} | Controls: ← → A/D (navigate), R (random), P (pos), N (neg)",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

        def on_key_press(event):
            if event.key == "right" or event.key == "d":
                current_idx[0] = min(current_idx[0] + 1, len(self.labels) - 1)
                update_plot()
            elif event.key == "left" or event.key == "a":
                current_idx[0] = max(current_idx[0] - 1, 0)
                update_plot()
            elif event.key == "r":
                current_idx[0] = random.randint(0, len(self.labels) - 1)
                update_plot()
            elif event.key == "p":
                # Jump to next positive sample
                pos_indices = np.where(self.labels == 1)[0]
                next_pos = pos_indices[pos_indices > current_idx[0]]
                if len(next_pos) > 0:
                    current_idx[0] = next_pos[0]
                else:
                    current_idx[0] = (
                        pos_indices[0] if len(pos_indices) > 0 else current_idx[0]
                    )
                update_plot()
            elif event.key == "n":
                # Jump to next negative sample
                neg_indices = np.where(self.labels == 0)[0]
                next_neg = neg_indices[neg_indices > current_idx[0]]
                if len(next_neg) > 0:
                    current_idx[0] = next_neg[0]
                else:
                    current_idx[0] = (
                        neg_indices[0] if len(neg_indices) > 0 else current_idx[0]
                    )
                update_plot()

        # Connect the key press event
        fig.canvas.mpl_connect("key_press_event", on_key_press)

        # Make sure the plot window has focus for key events
        plt.ion()  # Turn on interactive mode

        # Initial plot
        update_plot()

        print("Interactive browser launched!")
        print("Make sure the plot window has focus, then use:")
        print("  ← → or A/D: Navigate samples")
        print("  R: Random sample")
        print("  P: Next positive sample")
        print("  N: Next negative sample")
        print("  Close window to exit")

        plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument(
        "--dataset-path", required=True, type=str, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Specific sample index to visualize",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Show comparison grid of multiple samples",
    )
    parser.add_argument(
        "--n-samples", type=int, default=4, help="Number of samples for comparison view"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Launch interactive browser"
    )

    args = parser.parse_args()

    try:
        visualizer = DatasetVisualizer(args.dataset_path)

        if args.interactive:
            print("Launching interactive browser...")
            print("Use arrow keys (← →) or A/D to navigate")
            print("Press R for random sample, P for next positive, N for next negative")
            visualizer.interactive_browser()
        elif args.comparison:
            print(f"Showing comparison of {args.n_samples} samples...")
            fig = visualizer.visualize_comparison(args.n_samples)
            plt.show()
        elif args.sample_idx is not None:
            print(f"Visualizing sample {args.sample_idx}...")
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            visualizer.visualize_sample(args.sample_idx, ax)
            plt.show()
        else:
            print(
                "No specific visualization mode selected. Using interactive browser..."
            )
            visualizer.interactive_browser()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
