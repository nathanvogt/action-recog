import json
import sys
import numpy as np
from ml.sls_memoized import SlsMemoized
from ml.train_dataset import TrainDatasetLocal

# Default keypoints (matching TypeScript)
LEFT_LEG_NO_FEET = [4, 5, 6]
RIGHT_LEG_NO_FEET = [1, 2, 3]
LEFT_ARM_NO_HAND = [14, 15, 16]
RIGHT_ARM_NO_HAND = [11, 12, 13]
BACK = [0, 7]

DEFAULT_KEYPOINTS = (
    LEFT_LEG_NO_FEET + RIGHT_LEG_NO_FEET + LEFT_ARM_NO_HAND + RIGHT_ARM_NO_HAND + BACK
)


def test_sls_python(
    subject: str, exercise: str, c: int = 9, m: int = 4, max_frames: int = 100
):
    """Test Python SLS implementation and output results to JSON."""

    # Load dataset
    dataset = TrainDatasetLocal("train")
    poses = dataset.get_pose_array(subject, exercise)

    # Limit frames for testing
    poses = poses[:max_frames]

    # Initialize SLS
    sls = SlsMemoized(c=c, m=m)

    results = []

    # Process poses one by one
    for frame_idx, pose in enumerate(poses):
        # Process single pose
        lss_curves, total_loss = sls.process_poses([pose], DEFAULT_KEYPOINTS)

        # Get memo state
        memo_state = sls.get_memo_state()

        # Convert numpy arrays to lists for JSON serialization
        lss_curves_serializable = []
        for curve in lss_curves:
            curve_list = [
                point.tolist() if hasattr(point, "tolist") else list(point)
                for point in curve
            ]
            lss_curves_serializable.append(curve_list)

        frame_result = {
            "frame": frame_idx,
            "lss_curves": lss_curves_serializable,
            "total_loss": float(total_loss),
            "memo_state": memo_state,
            "num_keypoints": len(DEFAULT_KEYPOINTS),
        }

        results.append(frame_result)

        # Print progress
        print(
            f"Python - Frame {frame_idx}: loss={total_loss:.6f}, curves_count={len(lss_curves)}",
            file=sys.stderr,
        )

    # Output results to stdout as JSON
    output = {
        "implementation": "python",
        "subject": subject,
        "exercise": exercise,
        "c": c,
        "m": m,
        "keypoints": DEFAULT_KEYPOINTS,
        "total_frames": len(poses),
        "results": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_sls_python.py <subject> <exercise>", file=sys.stderr)
        sys.exit(1)

    subject = sys.argv[1]
    exercise = sys.argv[2]
    test_sls_python(subject, exercise)
