# compare_sls.py
import json
import subprocess
import sys
import time
import os
import numpy as np
from typing import Dict, List, Any


def run_python_test(subject: str, exercise: str) -> Dict[str, Any]:
    """Run Python SLS test and return results."""
    cmd = ["python", "test_sls_python.py", subject, exercise]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Python test failed: {result.stderr}")
        sys.exit(1)

    return json.loads(result.stdout)


def run_typescript_test(subject: str, exercise: str) -> Dict[str, Any]:
    """Run TypeScript SLS test and return results."""
    cmd = [
        "node",
        "--import",
        "tsx",
        "src/scripts/test_sls_typescript.ts",
        subject,
        exercise,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"TypeScript test failed: {result.stderr}")
        sys.exit(1)

    return json.loads(result.stdout)


def compare_points(
    py_points: List[List[float]], ts_points: List[List[float]], tolerance: float = 1e-6
) -> tuple[bool, float]:
    """Compare two sets of 3D points and return (is_equal, max_difference)."""
    if len(py_points) != len(ts_points):
        return False, float("inf")

    max_diff = 0.0
    for py_pt, ts_pt in zip(py_points, ts_points):
        if len(py_pt) != len(ts_pt):
            return False, float("inf")

        for py_coord, ts_coord in zip(py_pt, ts_pt):
            diff = abs(py_coord - ts_coord)
            max_diff = max(max_diff, diff)

    return max_diff <= tolerance, max_diff


def compare_curves(
    py_curves: List[List[List[float]]],
    ts_curves: List[List[List[float]]],
    tolerance: float = 1e-6,
) -> tuple[bool, Dict[str, Any]]:
    """Compare LSS curves from both implementations."""
    if len(py_curves) != len(ts_curves):
        return False, {
            "matches": False,
            "error": "Different number of curves",
            "py": len(py_curves),
            "ts": len(ts_curves),
            "max_difference": float("inf"),
            "curve_details": [],
        }

    comparison = {"matches": True, "max_difference": 0.0, "curve_details": []}

    for i, (py_curve, ts_curve) in enumerate(zip(py_curves, ts_curves)):
        curve_match, curve_diff = compare_points(py_curve, ts_curve, tolerance)

        comparison["curve_details"].append(
            {
                "curve_index": i,
                "matches": curve_match,
                "max_difference": curve_diff,
                "py_length": len(py_curve),
                "ts_length": len(ts_curve),
            }
        )

        if not curve_match:
            comparison["matches"] = False

        comparison["max_difference"] = max(comparison["max_difference"], curve_diff)

    return comparison["matches"], comparison


def compare_frame_results(
    py_frame: Dict[str, Any], ts_frame: Dict[str, Any], tolerance: float = 1e-6
) -> Dict[str, Any]:
    """Compare results from a single frame."""

    # Compare loss
    loss_diff = abs(py_frame["total_loss"] - ts_frame["total_loss"])
    loss_match = loss_diff <= tolerance

    # Compare curves
    curves_match, curves_comparison = compare_curves(
        py_frame["lss_curves"], ts_frame["lss_curves"], tolerance
    )

    # Compare memo state
    py_memo = py_frame["memo_state"]["mem_indices"]
    ts_memo = ts_frame["memo_state"]["memIndices"]

    memo_match = True
    memo_details = []

    if len(py_memo) != len(ts_memo):
        memo_match = False
    else:
        for i, (py_indices, ts_indices) in enumerate(zip(py_memo, ts_memo)):
            indices_match = py_indices == ts_indices
            memo_details.append(
                {
                    "keypoint_index": i,
                    "matches": indices_match,
                    "py_indices": py_indices,
                    "ts_indices": ts_indices,
                }
            )
            if not indices_match:
                memo_match = False

    return {
        "frame": py_frame["frame"],
        "loss": {
            "matches": loss_match,
            "difference": loss_diff,
            "py_loss": py_frame["total_loss"],
            "ts_loss": ts_frame["total_loss"],
        },
        "curves": curves_comparison,
        "memo": {"matches": memo_match, "details": memo_details},
        "overall_match": loss_match and curves_match and memo_match,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_sls.py <subject> <exercise>")
        sys.exit(1)

    subject, exercise = sys.argv[1], sys.argv[2]

    print(f"🔄 Testing SLS implementations for {subject}/{exercise}...")
    print()

    # Create temp directory in project folder for saving results
    temp_dir = os.path.join("temp", f"sls_comparison_{subject}_{exercise}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"📁 Saving results to: {temp_dir}")
    print()

    # Run both implementations with timing
    print("📊 Running TypeScript implementation...")
    ts_start_time = time.time()
    ts_results = run_typescript_test(subject, exercise)
    ts_end_time = time.time()
    ts_duration = ts_end_time - ts_start_time

    print("📊 Running Python implementation...")
    py_start_time = time.time()
    py_results = run_python_test(subject, exercise)
    py_end_time = time.time()
    py_duration = py_end_time - py_start_time

    # Save results to temp files
    ts_file = os.path.join(temp_dir, "typescript_results.json")
    py_file = os.path.join(temp_dir, "python_results.json")

    with open(ts_file, "w") as f:
        json.dump(ts_results, f, indent=2)

    with open(py_file, "w") as f:
        json.dump(py_results, f, indent=2)

    print(f"💾 Saved TypeScript results to: {ts_file}")
    print(f"💾 Saved Python results to: {py_file}")

    # Basic sanity checks
    if len(py_results["results"]) != len(ts_results["results"]):
        print("❌ ERROR: Different number of frames processed!")
        sys.exit(1)

    # Compare frame by frame
    print(f"\n🔍 Comparing {len(py_results['results'])} frames...")

    all_match = True
    tolerance = 1e-6

    for i, (py_frame, ts_frame) in enumerate(
        zip(py_results["results"], ts_results["results"])
    ):
        comparison = compare_frame_results(py_frame, ts_frame, tolerance)

        if not comparison["overall_match"]:
            all_match = False
            print(f"❌ Frame {i}: MISMATCH")
            if not comparison["loss"]["matches"]:
                print(f"   Loss difference: {comparison['loss']['difference']:.8f}")
            if not comparison["curves"]["matches"]:
                print(
                    f"   Curves max difference: {comparison['curves']['max_difference']:.8f}"
                )
            if not comparison["memo"]["matches"]:
                print(f"   Memo indices differ")
        else:
            print(
                f"✅ Frame {i}: MATCH (loss diff: {comparison['loss']['difference']:.8f})"
            )

    print(f"\n📋 Summary:")
    print(f"   Total frames: {len(py_results['results'])}")
    print(f"   Parameters: c={py_results['c']}, m={py_results['m']}")
    print(f"   Keypoints: {len(py_results['keypoints'])}")
    print(f"   Tolerance: {tolerance}")
    print(f"\n⏱️  Performance:")
    print(f"   TypeScript: {ts_duration:.3f}s")
    print(f"   Python: {py_duration:.3f}s")
    print(f"   Ratio (TS/Python): {ts_duration/py_duration:.2f}x")

    print(f"\n📁 Results saved to:")
    print(f"   Directory: {temp_dir}")
    print(f"   TypeScript: typescript_results.json")
    print(f"   Python: python_results.json")

    if all_match:
        print("\n✅ All frames match! Implementations are identical.")
    else:
        print("\n❌ Some frames don't match. Check implementation differences.")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
