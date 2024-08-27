import os
import numpy as np
import matplotlib.pyplot as plt
from sls_arcive import create_sls_with_memo
import json
import time


def process_keypoint(kp, curve, c):
    start = c + 1
    sls_points = create_sls_with_memo(m=3)
    mem_indices = np.arange(0, start)
    points = []
    processing_times = []

    for i in range(start + 1, len(curve)):
        start_time = time.time()
        window = curve[:i]
        mem_indices = np.append(mem_indices, i - 1)
        points_, mem_indices_, _ = sls_points(window, mem_indices, c)
        points = points_
        mem_indices = mem_indices_
        end_time = time.time()
        processing_times.append(end_time - start_time)

    return points, processing_times


def main():
    path = os.path.join("train", "s03", "joints3d_25", "squat.json")
    with open(path) as f:
        data = json.load(f)
    poses = np.array(data["joints3d_25"])
    poses = poses

    # Use only a single point on the body (e.g., the first point of BACK)
    keypoint = 0  # You can change this to any other keypoint index if desired
    curve = np.swapaxes(poses, 0, 1)[keypoint]

    c = 10

    lss_curve, processing_times = process_keypoint(keypoint, curve, c)

    # Plot the processing times
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(processing_times)), processing_times)
    plt.xlabel("Number of Points Processed")
    plt.ylabel("Processing Time (seconds)")
    plt.title("Processing Time per New Keypoint")
    plt.grid(True)
    plt.show()

    # Plot the 3D trajectory of the single keypoint
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x = lss_curve[:, 0]
    y = lss_curve[:, 1]
    z = lss_curve[:, 2]
    ax.plot(x, y, z, label=f"Keypoint {keypoint}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory of Single Keypoint")
    ax.legend()
    ax.view_init(elev=20, azim=45)
    plt.show()


if __name__ == "__main__":
    main()
