import os
import numpy as np
import matplotlib.pyplot as plt
from sls_arcive import create_sls_with_memo
import json
from multiprocessing import Pool, cpu_count
from data import (
    LEFT_LEG_NO_FEET,
    RIGHT_LEG_NO_FEET,
    LEFT_ARM_NO_HAND,
    RIGHT_ARM_NO_HAND,
    BACK,
    KP_TO_NAME,
)


def process_keypoint(args):
    kp, curve, c = args
    start = c + 1
    sls_points = create_sls_with_memo(m=3)
    mem_indices = np.arange(0, start)
    points = []
    for i in range(start + 1, len(curve)):
        window = curve[:i]
        mem_indices = np.append(mem_indices, i - 1)
        points_, mem_indices_, _ = sls_points(window, mem_indices, c)
        points = points_
        mem_indices = mem_indices_
    return kp, points


def main():
    path = os.path.join("train", "s03", "joints3d_25", "squat.json")
    with open(path) as f:
        data = json.load(f)
    poses = np.array(data["joints3d_25"])
    poses = poses[380:528]

    keypoints = (
        LEFT_LEG_NO_FEET
        + RIGHT_LEG_NO_FEET
        + LEFT_ARM_NO_HAND
        + RIGHT_ARM_NO_HAND
        + BACK
    )
    curves = np.swapaxes(poses, 0, 1)

    c = 10

    args = [(kp, curves[kp], c) for kp in keypoints]

    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores")

    with Pool(num_cores) as pool:
        results = pool.map(process_keypoint, args)

    lss_curves = np.zeros((25, c, 3))
    for kp, result in results:
        lss_curves[kp] = result

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for kp in keypoints:
        x = lss_curves[kp, :, 0]
        y = lss_curves[kp, :, 1]
        z = lss_curves[kp, :, 2]
        ax.plot(x, y, z, label=f"{KP_TO_NAME[kp]}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectories of Keypoints")

    ax.legend()

    ax.view_init(elev=20, azim=45)

    plt.show()


if __name__ == "__main__":
    main()
