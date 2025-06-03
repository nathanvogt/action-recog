import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import json
import time
import cProfile
import pstats
from sls_arcive import create_sls_with_memo, process_poses
from data import (
    LEFT_LEG_NO_FEET,
    RIGHT_LEG_NO_FEET,
    LEFT_ARM_NO_HAND,
    RIGHT_ARM_NO_HAND,
    BACK,
)


def main():
    split = "train"
    subject = "s03"
    exercise_type = "squat"
    poses_path = os.path.join(split, subject, "joints3d_25", f"{exercise_type}.json")
    with open(poses_path) as f:
        poses = np.array(json.load(f)["joints3d_25"])
    reps_path = os.path.join(split, subject, "rep_ann.json")
    with open(reps_path) as f:
        reps = json.load(f)[exercise_type]
        reps = [(reps[i], reps[i + 1]) for i in range(len(reps) - 1)]

    keypoints = (
        BACK
        + LEFT_ARM_NO_HAND
        + RIGHT_ARM_NO_HAND
        + LEFT_LEG_NO_FEET
        + RIGHT_LEG_NO_FEET
    )
    c = 6
    m = 3

    rep_poses = [poses[start:end, :, :] for start, end in reps]
    for rep in rep_poses[:1]:
        start_time = time.time()
        processed = process_poses(rep, c, m, keypoints)
        end_time = time.time()
        print(f"Processing time: {end_time - start_time}")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats()
