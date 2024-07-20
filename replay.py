import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import json
from sls_arcive import create_sls_with_memo
from data import (
    LEFT_LEG_NO_FEET,
    RIGHT_LEG_NO_FEET,
    LEFT_ARM_NO_HAND,
    RIGHT_ARM_NO_HAND,
    BACK,
    KP_TO_NAME,
)

path = os.path.join("train", "s03", "joints3d_25", "squat.json")
with open(path) as f:
    data = json.load(f)
poses = np.array(data["joints3d_25"])

connections = [
    (0, 1),
    (1, 2),
    (2, 3),  # right leg
    (0, 4),
    (4, 5),
    (5, 6),  # left leg
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),  # spine and head
    (8, 11),
    (11, 12),
    (12, 13),  # right arm
    (8, 14),
    (14, 15),
    (15, 16),  # left arm
]

keypoints = (
    LEFT_LEG_NO_FEET + RIGHT_LEG_NO_FEET + LEFT_ARM_NO_HAND + RIGHT_ARM_NO_HAND + BACK
)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

lines = [ax.plot([], [], [])[0] for _ in connections]
scatter = ax.scatter([], [], [])
lss_lines = [ax.plot([], [], [], linestyle="--", alpha=0.5)[0] for _ in keypoints]

max_range = (
    np.array(
        [
            poses[:, :, 0].max() - poses[:, :, 0].min(),
            poses[:, :, 1].max() - poses[:, :, 1].min(),
            poses[:, :, 2].max() - poses[:, :, 2].min(),
        ]
    ).max()
    / 2.0
)
mid_x = (poses[:, :, 0].max() + poses[:, :, 0].min()) * 0.5
mid_y = (poses[:, :, 1].max() + poses[:, :, 1].min()) * 0.5
mid_z = (poses[:, :, 2].max() + poses[:, :, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Global variables for animation control
paused = False
current_frame = 0


curves = np.swapaxes(poses, 0, 1)

# Initialize LSS curves
c = 8
sls_points = [create_sls_with_memo(m=3) for _ in keypoints]
lss_curves = [[] for _ in keypoints]
mem_indices = [np.arange(c, dtype=int) for _ in keypoints]


def update_lss_curves(frame):
    global lss_curves, mem_indices
    for i, kp in enumerate(keypoints):
        curve = curves[kp]
        if frame < c:
            lss_curves[i] = curve[0:1]
        else:
            mem_indices[i] = np.append(mem_indices[i], frame)
            points_, mem_indices[i], _ = sls_points[i](curve, mem_indices[i], c)
            lss_curves[i] = points_


def update(frame):
    global current_frame
    current_frame = frame

    # Update pose lines and scatter
    for line, connection in zip(lines, connections):
        start, end = connection
        line.set_data(
            [poses[frame, start, 0], poses[frame, end, 0]],
            [poses[frame, start, 1], poses[frame, end, 1]],
        )
        line.set_3d_properties([poses[frame, start, 2], poses[frame, end, 2]])

    scatter._offsets3d = (poses[frame, :, 0], poses[frame, :, 1], poses[frame, :, 2])

    # Update LSS curves
    update_lss_curves(frame)
    for i, lss_line in enumerate(lss_lines):
        lss_curve = lss_curves[i]
        if len(lss_curve) > 0:
            lss_line.set_data(lss_curve[:, 0], lss_curve[:, 1])
            lss_line.set_3d_properties(lss_curve[:, 2])

    ax.set_title(f"Frame {frame}")

    return lines + [scatter] + lss_lines


def on_key(event):
    global paused, current_frame
    if event.key == " ":
        paused = not paused
        if paused:
            anim.event_source.stop()
        else:
            anim.event_source.start()
    elif event.key == "right" and paused:
        current_frame = (current_frame + 1) % poses.shape[0]
        update(current_frame)
        fig.canvas.draw()
    elif event.key == "left" and paused:
        current_frame = (current_frame - 1) % poses.shape[0]
        update(current_frame)
        fig.canvas.draw()


fig.canvas.mpl_connect("key_press_event", on_key)

anim = FuncAnimation(
    fig, update, frames=poses.shape[0], interval=33.33, blit=False, repeat=True
)

plt.show()
