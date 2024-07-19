import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import json


path = os.path.join("train", "s03", "joints3d_25", "squat.json")
with open(path) as f:
    data = json.load(f)
poses = np.array(data["joints3d_25"])


connections = [
    (0, 1),
    (1, 2),
    (2, 3),  # right ankle
    (0, 4),
    (4, 5),
    (5, 6),  # left ankle
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),  # tip of head
    (8, 11),
    (11, 12),
    (12, 13),  # right wrist
    (8, 14),
    (14, 15),
    (15, 16),  # left wrist
    (3, 17),
    (17, 18),
    (6, 19),
    (19, 20),
    (13, 21),
    (13, 22),
    (16, 23),
    (16, 24),
]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

lines = [ax.plot([], [], [])[0] for _ in connections]
scatter = ax.scatter([], [], [])

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


def update(frame):
    global current_frame
    current_frame = frame
    for line, connection in zip(lines, connections):
        start, end = connection
        line.set_data(
            [poses[frame, start, 0], poses[frame, end, 0]],
            [poses[frame, start, 1], poses[frame, end, 1]],
        )
        line.set_3d_properties([poses[frame, start, 2], poses[frame, end, 2]])

    scatter._offsets3d = (poses[frame, :, 0], poses[frame, :, 1], poses[frame, :, 2])

    ax.set_title(f"Frame {frame}")

    return lines + [scatter]


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
