import os
import json
import numpy as np

path = os.path.join("train", "s03", "joints3d_25", "band_pull_apart.json")

with open(path) as f:
    data = json.load(f)

poses = np.array(data["joints3d_25"])
print(poses.shape)
