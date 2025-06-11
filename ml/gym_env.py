import gymnasium as gym
import numpy as np
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


class RepDetectionEnv(gym.Env):
    def __init__(
        self,
        subject,
        exercise,
        dataset_root="train",
        c=DEFAULT_C,
        m=DEFAULT_M,
        keypoints=(
            LEFT_LEG_NO_FEET
            + RIGHT_LEG_NO_FEET
            + LEFT_ARM_NO_HAND
            + RIGHT_ARM_NO_HAND
            + BACK
        ),
        tol=10,
    ):
        self.dataset = TrainDatasetLocal(dataset_root)
        self.poses = self.dataset.get_pose_array(subject, exercise)
        self.rep_idx = self.dataset.get_rep_timings(subject, exercise) or []
        self.tol = tol
        self.keypoints = keypoints
        self.sls = SlsMemoized(c=c, m=m)
        self.cur_idx = 0
        self.used_rep_boundaries = (
            set()
        )  # Track which rep boundaries have been correctly identified
        self.action_space = gym.spaces.Discrete(2)  # 0=no rep, 1=rep
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.keypoints) * c * 3,),
            dtype=np.float32,
        )

    def _get_obs(self):
        curves = self.sls.sls_curves
        if not curves:
            return np.zeros(self.observation_space.shape)

        out = []
        for curve in curves:
            pts = np.array(curve, dtype=np.float32)
            out.append(pts)
        return np.concatenate(out).ravel()

    def reset(self, **kwargs):
        self.sls.reset()
        self.cur_idx = 0
        self.used_rep_boundaries = set()
        return self._get_obs(), {}

    def step(self, action):
        reward = self._compute_reward(action)
        if action == 1:
            self.sls.reset()
        self.cur_idx += 1
        terminated = self.cur_idx >= len(self.poses)
        truncated = False  # We don't truncate episodes in this environment
        if not terminated:
            self.sls.process_poses([self.poses[self.cur_idx]])
        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape)
        )
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, action):
        if len(self.rep_idx) <= 1:
            near_boundary_idx = None
            near = False
        else:
            # Find the closest rep boundary and its index
            distances = [
                (abs(self.cur_idx - r), i) for i, r in enumerate(self.rep_idx[1:])
            ]
            min_distance, closest_boundary_idx = min(distances)
            near = min_distance <= self.tol
            # Adjust index to account for skipping first boundary
            near_boundary_idx = closest_boundary_idx + 1 if near else None

        if action == 1 and near:
            # Predicting a rep near a boundary
            if near_boundary_idx in self.used_rep_boundaries:
                # This boundary was already correctly identified - penalize
                return -1.0
            else:
                # New boundary correctly identified - reward and mark as used
                self.used_rep_boundaries.add(near_boundary_idx)
                return 10.0
        elif action == 0 and not near:
            # Correctly predicting no rep when not near a boundary
            return 0.05
        elif action == 0 and near:
            # Predicting no rep when near a boundary
            if near_boundary_idx in self.used_rep_boundaries:
                # Near an already-used boundary - neutral
                return 0.0
            else:
                # Near an unused boundary - false negative, penalize
                return -1.0
        else:
            # Action == 1 and not near - false positive
            return -4.0
