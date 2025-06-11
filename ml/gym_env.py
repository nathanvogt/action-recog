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
        self.start_idx = 0
        self.cur_idx = 0
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
            pts = np.array(curve[-self.sls._c :], dtype=np.float32)
            pad = np.zeros((self.sls._c - len(pts), 3), dtype=np.float32)
            out.append(np.concatenate([pad, pts], axis=0))
        return np.concatenate(out).ravel()

    def reset(self, **kwargs):
        self.sls.reset()
        self.start_idx = self.cur_idx = 0
        return self._get_obs(), {}

    def step(self, action):
        reward = self._compute_reward(action)
        if action == 1:  # flush window on rep
            self.start_idx = self.cur_idx
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
        # positive if action matches presence/absence of rep boundary
        near = min(abs(self.cur_idx - r) for r in self.rep_idx) <= self.tol
        if action == 1 and near:
            return 1.0
        if action == 0 and not near:
            return 0.1
        return -1.0
