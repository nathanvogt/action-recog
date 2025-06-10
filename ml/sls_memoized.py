from __future__ import annotations

import math
from typing import List, Tuple, Dict, Sequence

# ────────────────────────────── generic types ──────────────────────────────
Point = Tuple[float, float, float]

# ───────────────────────────── pose metadata ───────────────────────────────
KP_TO_NAME: Dict[int, str] = {
    0: "center_hip",
    1: "right_hip",
    2: "right_knee",
    3: "right_ankle",
    4: "left_hip",
    5: "left_knee",
    6: "left_ankle",
    7: "center_spine",
    8: "neck",
    9: "tip_of_neck",
    10: "head",
    11: "right_shoulder",
    12: "right_elbow",
    13: "right_wrist",
    14: "left_shoulder",
    15: "left_elbow",
    16: "left_wrist",
    17: "right_middle_foot",
    18: "right_tip_foot",
    19: "left_middle_foot",
    20: "left_tip_foot",
    21: "right_hand_1",
    22: "right_hand_2",
    23: "left_hand_1",
    24: "left_hand_2",
}

LEFT_LEG_NO_FEET: List[int] = [4, 5, 6]
RIGHT_LEG_NO_FEET: List[int] = [1, 2, 3]
LEFT_ARM_NO_HAND: List[int] = [14, 15, 16]
RIGHT_ARM_NO_HAND: List[int] = [11, 12, 13]
LEFT_HAND: List[int] = [22, 24]
RIGHT_HAND: List[int] = [21, 23]
LEFT_FOOT: List[int] = [19, 20]
RIGHT_FOOT: List[int] = [17, 18]
BACK: List[int] = [0, 7]

CONNECTIONS: List[Tuple[int, int]] = [
    # spine / torso
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    # hips
    (0, 1),
    (0, 4),
    # right leg
    (1, 2),
    (2, 3),
    # left leg
    (4, 5),
    (5, 6),
    # shoulders
    (8, 11),
    (8, 14),
    # right arm
    (11, 12),
    (12, 13),
    # left arm
    (14, 15),
    (15, 16),
    # right hand
    (13, 21),
    (13, 22),
    # left hand
    (16, 23),
    (16, 24),
    # right foot
    (3, 17),
    (17, 18),
    # left foot
    (6, 19),
    (19, 20),
]


# ───────────────────────────── math helpers ────────────────────────────────
def shortest_distance(p1: Point, p2: Point, p: Point) -> float:
    """Point‐to‐segment distance in 3-D (used by SLS loss)."""
    line_vec = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    point_vec = (p[0] - p1[0], p[1] - p1[1], p[2] - p1[2])

    cross = (
        line_vec[1] * point_vec[2] - line_vec[2] * point_vec[1],
        line_vec[2] * point_vec[0] - line_vec[0] * point_vec[2],
        line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0],
    )

    norm = math.sqrt(line_vec[0] ** 2 + line_vec[1] ** 2 + line_vec[2] ** 2) or 1e-5
    dist = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) / norm

    p1_dist = math.dist(p, p1)
    p2_dist = math.dist(p, p2)
    closer, farther = (p1, p2) if p1_dist <= p2_dist else (p2, p1)
    vec = (farther[0] - closer[0], farther[1] - closer[1], farther[2] - closer[2])
    dot = vec[0] * point_vec[0] + vec[1] * point_vec[1] + vec[2] * point_vec[2]

    if dot < 0:  # projection falls outside the segment
        dist = min(p1_dist, p2_dist)

    return dist


def segmented_least_squares_fixed_segments(
    points: Sequence[Point], num_segments: int
) -> Tuple[float, List[int]]:
    """
    Classic dynamic-programming SLS with a *fixed* number of segments.
    Returns (total_loss, segment_end_indices).
    """
    n = len(points)
    if num_segments >= n:
        return 0.0, []

    # pre-compute error table E[i][j]  (cost of fitting a single segment i…j)
    E = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for i in range(j + 1):
            E[i][j] = sum(
                shortest_distance(points[i], points[j], points[k])
                for k in range(i, j + 1)
            )

    # DP tables
    dp = [[math.inf] * (num_segments + 1) for _ in range(n)]
    backptr = [[0] * (num_segments + 1) for _ in range(n)]

    for j in range(n):
        dp[j][1] = E[0][j]
        backptr[j][1] = 0

    for k in range(2, num_segments + 1):
        for j in range(k - 1, n):
            best_cost, best_i = math.inf, 0
            for i in range(k - 1, j + 1):
                cost = E[i][j] + dp[i - 1][k - 1]
                if cost < best_cost:
                    best_cost, best_i = cost, i
            dp[j][k] = best_cost
            backptr[j][k] = best_i

    # reconstruct indices (segment ends wrt *points* array)
    idx: List[int] = []
    k, j = num_segments - 1, n - 1
    while k > 0:
        idx.append(j)
        j = backptr[j][k] - 1
        k -= 1
    idx.append(0)
    idx.reverse()
    return dp[n - 1][num_segments], idx


DEFAULT_C = 9
DEFAULT_M = 4


# ────────────────────────── public “functional” helpers ────────────────────
def sls_points(points: Sequence[Point], c: int = DEFAULT_C) -> List[Point]:
    """Plain SLS (no streaming, no memo)."""
    _, idx = segmented_least_squares_fixed_segments(points, c)
    return [points[i] for i in idx]


# ───────────────────────── streaming / memoised implementation ─────────────
class SlsMemoized:
    """
    Streaming Segmented-Least-Squares with memoisation – 1-to-1 port of the
    TypeScript `SlsMemoized` class.
    """

    # ───────── constructor / config ─────────
    def __init__(self, c: int = DEFAULT_C, m: int = DEFAULT_M) -> None:
        self._c = c
        self._m = m

        self._frame_count: int = 0  # == len(self._poses)
        self._poses: List[List[Point]] = []  # global frame buffer
        self._mem_indices: List[List[int]] = []  # per-kp memo index lists
        self._kp_list: List[int] = []  # active keypoints

    # ─────────────── pure, stateless helpers ───────────────
    @staticmethod
    def _process_points(points: Sequence[Point], c: int) -> List[Point]:
        _, idx = segmented_least_squares_fixed_segments(points, c)
        return [points[i] for i in idx]

    @staticmethod
    def _process_points_with_cost(
        points: Sequence[Point], c: int
    ) -> Tuple[List[Point], float]:
        loss, idx = segmented_least_squares_fixed_segments(points, c)
        return [points[i] for i in idx], loss

    # ---------- memoised helper (core of streaming variant) ----------
    def _process_points_with_memo(
        self, points_window: Sequence[Point], memo: List[int]
    ) -> Tuple[List[Point], List[int], float]:
        input_points = [points_window[i] for i in memo]
        loss, seg_idx = segmented_least_squares_fixed_segments(input_points, self._c)
        pose_window_idx = [memo[i] for i in seg_idx]

        # build *evenly spaced* memo indices between successive segment endpoints
        even: List[int] = [pose_window_idx[0]]
        for k in range(len(pose_window_idx) - 1):
            start, end = pose_window_idx[k], pose_window_idx[k + 1]
            gap = end - start
            if gap <= self._m:
                even.extend(range(start, end))  # inclusive of start, exclusive end
            else:
                step = gap / self._m
                even.extend(math.floor(start + 1 + j * step) for j in range(self._m))

        uniq = sorted(set(even))
        seg_points = [points_window[i] for i in pose_window_idx]
        return seg_points, uniq, loss

    # ─────────────── public streaming API ───────────────
    def process_poses(
        self,
        new_poses: Sequence[Sequence[Point]],
        keypoints: Sequence[int] | None = None,
    ) -> Tuple[List[List[Point]], float]:
        """
        Incrementally ingest *new* frames (each frame = full list of keypoints).

        Returns:
          - `lss_curves[i]` – updated SLS curve for the *i-th* requested keypoint
          - `total_loss`   – sum of approximation errors over this batch
        """
        if keypoints is None:
            keypoints = (
                LEFT_LEG_NO_FEET
                + RIGHT_LEG_NO_FEET
                + LEFT_ARM_NO_HAND
                + RIGHT_ARM_NO_HAND
                + BACK
            )

        # fast path: nothing to do
        if not new_poses:
            return [], 0.0

        keypoints = list(keypoints)
        self._kp_list = keypoints

        # (re-)initialise state on first call or kp-set change
        if not self._mem_indices or len(self._mem_indices) != len(keypoints):
            self._mem_indices = [[] for _ in keypoints]
            self._poses = []
            self._frame_count = 0

        # append new frames to the global buffer
        start_frame = self._frame_count
        self._poses.extend(list(map(list, new_poses)))
        self._frame_count += len(new_poses)

        # update memo index arrays with indices of *new* frames just appended
        for kp_memo in self._mem_indices:
            kp_memo.extend(range(start_frame, self._frame_count))

        # run SLS+memo for each requested keypoint
        lss_curves: List[List[Point]] = [[] for _ in keypoints]
        total_loss = 0.0

        for idx, kp in enumerate(keypoints):
            # build full trajectory of that kp so far
            curve = [frame[kp] for frame in self._poses]
            memo = self._mem_indices[idx]

            if len(curve) <= self._c:  # warm-up phase – nothing to fit yet
                lss_curves[idx] = [curve[0]] if curve else []
                continue

            seg_points, new_memo, loss = self._process_points_with_memo(curve, memo)
            self._mem_indices[idx] = new_memo
            lss_curves[idx] = seg_points
            total_loss += loss

        return lss_curves, total_loss

    # ─────────────── housekeeping / introspection ───────────────
    def reset(self) -> None:
        """Forget *all* state (use between independent sequences)."""
        self._frame_count = 0
        self._poses.clear()
        self._mem_indices.clear()
        self._kp_list.clear()

    def get_memo_state(self) -> Dict[str, object]:
        """Return *copies* of internal mutable state for debugging."""
        return {
            "current_frame": self._frame_count,
            "mem_indices": [memo.copy() for memo in self._mem_indices],
        }

    def get_mem_points(self) -> List[List[Point]]:
        """Return the actual memoised points (useful for visual debugging)."""
        if not self._kp_list:
            return []

        return [
            [self._poses[frame_idx][self._kp_list[kp_idx]] for frame_idx in memo]
            for kp_idx, memo in enumerate(self._mem_indices)
        ]

    def get_config(self) -> Dict[str, int]:
        return {"c": self._c, "m": self._m}
