import numpy as np
from typing import Callable, Tuple

SVM_INCLUDED_KPS = np.arange(25)


def segmented_least_squares_fixed_segments(
    points: np.ndarray, num_segments: int
) -> Tuple[float, np.ndarray]:
    n = len(points)
    if num_segments >= n:
        return 0, np.array([])

    E = np.zeros((n, n))
    dp = np.full((n, num_segments + 1), np.inf)
    result = np.zeros((n, num_segments + 1), dtype=int)

    for j in range(n):
        for i in range(j + 1):
            E[i, j] = np.sum(
                [
                    shortest_distance(points[i], points[j], point)
                    for point in points[i : j + 1]
                ]
            )

    dp[:, 1] = E[0, :]
    dp[1:, num_segments] = np.inf
    result[:, 1] = 0

    for k in range(2, num_segments + 1):
        for j in range(k - 1, n):
            costs = E[k - 1 : j + 1, j] + dp[k - 2 : j, k - 1]
            min_idx = np.argmin(costs)
            dp[j, k] = costs[min_idx]
            result[j, k] = min_idx + k - 1

    indices = []
    k, idx = num_segments - 1, n - 1
    while k > 0:
        indices.append(idx)
        idx = result[idx, k] - 1
        k -= 1
    indices.append(0)
    return dp[n - 1, num_segments], np.array(indices[::-1])


def shortest_distance(
    line_point1: np.ndarray, line_point2: np.ndarray, point: np.ndarray
) -> float:
    line_vector = line_point2 - line_point1
    point_vector = point - line_point1

    cross_product = np.cross(line_vector, point_vector)
    EPSILON = 0.00001  # avoid divide by zero
    distance = np.linalg.norm(cross_product) / (np.linalg.norm(line_vector) + EPSILON)

    p1_distance = np.linalg.norm(point - line_point1)
    p2_distance = np.linalg.norm(point - line_point2)
    closer, farther = (
        (line_point1, line_point2)
        if p1_distance <= p2_distance
        else (line_point2, line_point1)
    )
    vec = farther - closer
    dot = np.dot(vec, point_vector)

    if dot < 0:
        distance = min(p1_distance, p2_distance)

    scale = 1
    distance *= scale  # scale up to avoid floating point errors

    return distance


def segmented_least_squares(points: np.ndarray, c: float) -> Tuple[float, np.ndarray]:
    n = len(points)
    E = np.zeros((n, n))
    opt = np.full(n, np.inf)
    result = np.zeros(n, dtype=int)

    for j in range(n):
        for i in range(j + 1):
            E[i, j] = np.sum(
                [
                    shortest_distance(points[i], points[j], point)
                    for point in points[i : j + 1]
                ]
            )

    opt[0] = 0

    for j in range(1, n):
        costs = E[:j, j] + c + opt[:j]
        min_idx = np.argmin(costs)
        opt[j] = costs[min_idx]
        result[j] = min_idx

    indices = []
    k = n - 1
    while k > 0:
        indices.append(k)
        k = result[k] - 1

    return opt[n - 1], np.array(indices[::-1])


DEFAULT_C = 9
DEFAULT_M = 4


def create_sls_points_memo(
    m: int = DEFAULT_M,
) -> Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray]]:
    M = m
    pose_window = np.array([])
    mem_indices = np.array([], dtype=int)

    def sls_points_test(
        points: np.ndarray, c: int = DEFAULT_C
    ) -> Tuple[np.ndarray, np.ndarray]:
        nonlocal pose_window, mem_indices
        if len(points) < c + 1:
            return np.array([]), np.array([], dtype=int)

        new_indices = np.arange(len(pose_window), len(points))
        pose_window = points
        mem_indices = np.concatenate((mem_indices, new_indices))
        input_points = points[mem_indices]
        loss, indices = segmented_least_squares_fixed_segments(input_points, c)
        pose_window_indices = mem_indices[indices]

        evenly_spaced_indices = []
        for i in range(len(pose_window_indices) - 1):
            start, end = pose_window_indices[i], pose_window_indices[i + 1]
            diff = end - start
            if diff < M:
                evenly_spaced_indices.extend(range(start, end))
            else:
                step = diff / M
                evenly_spaced_indices.extend(int(start + j * step) for j in range(M))

        mem_indices = np.unique(evenly_spaced_indices)
        output_points = points[pose_window_indices]

        return output_points, mem_indices

    return sls_points_test


def create_sls_points_with_losses_memo(
    m: int = DEFAULT_M,
) -> Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray, float]]:
    M = m
    pose_window = np.array([])
    mem_indices = np.array([], dtype=int)

    def sls_points_test(
        points: np.ndarray, c: int = DEFAULT_C
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        nonlocal pose_window, mem_indices
        if len(points) < c + 1:
            return np.array([]), np.array([], dtype=int), 0

        new_indices = np.arange(len(pose_window), len(points))
        pose_window = points
        mem_indices = np.concatenate((mem_indices, new_indices))
        input_points = points[mem_indices]
        loss, indices = segmented_least_squares_fixed_segments(input_points, c)
        pose_window_indices = mem_indices[indices]

        evenly_spaced_indices = []
        for i in range(len(pose_window_indices) - 1):
            start, end = pose_window_indices[i], pose_window_indices[i + 1]
            diff = end - start
            if diff < M:
                evenly_spaced_indices.extend(range(start, end))
            else:
                step = diff / M
                evenly_spaced_indices.extend(int(start + j * step) for j in range(M))

        mem_indices = np.unique(evenly_spaced_indices)
        output_points = points[pose_window_indices]

        return output_points, mem_indices, loss

    return sls_points_test


def sls_points(points: np.ndarray, c: int = DEFAULT_C) -> np.ndarray:
    number_of_segments = c
    loss, indices = segmented_least_squares_fixed_segments(points, number_of_segments)
    return points[indices]


def sls_points_with_loss(
    points: np.ndarray, c: int = DEFAULT_C
) -> Tuple[np.ndarray, float]:
    loss, indices = segmented_least_squares_fixed_segments(points, c)
    return points[indices], loss


def create_approximate_all_curves(
    m: int = DEFAULT_M, c: int = DEFAULT_C
) -> Callable[[np.ndarray], np.ndarray]:
    compute_sls_points = [create_sls_points_memo(m) for _ in range(16)]
    c_ = c

    def approximate_all_curves(curves: np.ndarray, c: int = c_) -> np.ndarray:
        return np.array(
            [
                (
                    compute_sls_points[kp](curve, c)
                    if kp in SVM_INCLUDED_KPS
                    else (np.array([]), np.array([], dtype=int))
                )
                for kp, curve in enumerate(curves)
            ]
        )

    return approximate_all_curves


def create_approximate_all_curves_with_losses(
    m: int = DEFAULT_M, c: int = DEFAULT_C
) -> Callable[[np.ndarray], np.ndarray]:
    compute_sls_points = [create_sls_points_with_losses_memo(m) for _ in range(16)]
    c_ = c

    def approximate_all_curves(curves: np.ndarray, c: int = c_) -> np.ndarray:
        return np.array(
            [
                (
                    compute_sls_points[kp](curve, c)
                    if kp in SVM_INCLUDED_KPS
                    else (np.array([]), np.array([], dtype=int), 0)
                )
                for kp, curve in enumerate(curves)
            ]
        )

    return approximate_all_curves


def create_sls_with_memo(
    m: int = DEFAULT_M,
) -> Callable[[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray, float]]:
    M = m

    def sls_points_test(
        points_window: np.ndarray, mem_indices: np.ndarray, c: int = DEFAULT_C
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        input_points = points_window[mem_indices]
        loss, indices = segmented_least_squares_fixed_segments(input_points, c)
        pose_window_indices = mem_indices[indices]

        evenly_spaced_indices = [pose_window_indices[0]]
        for i in range(len(pose_window_indices) - 1):
            start, end = pose_window_indices[i], pose_window_indices[i + 1]
            diff = end - start
            if diff < M:
                evenly_spaced_indices.extend(range(start, end + 1))
            else:
                step = diff / M
                evenly_spaced_indices.extend(
                    int(start + 1 + j * step) for j in range(M)
                )
        mem_indices = np.unique(evenly_spaced_indices)
        output_points = points_window[pose_window_indices]

        return output_points, mem_indices, loss

    return sls_points_test


def create_sls_each_curve_with_memo(
    m: int = DEFAULT_M, c: int = DEFAULT_C, included_kps: list[int] = SVM_INCLUDED_KPS
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    compute_sls = create_sls_with_memo(m)
    c_ = c

    def sls_each_curve(
        point_windows: np.ndarray, curve_mem_indices: np.ndarray, c: int = c_
    ) -> np.ndarray:
        return np.array(
            [
                (
                    compute_sls(curve, curve_mem_indices[kp], c)
                    if kp in included_kps
                    else (np.array([]), np.array([], dtype=int), 0)
                )
                for kp, curve in enumerate(point_windows)
            ]
        )

    return sls_each_curve
