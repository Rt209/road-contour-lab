"""
提供道路左右邊界的全局最佳路徑平滑工具。

核心概念：
1. 邊界是一條從下到上連續的路徑，不是每列獨立挑點。
2. 每一列可以有多個候選點。
3. 使用動態規劃同時考慮：
   - 與 geometry baseline 的距離
   - 候選 evidence 強度
   - 與前一列位置差
"""

from typing import Dict, List

import numpy as np


def solve_boundary_path(
    candidates_per_row: List[List[Dict[str, float]]],
    row_order: np.ndarray,
    geometry_weight: float = 1.0,
    delta_weight: float = 1.8,
    confidence_weight: float = 1.2,
    max_delta: float = 28.0,
) -> Dict[str, np.ndarray]:
    """以動態規劃求解整條邊界的最佳路徑。"""
    num_rows = len(row_order)
    if num_rows == 0:
        empty = np.array([], dtype=np.int32)
        return {
            "xs": empty,
            "confidences": np.array([], dtype=np.float32),
            "baseline_xs": empty,
        }

    costs: List[np.ndarray] = []
    parents: List[np.ndarray] = []

    first_candidates = candidates_per_row[0]
    first_cost = np.array(
        [
            (abs(candidate["x"] - candidate["baseline_x"]) * geometry_weight)
            - (candidate["confidence"] * confidence_weight)
            for candidate in first_candidates
        ],
        dtype=np.float64,
    )
    costs.append(first_cost)
    parents.append(np.full(first_cost.shape, -1, dtype=np.int32))

    for row_idx in range(1, num_rows):
        candidates = candidates_per_row[row_idx]
        prev_candidates = candidates_per_row[row_idx - 1]
        row_cost = np.full(len(candidates), np.inf, dtype=np.float64)
        row_parent = np.full(len(candidates), -1, dtype=np.int32)

        for curr_idx, candidate in enumerate(candidates):
            data_cost = (
                abs(candidate["x"] - candidate["baseline_x"]) * geometry_weight
                - (candidate["confidence"] * confidence_weight)
            )

            for prev_idx, prev_candidate in enumerate(prev_candidates):
                delta = abs(candidate["x"] - prev_candidate["x"])
                transition_cost = delta_weight * min(delta, max_delta)
                if delta > max_delta:
                    transition_cost += (delta - max_delta) * (delta_weight * 4.0)

                total_cost = costs[row_idx - 1][prev_idx] + data_cost + transition_cost
                if total_cost < row_cost[curr_idx]:
                    row_cost[curr_idx] = total_cost
                    row_parent[curr_idx] = prev_idx

        costs.append(row_cost)
        parents.append(row_parent)

    last_idx = int(np.argmin(costs[-1]))
    xs = np.zeros(num_rows, dtype=np.int32)
    confidences = np.zeros(num_rows, dtype=np.float32)
    baseline_xs = np.zeros(num_rows, dtype=np.int32)

    for row_idx in range(num_rows - 1, -1, -1):
        candidate = candidates_per_row[row_idx][last_idx]
        xs[row_idx] = int(candidate["x"])
        confidences[row_idx] = float(candidate["confidence"])
        baseline_xs[row_idx] = int(candidate["baseline_x"])
        last_idx = int(parents[row_idx][last_idx])
        if row_idx > 0 and last_idx < 0:
            last_idx = 0

    return {
        "xs": xs,
        "confidences": confidences,
        "baseline_xs": baseline_xs,
    }
