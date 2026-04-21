"""
道路邊界的全局最佳路徑修正。

這個模組把左右邊界視為兩條從下到上連續的曲線，並且：
1. 左右分開建模與估計 confidence。
2. geometry 只作為 baseline，segmentation 只在 baseline 附近修正。
3. 以動態規劃挑出整條邊界的最佳路徑，避免單列局部噪聲主導。
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.boundary_smoothing import solve_boundary_path


def _compute_overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def _find_row_segments(row: np.ndarray) -> List[Tuple[int, int]]:
    xs = np.where(row > 0)[0]
    if xs.size == 0:
        return []

    segments = []
    start = int(xs[0])
    prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
            continue
        segments.append((start, prev))
        start = x
        prev = x
    segments.append((start, prev))
    return segments


def _line_x_at_y(line: Optional[Tuple[float, float]], y: int, width: int) -> Optional[int]:
    if line is None:
        return None
    slope, intercept = line
    if abs(slope) < 1e-6:
        return None
    x = int(round((y - intercept) / slope))
    return int(np.clip(x, 0, width - 1))


def _trace_geometry_baseline(
    geometry_result: Dict,
    shape: Tuple[int, int],
    min_top_width_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    left = np.full(height, -1, dtype=np.int32)
    right = np.full(height, -1, dtype=np.int32)
    valid = np.zeros(height, dtype=bool)

    geometry_mask = (geometry_result["mask"] > 0).astype(np.uint8) * 255
    geom_rows = np.where(np.any(geometry_mask > 0, axis=1))[0]
    if geom_rows.size == 0:
        return left, right, valid

    left_line = geometry_result.get("left_line")
    right_line = geometry_result.get("right_line")
    min_top_width = max(4, int(width * min_top_width_ratio))

    for y in range(int(geom_rows[0]), int(geom_rows[-1]) + 1):
        left_x = _line_x_at_y(left_line, y, width)
        right_x = _line_x_at_y(right_line, y, width)

        if left_x is None or right_x is None:
            xs = np.where(geometry_mask[y] > 0)[0]
            if xs.size == 0:
                continue
            left_x = int(xs[0])
            right_x = int(xs[-1])

        if right_x - left_x < min_top_width:
            continue

        left[y] = left_x
        right[y] = right_x
        valid[y] = True

    return left, right, valid


def _collect_side_candidates(
    supported_region: np.ndarray,
    baseline_xs: np.ndarray,
    valid_rows: np.ndarray,
    side: str,
    search_margin: int,
    width: int,
) -> Tuple[List[List[Dict[str, float]]], np.ndarray]:
    assert side in {"left", "right"}

    candidates_per_row: List[List[Dict[str, float]]] = []
    row_order = np.where(valid_rows)[0]

    for y in row_order:
        baseline_x = int(baseline_xs[y])
        row = supported_region[y]
        segments = _find_row_segments(row)
        row_candidates: List[Dict[str, float]] = [
            {
                "x": baseline_x,
                "confidence": 0.35,
                "baseline_x": baseline_x,
                "source": 0,
                "y": int(y),
            }
        ]

        for start, end in segments:
            seg_len = end - start + 1
            if side == "left":
                edge_x = start
            else:
                edge_x = end

            distance = abs(edge_x - baseline_x)
            if distance > search_margin:
                continue

            confidence = 1.0 - (distance / max(search_margin, 1))
            confidence += min(seg_len / max(width * 0.08, 1.0), 1.0) * 0.4
            row_candidates.append(
                {
                    "x": int(edge_x),
                    "confidence": float(confidence),
                    "baseline_x": baseline_x,
                    "source": 1,
                    "y": int(y),
                }
            )

        row_candidates.sort(key=lambda item: (item["confidence"], -abs(item["x"] - baseline_x)), reverse=True)
        dedup: List[Dict[str, float]] = []
        seen = set()
        for candidate in row_candidates:
            if candidate["x"] in seen:
                continue
            dedup.append(candidate)
            seen.add(candidate["x"])
            if len(dedup) >= 6:
                break

        candidates_per_row.append(dedup)

    return candidates_per_row, row_order


def _fill_series_from_solution(
    solution_xs: np.ndarray,
    confidences: np.ndarray,
    row_order: np.ndarray,
    height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.full(height, -1, dtype=np.int32)
    conf = np.zeros(height, dtype=np.float32)
    valid = np.zeros(height, dtype=bool)

    for idx, y in enumerate(row_order):
        xs[int(y)] = int(solution_xs[idx])
        conf[int(y)] = float(confidences[idx])
        valid[int(y)] = True

    return xs, conf, valid


def _build_mask_from_boundaries(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
    shape: Tuple[int, int],
    width_alpha: float = 0.92,
    min_width_px: int = 14,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)

    valid_rows = np.where(valid)[0]
    if valid_rows.size == 0:
        return mask

    top_y = int(valid_rows[0])
    top_width = max(int((right[top_y] - left[top_y]) * width_alpha), min_width_px)
    center_x = int(round((left[top_y] + right[top_y]) / 2.0))
    adjusted_left_top = max(0, center_x - top_width // 2)
    adjusted_right_top = min(width - 1, center_x + top_width // 2)

    left_adj = left.copy()
    right_adj = right.copy()
    left_adj[top_y] = adjusted_left_top
    right_adj[top_y] = adjusted_right_top

    for y in valid_rows:
        if left_adj[y] < right_adj[y]:
            mask[y, left_adj[y]:right_adj[y] + 1] = 255

    return mask


def _visualize_candidates(
    shape: Tuple[int, int],
    candidates_per_row: List[List[Dict[str, float]]],
    chosen_xs: np.ndarray,
    row_order: np.ndarray,
    baseline_xs: np.ndarray,
    side: str,
) -> np.ndarray:
    height, width = shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    candidate_color = (0, 140, 255) if side == "left" else (255, 140, 0)
    chosen_color = (0, 0, 255) if side == "left" else (0, 255, 0)
    baseline_color = (255, 255, 255)

    for idx, y in enumerate(row_order):
        for candidate in candidates_per_row[idx]:
            cv2.circle(canvas, (int(candidate["x"]), int(y)), 1, candidate_color, -1)
        cv2.circle(canvas, (int(baseline_xs[idx]), int(y)), 1, baseline_color, -1)
        cv2.circle(canvas, (int(chosen_xs[idx]), int(y)), 2, chosen_color, -1)

    return canvas


def _visualize_confidence_heatmap(
    shape: Tuple[int, int],
    left_confidence: np.ndarray,
    right_confidence: np.ndarray,
    left_x: np.ndarray,
    right_x: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    height, width = shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for y in np.where(valid)[0]:
        lx = int(left_x[y])
        rx = int(right_x[y])
        left_intensity = int(np.clip(left_confidence[y], 0.0, 1.0) * 255)
        right_intensity = int(np.clip(right_confidence[y], 0.0, 1.0) * 255)
        cv2.circle(canvas, (lx, int(y)), 2, (0, 0, left_intensity), -1)
        cv2.circle(canvas, (rx, int(y)), 2, (0, right_intensity, 0), -1)
    return canvas


def _plot_series(
    values_a: np.ndarray,
    values_b: Optional[np.ndarray],
    valid_rows: np.ndarray,
    height: int,
    max_value: float,
    color_a: Tuple[int, int, int],
    color_b: Tuple[int, int, int],
) -> np.ndarray:
    plot_height = 320
    plot_width = height
    canvas = np.zeros((plot_height, plot_width, 3), dtype=np.uint8)
    rows = valid_rows
    if rows.size <= 1:
        return canvas

    def _points(values: np.ndarray):
        pts = []
        for idx, y in enumerate(rows):
            x = idx
            scaled = 1.0 - (values[y] / max(max_value, 1e-6))
            py = int(np.clip(scaled * (plot_height - 20), 0, plot_height - 1))
            pts.append([x, py])
        return np.array(pts, dtype=np.int32)

    cv2.polylines(canvas, [_points(values_a)], False, color_a, 2)
    if values_b is not None:
        cv2.polylines(canvas, [_points(values_b)], False, color_b, 2)
    return canvas


def _visualize_offset_series(
    left_offset: np.ndarray,
    right_offset: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    max_value = float(max(np.max(np.abs(left_offset[valid_rows])), np.max(np.abs(right_offset[valid_rows])), 1.0))
    return _plot_series(
        np.abs(left_offset),
        np.abs(right_offset),
        valid_rows,
        int(valid_rows.size),
        max_value,
        (0, 0, 255),
        (0, 255, 0),
    )


def refine_mask_with_boundaries(
    region_mask: np.ndarray,
    geometry_result: Dict,
    min_overlap: float = 0.03,
    support_margin_x: int = 36,
    support_margin_y: int = 9,
    smoothing_window: int = 21,
    geometry_extension_limit: int = 90,
    min_top_width_ratio: float = 0.015,
    geometry_confidence_threshold: float = 0.45,
    left_search_margin: int = 52,
    right_search_margin: int = 40,
    left_max_delta: int = 26,
    right_max_delta: int = 22,
    width_alpha: float = 0.92,
) -> Dict[str, np.ndarray]:
    region = (region_mask > 0).astype(np.uint8) * 255
    geometry_mask = (geometry_result["mask"] > 0).astype(np.uint8) * 255
    height, width = region.shape[:2]

    overlap = _compute_overlap_ratio(region, geometry_mask)
    confidence = float(geometry_result.get("confidence", 0.0))

    if overlap < min_overlap and confidence < geometry_confidence_threshold:
        rows = np.where(np.any(region > 0, axis=1))[0]
        left = np.full(height, -1, dtype=np.int32)
        right = np.full(height, -1, dtype=np.int32)
        valid = np.zeros(height, dtype=bool)
        left_conf = np.zeros(height, dtype=np.float32)
        right_conf = np.zeros(height, dtype=np.float32)
        for y in rows:
            xs = np.where(region[y] > 0)[0]
            left[y] = int(xs[0])
            right[y] = int(xs[-1])
            valid[y] = True
            left_conf[y] = 0.2
            right_conf[y] = 0.2
        fallback_mask = region.copy()
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        return {
            "fused_before_regularization": fallback_mask,
            "regularized_mask": fallback_mask,
            "top_edge_visualization": blank.copy(),
            "boundary_trace_visualization": blank.copy(),
            "left_boundary_candidates_visualization": blank.copy(),
            "right_boundary_candidates_visualization": blank.copy(),
            "confidence_heatmap_visualization": blank.copy(),
            "boundary_delta_visualization": blank.copy(),
            "width_curve_visualization": blank.copy(),
            "geometry_offset_visualization": blank.copy(),
            "left_boundary": left,
            "right_boundary": right,
            "left_confidence": left_conf,
            "right_confidence": right_conf,
            "valid_rows": valid,
            "supported_region": region.copy(),
            "overlap_ratio": overlap,
            "geometry_confidence": confidence,
            "left_offset": np.zeros(height, dtype=np.int32),
            "right_offset": np.zeros(height, dtype=np.int32),
        }

    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, support_margin_x * 2 + 1), max(3, support_margin_y * 2 + 1)),
    )
    geometry_band = cv2.dilate(geometry_mask, band_kernel, iterations=1)
    supported_region = cv2.bitwise_and(region, geometry_band)

    baseline_left, baseline_right, valid = _trace_geometry_baseline(
        geometry_result,
        region.shape,
        min_top_width_ratio=min_top_width_ratio,
    )
    row_order = np.where(valid)[0]

    left_candidates, left_rows = _collect_side_candidates(
        supported_region,
        baseline_left,
        valid,
        side="left",
        search_margin=left_search_margin,
        width=width,
    )
    right_candidates, right_rows = _collect_side_candidates(
        supported_region,
        baseline_right,
        valid,
        side="right",
        search_margin=right_search_margin,
        width=width,
    )

    left_solution = solve_boundary_path(
        left_candidates,
        left_rows,
        geometry_weight=1.2,
        delta_weight=2.1,
        confidence_weight=1.5,
        max_delta=float(left_max_delta),
    )
    right_solution = solve_boundary_path(
        right_candidates,
        right_rows,
        geometry_weight=1.0,
        delta_weight=1.7,
        confidence_weight=1.4,
        max_delta=float(right_max_delta),
    )

    left_boundary, left_confidence, left_valid = _fill_series_from_solution(
        left_solution["xs"],
        left_solution["confidences"],
        left_rows,
        height,
    )
    right_boundary, right_confidence, right_valid = _fill_series_from_solution(
        right_solution["xs"],
        right_solution["confidences"],
        right_rows,
        height,
    )
    valid = np.logical_and(left_valid, right_valid)

    fused_before = _build_mask_from_boundaries(
        left_boundary,
        right_boundary,
        valid,
        region.shape,
        width_alpha=width_alpha,
    )

    left_delta = np.zeros(height, dtype=np.int32)
    right_delta = np.zeros(height, dtype=np.int32)
    valid_rows = np.where(valid)[0]
    for idx in range(1, valid_rows.size):
        curr_y = int(valid_rows[idx])
        prev_y = int(valid_rows[idx - 1])
        left_delta[curr_y] = abs(left_boundary[curr_y] - left_boundary[prev_y])
        right_delta[curr_y] = abs(right_boundary[curr_y] - right_boundary[prev_y])

    width_curve = np.zeros(height, dtype=np.float32)
    geometry_width_curve = np.zeros(height, dtype=np.float32)
    for y in valid_rows:
        width_curve[y] = float(right_boundary[y] - left_boundary[y])
        geometry_width_curve[y] = float(baseline_right[y] - baseline_left[y])

    left_offset = np.zeros(height, dtype=np.int32)
    right_offset = np.zeros(height, dtype=np.int32)
    for y in valid_rows:
        left_offset[y] = int(left_boundary[y] - baseline_left[y])
        right_offset[y] = int(right_boundary[y] - baseline_right[y])

    top_edge_visual = cv2.cvtColor(fused_before, cv2.COLOR_GRAY2BGR)
    top_rows = valid_rows[: max(1, min(24, valid_rows.size))]
    for y in top_rows:
        cv2.circle(top_edge_visual, (int(left_boundary[y]), int(y)), 2, (0, 0, 255), -1)
        cv2.circle(top_edge_visual, (int(right_boundary[y]), int(y)), 2, (0, 255, 0), -1)

    boundary_trace_visual = np.zeros((height, width, 3), dtype=np.uint8)
    left_points = np.array([[int(left_boundary[y]), int(y)] for y in valid_rows], dtype=np.int32)
    right_points = np.array([[int(right_boundary[y]), int(y)] for y in valid_rows], dtype=np.int32)
    cv2.polylines(boundary_trace_visual, [left_points], False, (0, 0, 255), 2)
    cv2.polylines(boundary_trace_visual, [right_points], False, (0, 255, 0), 2)

    left_candidates_visual = _visualize_candidates(
        region.shape,
        left_candidates,
        left_solution["xs"],
        left_rows,
        left_solution["baseline_xs"],
        "left",
    )
    right_candidates_visual = _visualize_candidates(
        region.shape,
        right_candidates,
        right_solution["xs"],
        right_rows,
        right_solution["baseline_xs"],
        "right",
    )
    confidence_heatmap = _visualize_confidence_heatmap(
        region.shape,
        left_confidence,
        right_confidence,
        left_boundary,
        right_boundary,
        valid,
    )
    boundary_delta_visual = _plot_series(
        left_delta,
        right_delta,
        valid_rows,
        int(valid_rows.size),
        float(max(np.max(left_delta[valid_rows]), np.max(right_delta[valid_rows]), 1.0)),
        (0, 0, 255),
        (0, 255, 0),
    )
    width_curve_visual = _plot_series(
        width_curve,
        geometry_width_curve,
        valid_rows,
        int(valid_rows.size),
        float(max(np.max(width_curve[valid_rows]), np.max(geometry_width_curve[valid_rows]), 1.0)),
        (255, 255, 255),
        (255, 255, 0),
    )
    geometry_offset_visual = _visualize_offset_series(left_offset, right_offset, valid_rows)

    return {
        "fused_before_regularization": fused_before,
        "regularized_mask": fused_before.copy(),
        "top_edge_visualization": top_edge_visual,
        "boundary_trace_visualization": boundary_trace_visual,
        "left_boundary_candidates_visualization": left_candidates_visual,
        "right_boundary_candidates_visualization": right_candidates_visual,
        "confidence_heatmap_visualization": confidence_heatmap,
        "boundary_delta_visualization": boundary_delta_visual,
        "width_curve_visualization": width_curve_visual,
        "geometry_offset_visualization": geometry_offset_visual,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "left_confidence": left_confidence,
        "right_confidence": right_confidence,
        "valid_rows": valid,
        "supported_region": supported_region,
        "overlap_ratio": overlap,
        "geometry_confidence": confidence,
        "left_offset": left_offset,
        "right_offset": right_offset,
    }
