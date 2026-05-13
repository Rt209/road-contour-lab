"""Curve-aware road boundary tracing.

This module refines a candidate road region without assuming that the road can
be represented by two straight Hough lines.  It traces the left and right road
edges row-by-row, smooths those traces, and rebuilds a filled road mask from
the curved boundaries.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _find_row_segments(row: np.ndarray) -> List[Tuple[int, int]]:
    xs = np.where(row > 0)[0]
    if xs.size == 0:
        return []

    segments: List[Tuple[int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for x_value in xs[1:]:
        x = int(x_value)
        if x == prev + 1:
            prev = x
            continue
        segments.append((start, prev))
        start = x
        prev = x
    segments.append((start, prev))
    return segments


def _segment_score(
    segment: Tuple[int, int],
    predicted_center: float,
    previous_width: float,
    image_width: int,
    center_bias_weight: float,
    width_change_weight: float,
) -> float:
    left, right = segment
    width = max(float(right - left + 1), 1.0)
    center = (left + right) / 2.0
    center_distance = abs(center - predicted_center) / max(image_width, 1)
    width_distance = abs(width - previous_width) / max(previous_width, 1.0)
    return width - (center_distance * image_width * center_bias_weight) - (
        width_distance * previous_width * width_change_weight
    )


def _choose_bottom_segment(
    mask: np.ndarray,
    min_row_coverage: int,
    bottom_search_ratio: float,
) -> Optional[Tuple[int, Tuple[int, int]]]:
    height, width = mask.shape[:2]
    center_x = width / 2.0
    start_y = max(0, int(height * (1.0 - bottom_search_ratio)))

    best: Optional[Tuple[float, int, Tuple[int, int]]] = None
    for y in range(height - 1, start_y - 1, -1):
        for segment in _find_row_segments(mask[y]):
            seg_width = segment[1] - segment[0] + 1
            if seg_width < min_row_coverage:
                continue
            seg_center = (segment[0] + segment[1]) / 2.0
            center_score = 1.0 - abs(seg_center - center_x) / max(center_x, 1.0)
            score = seg_width + center_score * width * 0.35
            if best is None or score > best[0]:
                best = (score, y, segment)

    if best is None:
        return None
    return best[1], best[2]


def _interpolate_series(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    height = values.shape[0]
    valid_rows = np.where(valid)[0]
    if valid_rows.size == 0:
        return values.astype(np.float32)
    if valid_rows.size == 1:
        return np.full(height, float(values[valid_rows[0]]), dtype=np.float32)

    rows = np.arange(height)
    return np.interp(rows, valid_rows, values[valid_rows]).astype(np.float32)


def _smooth_series(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)

    window = int(window)
    if window % 2 == 0:
        window += 1

    filled = _interpolate_series(values, valid)
    return _smooth_1d(filled, window)


def refine_curve_road_mask_from_lane_markers(
    original_image: np.ndarray,
    min_valid_rows: int = 90,
    min_boundary_width_ratio: float = 0.16,
    bottom_extension_ratio: float = 0.98,
    smoothing_window: int = 31,
    boundary_padding: int = 6,
) -> Dict[str, np.ndarray]:
    """Build a road mask from visible white left/right road edge markings."""
    height, width = original_image.shape[:2]
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    hue = hsv[:, :, 0]
    white_markers = (saturation < 75) & (value > 150)
    yellow_markers = (hue >= 12) & (hue <= 42) & (saturation > 75) & (value > 110)
    marker_mask = (white_markers | yellow_markers).astype(np.uint8) * 255
    marker_mask[: int(height * 0.55), :] = 0
    marker_mask = cv2.morphologyEx(
        marker_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)),
        iterations=1,
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(marker_mask, 8)
    left_labels: List[int] = []
    right_labels: List[int] = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < 80:
            continue

        centroid_x = centroids[label_idx][0]
        touches_left_entry = x < width * 0.04 or (
            y + h > height * 0.95 and centroid_x < width * 0.45
        )
        touches_right_entry = x + w > width * 0.96 or (
            y + h > height * 0.70 and centroid_x > width * 0.65
        )
        if touches_left_entry:
            left_labels.append(label_idx)
        if touches_right_entry:
            right_labels.append(label_idx)

    blank = np.zeros((height, width, 3), dtype=np.uint8)
    empty = np.zeros((height, width), dtype=np.uint8)
    if not left_labels or not right_labels:
        return {
            "regularized_mask": empty,
            "marker_mask": marker_mask,
            "boundary_trace_visualization": blank,
            "width_curve_visualization": blank.copy(),
            "valid_rows": np.zeros(height, dtype=bool),
            "confidence": 0.0,
        }

    left_mask = np.isin(labels, left_labels)
    right_mask = np.isin(labels, right_labels)
    left = np.full(height, np.nan, dtype=np.float32)
    right = np.full(height, np.nan, dtype=np.float32)

    for y in range(int(height * 0.55), height):
        left_xs = np.where(left_mask[y])[0]
        right_xs = np.where(right_mask[y])[0]
        if left_xs.size > 0:
            left[y] = float(left_xs.min())
        if right_xs.size > 0:
            right[y] = float(right_xs.max())

    min_boundary_width = width * min_boundary_width_ratio
    valid = (~np.isnan(left)) & (~np.isnan(right)) & ((right - left) > min_boundary_width)
    valid_rows = np.where(valid)[0]
    if valid_rows.size < min_valid_rows:
        return {
            "regularized_mask": empty,
            "marker_mask": marker_mask,
            "boundary_trace_visualization": blank,
            "width_curve_visualization": blank.copy(),
            "valid_rows": valid,
            "confidence": float(valid_rows.size / max(min_valid_rows, 1)),
        }

    end_y = max(int(height * bottom_extension_ratio), int(valid_rows[-1]))
    rows = np.arange(int(valid_rows[0]), min(height, end_y + 1))
    left_interp = np.interp(rows, valid_rows, left[valid_rows])
    right_interp = np.interp(rows, valid_rows, right[valid_rows])
    left_interp = _smooth_1d(left_interp, smoothing_window)
    right_interp = _smooth_1d(right_interp, smoothing_window)

    top_width_ratio = float((right_interp[0] - left_interp[0]) / max(width, 1))
    edge_touch = (left_interp <= 2.0) | (right_interp >= (width - 3.0))
    edge_touch_ratio = float(np.mean(edge_touch))
    if top_width_ratio > 0.78 or edge_touch_ratio > 0.28:
        return {
            "regularized_mask": empty,
            "marker_mask": marker_mask,
            "boundary_trace_visualization": blank,
            "width_curve_visualization": blank.copy(),
            "valid_rows": valid,
            "confidence": 0.0,
        }

    regularized = np.zeros((height, width), dtype=np.uint8)
    for idx, y in enumerate(rows):
        row_left = int(np.clip(left_interp[idx] - boundary_padding, 0, width - 1))
        row_right = int(np.clip(right_interp[idx] + boundary_padding, 0, width - 1))
        if row_left < row_right:
            regularized[int(y), row_left:row_right + 1] = 255

    trace = np.zeros((height, width, 3), dtype=np.uint8)
    left_points = np.array(
        [[int(np.clip(left_interp[idx], 0, width - 1)), int(y)] for idx, y in enumerate(rows)],
        dtype=np.int32,
    )
    right_points = np.array(
        [[int(np.clip(right_interp[idx], 0, width - 1)), int(y)] for idx, y in enumerate(rows)],
        dtype=np.int32,
    )
    cv2.polylines(trace, [left_points], False, (0, 0, 255), 2)
    cv2.polylines(trace, [right_points], False, (0, 255, 0), 2)

    width_visual = np.zeros((320, max(height, 1), 3), dtype=np.uint8)
    row_widths = right_interp - left_interp
    max_width = max(float(np.max(row_widths)), 1.0)
    points = []
    for idx, _ in enumerate(rows):
        py = int(np.clip((1.0 - row_widths[idx] / max_width) * 300, 0, 319))
        points.append([idx, py])
    if len(points) > 1:
        cv2.polylines(width_visual, [np.array(points, dtype=np.int32)], False, (255, 255, 255), 2)

    output_valid = np.zeros(height, dtype=bool)
    output_valid[rows] = True
    confidence = float(np.clip(valid_rows.size / max(height - valid_rows[0], 1), 0.0, 1.0))
    return {
        "regularized_mask": regularized,
        "marker_mask": marker_mask,
        "boundary_trace_visualization": trace,
        "width_curve_visualization": width_visual,
        "valid_rows": output_valid,
        "confidence": confidence,
    }


def _build_mask_from_curves(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
    shape: Tuple[int, int],
    width_alpha: float,
    top_width_alpha: float,
    min_width: int,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    valid_rows = np.where(valid)[0]
    if valid_rows.size == 0:
        return mask

    top_y = int(valid_rows[0])
    bottom_y = int(valid_rows[-1])
    span = max(bottom_y - top_y, 1)

    for y in valid_rows:
        curve_width = max(float(right[y] - left[y]), float(min_width))
        t = (float(y) - top_y) / span
        row_alpha = top_width_alpha + (width_alpha - top_width_alpha) * t
        target_width = max(curve_width * row_alpha, float(min_width))
        center = (float(left[y]) + float(right[y])) / 2.0
        row_left = int(np.clip(round(center - target_width / 2.0), 0, width - 1))
        row_right = int(np.clip(round(center + target_width / 2.0), 0, width - 1))
        if row_left < row_right:
            mask[int(y), row_left:row_right + 1] = 255

    return mask


def refine_curve_road_mask(
    region_mask: np.ndarray,
    min_row_coverage_ratio: float = 0.025,
    bottom_search_ratio: float = 0.28,
    center_bias_weight: float = 0.35,
    width_change_weight: float = 0.25,
    max_center_shift_ratio: float = 0.12,
    max_gap_rows: int = 18,
    smoothing_window: int = 31,
    width_alpha: float = 1.02,
    top_width_alpha: float = 0.92,
    min_width_ratio: float = 0.035,
    close_kernel_size: int = 9,
    close_iterations: int = 2,
) -> Dict[str, np.ndarray]:
    region = (region_mask > 0).astype(np.uint8) * 255
    height, width = region.shape[:2]
    min_row_coverage = max(3, int(width * min_row_coverage_ratio))
    min_width = max(4, int(width * min_width_ratio))
    max_center_shift = max(4.0, width * max_center_shift_ratio)

    start = _choose_bottom_segment(region, min_row_coverage, bottom_search_ratio)
    empty_boundary = np.full(height, -1, dtype=np.int32)
    empty_float = np.zeros(height, dtype=np.float32)
    if start is None:
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        return {
            "regularized_mask": region.copy(),
            "boundary_trace_visualization": blank,
            "width_curve_visualization": blank.copy(),
            "left_boundary": empty_boundary.copy(),
            "right_boundary": empty_boundary.copy(),
            "center_curve": empty_float.copy(),
            "width_curve": empty_float.copy(),
            "valid_rows": np.zeros(height, dtype=bool),
            "confidence": 0.0,
        }

    start_y, start_segment = start
    left = np.full(height, -1, dtype=np.int32)
    right = np.full(height, -1, dtype=np.int32)
    valid = np.zeros(height, dtype=bool)

    predicted_center = (start_segment[0] + start_segment[1]) / 2.0
    previous_width = float(start_segment[1] - start_segment[0] + 1)
    gap_rows = 0

    for y in range(start_y, -1, -1):
        segments = [
            segment
            for segment in _find_row_segments(region[y])
            if segment[1] - segment[0] + 1 >= min_row_coverage
        ]

        chosen: Optional[Tuple[int, int]] = None
        if segments:
            scored = sorted(
                segments,
                key=lambda segment: _segment_score(
                    segment,
                    predicted_center,
                    previous_width,
                    width,
                    center_bias_weight,
                    width_change_weight,
                ),
                reverse=True,
            )
            candidate = scored[0]
            candidate_center = (candidate[0] + candidate[1]) / 2.0
            if abs(candidate_center - predicted_center) <= max_center_shift or gap_rows > 0:
                chosen = candidate

        if chosen is None:
            gap_rows += 1
            if gap_rows > max_gap_rows:
                break
            continue

        gap_rows = 0
        left[y] = int(chosen[0])
        right[y] = int(chosen[1])
        valid[y] = True
        measured_center = (chosen[0] + chosen[1]) / 2.0
        measured_width = float(chosen[1] - chosen[0] + 1)
        predicted_center = 0.72 * measured_center + 0.28 * predicted_center
        previous_width = 0.72 * measured_width + 0.28 * previous_width

    smooth_left = _smooth_series(left.astype(np.float32), valid, smoothing_window)
    smooth_right = _smooth_series(right.astype(np.float32), valid, smoothing_window)
    smooth_left = np.clip(np.rint(smooth_left), 0, width - 1).astype(np.int32)
    smooth_right = np.clip(np.rint(smooth_right), 0, width - 1).astype(np.int32)

    regularized = _build_mask_from_curves(
        smooth_left,
        smooth_right,
        valid,
        region.shape,
        width_alpha=width_alpha,
        top_width_alpha=top_width_alpha,
        min_width=min_width,
    )

    if close_kernel_size > 1 and close_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(close_kernel_size), int(close_kernel_size)),
        )
        regularized = cv2.morphologyEx(
            regularized,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=int(close_iterations),
        )

    valid_rows = np.where(valid)[0]
    trace = np.zeros((height, width, 3), dtype=np.uint8)
    if valid_rows.size > 1:
        left_points = np.array([[int(smooth_left[y]), int(y)] for y in valid_rows], dtype=np.int32)
        right_points = np.array([[int(smooth_right[y]), int(y)] for y in valid_rows], dtype=np.int32)
        cv2.polylines(trace, [left_points], False, (0, 0, 255), 2)
        cv2.polylines(trace, [right_points], False, (0, 255, 0), 2)

    center_curve = np.zeros(height, dtype=np.float32)
    width_curve = np.zeros(height, dtype=np.float32)
    for y in valid_rows:
        center_curve[y] = (smooth_left[y] + smooth_right[y]) / 2.0
        width_curve[y] = max(float(smooth_right[y] - smooth_left[y]), 0.0)

    width_visual = np.zeros((320, max(height, 1), 3), dtype=np.uint8)
    if valid_rows.size > 1:
        max_width = max(float(np.max(width_curve[valid_rows])), 1.0)
        points = []
        for idx, y in enumerate(valid_rows):
            py = int(np.clip((1.0 - width_curve[y] / max_width) * 300, 0, 319))
            points.append([idx, py])
        cv2.polylines(width_visual, [np.array(points, dtype=np.int32)], False, (255, 255, 255), 2)

    confidence = float(valid_rows.size / max(start_y + 1, 1))
    return {
        "regularized_mask": regularized,
        "boundary_trace_visualization": trace,
        "width_curve_visualization": width_visual,
        "left_boundary": smooth_left,
        "right_boundary": smooth_right,
        "center_curve": center_curve,
        "width_curve": width_curve,
        "valid_rows": valid,
        "confidence": np.float32(np.clip(confidence, 0.0, 1.0)),
    }
