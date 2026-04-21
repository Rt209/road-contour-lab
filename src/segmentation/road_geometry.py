"""
此模組利用道路在透視畫面中的幾何特性，估計左右路緣線並重建道路區域。
當一般紋理或連通元件方法容易被天空、山脊或地表紋理干擾時，
可將本模組作為後段修正，產生更貼近真實馬路範圍的幾何遮罩。
"""

from typing import Dict, Optional

import cv2
import numpy as np


def _build_roi_mask(
    image_shape: tuple,
    top_y_ratio: float,
    top_width_ratio: float,
    bottom_width_ratio: float
) -> np.ndarray:
    """
    建立道路透視常見的梯形興趣區域，限制線段搜尋範圍。
    """
    height, width = image_shape[:2]
    center_x = width // 2
    top_y = int(height * top_y_ratio)
    top_half_width = int(width * top_width_ratio / 2.0)
    bottom_half_width = int(width * bottom_width_ratio / 2.0)

    polygon = np.array(
        [
            [center_x - bottom_half_width, height - 1],
            [center_x - top_half_width, top_y],
            [center_x + top_half_width, top_y],
            [center_x + bottom_half_width, height - 1],
        ],
        dtype=np.int32,
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def _average_lane_line(lines: list) -> Optional[tuple]:
    """
    將同一側的候選線段依長度加權平均，得到穩定的代表線。
    """
    if not lines:
        return None

    weights = np.array([line["length"] for line in lines], dtype=np.float64)
    slopes = np.array([line["slope"] for line in lines], dtype=np.float64)
    intercepts = np.array([line["intercept"] for line in lines], dtype=np.float64)

    slope = float(np.average(slopes, weights=weights))
    intercept = float(np.average(intercepts, weights=weights))
    return slope, intercept


def _normalize_illumination(grayscale_image: np.ndarray) -> np.ndarray:
    """
    使用 CLAHE 強化局部對比，降低逆光、陰影或亮部過曝對邊界偵測的影響。
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(grayscale_image)


def _select_lane_line(
    lines: list,
    image_width: int,
    side: str
) -> Optional[tuple]:
    """
    從候選線段中挑出最像外側路緣的代表線。
    左側偏好底部投影更靠左的長線段，右側偏好底部投影更靠右的長線段。
    """
    if not lines:
        return None

    if side == "left":
        outward_reference = image_width * 0.10
        scored = sorted(
            lines,
            key=lambda line: (
                line["length"] - abs(line["bottom_x"] - outward_reference) * 0.35
            ),
            reverse=True,
        )
    else:
        outward_reference = image_width * 0.90
        scored = sorted(
            lines,
            key=lambda line: (
                line["length"] - abs(line["bottom_x"] - outward_reference) * 0.35
            ),
            reverse=True,
        )

    top_candidates = scored[: min(5, len(scored))]
    return _average_lane_line(top_candidates)


def _draw_lines(
    base_image: np.ndarray,
    lines: list,
    color: tuple,
    thickness: int = 2
) -> np.ndarray:
    """在視覺化影像上畫出線段。"""
    canvas = base_image.copy()
    for line in lines:
        x1, y1, x2, y2 = [int(value) for value in line["points"]]
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
    return canvas


def _draw_representative_line(
    canvas: np.ndarray,
    slope: float,
    intercept: float,
    color: tuple,
    thickness: int = 3
) -> np.ndarray:
    """將代表性路緣線延伸成完整視覺化線段。"""
    result = canvas.copy()
    height, width = result.shape[:2]
    y_bottom = height - 1
    y_top = int(height * 0.55)

    x_bottom = int(np.clip((y_bottom - intercept) / slope, 0, width - 1))
    x_top = int(np.clip((y_top - intercept) / slope, 0, width - 1))
    cv2.line(result, (x_bottom, y_bottom), (x_top, y_top), color, thickness)
    return result


def detect_road_geometry_mask(
    grayscale_image: np.ndarray,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    hough_threshold: int = 60,
    min_line_length: int = 120,
    max_line_gap: int = 40,
    roi_top_y_ratio: float = 0.42,
    roi_top_width_ratio: float = 0.10,
    roi_bottom_width_ratio: float = 0.78,
    line_top_offset_ratio: float = -0.02,
    bottom_left_min_ratio: float = 0.14,
    bottom_left_max_ratio: float = 0.30,
    bottom_right_min_ratio: float = 0.70,
    bottom_right_max_ratio: float = 0.86,
    top_width_scale: float = 1.28,
    top_width_min_ratio: float = 0.04,
    top_width_max_ratio: float = 0.20,
    top_y_min_ratio: float = 0.42,
    top_y_max_ratio: float = 0.52,
    bottom_expand_ratio: float = 0.12,
    top_expand_ratio: float = 0.28,
    outer_bottom_offset_ratio: float = 0.20,
    outer_top_offset_ratio: float = 0.08,
) -> Dict[str, Optional[np.ndarray]]:
    """
    偵測左右路緣線，並以兩條線重建道路梯形遮罩。
    Args:
        grayscale_image: 灰階輸入影像。
        canny_threshold1: Canny 低門檻。
        canny_threshold2: Canny 高門檻。
        hough_threshold: HoughLinesP 投票門檻。
        min_line_length: 最短線段長度。
        max_line_gap: 可接受的線段缺口。
        roi_top_y_ratio: 梯形 ROI 上緣高度比例。
        roi_top_width_ratio: 梯形 ROI 上緣寬度比例。
        roi_bottom_width_ratio: 梯形 ROI 下緣寬度比例。
        line_top_offset_ratio: 幾何遮罩頂點相對交會點往下偏移的比例。
        bottom_left_min_ratio: 左下頂點最小水平比例。
        bottom_left_max_ratio: 左下頂點最大水平比例。
        bottom_right_min_ratio: 右下頂點最小水平比例。
        bottom_right_max_ratio: 右下頂點最大水平比例。
        top_width_scale: 以原始頂部寬度縮放的比例。
        top_width_min_ratio: 頂部最小寬度比例。
        top_width_max_ratio: 頂部最大寬度比例。
        top_y_min_ratio: 頂部最小高度比例。
        top_y_max_ratio: 頂部最大高度比例。
        bottom_expand_ratio: 底部道路寬度向外擴張比例。
        top_expand_ratio: 頂部道路寬度向外擴張比例。
        outer_bottom_offset_ratio: 以影像寬度為基準，額外向外推估底部路肩的比例。
        outer_top_offset_ratio: 以影像寬度為基準，額外向外推估遠端路肩的比例。
    Returns:
        包含道路遮罩、ROI、邊緣圖與偵測狀態的字典。
    """
    height, width = grayscale_image.shape[:2]
    normalized_gray = _normalize_illumination(grayscale_image)
    blurred = cv2.GaussianBlur(normalized_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    roi_mask = _build_roi_mask(
        grayscale_image.shape,
        top_y_ratio=roi_top_y_ratio,
        top_width_ratio=roi_top_width_ratio,
        bottom_width_ratio=roi_bottom_width_ratio,
    )
    roi_edges = cv2.bitwise_and(edges, roi_mask)
    debug_base = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    empty_mask = np.zeros_like(grayscale_image, dtype=np.uint8)
    result = {
        "mask": empty_mask,
        "roi_mask": roi_mask,
        "roi_edges": roi_edges,
        "clahe_input": normalized_gray,
        "raw_lines_visualization": debug_base.copy(),
        "filtered_lines_visualization": debug_base.copy(),
        "vanishing_point_visualization": debug_base.copy(),
        "trapezoid_mask": empty_mask.copy(),
        "outer_expanded_mask": empty_mask.copy(),
        "left_line": None,
        "right_line": None,
        "vanishing_point": None,
        "confidence": 0.0,
        "success": False,
    }

    if lines is None:
        return result

    left_lines = []
    right_lines = []
    raw_line_entries = []

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(value) for value in line]
        raw_line_entries.append({"points": (x1, y1, x2, y2)})
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3 or abs(slope) > 3.0:
            continue

        length = float(np.hypot(x2 - x1, y2 - y1))
        intercept = y1 - slope * x1
        bottom_x = (height - 1 - intercept) / slope

        if slope < 0 and max(x1, x2) < width * 0.62:
            left_lines.append(
                {
                    "slope": slope,
                    "intercept": intercept,
                    "length": length,
                    "bottom_x": bottom_x,
                    "points": (x1, y1, x2, y2),
                }
            )
        elif slope > 0 and min(x1, x2) > width * 0.38:
            right_lines.append(
                {
                    "slope": slope,
                    "intercept": intercept,
                    "length": length,
                    "bottom_x": bottom_x,
                    "points": (x1, y1, x2, y2),
                }
            )

    result["raw_lines_visualization"] = _draw_lines(debug_base, raw_line_entries, (160, 160, 160), 1)
    filtered_vis = _draw_lines(debug_base, left_lines, (0, 0, 255), 2)
    filtered_vis = _draw_lines(filtered_vis, right_lines, (0, 255, 0), 2)

    left_line = _select_lane_line(left_lines, width, "left")
    right_line = _select_lane_line(right_lines, width, "right")
    if left_line is None or right_line is None:
        result["filtered_lines_visualization"] = filtered_vis
        return result

    left_slope, left_intercept = left_line
    right_slope, right_intercept = right_line
    result["left_line"] = left_line
    result["right_line"] = right_line
    filtered_vis = _draw_representative_line(filtered_vis, left_slope, left_intercept, (0, 0, 255), 4)
    filtered_vis = _draw_representative_line(filtered_vis, right_slope, right_intercept, (0, 255, 0), 4)
    result["filtered_lines_visualization"] = filtered_vis

    intersection_x = (right_intercept - left_intercept) / (left_slope - right_slope)
    intersection_y = left_slope * intersection_x + left_intercept
    if not np.isfinite(intersection_x) or not np.isfinite(intersection_y):
        return result
    result["vanishing_point"] = (float(intersection_x), float(intersection_y))
    vanishing_vis = filtered_vis.copy()
    cv2.circle(
        vanishing_vis,
        (int(np.clip(intersection_x, 0, width - 1)), int(np.clip(intersection_y, 0, height - 1))),
        8,
        (255, 255, 0),
        -1,
    )
    result["vanishing_point_visualization"] = vanishing_vis

    top_y = int(
        np.clip(
            intersection_y + (height * line_top_offset_ratio),
            height * top_y_min_ratio,
            height * top_y_max_ratio,
        )
    )
    bottom_y = height - 1

    raw_left_bottom_x = (bottom_y - left_intercept) / left_slope
    raw_right_bottom_x = (bottom_y - right_intercept) / right_slope
    raw_left_top_x = (top_y - left_intercept) / left_slope
    raw_right_top_x = (top_y - right_intercept) / right_slope

    raw_bottom_width = max(raw_right_bottom_x - raw_left_bottom_x, 1.0)
    raw_top_width_before_expand = max(raw_right_top_x - raw_left_top_x, 1.0)
    bottom_expand = raw_bottom_width * bottom_expand_ratio
    top_expand = raw_top_width_before_expand * top_expand_ratio

    left_bottom_x = int(
        np.clip(
            raw_left_bottom_x - bottom_expand,
            width * bottom_left_min_ratio,
            width * bottom_left_max_ratio,
        )
    )
    right_bottom_x = int(
        np.clip(
            raw_right_bottom_x + bottom_expand,
            width * bottom_right_min_ratio,
            width * bottom_right_max_ratio,
        )
    )

    raw_left_top_x -= top_expand
    raw_right_top_x += top_expand

    raw_top_width = max(raw_right_top_x - raw_left_top_x, 1.0)
    target_top_width = raw_top_width * top_width_scale
    target_top_width = float(
        np.clip(
            target_top_width,
            width * top_width_min_ratio,
            width * top_width_max_ratio,
        )
    )
    center_x = float(np.clip(intersection_x, width * 0.45, width * 0.55))
    left_top_x = int(np.clip(center_x - (target_top_width / 2.0), 0, width - 1))
    right_top_x = int(np.clip(center_x + (target_top_width / 2.0), 0, width - 1))

    trapezoid_polygon = np.array(
        [
            [left_bottom_x, bottom_y],
            [left_top_x, top_y],
            [right_top_x, top_y],
            [right_bottom_x, bottom_y],
        ],
        dtype=np.int32,
    )
    trapezoid_mask = np.zeros_like(grayscale_image, dtype=np.uint8)
    cv2.fillConvexPoly(trapezoid_mask, trapezoid_polygon, 255)
    result["trapezoid_mask"] = trapezoid_mask

    # 依透視效果額外向外推估整體路面外緣，讓輪廓更接近白線外側的路肩。
    outer_bottom_offset = int(width * outer_bottom_offset_ratio)
    outer_top_offset = int(width * outer_top_offset_ratio)
    left_bottom_x = int(np.clip(left_bottom_x - outer_bottom_offset, 0, width - 1))
    right_bottom_x = int(np.clip(right_bottom_x + outer_bottom_offset, 0, width - 1))
    left_top_x = int(np.clip(left_top_x - outer_top_offset, 0, width - 1))
    right_top_x = int(np.clip(right_top_x + outer_top_offset, 0, width - 1))

    if left_top_x >= right_top_x or left_bottom_x >= right_bottom_x:
        return result

    polygon = np.array(
        [
            [left_bottom_x, bottom_y],
            [left_top_x, top_y],
            [right_top_x, top_y],
            [right_bottom_x, bottom_y],
        ],
        dtype=np.int32,
    )

    road_mask = np.zeros_like(grayscale_image, dtype=np.uint8)
    cv2.fillConvexPoly(road_mask, polygon, 255)

    result["outer_expanded_mask"] = road_mask.copy()
    result["mask"] = road_mask
    line_support_score = min(len(left_lines), len(right_lines)) / 5.0
    vp_score = 1.0 if 0 <= intersection_x < width and intersection_y < height * 0.75 else 0.5
    result["confidence"] = float(np.clip((line_support_score * 0.6) + (vp_score * 0.4), 0.0, 1.0))
    result["success"] = True
    return result
