"""
提供 geometry 與 segmentation 的關係量測與融合入口。

這一層刻意保持輕量，真正的邊界式修正邏輯放在
`boundary_refinement.py`，避免在這裡再次引入 row-span fill、
convex hull 或其他容易造成平台化上邊界的操作。
"""

import numpy as np

from src.segmentation.boundary_refinement import refine_mask_with_boundaries


def compute_overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """計算兩張二值遮罩的 IoU。"""
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0

    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def create_overlap_visualization(
    region_mask: np.ndarray,
    geometry_mask: np.ndarray
) -> np.ndarray:
    """建立 geometry 與 segmentation 的重疊可視化圖。"""
    canvas = np.zeros((*region_mask.shape, 3), dtype=np.uint8)
    region = region_mask > 0
    geometry = geometry_mask > 0

    canvas[region] = (0, 0, 255)
    canvas[geometry] = np.maximum(
        canvas[geometry],
        np.array([0, 255, 0], dtype=np.uint8),
    )
    canvas[np.logical_and(region, geometry)] = (0, 255, 255)
    return canvas


def repair_region_with_geometry(
    region_mask: np.ndarray,
    geometry_result: dict,
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
) -> dict:
    """
    以邊界追蹤方式整合 segmentation 與 geometry。

    回傳值包含修正前後遮罩與診斷圖，讓 pipeline 可以直接輸出 debug artifacts。
    """
    return refine_mask_with_boundaries(
        region_mask=region_mask,
        geometry_result=geometry_result,
        min_overlap=min_overlap,
        support_margin_x=support_margin_x,
        support_margin_y=support_margin_y,
        smoothing_window=smoothing_window,
        geometry_extension_limit=geometry_extension_limit,
        min_top_width_ratio=min_top_width_ratio,
        geometry_confidence_threshold=geometry_confidence_threshold,
        left_search_margin=left_search_margin,
        right_search_margin=right_search_margin,
        left_max_delta=left_max_delta,
        right_max_delta=right_max_delta,
        width_alpha=width_alpha,
    )
