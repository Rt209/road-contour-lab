"""
此模組負責連通元件分析與面積過濾。
在區域成長完成後，畫面中仍可能包含多個破碎區塊或雜訊，
因此需要利用連通元件方法保留主要道路區域，去除不必要的小區塊。
"""

import cv2
import numpy as np


def analyze_connected_components(
    binary_mask: np.ndarray,
    connectivity: int = 8
) -> dict:
    """
    分析二值遮罩中的連通元件。

    Args:
        binary_mask: 輸入二值遮罩。
        connectivity: 連通方式，可為 4 或 8。

    Returns:
        包含 `labels`、`stats`、`centroids`、`num_labels` 的字典。
    """
    # 確保輸入為標準二值格式。
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=connectivity
    )

    return {
        "num_labels": num_labels,
        "labels": labels,
        "stats": stats,  # 每個標籤的統計資訊，例如面積與外接框。
        "centroids": centroids,  # 每個標籤的中心點座標。
    }


def filter_by_area(
    binary_mask: np.ndarray,
    min_area: int = 100,
    max_area: int = None,
    connectivity: int = 8
) -> np.ndarray:
    """
    依照面積條件過濾連通元件。

    Args:
        binary_mask: 輸入二值遮罩。
        min_area: 最小保留面積。
        max_area: 最大保留面積，`None` 表示不限制。
        connectivity: 連通方式，可為 4 或 8。

    Returns:
        過濾後的二值遮罩。
    """
    cc_info = analyze_connected_components(binary_mask, connectivity)

    result = np.zeros_like(binary_mask, dtype=np.uint8)
    labels = cc_info["labels"]
    stats = cc_info["stats"]

    for label_idx in range(1, cc_info["num_labels"]):
        area = stats[label_idx, cv2.CC_STAT_AREA]

        if area >= min_area:
            if max_area is None or area <= max_area:
                result[labels == label_idx] = 255

    return result


def keep_largest_component(
    binary_mask: np.ndarray,
    connectivity: int = 8
) -> np.ndarray:
    """
    只保留最大連通元件。

    Args:
        binary_mask: 輸入二值遮罩。
        connectivity: 連通方式，可為 4 或 8。

    Returns:
        僅保留最大連通元件的二值遮罩。
    """
    cc_info = analyze_connected_components(binary_mask, connectivity)

    if cc_info["num_labels"] <= 1:
        return binary_mask

    # 尋找最大連通元件，背景標籤 0 不列入比較。
    labels = cc_info["labels"]
    stats = cc_info["stats"]

    largest_label = 1
    largest_area = 0

    for label_idx in range(1, cc_info["num_labels"]):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label_idx

    result = np.zeros_like(binary_mask, dtype=np.uint8)
    result[labels == largest_label] = 255
    return result


def keep_road_like_component(
    binary_mask: np.ndarray,
    connectivity: int = 8,
    bottom_bias_weight: float = 2.5,
    center_bias_weight: float = 0.7,
    area_weight: float = 1.0,
    height_weight: float = 0.9
) -> np.ndarray:
    """
    保留最符合道路位置先驗的連通區域。
    評分會偏向畫面較下方、較接近水平中線，且面積足夠的區域。
    Args:
        binary_mask: 二值化遮罩。
        connectivity: 連通性，支援 4 或 8。
        bottom_bias_weight: 區域靠近下方的加權比例。
        center_bias_weight: 區域靠近畫面中線的加權比例。
        area_weight: 區域面積的加權比例。
        height_weight: 區域高度的加權比例，用來保留較長的縱向道路區域。
    Returns:
        保留下來的最佳道路型連通區域遮罩。
    """
    cc_info = analyze_connected_components(binary_mask, connectivity)

    if cc_info["num_labels"] <= 1:
        return binary_mask

    labels = cc_info["labels"]
    stats = cc_info["stats"]
    centroids = cc_info["centroids"]
    height, width = binary_mask.shape[:2]
    image_area = float(height * width)

    best_label = 1
    best_score = -np.inf

    for label_idx in range(1, cc_info["num_labels"]):
        x, y, w, h, area = stats[label_idx]
        centroid_x, _ = centroids[label_idx]

        bottom_ratio = (y + h) / max(height, 1)
        center_score = 1.0 - abs(centroid_x - (width / 2.0)) / max(width / 2.0, 1.0)
        area_ratio = area / max(image_area, 1.0)
        height_ratio = h / max(height, 1.0)

        score = (
            area_ratio * area_weight
            + bottom_ratio * bottom_bias_weight
            + center_score * center_bias_weight
            + height_ratio * height_weight
        )

        if score > best_score:
            best_score = score
            best_label = label_idx

    result = np.zeros_like(binary_mask, dtype=np.uint8)
    result[labels == best_label] = 255
    return result


def remove_small_components(
    binary_mask: np.ndarray,
    min_area: int = 100,
    connectivity: int = 8
) -> np.ndarray:
    """
    移除小面積連通元件。

    Args:
        binary_mask: 輸入二值遮罩。
        min_area: 最小保留面積。
        connectivity: 連通方式，可為 4 或 8。

    Returns:
        過濾後的二值遮罩。
    """
    return filter_by_area(binary_mask, min_area, None, connectivity)
