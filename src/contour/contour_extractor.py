"""
此模組負責從最終道路區域遮罩中擷取 contour，並計算輪廓幾何性質。
它是從分割結果走向可視化與後續分析的重要橋接模組。
"""

import cv2
import numpy as np


def extract_contours(
    binary_mask: np.ndarray,
    min_area: int = 100
) -> list:
    """
    從二值遮罩中擷取輪廓。

    Args:
        binary_mask: 輸入二值遮罩。
        min_area: 最小輪廓面積。

    Returns:
        輪廓列表，每個輪廓為座標陣列。
    """
    # 確保輸入為標準二值格式。
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # 依照面積門檻過濾輪廓。
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered_contours.append(contour)

    _ = hierarchy
    return filtered_contours


def get_contour_properties(contour: np.ndarray) -> dict:
    """
    計算單一輪廓的幾何屬性。

    Args:
        contour: 輪廓座標陣列。

    Returns:
        輪廓屬性字典。
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # 計算外接矩形。
    x, y, w, h = cv2.boundingRect(contour)

    # 若點數足夠，額外擬合橢圓。
    ellipse = None
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)

    # 透過 moments 計算輪廓中心點。
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    return {
        "area": area,
        "perimeter": perimeter,
        "bbox": (x, y, w, h),
        "centroid": (cx, cy),
        "ellipse": ellipse,
        "aspect_ratio": float(w) / h if h > 0 else 0,
        "circularity": 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,
    }


def approximate_contour(
    contour: np.ndarray,
    epsilon_ratio: float = 0.02
) -> np.ndarray:
    """
    使用 Douglas-Peucker 演算法近似輪廓。

    Args:
        contour: 輸入輪廓。
        epsilon_ratio: 相對於周長的近似比例。

    Returns:
        近似後的輪廓。
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * perimeter
    approximated = cv2.approxPolyDP(contour, epsilon, True)
    return approximated


def filter_contours_by_property(
    contours: list,
    min_area: int = None,
    max_area: int = None,
    min_perimeter: float = None,
    max_perimeter: float = None
) -> list:
    """
    依照指定屬性過濾輪廓。

    Args:
        contours: 輪廓列表。
        min_area: 最小面積。
        max_area: 最大面積。
        min_perimeter: 最小周長。
        max_perimeter: 最大周長。

    Returns:
        過濾後的輪廓列表。
    """
    filtered = []

    for contour in contours:
        props = get_contour_properties(contour)

        # 檢查面積條件。
        if min_area and props["area"] < min_area:
            continue
        if max_area and props["area"] > max_area:
            continue

        # 檢查周長條件。
        if min_perimeter and props["perimeter"] < min_perimeter:
            continue
        if max_perimeter and props["perimeter"] > max_perimeter:
            continue

        filtered.append(contour)

    return filtered
