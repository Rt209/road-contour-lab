"""
此模組負責將輪廓結果畫回影像，提供後續檢查與展示用的視覺化輸出。
除了基本 contour 疊圖，也提供外接框、中心點與索引標記等輔助資訊。
"""

import cv2
import numpy as np


def draw_contours(
    image: np.ndarray,
    contours: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    copy: bool = True
) -> np.ndarray:
    """
    在影像上繪製多個輪廓。

    Args:
        image: 輸入影像。
        contours: 輪廓列表。
        color: 輪廓顏色，格式為 BGR。
        thickness: 線條粗細，`-1` 代表填滿。
        copy: 是否先複製影像再繪製。

    Returns:
        已繪上輪廓的影像。
    """
    result = image.copy() if copy else image
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def draw_contour_with_info(
    image: np.ndarray,
    contour: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    繪製單一輪廓並附上其幾何資訊。

    Args:
        image: 輸入影像。
        contour: 單一輪廓。
        color: 輪廓顏色，格式為 BGR。
        thickness: 輪廓線條粗細。

    Returns:
        已標註的影像。
    """
    from src.contour.contour_extractor import get_contour_properties

    result = image.copy()

    # 畫出輪廓本體。
    cv2.drawContours(result, [contour], 0, color, thickness)

    # 取得輪廓幾何屬性。
    props = get_contour_properties(contour)

    # 標示中心點。
    cx, cy = props["centroid"]
    cv2.circle(result, (cx, cy), 5, (0, 255, 255), -1)

    # 標示外接框。
    x, y, w, h = props["bbox"]
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 顯示面積資訊。
    cv2.putText(
        result,
        f"Area: {props['area']:.0f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    return result


def visualize_contours_grid(
    image: np.ndarray,
    contours: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    以網格式資訊視角標示多個輪廓。

    Args:
        image: 輸入影像。
        contours: 輪廓列表。
        color: 輪廓顏色，格式為 BGR。
        thickness: 輪廓線條粗細。

    Returns:
        已標註的影像。
    """
    result = image.copy()

    for idx, contour in enumerate(contours):
        # 畫出輪廓。
        cv2.drawContours(result, [contour], 0, color, thickness)

        # 取得並畫出外接框。
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # 標示輪廓索引。
        cv2.putText(
            result,
            str(idx),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return result


def create_contour_mask(
    shape: tuple,
    contours: list
) -> np.ndarray:
    """
    由輪廓列表建立二值遮罩。

    Args:
        shape: 輸出尺寸 `(height, width)`。
        contours: 輪廓列表。

    Returns:
        二值遮罩。
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return mask
