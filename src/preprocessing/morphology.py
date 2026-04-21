"""
此模組提供形態學運算工具，用來清理二值遮罩中的小雜訊、孔洞與破碎區塊。
在候選道路區域生成後，通常會搭配 opening 與 closing 讓遮罩更平滑、連續。
"""

import cv2
import numpy as np


def get_morphology_kernel(
    kernel_size: int,
    shape: str = "rect"
) -> np.ndarray:
    """
    建立形態學運算用的 kernel。

    Args:
        kernel_size: kernel 大小。
        shape: kernel 形狀，可為 `rect`、`ellipse` 或 `cross`。

    Returns:
        OpenCV 形態學 kernel。
    """
    if shape == "rect":
        return cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )
    if shape == "ellipse":
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
    if shape == "cross":
        return cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (kernel_size, kernel_size)
        )

    raise ValueError(f"Unknown kernel shape: {shape}")


def opening(
    image: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1
) -> np.ndarray:
    """
    套用 opening 運算。

    Opening 會先侵蝕再膨脹，常用於去除小面積雜點。

    Args:
        image: 輸入二值影像。
        kernel_size: kernel 大小。
        iterations: 執行次數。

    Returns:
        opening 後的影像。
    """
    kernel = get_morphology_kernel(kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def closing(
    image: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1
) -> np.ndarray:
    """
    套用 closing 運算。

    Closing 會先膨脹再侵蝕，常用於填補小孔洞與連接近鄰區塊。

    Args:
        image: 輸入二值影像。
        kernel_size: kernel 大小。
        iterations: 執行次數。

    Returns:
        closing 後的影像。
    """
    kernel = get_morphology_kernel(kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def erosion(
    image: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1
) -> np.ndarray:
    """對輸入影像套用侵蝕運算。"""
    kernel = get_morphology_kernel(kernel_size)
    return cv2.erode(image, kernel, iterations=iterations)


def dilation(
    image: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1
) -> np.ndarray:
    """對輸入影像套用膨脹運算。"""
    kernel = get_morphology_kernel(kernel_size)
    return cv2.dilate(image, kernel, iterations=iterations)
