"""
此模組負責影像平滑與降噪處理。
在道路輪廓擷取前，先透過模糊降低局部雜訊，能讓後續的邊緣與紋理特徵更穩定。
目前提供 Gaussian Blur 與雙邊濾波兩種常見方法。
"""

import cv2
import numpy as np


def gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    對影像套用 Gaussian Blur。

    Args:
        image: 輸入影像。
        kernel_size: 核心大小，必須為奇數。
        sigma: 高斯分布標準差。

    Returns:
        模糊後的影像。
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def bilateral_filter(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    套用雙邊濾波。

    Args:
        image: 輸入影像。
        diameter: 鄰域直徑。
        sigma_color: 色彩空間的 sigma。
        sigma_space: 座標空間的 sigma。

    Returns:
        濾波後的影像。
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
