"""
此模組負責以 Sobel 算子擷取影像中的梯度資訊。
它會計算水平與垂直方向的邊緣變化，進一步得到梯度強度與方向，
作為道路邊界偵測的重要特徵來源。
"""

import cv2
import numpy as np


def compute_sobel_gradient(
    image: np.ndarray,
    kernel_size: int = 3
) -> tuple:
    """
    計算影像在 x 與 y 方向的 Sobel 梯度。

    Args:
        image: 輸入灰階影像。
        kernel_size: Sobel kernel 大小，可為 1、3、5 或 7。

    Returns:
        `(gradient_x, gradient_y)`。
    """
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=kernel_size)

    return grad_x, grad_y


def compute_gradient_magnitude(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    由 x 與 y 方向梯度計算梯度強度。

    Args:
        grad_x: x 方向梯度。
        grad_y: y 方向梯度。
        normalize: 是否正規化到 0 到 255。

    Returns:
        梯度強度圖。
    """
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    if normalize:
        if magnitude.max() > magnitude.min():
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255
        magnitude = magnitude.astype(np.uint8)

    return magnitude


def compute_gradient_angle(
    grad_x: np.ndarray,
    grad_y: np.ndarray
) -> np.ndarray:
    """
    計算梯度方向角。

    Args:
        grad_x: x 方向梯度。
        grad_y: y 方向梯度。

    Returns:
        以弧度表示的方向角。
    """
    return np.arctan2(grad_y, grad_x)


def sobel_feature_extraction(
    image: np.ndarray,
    kernel_size: int = 3
) -> dict:
    """
    執行完整的 Sobel 特徵擷取流程。

    Args:
        image: 輸入灰階影像。
        kernel_size: Sobel kernel 大小。

    Returns:
        包含 `magnitude`、`angle`、`grad_x`、`grad_y` 的字典。
    """
    grad_x, grad_y = compute_sobel_gradient(image, kernel_size)
    magnitude = compute_gradient_magnitude(grad_x, grad_y, normalize=True)
    angle = compute_gradient_angle(grad_x, grad_y)

    return {
        "magnitude": magnitude,
        "angle": angle,
        "grad_x": grad_x,
        "grad_y": grad_y,
    }
