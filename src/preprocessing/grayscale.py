"""
此模組負責將輸入影像統一轉換成灰階格式。
道路輪廓流程中的 Sobel、LBP 與部分分割步驟都以灰階影像作為輸入，
因此這個模組扮演前處理中的格式標準化角色。
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    將影像轉成灰階。

    Args:
        image: 輸入影像，可為 BGR、BGRA 或灰階。

    Returns:
        灰階影像。
    """
    if len(image.shape) == 2:
        # 已經是灰階影像時直接回傳。
        return image

    if image.shape[2] == 3:
        # 三通道彩色影像。
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        # 四通道影像會先移除 alpha，再轉灰階。
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image channels: {image.shape[2]}")
