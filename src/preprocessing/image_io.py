"""
此模組負責影像的讀取與寫出，是整條道路輪廓流程最基礎的 I/O 入口。
主要用途包含：
1. 從磁碟讀入原始影像。
2. 將處理結果輸出到指定路徑。
3. 提供統一的影像尺寸資訊查詢介面。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union


def read_image(
    image_path: Union[str, Path]
) -> np.ndarray:
    """
    從檔案讀取影像。

    Args:
        image_path: 影像檔案路徑。

    Returns:
        影像陣列，彩色影像格式為 BGR。
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    return image


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path]
) -> None:
    """
    將影像寫出到檔案。

    Args:
        image: 要儲存的影像陣列。
        output_path: 輸出檔案路徑。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise IOError(f"Failed to save image: {output_path}")


def get_image_shape(image: np.ndarray) -> tuple:
    """回傳影像尺寸，可為 `(height, width)` 或 `(height, width, channels)`。"""
    return image.shape
