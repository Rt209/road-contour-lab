"""
此模組負責從融合後的特徵圖產生候選道路遮罩。
它會先做閾值化，再透過形態學運算清理雜訊，
讓後續的距離轉換與區域成長能在更穩定的二值結果上進行。
"""

import cv2
import numpy as np
from src.preprocessing.morphology import opening, closing


def create_candidate_mask(
    fused_feature: np.ndarray,
    threshold: float = 100.0
) -> np.ndarray:
    """
    由融合特徵圖建立二值候選遮罩。

    Args:
        fused_feature: 融合後特徵圖。
        threshold: 二值化門檻值。

    Returns:
        二值遮罩。
    """
    _, mask = cv2.threshold(fused_feature, threshold, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def clean_candidate_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 2,
    opening_enabled: bool = True,
    closing_enabled: bool = True
) -> np.ndarray:
    """
    以形態學方法清理候選遮罩。

    Args:
        mask: 輸入二值遮罩。
        kernel_size: 形態學 kernel 大小。
        iterations: 執行次數。
        opening_enabled: 是否套用 opening。
        closing_enabled: 是否套用 closing。

    Returns:
        清理後的遮罩。
    """
    result = mask.copy()

    if opening_enabled:
        result = opening(result, kernel_size, iterations)

    if closing_enabled:
        result = closing(result, kernel_size, iterations)

    return result


def generate_candidate_mask(
    fused_feature: np.ndarray,
    threshold: float = 100.0,
    kernel_size: int = 5,
    iterations: int = 2,
    opening_enabled: bool = True,
    closing_enabled: bool = True
) -> np.ndarray:
    """
    產生並清理候選道路遮罩。

    Args:
        fused_feature: 融合後特徵圖。
        threshold: 二值化門檻值。
        kernel_size: 形態學 kernel 大小。
        iterations: 形態學執行次數。
        opening_enabled: 是否套用 opening。
        closing_enabled: 是否套用 closing。

    Returns:
        清理完成的二值遮罩。
    """
    # 先建立初始候選遮罩。
    mask = create_candidate_mask(fused_feature, threshold)

    # 再以形態學方法清理雜訊與破洞。
    cleaned_mask = clean_candidate_mask(
        mask,
        kernel_size=kernel_size,
        iterations=iterations,
        opening_enabled=opening_enabled,
        closing_enabled=closing_enabled,
    )

    return cleaned_mask
