"""
此模組負責將 Sobel 邊緣特徵與 LBP 紋理特徵融合成單一特徵圖。
融合後的結果會交由後續候選遮罩與分割步驟使用，
目的是同時保留道路邊界與表面紋理資訊。
"""

import numpy as np


def normalize_feature(
    feature: np.ndarray,
    target_range: tuple = (0, 255)
) -> np.ndarray:
    """
    將特徵值正規化到指定範圍。

    Args:
        feature: 輸入特徵圖。
        target_range: 目標範圍 `(min, max)`。

    Returns:
        正規化後的特徵圖。
    """
    if feature.size == 0:
        return feature

    min_val = np.min(feature)
    max_val = np.max(feature)

    if max_val == min_val:
        return np.zeros_like(feature, dtype=np.float32)

    normalized = (feature - min_val) / (max_val - min_val)
    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]

    return normalized.astype(np.float32)


def fuse_features_weighted(
    sobel_magnitude: np.ndarray,
    lbp_map: np.ndarray,
    sobel_weight: float = 0.5,
    lbp_weight: float = 0.5
) -> np.ndarray:
    """
    以加權平均方式融合 Sobel 與 LBP。

    Args:
        sobel_magnitude: Sobel 強度圖。
        lbp_map: LBP 特徵圖。
        sobel_weight: Sobel 權重。
        lbp_weight: LBP 權重。

    Returns:
        融合後的特徵圖。
    """
    # 先把兩種特徵拉到相同數值範圍。
    sobel_norm = normalize_feature(sobel_magnitude, (0, 255))
    lbp_norm = normalize_feature(lbp_map, (0, 255))

    # 依照設定權重進行融合。
    total_weight = sobel_weight + lbp_weight
    fused = (sobel_norm * sobel_weight + lbp_norm * lbp_weight) / total_weight

    return fused.astype(np.uint8)


def fuse_features_product(
    sobel_magnitude: np.ndarray,
    lbp_map: np.ndarray
) -> np.ndarray:
    """
    以逐元素相乘的方式融合特徵。

    Args:
        sobel_magnitude: Sobel 強度圖。
        lbp_map: LBP 特徵圖。

    Returns:
        融合後的特徵圖。
    """
    sobel_norm = normalize_feature(sobel_magnitude, (0, 1))
    lbp_norm = normalize_feature(lbp_map, (0, 1))
    fused = sobel_norm * lbp_norm * 255
    return fused.astype(np.uint8)


def fuse_features_max(
    sobel_magnitude: np.ndarray,
    lbp_map: np.ndarray
) -> np.ndarray:
    """
    以逐像素最大值方式融合特徵。

    Args:
        sobel_magnitude: Sobel 強度圖。
        lbp_map: LBP 特徵圖。

    Returns:
        融合後的特徵圖。
    """
    sobel_norm = normalize_feature(sobel_magnitude, (0, 255))
    lbp_norm = normalize_feature(lbp_map, (0, 255))
    fused = np.maximum(sobel_norm, lbp_norm)
    return fused.astype(np.uint8)


def apply_vertical_position_prior(
    feature_map: np.ndarray,
    strength: float = 0.5,
    start_ratio: float = 0.35
) -> np.ndarray:
    """
    套用由上而下遞增的垂直位置先驗，降低天空與遠景對候選區域的干擾。
    Args:
        feature_map: 融合後的特徵圖。
        strength: 位置先驗強度，數值越大越偏向保留畫面下半部。
        start_ratio: 先驗開始明顯增強的位置比例，0 代表最上方、1 代表最下方。
    Returns:
        套用位置先驗後的特徵圖。
    """
    if feature_map.size == 0:
        return feature_map

    strength = float(np.clip(strength, 0.0, 1.0))
    start_ratio = float(np.clip(start_ratio, 0.0, 0.95))

    height = feature_map.shape[0]
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(-1, 1)
    vertical_prior = np.clip((rows - start_ratio) / (1.0 - start_ratio), 0.0, 1.0)
    weight_map = (1.0 - strength) + strength * vertical_prior

    weighted_feature = feature_map.astype(np.float32) * weight_map
    return np.clip(weighted_feature, 0, 255).astype(np.uint8)


def fuse_features(
    sobel_magnitude: np.ndarray,
    lbp_map: np.ndarray,
    method: str = "weighted",
    **kwargs
) -> np.ndarray:
    """
    依指定方式融合 Sobel 與 LBP 特徵。

    Args:
        sobel_magnitude: Sobel 強度圖。
        lbp_map: LBP 特徵圖。
        method: 融合方法，可為 `weighted`、`product`、`max`。
        **kwargs: 對應融合方法的額外參數。

    Returns:
        融合後的特徵圖。
    """
    if method == "weighted":
        return fuse_features_weighted(
            sobel_magnitude,
            lbp_map,
            kwargs.get("sobel_weight", 0.5),
            kwargs.get("lbp_weight", 0.5),
        )
    if method == "product":
        return fuse_features_product(sobel_magnitude, lbp_map)
    if method == "max":
        return fuse_features_max(sobel_magnitude, lbp_map)

    raise ValueError(f"Unknown fusion method: {method}")
