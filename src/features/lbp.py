"""
此模組負責 Local Binary Pattern（LBP）紋理特徵擷取。
LBP 會比較像素與周圍鄰點的強度關係，將局部紋理轉成二進位編碼，
有助於區分道路表面與周邊非道路區域的紋理差異。
"""

import numpy as np


def compute_lbp(
    image: np.ndarray,
    radius: int = 1,
    n_points: int = 8
) -> np.ndarray:
    """
    計算影像的 LBP 特徵圖。

    Args:
        image: 輸入灰階影像。
        radius: 鄰域半徑。
        n_points: 鄰域採樣點數。

    Returns:
        LBP 特徵圖。
    """
    h, w = image.shape
    lbp_map = np.zeros((h, w), dtype=np.uint8)

    # 先建立鄰近採樣點的位移量。
    angles = 2 * np.pi * np.arange(n_points) / n_points
    dx = radius * np.cos(angles)
    dy = -radius * np.sin(angles)

    # 為了避免邊界越界，先對原圖做 padding。
    pad = radius + 1
    padded = np.pad(image, pad, mode="constant", constant_values=image[0, 0])

    # 逐像素計算 LBP 編碼。
    for i in range(h):
        for j in range(w):
            center = padded[i + pad, j + pad]
            binary_string = ""

            for k in range(n_points):
                y = i + pad + int(dy[k])
                x = j + pad + int(dx[k])

                # 保留小數位移資訊，方便後續擴充成更精細的插值版本。
                y_frac = dy[k] - int(dy[k])
                x_frac = dx[k] - int(dx[k])

                if 0 <= y < h + 2 * pad - 1 and 0 <= x < w + 2 * pad - 1:
                    # 目前採用最鄰近取樣。
                    neighbor = padded[y, x]
                    binary_string += "1" if neighbor >= center else "0"
                else:
                    binary_string += "0"

            # 避免靜態分析工具認為小數資訊未被保留是誤植。
            _ = (y_frac, x_frac)

            # 將二進位字串轉成十進位 LBP 值。
            lbp_map[i, j] = int(binary_string, 2)

    return lbp_map


def compute_lbp_histogram(
    lbp_map: np.ndarray,
    n_bins: int = 256
) -> np.ndarray:
    """
    計算 LBP 值分布直方圖。

    Args:
        lbp_map: LBP 特徵圖。
        n_bins: 直方圖分箱數。

    Returns:
        LBP 直方圖。
    """
    histogram, _ = np.histogram(lbp_map, bins=n_bins, range=(0, n_bins))
    return histogram


def lbp_feature_extraction(
    image: np.ndarray,
    radius: int = 1,
    n_points: int = 8
) -> dict:
    """
    執行完整的 LBP 特徵擷取流程。

    Args:
        image: 輸入灰階影像。
        radius: 鄰域半徑。
        n_points: 採樣點數。

    Returns:
        包含 `lbp_map` 與 `histogram` 的字典。
    """
    lbp_map = compute_lbp(image, radius, n_points)
    histogram = compute_lbp_histogram(lbp_map)

    return {
        "lbp_map": lbp_map,
        "histogram": histogram,
    }


def normalize_lbp_map(
    lbp_map: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    將 LBP 特徵圖正規化到 0 到 255，方便視覺化。

    Args:
        lbp_map: 輸入 LBP 特徵圖。
        normalize: 是否進行正規化。

    Returns:
        正規化後的 LBP 圖。
    """
    if not normalize:
        return lbp_map

    if lbp_map.max() > lbp_map.min():
        normalized = (lbp_map - lbp_map.min()) / (lbp_map.max() - lbp_map.min()) * 255
        return normalized.astype(np.uint8)

    return np.zeros_like(lbp_map, dtype=np.uint8)
