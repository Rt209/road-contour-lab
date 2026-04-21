"""
此模組負責距離轉換與種子點選取。
在候選道路遮罩中，距離轉換可以幫助找出距離邊界較遠的穩定區域，
再從這些區域挑出種子點，提供 BFS 區域成長使用。
"""

import cv2
import numpy as np


def compute_distance_transform(
    binary_mask: np.ndarray,
    distance_type: str = "L2"
) -> np.ndarray:
    """
    對二值遮罩計算距離轉換。

    Args:
        binary_mask: 輸入二值遮罩。
        distance_type: 距離類型，可為 `L1`、`L2` 或 `C`。

    Returns:
        距離轉換圖。
    """
    if distance_type == "L2":
        dist_type = cv2.DIST_L2
    elif distance_type == "L1":
        dist_type = cv2.DIST_L1
    elif distance_type == "C":
        dist_type = cv2.DIST_C
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")

    # 確保輸入為標準二值格式。
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    dist_transform = cv2.distanceTransform(binary_mask, dist_type, cv2.DIST_MASK_PRECISE)
    return dist_transform


def find_local_maxima(
    distance_map: np.ndarray,
    threshold: float = 20.0
) -> list:
    """
    在距離圖中尋找局部極大值作為候選種子點。

    Args:
        distance_map: 距離轉換圖。
        threshold: 最小距離門檻。

    Returns:
        種子點座標列表，格式為 `(y, x)`。
    """
    # 先以門檻篩出距離較大的區域。
    _, binary = cv2.threshold(distance_map, threshold, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # 針對候選區域做連通元件分析。
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    seeds = []

    # 跳過背景標籤 0。
    for label in range(1, num_labels):
        # 在每個連通區內挑出距離最大的點。
        max_dist_idx = np.argmax(distance_map[labels == label])

        # 取得最大距離點的座標。
        component_coords = np.argwhere(labels == label)
        if len(component_coords) > 0:
            max_coord = component_coords[max_dist_idx]
            seeds.append(tuple(max_coord))

    _ = (stats, centroids)
    return seeds


def select_seed_points(
    distance_map: np.ndarray,
    threshold: float = 20.0,
    min_distance: float = 10.0,
    max_seeds: int = 5
) -> list:
    """
    從距離圖中挑選最適合的種子點。

    Args:
        distance_map: 距離轉換圖。
        threshold: 最小距離門檻。
        min_distance: 種子點彼此的最小距離。
        max_seeds: 最多保留的種子點數量。

    Returns:
        種子點座標列表，格式為 `(y, x)`。
    """
    seeds = find_local_maxima(distance_map, threshold)

    if len(seeds) == 0:
        # 若找不到局部峰值，退回全域最大值作為種子點。
        max_idx = np.argmax(distance_map)
        seed = np.unravel_index(max_idx, distance_map.shape)
        seeds = [seed]

    # 依照最小距離條件過濾彼此太近的種子點。
    filtered_seeds = []
    for seed in sorted(seeds, key=lambda s: distance_map[s[0], s[1]], reverse=True):
        # 檢查與既有種子點是否有足夠距離。
        is_valid = True
        for existing_seed in filtered_seeds:
            dist = np.sqrt((seed[0] - existing_seed[0]) ** 2 + (seed[1] - existing_seed[1]) ** 2)
            if dist < min_distance:
                is_valid = False
                break

        if is_valid:
            filtered_seeds.append(seed)
            if len(filtered_seeds) >= max_seeds:
                break

    return filtered_seeds if filtered_seeds else seeds[:max_seeds]


def visualize_seeds_on_distance(
    distance_map: np.ndarray,
    seeds: list
) -> np.ndarray:
    """
    將種子點標記在距離圖上，方便視覺化檢查。

    Args:
        distance_map: 距離轉換圖。
        seeds: 種子點座標列表。

    Returns:
        視覺化影像。
    """
    # 先將距離圖正規化到 0 到 255。
    dist_vis = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min() + 1e-6) * 255
    dist_vis = dist_vis.astype(np.uint8)

    # 轉成 BGR 方便畫上彩色標記。
    if len(dist_vis.shape) == 2:
        vis = cv2.cvtColor(dist_vis, cv2.COLOR_GRAY2BGR)
    else:
        vis = dist_vis.copy()

    # 把種子點畫到圖上。
    for seed in seeds:
        cv2.circle(vis, (seed[1], seed[0]), 5, (0, 255, 0), -1)
        cv2.circle(vis, (seed[1], seed[0]), 7, (0, 255, 0), 2)

    return vis
