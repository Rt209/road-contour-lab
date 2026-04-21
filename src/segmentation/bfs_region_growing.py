"""
此模組使用 BFS 進行區域成長，從種子點開始向外擴張候選道路區域。
它主要依賴候選遮罩與連通關係，把與種子點相連的可行區域保留下來，
作為較完整的道路分割結果。
"""

from collections import deque
import cv2
import numpy as np


def bfs_region_growing(
    candidate_mask: np.ndarray,
    seed_point: tuple,
    neighbor_threshold: float = 30.0,
    connectivity: int = 8
) -> np.ndarray:
    """
    由單一種子點執行 BFS 區域成長。

    Args:
        candidate_mask: 候選二值遮罩。
        seed_point: 種子點座標 `(y, x)`。
        neighbor_threshold: 鄰點門檻，保留擴充欄位，目前遮罩版流程未使用。
        connectivity: 連通方式，可為 4 或 8。

    Returns:
        區域成長後的二值遮罩。
    """
    h, w = candidate_mask.shape
    region = np.zeros_like(candidate_mask, dtype=np.uint8)

    # 依照連通方式定義鄰點方向。
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]

    # 初始化 BFS 佇列與已拜訪集合。
    queue = deque([seed_point])
    visited = {seed_point}
    region[seed_point[0], seed_point[1]] = 255

    while queue:
        y, x = queue.popleft()

        # 逐一檢查目前點的所有鄰居。
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx

            # 先確認座標仍在影像範圍內。
            if 0 <= ny < h and 0 <= nx < w:
                if (ny, nx) not in visited and candidate_mask[ny, nx] > 0:
                    visited.add((ny, nx))
                    region[ny, nx] = 255
                    queue.append((ny, nx))

    _ = neighbor_threshold
    return region


def multi_seed_region_growing(
    candidate_mask: np.ndarray,
    seed_points: list,
    neighbor_threshold: float = 30.0,
    connectivity: int = 8
) -> np.ndarray:
    """
    由多個種子點執行區域成長，並合併結果。

    Args:
        candidate_mask: 候選二值遮罩。
        seed_points: 種子點座標列表。
        neighbor_threshold: 鄰點門檻。
        connectivity: 連通方式，可為 4 或 8。

    Returns:
        合併後的區域遮罩。
    """
    region = np.zeros_like(candidate_mask, dtype=np.uint8)

    # 逐個種子點成長，最後做聯集。
    for seed in seed_points:
        if candidate_mask[seed[0], seed[1]] > 0:
            single_region = bfs_region_growing(
                candidate_mask,
                seed,
                neighbor_threshold,
                connectivity,
            )
            region = np.maximum(region, single_region)

    return region


def watershed_segmentation(
    image: np.ndarray,
    seed_points: list
) -> np.ndarray:
    """
    由種子點啟動 watershed 分割。

    Args:
        image: 輸入影像，可為灰階或彩色。
        seed_points: 種子點座標列表。

    Returns:
        Watershed 標籤圖。
    """
    if len(image.shape) == 2:
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color_image = image.copy()

    # 建立 watershed 所需的 marker 圖。
    markers = np.zeros(image.shape[:2], dtype=np.int32)

    for idx, seed in enumerate(seed_points, start=1):
        markers[seed[0], seed[1]] = idx

    # 執行 watershed 分割。
    markers = cv2.watershed(color_image, markers)
    return markers
