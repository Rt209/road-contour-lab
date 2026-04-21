"""
此模組提供影像儲存、顯示與正規化等視覺化工具。
它主要用於中間結果檢查與除錯，方便觀察各階段輸出。
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path]
) -> None:
    """
    使用 cv2 將影像儲存到檔案。

    Args:
        image: 影像陣列。
        output_path: 輸出路徑。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 若未來需要改成 RGB 顯示輸出，可在這裡統一處理。
    if len(image.shape) == 3 and image.shape[2] == 3:
        pass

    cv2.imwrite(str(output_path), image)


def display_images(
    images: dict,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    以網格方式顯示多張影像。

    Args:
        images: 影像字典，格式為 `{title: image_array}`。
        figsize: 畫布大小。
        save_path: 若提供則將圖表另存新檔。
    """
    n_images = len(images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_images > 1 else [axes]

    for idx, (title, image) in enumerate(images.items()):
        if idx < len(axes):
            ax = axes[idx]

            # 顯示彩色圖時轉成 matplotlib 慣用的 RGB。
            if len(image.shape) == 3 and image.shape[2] == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image

            if len(display_image.shape) == 2:
                ax.imshow(display_image, cmap="gray")
            else:
                ax.imshow(display_image)

            ax.set_title(title, fontsize=10)
            ax.axis("off")

    # 將未使用的子圖隱藏。
    for idx in range(n_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.close()


def normalize_image(
    image: np.ndarray,
    dtype: type = np.uint8
) -> np.ndarray:
    """
    將影像正規化到 0 到 255。

    Args:
        image: 輸入影像。
        dtype: 輸出資料型別。

    Returns:
        正規化後的影像。
    """
    if image.size == 0:
        return image

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        return np.zeros_like(image, dtype=dtype)

    normalized = (image - min_val) / (max_val - min_val) * 255
    return normalized.astype(dtype)
