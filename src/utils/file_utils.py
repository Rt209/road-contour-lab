"""
此模組提供輸出資料夾與檔案路徑相關的輔助工具。
它的目的在於統一路徑建立邏輯，避免主流程散落重複的檔案處理程式碼。
"""

from pathlib import Path
from typing import List, Union


def ensure_output_dir(output_dir: Union[str, Path]) -> Path:
    """
    確保輸出資料夾存在。

    Args:
        output_dir: 輸出資料夾路徑。

    Returns:
        `Path` 物件。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_path(
    output_dir: Union[str, Path],
    filename: str,
    extension: str = ".png"
) -> Path:
    """
    組合輸出檔案完整路徑。

    Args:
        output_dir: 輸出資料夾。
        filename: 不含副檔名的檔名。
        extension: 副檔名，預設為 `.png`。

    Returns:
        完整輸出路徑。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not extension.startswith("."):
        extension = "." + extension

    return output_dir / (filename + extension)


def list_image_files(
    input_dir: Union[str, Path],
    extensions: List[str] = None
) -> List[Path]:
    """
    列出資料夾中的影像檔案。

    Args:
        input_dir: 輸入資料夾。
        extensions: 要搜尋的副檔名列表，預設為 jpg、png、jpeg。

    Returns:
        影像檔案路徑列表。
    """
    if extensions is None:
        extensions = [".jpg", ".png", ".jpeg"]

    input_dir = Path(input_dir)
    image_files = []

    for ext in extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    return sorted(list(set(image_files)))
