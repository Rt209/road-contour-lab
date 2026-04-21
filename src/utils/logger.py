"""
此模組提供專案使用的 logging 工具。
它會建立終端輸出與可選的檔案輸出，讓 pipeline 執行過程更容易追蹤與除錯。
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    建立含終端輸出與可選檔案輸出的 logger。

    Args:
        name: logger 名稱。
        log_file: 可選的 log 檔路徑。
        level: logging 等級。

    Returns:
        已設定完成的 logger。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 建立終端輸出 handler。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 若有提供路徑，額外建立檔案輸出 handler。
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """取得既有 logger，若不存在則由 logging 系統建立。"""
    return logging.getLogger(name)
