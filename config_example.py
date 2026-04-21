"""
此檔案提供自訂設定範例，方便依照不同場景快速調整參數。
使用者可以複製這份內容並修改數值，作為實驗或特定影像條件的設定基礎。
"""

from configs.default_config import RoadContourConfig


def get_custom_config() -> RoadContourConfig:
    """建立適合自訂場景使用的設定物件。"""
    config = RoadContourConfig()

    # 範例 1：偏重邊緣資訊，提升 Sobel 影響力。
    # 若要啟用，請取消以下註解。
    # config.feature_fusion.sobel_weight = 0.7
    # config.feature_fusion.lbp_weight = 0.3
    # config.candidate_mask.threshold = 80

    # 範例 2：偏重紋理資訊，提升 LBP 影響力。
    # config.feature_fusion.sobel_weight = 0.3
    # config.feature_fusion.lbp_weight = 0.7
    # config.lbp.radius = 2
    # config.lbp.n_points = 16

    # 範例 3：保守模式，增加平滑與遮罩清理強度。
    # config.preprocessing.blur_kernel_size = 7
    # config.candidate_mask.morph_iterations = 3
    # config.candidate_mask.morph_kernel_size = 7
    # config.contour.min_area = 200

    # 範例 4：積極模式，增加種子點與搜尋範圍。
    # config.distance_seed.max_seeds = 10
    # config.distance_seed.threshold = 10.0
    # config.bfs.neighbor_threshold = 50.0

    return config


# 針對常見使用情境整理的預設組合。
CONFIGS = {
    "sensitive": {
        "feature_fusion": {"sobel_weight": 0.7, "lbp_weight": 0.3},
        "candidate_mask": {"threshold": 80},
    },
    "texture_focus": {
        "feature_fusion": {"sobel_weight": 0.3, "lbp_weight": 0.7},
        "lbp": {"radius": 2, "n_points": 16},
    },
    "conservative": {
        "preprocessing": {"blur_kernel_size": 7},
        "candidate_mask": {"morph_iterations": 3, "morph_kernel_size": 7},
        "contour": {"min_area": 200},
    },
    "aggressive": {
        "distance_seed": {"max_seeds": 10, "threshold": 10.0},
        "bfs": {"neighbor_threshold": 50.0},
    },
}


if __name__ == "__main__":
    # 列出可參考的情境設定名稱。
    print("Available predefined configurations:")
    for name in CONFIGS.keys():
        print(f"  - {name}")

    # 示範如何建立自訂設定。
    config = get_custom_config()
    print(f"\nCustom config created with output directory: {config.output_dir}")
