"""
此模組集中管理道路輪廓擷取流程的所有預設參數。
透過 dataclass 分層定義前處理、特徵擷取、分割、輪廓與輸出設定，
可以讓主流程更容易維護，也方便之後擴充與調參。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessingConfig:
    """前處理相關參數。"""

    blur_kernel_size: int = 5
    blur_sigma: float = 1.0


@dataclass
class SobelConfig:
    """Sobel 特徵擷取參數。"""

    kernel_size: int = 3
    threshold: float = 50.0  # 邊緣強度門檻。


@dataclass
class LBPConfig:
    """LBP 特徵擷取參數。"""

    radius: int = 1
    n_points: int = 8
    threshold: float = 50.0  # LBP 直方圖門檻。


@dataclass
class FeatureFusionConfig:
    """特徵融合參數。"""

    sobel_weight: float = 0.78
    lbp_weight: float = 0.22
    fusion_threshold: float = 100.0
    vertical_prior_enabled: bool = True
    vertical_prior_strength: float = 0.32
    vertical_prior_start_ratio: float = 0.25


@dataclass
class CandidateMaskConfig:
    """候選遮罩生成參數。"""

    threshold: float = 90.0
    morph_kernel_size: int = 13
    morph_iterations: int = 4
    opening_enabled: bool = False
    closing_enabled: bool = True


@dataclass
class DistanceSeedConfig:
    """距離轉換與種子點選取參數。"""

    threshold: float = 4.0  # 選取種子點時使用的最小距離門檻。
    min_distance: float = 10.0  # 種子點彼此之間的最小距離。
    max_seeds: int = 20  # 最多保留的種子點數量。


@dataclass
class BFSConfig:
    """BFS 區域成長參數。"""

    neighbor_threshold: float = 30.0  # 保留的鄰點相似度門檻。
    connectivity: int = 8  # 可為 4 或 8。


@dataclass
class ConnectedComponentsConfig:
    """連通元件分析參數。"""

    min_area: int = 100  # 最小保留面積。
    connectivity: int = 8
    selection_strategy: str = "road_prior"
    bottom_bias_weight: float = 2.5
    center_bias_weight: float = 0.7
    area_weight: float = 1.0
    height_weight: float = 0.9


@dataclass
class RoadGeometryConfig:
    """道路幾何修正參數。"""

    enabled: bool = True
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    hough_threshold: int = 60
    min_line_length: int = 120
    max_line_gap: int = 40
    roi_top_y_ratio: float = 0.42
    roi_top_width_ratio: float = 0.16
    roi_bottom_width_ratio: float = 0.88
    line_top_offset_ratio: float = -0.02
    bottom_left_min_ratio: float = 0.08
    bottom_left_max_ratio: float = 0.34
    bottom_right_min_ratio: float = 0.66
    bottom_right_max_ratio: float = 0.92
    top_width_scale: float = 1.55
    top_width_min_ratio: float = 0.07
    top_width_max_ratio: float = 0.28
    top_y_min_ratio: float = 0.42
    top_y_max_ratio: float = 0.52
    bottom_expand_ratio: float = 0.18
    top_expand_ratio: float = 0.42
    outer_bottom_offset_ratio: float = 0.24
    outer_top_offset_ratio: float = 0.14
    geometry_region_min_overlap: float = 0.03
    geometry_confidence_threshold: float = 0.45
    boundary_support_margin_x: int = 36
    boundary_support_margin_y: int = 9
    boundary_smoothing_window: int = 21
    geometry_extension_limit: int = 90
    min_top_width_ratio: float = 0.015
    left_boundary_search_margin: int = 64
    right_boundary_search_margin: int = 56
    left_boundary_max_delta: int = 34
    right_boundary_max_delta: int = 30
    boundary_width_alpha: float = 1.0


@dataclass
class CurveRoadConfig:
    """Configuration for curve-aware road mask refinement."""

    enabled: bool = True
    min_row_coverage_ratio: float = 0.025
    bottom_search_ratio: float = 0.28
    center_bias_weight: float = 0.35
    width_change_weight: float = 0.25
    max_center_shift_ratio: float = 0.12
    max_gap_rows: int = 18
    smoothing_window: int = 31
    width_alpha: float = 1.02
    top_width_alpha: float = 0.92
    min_width_ratio: float = 0.035
    close_kernel_size: int = 9
    close_iterations: int = 2


@dataclass
class ContourConfig:
    """輪廓擷取與繪製參數。"""

    min_area: int = 1500
    contour_color: tuple = (0, 255, 0)
    contour_thickness: int = 2
    contour_method: str = "approxDP"  # 可為 `exact` 或 `approxDP`。
    approximation_epsilon_ratio: float = 0.02


@dataclass
class OutputConfig:
    """輸出相關參數。"""

    save_intermediate: bool = True
    save_formats: list = None

    def __post_init__(self):
        if self.save_formats is None:
            self.save_formats = ["png"]


@dataclass
class RoadContourConfig:
    """整合所有子設定的主設定類別。"""

    pipeline_version: str = "v1"
    preprocessing: PreprocessingConfig = None
    sobel: SobelConfig = None
    lbp: LBPConfig = None
    feature_fusion: FeatureFusionConfig = None
    candidate_mask: CandidateMaskConfig = None
    distance_seed: DistanceSeedConfig = None
    bfs: BFSConfig = None
    connected_components: ConnectedComponentsConfig = None
    road_geometry: RoadGeometryConfig = None
    curve_road: CurveRoadConfig = None
    contour: ContourConfig = None
    output: OutputConfig = None

    # 路徑設定。
    input_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    def __post_init__(self):
        """若使用者未提供子設定，則自動填入預設值。"""
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.sobel is None:
            self.sobel = SobelConfig()
        if self.lbp is None:
            self.lbp = LBPConfig()
        if self.feature_fusion is None:
            self.feature_fusion = FeatureFusionConfig()
        if self.candidate_mask is None:
            self.candidate_mask = CandidateMaskConfig()
        if self.distance_seed is None:
            self.distance_seed = DistanceSeedConfig()
        if self.bfs is None:
            self.bfs = BFSConfig()
        if self.connected_components is None:
            self.connected_components = ConnectedComponentsConfig()
        if self.road_geometry is None:
            self.road_geometry = RoadGeometryConfig()
        if self.curve_road is None:
            self.curve_road = CurveRoadConfig()
        if self.contour is None:
            self.contour = ContourConfig()
        if self.output is None:
            self.output = OutputConfig()

        # 設定預設輸出資料夾。
        if self.output_dir is None:
            self.output_dir = Path("data/output")


def get_default_config() -> RoadContourConfig:
    """建立並回傳預設設定物件。"""
    return RoadContourConfig()
