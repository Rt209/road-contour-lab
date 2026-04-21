"""
此模組是整個道路輪廓擷取系統的主流程協調器。
它會依序串接前處理、特徵擷取、分割、連通元件過濾與輪廓視覺化，
並集中管理所有中間結果與最終輸出。
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from configs.default_config import RoadContourConfig
from src.contour.contour_extractor import extract_contours
from src.contour.contour_visualizer import draw_contours
from src.features.feature_fusion import (
    apply_vertical_position_prior,
    fuse_features,
)
from src.features.lbp import lbp_feature_extraction, normalize_lbp_map
from src.features.sobel import sobel_feature_extraction
from src.preprocessing.blur import gaussian_blur
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.image_io import read_image, save_image
from src.segmentation.bfs_region_growing import multi_seed_region_growing
from src.segmentation.candidate_mask import generate_candidate_mask
from src.segmentation.connected_components import (
    keep_largest_component,
    keep_road_like_component,
)
from src.segmentation.distance_seed import (
    compute_distance_transform,
    select_seed_points,
    visualize_seeds_on_distance,
)
from src.segmentation.mask_fusion import (
    compute_overlap_ratio,
    create_overlap_visualization,
    repair_region_with_geometry,
)
from src.segmentation.road_geometry import detect_road_geometry_mask
from src.utils.logger import setup_logger
from src.utils.plotting import normalize_image


class RoadContourPipeline:
    """道路輪廓擷取主流程類別。"""

    def __init__(self, config: RoadContourConfig):
        """
        初始化 pipeline。

        Args:
            config: 流程設定物件。
        """
        self.config = config
        self.logger = setup_logger(__name__)

        # 用來保存各階段中間結果，方便後續輸出與除錯。
        self.results = {}

    def run(self, input_path: str) -> Dict:
        """
        執行完整道路輪廓擷取流程。

        Args:
            input_path: 輸入影像路徑。

        Returns:
            各階段結果字典。
        """
        self.logger.info("Starting road contour extraction pipeline")
        self.logger.info(f"Input: {input_path}")

        # Step 1: 讀圖、轉灰階與模糊前處理。
        self.logger.info("Step 1: Reading and preprocessing image...")
        original = read_image(input_path)
        gray = to_grayscale(original)
        blurred = gaussian_blur(
            gray,
            kernel_size=self.config.preprocessing.blur_kernel_size,
            sigma=self.config.preprocessing.blur_sigma,
        )

        self.results["original"] = original
        self.results["grayscale"] = gray
        self.results["blurred"] = blurred
        self.results["debug_01_gray"] = gray
        self.results["debug_02_blur"] = blurred

        # Step 2: 擷取 Sobel 邊緣特徵。
        self.logger.info("Step 2: Sobel feature extraction...")
        sobel_features = sobel_feature_extraction(
            blurred,
            kernel_size=self.config.sobel.kernel_size,
        )
        sobel_magnitude = sobel_features["magnitude"]
        self.results["sobel_magnitude"] = sobel_magnitude
        self.results["debug_03_sobel_magnitude"] = sobel_magnitude

        # Step 3: 擷取 LBP 紋理特徵。
        self.logger.info("Step 3: LBP feature extraction...")
        lbp_features = lbp_feature_extraction(
            blurred,
            radius=self.config.lbp.radius,
            n_points=self.config.lbp.n_points,
        )
        lbp_map = normalize_lbp_map(lbp_features["lbp_map"])
        self.results["lbp_map"] = lbp_map
        self.results["debug_04_lbp_map"] = lbp_map

        # Step 4: 融合邊緣與紋理資訊。
        self.logger.info("Step 4: Feature fusion...")
        fused_features = fuse_features(
            sobel_magnitude,
            lbp_map,
            method="weighted",
            sobel_weight=self.config.feature_fusion.sobel_weight,
            lbp_weight=self.config.feature_fusion.lbp_weight,
        )

        if self.config.feature_fusion.vertical_prior_enabled:
            fused_features = apply_vertical_position_prior(
                fused_features,
                strength=self.config.feature_fusion.vertical_prior_strength,
                start_ratio=self.config.feature_fusion.vertical_prior_start_ratio,
            )

        self.results["fused_features"] = fused_features
        self.results["debug_05_fused_feature_map"] = fused_features

        # Step 5: 由融合特徵建立候選道路遮罩。
        self.logger.info("Step 5: Generating candidate mask...")
        candidate_mask = generate_candidate_mask(
            fused_features,
            threshold=self.config.candidate_mask.threshold,
            kernel_size=self.config.candidate_mask.morph_kernel_size,
            iterations=self.config.candidate_mask.morph_iterations,
            opening_enabled=self.config.candidate_mask.opening_enabled,
            closing_enabled=self.config.candidate_mask.closing_enabled,
        )

        candidate_mask_raw = (
            (fused_features > self.config.candidate_mask.threshold).astype(np.uint8) * 255
        )
        self.results["candidate_mask"] = candidate_mask
        self.results["candidate_mask_raw"] = candidate_mask_raw
        self.results["debug_06_candidate_mask_raw"] = candidate_mask_raw
        self.results["debug_07_candidate_mask_morph"] = candidate_mask

        # Step 6: 計算距離轉換並挑選種子點。
        self.logger.info("Step 6: Distance transform and seed selection...")
        distance_map = compute_distance_transform(candidate_mask, distance_type="L2")
        self.results["distance_transform"] = distance_map
        self.results["debug_08_distance_transform"] = distance_map

        seed_points = select_seed_points(
            distance_map,
            threshold=self.config.distance_seed.threshold,
            min_distance=self.config.distance_seed.min_distance,
            max_seeds=self.config.distance_seed.max_seeds,
        )

        self.logger.info(f"Found {len(seed_points)} seed points")
        seed_vis = visualize_seeds_on_distance(distance_map, seed_points)
        self.results["seed_visualization"] = seed_vis
        self.results["debug_09_seed_points"] = seed_vis

        # Step 7: 以種子點做 BFS 區域成長。
        self.logger.info("Step 7: BFS region growing...")
        if len(seed_points) == 0:
            self.logger.warning("No seed points found, using candidate mask directly")
            bfs_region = candidate_mask.copy()
        else:
            bfs_region = multi_seed_region_growing(
                candidate_mask,
                seed_points,
                neighbor_threshold=self.config.bfs.neighbor_threshold,
                connectivity=self.config.bfs.connectivity,
            )

        self.results["bfs_region"] = bfs_region
        self.results["debug_10_bfs_region"] = bfs_region

        # Step 8: 依設定保留最合理的連通元件。
        self.logger.info("Step 8: Filtering connected components...")
        if self.config.connected_components.selection_strategy == "road_prior":
            final_region = keep_road_like_component(
                bfs_region,
                connectivity=self.config.connected_components.connectivity,
                bottom_bias_weight=self.config.connected_components.bottom_bias_weight,
                center_bias_weight=self.config.connected_components.center_bias_weight,
                area_weight=self.config.connected_components.area_weight,
                height_weight=self.config.connected_components.height_weight,
            )
        else:
            final_region = keep_largest_component(
                bfs_region,
                connectivity=self.config.connected_components.connectivity,
            )
        selected_region = final_region.copy()
        self.results["debug_11_connected_component_selected"] = selected_region

        # Step 8.5: 利用道路幾何修正最後區域，避免只抓到零碎底部區塊。
        if self.config.road_geometry.enabled:
            self.logger.info("Step 8.5: Refining region with road geometry...")
            geometry_result = detect_road_geometry_mask(
                gray,
                canny_threshold1=self.config.road_geometry.canny_threshold1,
                canny_threshold2=self.config.road_geometry.canny_threshold2,
                hough_threshold=self.config.road_geometry.hough_threshold,
                min_line_length=self.config.road_geometry.min_line_length,
                max_line_gap=self.config.road_geometry.max_line_gap,
                roi_top_y_ratio=self.config.road_geometry.roi_top_y_ratio,
                roi_top_width_ratio=self.config.road_geometry.roi_top_width_ratio,
                roi_bottom_width_ratio=self.config.road_geometry.roi_bottom_width_ratio,
                line_top_offset_ratio=self.config.road_geometry.line_top_offset_ratio,
                bottom_left_min_ratio=self.config.road_geometry.bottom_left_min_ratio,
                bottom_left_max_ratio=self.config.road_geometry.bottom_left_max_ratio,
                bottom_right_min_ratio=self.config.road_geometry.bottom_right_min_ratio,
                bottom_right_max_ratio=self.config.road_geometry.bottom_right_max_ratio,
                top_width_scale=self.config.road_geometry.top_width_scale,
                top_width_min_ratio=self.config.road_geometry.top_width_min_ratio,
                top_width_max_ratio=self.config.road_geometry.top_width_max_ratio,
                top_y_min_ratio=self.config.road_geometry.top_y_min_ratio,
                top_y_max_ratio=self.config.road_geometry.top_y_max_ratio,
                bottom_expand_ratio=self.config.road_geometry.bottom_expand_ratio,
                top_expand_ratio=self.config.road_geometry.top_expand_ratio,
                outer_bottom_offset_ratio=self.config.road_geometry.outer_bottom_offset_ratio,
                outer_top_offset_ratio=self.config.road_geometry.outer_top_offset_ratio,
            )
            self.results["road_geometry_mask"] = geometry_result["mask"]
            self.results["road_geometry_roi"] = geometry_result["roi_mask"]
            self.results["road_geometry_edges"] = geometry_result["roi_edges"]
            self.results["debug_12_clahe_geometry_input"] = geometry_result["clahe_input"]
            self.results["debug_13_canny_edges_for_geometry"] = geometry_result["roi_edges"]
            self.results["debug_14_hough_lines_raw"] = geometry_result["raw_lines_visualization"]
            self.results["debug_15_hough_lines_filtered_left_right"] = geometry_result["filtered_lines_visualization"]
            self.results["debug_16_vanishing_point_visualization"] = geometry_result["vanishing_point_visualization"]
            self.results["debug_17_geometry_trapezoid_mask"] = geometry_result["trapezoid_mask"]
            self.results["debug_18_geometry_outer_expanded_mask"] = geometry_result["outer_expanded_mask"]

            if geometry_result["success"]:
                overlap_ratio = compute_overlap_ratio(selected_region, geometry_result["mask"])
                overlap_visual = create_overlap_visualization(
                    selected_region,
                    geometry_result["mask"],
                )
                self.results["debug_19_geometry_vs_region_overlap"] = overlap_visual
                self.results["geometry_region_overlap_ratio"] = overlap_ratio

                fusion_result = repair_region_with_geometry(
                    selected_region,
                    geometry_result,
                    min_overlap=self.config.road_geometry.geometry_region_min_overlap,
                    support_margin_x=self.config.road_geometry.boundary_support_margin_x,
                    support_margin_y=self.config.road_geometry.boundary_support_margin_y,
                    smoothing_window=self.config.road_geometry.boundary_smoothing_window,
                    geometry_extension_limit=self.config.road_geometry.geometry_extension_limit,
                    min_top_width_ratio=self.config.road_geometry.min_top_width_ratio,
                    geometry_confidence_threshold=self.config.road_geometry.geometry_confidence_threshold,
                    left_search_margin=self.config.road_geometry.left_boundary_search_margin,
                    right_search_margin=self.config.road_geometry.right_boundary_search_margin,
                    left_max_delta=self.config.road_geometry.left_boundary_max_delta,
                    right_max_delta=self.config.road_geometry.right_boundary_max_delta,
                    width_alpha=self.config.road_geometry.boundary_width_alpha,
                )
                self.results["debug_19b_fused_mask_before_shape_regularization"] = (
                    fusion_result["fused_before_regularization"]
                )
                self.results["debug_19c_mask_after_shape_regularization"] = (
                    fusion_result["regularized_mask"]
                )
                self.results["debug_19d_top_edge_points_visualization"] = (
                    fusion_result["top_edge_visualization"]
                )
                self.results["debug_19e_left_right_boundary_trace"] = (
                    fusion_result["boundary_trace_visualization"]
                )
                self.results["debug_19f_left_boundary_candidates"] = (
                    fusion_result["left_boundary_candidates_visualization"]
                )
                self.results["debug_19g_right_boundary_candidates"] = (
                    fusion_result["right_boundary_candidates_visualization"]
                )
                self.results["debug_19h_left_right_confidence_heatmap"] = (
                    fusion_result["confidence_heatmap_visualization"]
                )
                self.results["debug_19i_boundary_delta_per_row"] = (
                    fusion_result["boundary_delta_visualization"]
                )
                self.results["debug_19j_boundary_width_curve"] = (
                    fusion_result["width_curve_visualization"]
                )
                self.results["debug_19k_geometry_vs_refined_boundary_offset"] = (
                    fusion_result["geometry_offset_visualization"]
                )
                self.results["geometry_confidence"] = fusion_result["geometry_confidence"]
                final_region = fusion_result["regularized_mask"]
                self.logger.info(
                    "Road geometry refinement applied with overlap ratio %.4f and confidence %.4f",
                    overlap_ratio,
                    fusion_result["geometry_confidence"],
                )
            else:
                self.logger.info("Road geometry refinement skipped")

        self.results["final_region"] = final_region
        self.results["debug_20_final_mask"] = final_region

        # Step 9: 從最終區域擷取輪廓。
        self.logger.info("Step 9: Contour extraction...")
        contours = extract_contours(
            final_region,
            min_area=self.config.contour.min_area,
        )

        self.logger.info(f"Extracted {len(contours)} contours")
        self.results["contours"] = contours

        # Step 10: 將輪廓疊回原圖。
        self.logger.info("Step 10: Creating visualizations...")
        contour_overlay = draw_contours(
            original,
            contours,
            color=self.config.contour.contour_color,
            thickness=self.config.contour.contour_thickness,
        )
        self.results["contour_overlay"] = contour_overlay
        self.results["debug_21_contour_overlay"] = contour_overlay

        self.logger.info("Pipeline completed successfully!")
        return self.results

    def save_intermediate_results(
        self,
        output_dir: Optional[str] = None
    ) -> None:
        """
        將中間結果輸出到資料夾。

        Args:
            output_dir: 輸出資料夾，若未提供則使用設定值。
        """
        if output_dir is None:
            output_dir = self.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 各階段結果對應的輸出檔名。
        save_mapping = {
            "debug_01_gray": "01_gray.png",
            "debug_02_blur": "02_blur.png",
            "debug_03_sobel_magnitude": "03_sobel_magnitude.png",
            "debug_04_lbp_map": "04_lbp_map.png",
            "debug_05_fused_feature_map": "05_fused_feature_map.png",
            "debug_06_candidate_mask_raw": "06_candidate_mask_raw.png",
            "debug_07_candidate_mask_morph": "07_candidate_mask_morph.png",
            "debug_08_distance_transform": "08_distance_transform.png",
            "debug_09_seed_points": "09_seed_points.png",
            "debug_10_bfs_region": "10_bfs_region.png",
            "debug_11_connected_component_selected": "11_connected_component_selected.png",
            "debug_12_clahe_geometry_input": "12_clahe_geometry_input.png",
            "debug_13_canny_edges_for_geometry": "13_canny_edges_for_geometry.png",
            "debug_14_hough_lines_raw": "14_hough_lines_raw.png",
            "debug_15_hough_lines_filtered_left_right": "15_hough_lines_filtered_left_right.png",
            "debug_16_vanishing_point_visualization": "16_vanishing_point_visualization.png",
            "debug_17_geometry_trapezoid_mask": "17_geometry_trapezoid_mask.png",
            "debug_18_geometry_outer_expanded_mask": "18_geometry_outer_expanded_mask.png",
            "debug_19_geometry_vs_region_overlap": "19_geometry_vs_region_overlap.png",
            "debug_19b_fused_mask_before_shape_regularization": "19b_fused_mask_before_shape_regularization.png",
            "debug_19c_mask_after_shape_regularization": "19c_mask_after_shape_regularization.png",
            "debug_19d_top_edge_points_visualization": "19d_top_edge_points_visualization.png",
            "debug_19e_left_right_boundary_trace": "19e_left_right_boundary_trace.png",
            "debug_19f_left_boundary_candidates": "19f_left_boundary_candidates.png",
            "debug_19g_right_boundary_candidates": "19g_right_boundary_candidates.png",
            "debug_19h_left_right_confidence_heatmap": "19h_left_right_confidence_heatmap.png",
            "debug_19i_boundary_delta_per_row": "19i_boundary_delta_per_row.png",
            "debug_19j_boundary_width_curve": "19j_boundary_width_curve.png",
            "debug_19k_geometry_vs_refined_boundary_offset": "19k_geometry_vs_refined_boundary_offset.png",
            "debug_20_final_mask": "20_final_mask.png",
            "debug_21_contour_overlay": "21_contour_overlay.png",
        }

        for result_name, filename in save_mapping.items():
            if result_name in self.results:
                output_path = output_dir / filename
                image = self.results[result_name]

                # 非 uint8 影像先做正規化，避免輸出異常。
                if image.dtype != np.uint8:
                    image = normalize_image(image, dtype=np.uint8)

                save_image(image, output_path)
                self.logger.info(f"Saved: {filename}")

    def get_results(self) -> Dict:
        """回傳目前保存的所有結果。"""
        return self.results

    def get_result(self, key: str) -> Optional[np.ndarray]:
        """依 key 取得單一結果。"""
        return self.results.get(key)


def run_pipeline(
    input_path: str,
    output_dir: str = "data/output",
    config: Optional[RoadContourConfig] = None,
    save_intermediate: bool = True
) -> Dict:
    """
    提供簡化版的 pipeline 呼叫介面。

    Args:
        input_path: 輸入影像路徑。
        output_dir: 輸出資料夾。
        config: 流程設定，未提供時使用預設值。
        save_intermediate: 是否儲存中間結果。

    Returns:
        各階段結果字典。
    """
    if config is None:
        from configs.default_config import get_default_config

        config = get_default_config()

    config.output_dir = Path(output_dir)

    pipeline = RoadContourPipeline(config)
    results = pipeline.run(input_path)

    if save_intermediate:
        pipeline.save_intermediate_results(output_dir)

    return results
