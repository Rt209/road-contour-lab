#!/usr/bin/env python3
"""
此檔案是道路輪廓擷取系統的命令列入口。
使用者可透過 CLI 指定輸入影像、輸出資料夾與部分重要參數，
再交由主流程類別執行完整的道路輪廓擷取。
"""

import argparse
import sys
from pathlib import Path

from configs.default_config import get_default_config
from src.pipeline.road_contour_pipeline import RoadContourPipeline
from src.utils.logger import setup_logger


def parse_arguments():
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description="Road Contour Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/input/road.jpg --output_dir data/output/
  python main.py --input data/input/road.jpg --output_dir data/output/ --save_intermediate
  python main.py --input data/input/road.jpg --config my_config.py
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output",
        help="Output directory for result images (default: data/output)",
    )

    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        default=False,
        help="Save intermediate processing results (default: False)",
    )

    parser.add_argument(
        "--no_intermediate",
        dest="save_intermediate",
        action="store_false",
        help="Do not save intermediate results",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration file",
    )

    parser.add_argument(
        "--threshold_fusion",
        type=float,
        default=None,
        help="Feature fusion threshold",
    )

    parser.add_argument(
        "--threshold_candidate",
        type=float,
        default=None,
        help="Candidate mask threshold",
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=None,
        help="Minimum contour area",
    )

    return parser.parse_args()


def build_default_output_path(input_path: Path, output_dir: Path) -> Path:
    """依輸入檔名建立預設輸出檔名 `檔名-result.jpg`。"""
    return output_dir / f"{input_path.stem}-result.jpg"


def main():
    """命令列主入口。"""
    args = parse_arguments()

    # 建立 logger，方便顯示流程進度與錯誤資訊。
    logger = setup_logger(__name__)

    logger.info("=" * 60)
    logger.info("Road Contour Extraction System - Method A")
    logger.info("=" * 60)

    # 驗證輸入影像是否存在。
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    logger.info(f"Input image: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    output_dir = Path(args.output_dir)
    final_output_path = build_default_output_path(input_path, output_dir)
    logger.info(f"Final output image: {final_output_path}")

    # 載入預設設定，並套用命令列覆寫值。
    try:
        config = get_default_config()

        if args.threshold_fusion is not None:
            config.feature_fusion.fusion_threshold = args.threshold_fusion
            logger.info(f"Feature fusion threshold: {args.threshold_fusion}")

        if args.threshold_candidate is not None:
            config.candidate_mask.threshold = args.threshold_candidate
            logger.info(f"Candidate mask threshold: {args.threshold_candidate}")

        if args.min_area is not None:
            config.contour.min_area = args.min_area
            logger.info(f"Minimum contour area: {args.min_area}")

        config.output_dir = output_dir
        _ = args.config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # 執行主流程並輸出結果。
    try:
        logger.info("\nStarting pipeline execution...")

        pipeline = RoadContourPipeline(config)
        results = pipeline.run(str(input_path))

        logger.info(f"\nExtracted {len(results['contours'])} contours")

        if args.save_intermediate:
            logger.info("\nSaving intermediate results...")
            pipeline.save_intermediate_results(str(args.output_dir))
        else:
            # 預設只保留自動命名的最終結果圖，並清除舊的中間圖片。
            output_dir.mkdir(parents=True, exist_ok=True)
            for file_path in output_dir.iterdir():
                if file_path.name == ".gitkeep":
                    continue
                if file_path.is_file():
                    file_path.unlink()

            from src.preprocessing.image_io import save_image

            save_image(
                results["contour_overlay"],
                final_output_path,
            )
            logger.info(f"Saved {final_output_path.name}")

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {final_output_path}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"\nPipeline execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
