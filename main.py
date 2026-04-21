#!/usr/bin/env python3
"""
命令列入口點，負責載入設定並執行道路輪廓擷取流程。
"""

import argparse
import sys
from datetime import datetime
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


def _build_timestamp_label() -> str:
    """建立可用於輸出命名的時間戳記。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_unique_path(path: Path) -> Path:
    """若目標已存在，自動附加流水號避免覆蓋。"""
    if not path.exists():
        return path

    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def build_default_output_path(input_path: Path, output_dir: Path) -> Path:
    """建立不覆蓋既有結果圖的輸出路徑。"""
    timestamp = _build_timestamp_label()
    return _ensure_unique_path(output_dir / f"{input_path.stem}-result-{timestamp}.jpg")


def build_intermediate_output_dir(input_path: Path, output_dir: Path) -> Path:
    """為每次執行建立獨立的中間結果資料夾。"""
    timestamp = _build_timestamp_label()
    return _ensure_unique_path(output_dir / f"{input_path.stem}-run-{timestamp}")


def main():
    """程式主入口。"""
    args = parse_arguments()

    logger = setup_logger(__name__)

    logger.info("=" * 60)
    logger.info("Road Contour Extraction System - Method A")
    logger.info("=" * 60)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    logger.info(f"Input image: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    output_dir = Path(args.output_dir)

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

    try:
        logger.info("\nStarting pipeline execution...")

        pipeline = RoadContourPipeline(config)
        results = pipeline.run(str(input_path))

        logger.info(f"\nExtracted {len(results['contours'])} contours")

        if args.save_intermediate:
            run_output_dir = build_intermediate_output_dir(input_path, output_dir)
            logger.info(f"Saving intermediate results to: {run_output_dir}")
            pipeline.save_intermediate_results(str(run_output_dir))
            final_output_path = run_output_dir / "21_contour_overlay.png"
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            final_output_path = build_default_output_path(input_path, output_dir)

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
