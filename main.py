"""
CampusNetra product0.3 — Vision-Based Classroom Occupancy & Entry/Exit Counting.

Entry-point CLI application.

Usage
-----
::

    # Default run
    python main.py

    # Override config path
    python main.py --config config/config.yaml

    # Override input and output videos
    python main.py --input data/input/classroom.mp4 --output data/output/classroom_processed.mp4

    # Enable live preview window
    python main.py --show
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config.loader import AppConfig, ConfigLoader
from src.pipeline.models import PipelineConfig
from src.pipeline.pipeline import Pipeline

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("campusnetra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CampusNetra product0.3 — Vision-Based Classroom Occupancy & Entry/Exit Counting"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input video file (overrides config/config.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output processed video file (overrides config/config.yaml)",
    )
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Display live OpenCV preview window during processing",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Inference device: 'cpu' or '0' for CUDA GPU",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        loader = ConfigLoader(args.config)
        app_config = loader.load()

        # Apply CLI overrides
        input_path = args.input or app_config.pipeline.input_path
        output_path = args.output or app_config.pipeline.output_path
        show_window = args.show or app_config.pipeline.show_live_window

        pipeline_cfg = PipelineConfig(
            input_path=input_path,
            output_path=output_path,
            show_live_window=show_window,
        )

        detection_cfg = app_config.detection
        if args.device:
            from dataclasses import replace
            detection_cfg = replace(detection_cfg, device=args.device)

        resolved_config = AppConfig(
            detection=detection_cfg,
            tracking=app_config.tracking,
            counting=app_config.counting,
            visualization=app_config.visualization,
            pipeline=pipeline_cfg,
            metrics=app_config.metrics,
        )

        pipeline = Pipeline(resolved_config)
        pipeline.run()
        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user.")
        return 130
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
