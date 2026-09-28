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
import threading
from src.api_server import start_api_server

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

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
PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


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
    parser.add_argument(
        "--max-frames",
        "-n",
        type=int,
        default=None,
        help="Maximum number of frames to process before exiting",
    )
    parser.add_argument(
        "--rtsp",
        action="store_true",
        help="Use RTSP live camera feed instead of video file",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera ID to use with --rtsp (e.g., 'A14-419')",
    )
    return parser.parse_args()


def interactive_menu() -> tuple[bool, Optional[str], Optional[str]]:
    print("=" * 60)
    print(" CampusNetra I-LAB — Select Input Source")
    print("=" * 60)
    print("1. Process Uploaded Video")
    print("2. Process Live RTSP Camera")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        try:
            import yaml
            with open(project_path("config/cameras.yaml"), "r", encoding="utf-8") as f:
                cameras_config = yaml.safe_load(f)
            cameras = cameras_config.get("cameras", {})
        except Exception as e:
            print(f"Error loading cameras.yaml: {e}")
            return False, None, None

        if not cameras:
            print("No cameras configured.")
            return False, None, None

        print("\nAvailable Cameras:")
        cam_ids = list(cameras.keys())
        for i, cid in enumerate(cam_ids, start=1):
            name = cameras[cid].get("name", "Unknown")
            print(f"{i}. {cid} — {name}")

        cam_choice = input(f"Select Camera (1-{len(cam_ids)}): ").strip()
        try:
            cam_idx = int(cam_choice) - 1
            if 0 <= cam_idx < len(cam_ids):
                return True, cam_ids[cam_idx], None
        except ValueError:
            pass
        print("Invalid camera selection.")
        return False, None, None
    elif choice == "1":
        video_path = input("Enter path to video file (leave empty for default): ").strip()
        if not video_path:
            video_path = None
        return False, None, video_path
    else:
        print("Invalid choice.")
        return False, None, None


def main() -> int:
    args = parse_args()

    try:
        loader = ConfigLoader(project_path(args.config))
        app_config = loader.load()

        use_rtsp = args.rtsp
        camera_id = args.camera
        input_path = args.input or app_config.pipeline.input_path

        # Interactive menu if no source args provided explicitly
        if len(sys.argv) == 1:
            is_rtsp, menu_cam, menu_input = interactive_menu()
            if is_rtsp:
                use_rtsp = True
                camera_id = menu_cam
            else:
                use_rtsp = False
                if menu_input:
                    input_path = menu_input

        # Determine output
        output_path = args.output or app_config.pipeline.output_path
        show_window = args.show or app_config.pipeline.show_live_window
        max_frames = args.max_frames if args.max_frames is not None else app_config.pipeline.max_frames

        pipeline_cfg = PipelineConfig(
            input_path=input_path,
            output_path=output_path,
            record_output=app_config.pipeline.record_output or args.output is not None,
            show_live_window=show_window,
            max_frames=max_frames,
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

        # Setup Video Source
        video_source = None
        if use_rtsp:
            from src.video.rtsp_video_source import RTSPVideoSource
            if not camera_id:
                print("Error: --camera must be specified when using --rtsp, unless using the interactive menu.")
                return 1
            video_source = RTSPVideoSource(
                camera_id=camera_id,
                config_path=str(project_path("config/cameras.yaml")),
            )
            print(f"[INFO] Using Live RTSP Camera: {camera_id}")
        else:
            from src.video.file_video_source import FileVideoSource
            video_source = FileVideoSource(input_path)
            print(f"[INFO] Using Video File: {input_path}")

        pipeline = Pipeline(resolved_config, video_source=video_source)
        api_thread = threading.Thread(
            target=start_api_server,
            daemon=True
        )

        api_thread.start()


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
