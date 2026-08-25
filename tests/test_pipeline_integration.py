"""
Integration tests for the full CampusNetra product0.3 computer vision pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config.models import AppConfig
from src.counting.models import CountingConfig
from src.detection.models import DetectionConfig
from src.pipeline.models import MetricsConfig, PipelineConfig
from src.pipeline.pipeline import Pipeline
from src.tracking.models import TrackerConfig
from src.visualization.models import VisualizationConfig


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Generate a synthetic test video with a moving rectangular object."""
    video_path = tmp_path / "synthetic_classroom.mp4"
    w, h, fps, num_frames = 640, 480, 25.0, 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for i in range(num_frames):
        frame = np.full((h, w, 3), 120, dtype=np.uint8)
        # Draw a synthetic moving person figure (circle + rectangle)
        y_pos = int(50 + i * 10)
        cv2.circle(frame, (320, y_pos), 20, (50, 50, 200), -1)
        cv2.rectangle(frame, (300, y_pos + 20), (340, y_pos + 100), (200, 50, 50), -1)
        writer.write(frame)

    writer.release()
    return video_path


class TestPipelineIntegration:
    def test_pipeline_runs_to_completion_on_synthetic_video(self, tmp_path: Path, synthetic_video: Path):
        out_video = tmp_path / "processed_output.mp4"

        config = AppConfig(
            detection=DetectionConfig(model_path="yolo11n.pt", confidence_threshold=0.3),
            tracking=TrackerConfig(tracker_type="bytetrack"),
            counting=CountingConfig(line_start=(0.0, 240.0), line_end=(640.0, 240.0)),
            visualization=VisualizationConfig(),
            pipeline=PipelineConfig(
                input_path=str(synthetic_video),
                output_path=str(out_video),
                show_live_window=False,
            ),
            metrics=MetricsConfig(console_every_n_frames=10),
        )

        pipeline = Pipeline(config)
        metrics = pipeline.run()

        # Invariants & Assertions
        assert metrics.processed_frames == 30
        assert metrics.total_frames == 30
        assert metrics.average_fps > 0
        assert metrics.final_occupancy >= 0
        assert metrics.final_occupancy == max(0, metrics.total_entries - metrics.total_exits)
        assert out_video.exists()
        assert out_video.stat().st_size > 0

        # Validate that output video is decodable
        cap = cv2.VideoCapture(str(out_video))
        assert cap.isOpened()
        ret, frame = cap.read()
        assert ret is True
        assert frame.shape == (480, 640, 3)
        cap.release()
