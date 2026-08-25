"""
Data models for the Pipeline Orchestrator module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.occupancy.models import ClassroomState


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for video input, output, and execution mode.

    Attributes
    ----------
    input_path : str
        Path to the input CCTV video file.
    output_path : str
        Path to write the processed output video.
    show_live_window : bool
        If True, displays an OpenCV preview window during processing.
    """

    input_path: str = "data/input/classroom.mp4"
    output_path: str = "data/output/classroom_processed.mp4"
    show_live_window: bool = False


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Configuration for telemetry and console logging."""

    console_every_n_frames: int = 30


@dataclass(slots=True)
class PipelineMetrics:
    """Aggregate execution statistics and performance metrics."""

    total_frames: int = 0
    processed_frames: int = 0
    total_wall_time_s: float = 0.0
    average_fps: float = 0.0

    # Fine-grained stage timings (average ms per frame)
    avg_read_ms: float = 0.0
    avg_detect_ms: float = 0.0
    avg_track_ms: float = 0.0
    avg_count_ms: float = 0.0
    avg_render_ms: float = 0.0
    avg_write_ms: float = 0.0

    # Final domain statistics
    total_entries: int = 0
    total_exits: int = 0
    final_occupancy: int = 0
    final_state: ClassroomState = ClassroomState.EMPTY
