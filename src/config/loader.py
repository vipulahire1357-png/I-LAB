"""
Configuration loader for CampusNetra product0.3.

Reads config/config.yaml and constructs typed configuration dataclass
objects for each module. All modules receive their configuration through
these typed objects — no module reads YAML directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.config.models import AppConfig
from src.counting.models import CountingConfig
from src.detection.models import DetectionConfig
from src.pipeline.models import MetricsConfig, PipelineConfig
from src.tracking.models import TrackerConfig
from src.visualization.models import VisualizationConfig


class ConfigLoadError(Exception):
    """Raised when the configuration file cannot be loaded or parsed."""


class ConfigLoader:
    """Loads and validates the runtime configuration YAML file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file (default: ``config/config.yaml``).
    """

    DEFAULT_CONFIG_PATH = "config/config.yaml"

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._path = str(
            Path(config_path) if config_path else Path(self.DEFAULT_CONFIG_PATH)
        )

    def load(self) -> AppConfig:
        """Parse the YAML file and construct all module config objects.

        Returns
        -------
        AppConfig
            Fully populated configuration.

        Raises
        ------
        ConfigLoadError
            If the file is missing, unreadable, or contains invalid values.
        """
        raw = self._read_yaml()
        return AppConfig(
            detection=self._build_detection(raw),
            tracking=self._build_tracking(raw),
            counting=self._build_counting(raw),
            visualization=self._build_visualization(raw),
            pipeline=self._build_pipeline(raw),
            metrics=self._build_metrics(raw),
        )

    # ------------------------------------------------------------------
    # Private — file I/O
    # ------------------------------------------------------------------

    def _read_yaml(self) -> dict[str, Any]:
        if not os.path.exists(self._path):
            raise ConfigLoadError(
                f"Configuration file not found: {self._path}\n"
                f"Create it or pass a valid path to ConfigLoader()."
            )
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigLoadError(
                f"Failed to parse YAML from {self._path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ConfigLoadError(
                f"Configuration file {self._path} must be a YAML mapping, "
                f"got {type(data).__name__}."
            )
        return data

    # ------------------------------------------------------------------
    # Private — section builders
    # ------------------------------------------------------------------

    def _build_detection(self, raw: dict[str, Any]) -> DetectionConfig:
        d = raw.get("detection", {})
        try:
            return DetectionConfig(
                model_path=str(d.get("model_path", "yolo11n.pt")),
                confidence_threshold=float(d.get("confidence_threshold", 0.35)),
                iou_threshold=float(d.get("iou_threshold", 0.45)),
                input_size=int(d.get("input_size", 640)),
                device=str(d.get("device", "cpu")),
                scene_profile=str(d.get("scene_profile", "general")),
            )
        except Exception as exc:
            raise ConfigLoadError(f"Invalid [detection] section: {exc}") from exc

    def _build_tracking(self, raw: dict[str, Any]) -> TrackerConfig:
        d = raw.get("tracking", {})
        try:
            return TrackerConfig(
                tracker_type=str(d.get("tracker_type", "bytetrack")),
                track_high_threshold=float(d.get("track_high_threshold", 0.25)),
                track_low_threshold=float(d.get("track_low_threshold", 0.10)),
                match_threshold=float(d.get("match_threshold", 0.80)),
                max_age=int(d.get("max_age", 30)),
                min_hits=int(d.get("min_hits", 3)),
            )
        except Exception as exc:
            raise ConfigLoadError(f"Invalid [tracking] section: {exc}") from exc

    def _build_counting(self, raw: dict[str, Any]) -> CountingConfig:
        d = raw.get("counting", {})
        try:
            start_raw = d.get("line_start", [0, 400])
            end_raw = d.get("line_end", [1024, 400])
            if len(start_raw) != 2 or len(end_raw) != 2:
                raise ValueError(
                    "line_start and line_end must each be [x, y] with 2 values."
                )
            line_start = (float(start_raw[0]), float(start_raw[1]))
            line_end = (float(end_raw[0]), float(end_raw[1]))
            if line_start == line_end:
                raise ValueError(
                    "line_start and line_end must not be the same point."
                )
            inside_direction = str(d.get("inside_direction", "positive"))
            if inside_direction not in ("positive", "negative"):
                raise ValueError(
                    f"inside_direction must be 'positive' or 'negative', "
                    f"got {inside_direction!r}."
                )
            hysteresis_dist = float(d.get("hysteresis_distance", 12.0))
            return CountingConfig(
                line_start=line_start,
                line_end=line_end,
                inside_direction=inside_direction,
                hysteresis_distance=hysteresis_dist,
            )
        except Exception as exc:
            raise ConfigLoadError(f"Invalid [counting] section: {exc}") from exc

    def _build_visualization(self, raw: dict[str, Any]) -> VisualizationConfig:
        d = raw.get("visualization", {})
        try:
            def _bgr(key: str, default: list[int]) -> tuple[int, int, int]:
                v = d.get(key, default)
                if len(v) != 3:
                    raise ValueError(f"{key} must be [B, G, R] with 3 values.")
                return (int(v[0]), int(v[1]), int(v[2]))

            op_raw = d.get("overlay_position", [10, 10])
            return VisualizationConfig(
                line_color=_bgr("line_color", [0, 255, 255]),
                line_thickness=int(d.get("line_thickness", 2)),
                bbox_color_confirmed=_bgr("bbox_color_confirmed", [0, 255, 0]),
                bbox_color_lost=_bgr("bbox_color_lost", [128, 128, 128]),
                event_flash_frames=int(d.get("event_flash_frames", 10)),
                font_scale=float(d.get("font_scale", 0.65)),
                font_thickness=int(d.get("font_thickness", 2)),
                overlay_position=(int(op_raw[0]), int(op_raw[1])),
            )
        except Exception as exc:
            raise ConfigLoadError(f"Invalid [visualization] section: {exc}") from exc

    def _build_pipeline(self, raw: dict[str, Any]) -> PipelineConfig:
        d = raw.get("video", {})
        try:
            return PipelineConfig(
                input_path=str(d.get("input_path", "data/input/classroom.mp4")),
                output_path=str(d.get("output_path", "data/output/classroom_processed.mp4")),
                show_live_window=bool(d.get("show_live_window", False)),
            )
        except Exception as exc:
            raise ConfigLoadError(f"Invalid [video] section: {exc}") from exc

    def _build_metrics(self, raw: dict[str, Any]) -> MetricsConfig:
        d = raw.get("metrics", {})
        try:
            return MetricsConfig(
                console_every_n_frames=int(d.get("console_every_n_frames", 30)),
            )
        except Exception as exc:
            raise ConfigLoadError(f"Invalid [metrics] section: {exc}") from exc

    def __repr__(self) -> str:
        return f"ConfigLoader(path={self._path!r})"
