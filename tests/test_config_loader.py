"""
Unit tests for the ConfigLoader and AppConfig classes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoader, ConfigLoadError


class TestConfigLoader:
    def test_load_default_config(self):
        loader = ConfigLoader("config/config.yaml")
        cfg = loader.load()

        assert cfg.detection.model_path == "yolo11n.pt"
        assert 0.0 < cfg.detection.confidence_threshold < 1.0
        assert cfg.tracking.tracker_type == "bytetrack"
        assert cfg.counting.inside_direction in ("positive", "negative")
        assert len(cfg.counting.line_start) == 2
        assert len(cfg.counting.line_end) == 2
        assert cfg.visualization.line_thickness >= 1
        assert cfg.pipeline.input_path.endswith(".mp4")
        assert cfg.pipeline.output_path.endswith(".mp4")

    def test_missing_config_raises_error(self):
        loader = ConfigLoader("non_existent_config.yaml")
        with pytest.raises(ConfigLoadError):
            loader.load()

    def test_invalid_yaml_raises_error(self, tmp_path: Path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("detection: [unclosed list", encoding="utf-8")

        loader = ConfigLoader(bad_yaml)
        with pytest.raises(ConfigLoadError):
            loader.load()

    def test_custom_values_loaded_properly(self, tmp_path: Path):
        custom_yaml = tmp_path / "custom.yaml"
        data = {
            "video": {
                "input_path": "custom_in.mp4",
                "output_path": "custom_out.mp4",
                "show_live_window": True,
            },
            "detection": {
                "model_path": "yolo11s.pt",
                "confidence_threshold": 0.5,
                "input_size": 1280,
                "device": "cpu",
            },
            "tracking": {
                "max_age": 50,
                "min_hits": 4,
            },
            "counting": {
                "line_start": [10, 20],
                "line_end": [100, 200],
                "inside_direction": "negative",
            },
            "visualization": {
                "line_color": [255, 0, 0],
                "line_thickness": 3,
            },
            "metrics": {
                "console_every_n_frames": 10,
            },
        }
        with open(custom_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

        loader = ConfigLoader(custom_yaml)
        cfg = loader.load()

        assert cfg.pipeline.input_path == "custom_in.mp4"
        assert cfg.pipeline.output_path == "custom_out.mp4"
        assert cfg.pipeline.show_live_window is True
        assert cfg.detection.model_path == "yolo11s.pt"
        assert cfg.detection.confidence_threshold == 0.5
        assert cfg.tracking.max_age == 50
        assert cfg.counting.line_start == (10.0, 20.0)
        assert cfg.counting.inside_direction == "negative"
        assert cfg.visualization.line_color == (255, 0, 0)
        assert cfg.metrics.console_every_n_frames == 10
