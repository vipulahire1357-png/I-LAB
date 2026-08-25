"""
Data models for the configuration system.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.counting.models import CountingConfig
from src.detection.models import DetectionConfig
from src.pipeline.models import MetricsConfig, PipelineConfig
from src.tracking.models import TrackerConfig
from src.visualization.models import VisualizationConfig


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration assembled from config.yaml.

    Attributes
    ----------
    detection : DetectionConfig
    tracking : TrackerConfig
    counting : CountingConfig
    visualization : VisualizationConfig
    pipeline : PipelineConfig
    metrics : MetricsConfig
    """

    detection: DetectionConfig
    tracking: TrackerConfig
    counting: CountingConfig
    visualization: VisualizationConfig
    pipeline: PipelineConfig
    metrics: MetricsConfig
