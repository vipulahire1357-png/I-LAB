"""
Detection Module (Module 3) — CampusNetra

Provides YOLO-based person detection with optional classroom scene
awareness (dual-pass global + far-ROI detection).  Downstream modules
consume ``DetectionResult`` and ``PersonDetection`` — never YOLO directly.

Public API:
    - PersonDetector        (YOLO-backed detection engine)
    - DetectionConfig       (model path, thresholds, device, scene profile)
    - SceneProfile          (GENERAL / CLASSROOM enum)
    - PersonDetection       (single bbox + confidence)
    - DetectionResult       (per-frame aggregate result)
    - PersonDetectionError  (base exception)
    - ModelLoadError         (model loading failures)
    - InferenceError         (runtime inference failures)
    - InvalidFrameError      (bad input frame)
"""

from src.detection.exceptions import (
    InferenceError,
    InvalidFrameError,
    ModelLoadError,
    PersonDetectionError,
)
from src.detection.models import (
    DetectionConfig,
    DetectionResult,
    PersonDetection,
    SceneProfile,
)
from src.detection.person_detector import PersonDetector

__all__ = [
    "PersonDetector",
    "DetectionConfig",
    "SceneProfile",
    "PersonDetection",
    "DetectionResult",
    "PersonDetectionError",
    "ModelLoadError",
    "InferenceError",
    "InvalidFrameError",
]

