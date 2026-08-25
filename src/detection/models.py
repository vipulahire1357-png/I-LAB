"""
Data models for the Detection module.

Contains the detection configuration, scene profiles, per-detection
result, and per-frame aggregate result.  All models are plain data
containers with no YOLO/Ultralytics dependency.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from src.detection.exceptions import PersonDetectionError


# ======================================================================
# Scene profile
# ======================================================================

class SceneProfile(enum.Enum):
    """Predefined scene profiles.

    ``GENERAL``
        Default balanced configuration (outdoor / street / corridor).

    ``CLASSROOM``
        Fixed overhead camera in a classroom.  Activates far-region
        recovery detection for small/distant students.
    """

    GENERAL = "general"
    CLASSROOM = "classroom"


# ======================================================================
# Configuration
# ======================================================================

@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Immutable configuration for :class:`PersonDetector`.

    Parameters
    ----------
    model_path : str
        Path (or Ultralytics model name) for the YOLO weights.
        Examples: ``"yolo11n.pt"``, ``"yolo11s.pt"``.
    confidence_threshold : float
        Minimum confidence to keep a detection.  Range [0, 1].
    iou_threshold : float
        IoU threshold for non-maximum suppression.  Range [0, 1].
    input_size : int
        Square input dimension for the model (e.g. 640, 1280).
    device : str
        Inference device.  ``"cpu"`` for CPU, ``"0"`` for GPU 0, etc.
    person_class_id : int
        COCO class ID for ``person``.  Default 0.
    scene_profile : str
        Scene profile name.  ``"general"`` or ``"classroom"``.
    classroom_far_roi : tuple[float, float, float, float] | None
        Normalised ``(x1, y1, x2, y2)`` ROI for the far/back classroom
        region.  Values in ``[0, 1]`` relative to frame dimensions.
        Used only when ``scene_profile="classroom"``.
    classroom_roi_input_size : int
        Inference input size for the far-ROI second pass.
    classroom_roi_confidence : float
        Confidence threshold for the far-ROI second pass (typically
        lower than global to recover weak small-person detections).
    fusion_iou_threshold : float
        IoU threshold for duplicate suppression during detection fusion.
    """

    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    input_size: int = 640
    device: str = "cpu"
    person_class_id: int = 0

    # Scene-aware settings
    scene_profile: str = "general"
    classroom_far_roi: Optional[Tuple[float, float, float, float]] = None
    classroom_roi_input_size: int = 640
    classroom_roi_confidence: float = 0.15
    fusion_iou_threshold: float = 0.50

    def __post_init__(self) -> None:
        if not self.model_path:
            raise PersonDetectionError(
                "model_path must not be empty.", operation="configure"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise PersonDetectionError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}",
                operation="configure",
            )
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise PersonDetectionError(
                f"iou_threshold must be in [0, 1], got {self.iou_threshold}",
                operation="configure",
            )
        if self.input_size <= 0:
            raise PersonDetectionError(
                f"input_size must be > 0, got {self.input_size}",
                operation="configure",
            )
        if self.scene_profile not in ("general", "classroom"):
            raise PersonDetectionError(
                f"scene_profile must be 'general' or 'classroom', got {self.scene_profile!r}",
                operation="configure",
            )
        if self.classroom_far_roi is not None:
            rx1, ry1, rx2, ry2 = self.classroom_far_roi
            if not (0.0 <= rx1 < rx2 <= 1.0 and 0.0 <= ry1 < ry2 <= 1.0):
                raise PersonDetectionError(
                    f"classroom_far_roi must be normalised (x1,y1,x2,y2) in [0,1] "
                    f"with x1<x2 and y1<y2, got {self.classroom_far_roi}",
                    operation="configure",
                )

    @property
    def is_classroom(self) -> bool:
        """Whether the classroom scene profile is active."""
        return self.scene_profile == "classroom"

    @staticmethod
    def classroom_default() -> "DetectionConfig":
        """Factory for a tuned classroom configuration.

        Uses:
        - ``yolo11s.pt`` for better recall on small persons
        - ``input_size=1280`` for high-resolution global detection
        - ``confidence_threshold=0.20`` for recall-oriented filtering
        - Far/back ROI covering approximately the top 45% of the frame
        - ROI second-pass at 640 with conf=0.15
        """
        return DetectionConfig(
            model_path="yolo11s.pt",
            confidence_threshold=0.20,
            iou_threshold=0.50,
            input_size=1280,
            device="cpu",
            scene_profile="classroom",
            classroom_far_roi=(0.0, 0.0, 1.0, 0.45),
            classroom_roi_input_size=640,
            classroom_roi_confidence=0.15,
            fusion_iou_threshold=0.50,
        )


# ======================================================================
# Single detection
# ======================================================================

@dataclass(frozen=True, slots=True)
class PersonDetection:
    """A single person detection in original-frame coordinates.

    Coordinate system
    -----------------
    ::

        (x1, y1) ──────────────┐
        │                      │
        │       center         │
        │                      │
        └──────────────── (x2, y2)

    ``(x1, y1)`` = top-left, ``(x2, y2)`` = bottom-right,
    in the coordinate space of the **original input frame** (not the
    model's internal inference resolution).

    Attributes
    ----------
    bbox : tuple[float, float, float, float]
        ``(x1, y1, x2, y2)`` bounding box.
    confidence : float
        Detection confidence in ``[0, 1]``.
    class_id : int
        COCO class ID (``0`` for person).
    """

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Tuple[float, float]:
        """``(cx, cy)`` centre point of the bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    @property
    def area(self) -> float:
        return self.width * self.height

    def __repr__(self) -> str:
        cx, cy = self.center
        return (
            f"PersonDetection("
            f"bbox=({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}), "
            f"conf={self.confidence:.2f}, "
            f"center=({cx:.0f},{cy:.0f}), "
            f"size={self.width:.0f}×{self.height:.0f})"
        )


# ======================================================================
# Per-frame result
# ======================================================================

@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Aggregate detection result for a single frame.

    Attributes
    ----------
    frame_id : int
        Matches :attr:`Frame.frame_id` from Module 1.
    timestamp : float
        Matches :attr:`Frame.timestamp` from Module 1.
    detections : tuple[PersonDetection, ...]
        Zero or more person detections, sorted by confidence descending.
    inference_time_ms : float
        Time spent on model inference (milliseconds).
    total_time_ms : float
        Total detect() wall time including pre/post-processing (ms).
    """

    frame_id: int
    timestamp: float
    detections: Tuple[PersonDetection, ...]
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def count(self) -> int:
        """Number of person detections."""
        return len(self.detections)

    def __repr__(self) -> str:
        return (
            f"DetectionResult("
            f"frame_id={self.frame_id}, "
            f"timestamp={self.timestamp:.3f}s, "
            f"people={self.count}, "
            f"inference={self.inference_time_ms:.1f}ms)"
        )

