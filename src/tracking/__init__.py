"""
Tracking Module (Module 4) — CampusNetra

ByteTrack-based multi-object person tracking.  Assigns stable
temporary Track IDs to person detections across frames.

Public API:
    - PersonTracker       (ByteTrack-backed tracker)
    - TrackerConfig       (algorithm parameters)
    - Track               (single-track snapshot)
    - TrackState           (TENTATIVE / CONFIRMED / LOST / REMOVED)
    - TrackingResult       (per-frame aggregate result)
    - TrackingMetrics      (aggregate timing / count stats)
    - TrackerError         (base exception)
    - TrackerConfigError   (invalid configuration)
    - TrackerUpdateError   (runtime update failure)
"""

from src.tracking.exceptions import (
    TrackerConfigError,
    TrackerError,
    TrackerUpdateError,
)
from src.tracking.models import (
    Track,
    TrackerConfig,
    TrackingMetrics,
    TrackingResult,
    TrackState,
)
from src.tracking.person_tracker import PersonTracker

__all__ = [
    "PersonTracker",
    "TrackerConfig",
    "Track",
    "TrackState",
    "TrackingResult",
    "TrackingMetrics",
    "TrackerError",
    "TrackerConfigError",
    "TrackerUpdateError",
]

