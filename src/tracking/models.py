"""
Data models for the Tracking module.

Contains track lifecycle states, configuration, per-track data,
per-frame tracking result, and aggregate metrics.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.tracking.exceptions import TrackerConfigError


# ======================================================================
# Track lifecycle state
# ======================================================================

class TrackState(enum.Enum):
    """Lifecycle state of a tracked person.

    State transitions::

        TENTATIVE ──▸ CONFIRMED ──▸ LOST ──▸ REMOVED
                         ▲            │
                         └────────────┘  (re-associated)
    """

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    REMOVED = "REMOVED"


# ======================================================================
# Configuration
# ======================================================================

@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Immutable configuration for :class:`PersonTracker`.

    Parameters
    ----------
    tracker_type : str
        Identifier for the tracking algorithm.  Currently ``"bytetrack"``.
    track_high_threshold : float
        High confidence threshold for first-pass association (ByteTrack).
    track_low_threshold : float
        Low confidence threshold for second-pass association (ByteTrack).
    match_threshold : float
        Minimum IoU overlap for matching detections to existing tracks.
    max_age : int
        Maximum number of frames a lost track is kept before removal.
    min_hits : int
        Minimum consecutive detections before a track is confirmed.
    frame_rate : int
        Expected video frame rate (used by Kalman filter motion model).
    """

    tracker_type: str = "bytetrack"
    track_high_threshold: float = 0.25
    track_low_threshold: float = 0.1
    match_threshold: float = 0.8
    max_age: int = 30
    min_hits: int = 3
    frame_rate: int = 30

    def __post_init__(self) -> None:
        if self.tracker_type not in ("bytetrack",):
            raise TrackerConfigError(
                f"Unsupported tracker_type: {self.tracker_type!r}. "
                f"Supported: 'bytetrack'."
            )
        if not 0.0 <= self.track_high_threshold <= 1.0:
            raise TrackerConfigError(
                f"track_high_threshold must be in [0, 1], got {self.track_high_threshold}"
            )
        if self.max_age < 1:
            raise TrackerConfigError(
                f"max_age must be >= 1, got {self.max_age}"
            )
        if self.min_hits < 1:
            raise TrackerConfigError(
                f"min_hits must be >= 1, got {self.min_hits}"
            )

    @staticmethod
    def classroom_default(frame_rate: int = 25) -> "TrackerConfig":
        """Factory for a classroom-tuned ByteTrack configuration.

        Classroom optimisations:
        - ``max_age=60`` — tolerate ~2 s of occlusion behind desks/students.
        - ``min_hits=2`` — faster track confirmation for seated students.
        - ``track_high_threshold=0.15`` — accept low-confidence detections
          from the far/back region.
        - ``match_threshold=0.70`` — more lenient IoU matching for
          near-stationary / overlapping seated students.
        """
        return TrackerConfig(
            tracker_type="bytetrack",
            track_high_threshold=0.15,
            track_low_threshold=0.05,
            match_threshold=0.70,
            max_age=60,
            min_hits=2,
            frame_rate=frame_rate,
        )


# ======================================================================
# Single track snapshot
# ======================================================================

@dataclass(frozen=True, slots=True)
class Track:
    """Immutable snapshot of a tracked person for one frame.

    Coordinate system matches Module 3: ``(x1, y1)`` = top-left,
    ``(x2, y2)`` = bottom-right in original-frame pixels.

    Attributes
    ----------
    track_id : int
        Unique monotonic ID within the current session.  Never reused.
    bbox : tuple[float, float, float, float]
        ``(x1, y1, x2, y2)`` bounding box.
    confidence : float
        Detection confidence of the associated detection.
    state : TrackState
        Current lifecycle state.
    age : int
        Frames since this track was first created.
    hits : int
        Total frames in which this track was matched to a detection.
    misses : int
        Consecutive frames without a matching detection.
    first_frame : int
        ``frame_id`` when this track was created.
    last_frame : int
        ``frame_id`` of the most recent match.
    first_timestamp : float
        Video timestamp when this track was created (seconds).
    last_timestamp : float
        Video timestamp of the most recent match (seconds).
    """

    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    state: TrackState
    age: int = 0
    hits: int = 0
    misses: int = 0
    first_frame: int = 0
    last_frame: int = 0
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0

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
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    def __repr__(self) -> str:
        cx, cy = self.center
        return (
            f"Track(id={self.track_id}, "
            f"state={self.state.value}, "
            f"bbox=({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}), "
            f"conf={self.confidence:.2f}, "
            f"age={self.age}, hits={self.hits})"
        )


# ======================================================================
# Per-frame tracking result
# ======================================================================

@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Aggregate tracking result for a single frame.

    Attributes
    ----------
    frame_id : int
        Matches the input ``DetectionResult.frame_id``.
    timestamp : float
        Matches the input ``DetectionResult.timestamp``.
    tracks : tuple[Track, ...]
        Active tracks in this frame (CONFIRMED and LOST).
    tracking_time_ms : float
        Wall-clock time spent on the tracking update (milliseconds).
    active_count : int
        Number of CONFIRMED tracks in this frame.
    lost_count : int
        Number of LOST tracks in this frame.
    """

    frame_id: int
    timestamp: float
    tracks: Tuple[Track, ...]
    tracking_time_ms: float = 0.0
    active_count: int = 0
    lost_count: int = 0

    def __repr__(self) -> str:
        return (
            f"TrackingResult("
            f"frame_id={self.frame_id}, "
            f"active={self.active_count}, "
            f"lost={self.lost_count}, "
            f"tracking={self.tracking_time_ms:.1f}ms)"
        )


# ======================================================================
# Aggregate metrics
# ======================================================================

@dataclass(slots=True)
class TrackingMetrics:
    """Live metrics collected by the tracker across frames."""

    frames_processed: int = 0
    total_tracks_created: int = 0
    total_tracks_removed: int = 0
    current_active_tracks: int = 0
    max_active_tracks: int = 0

    _tracking_times: list[float] = field(default_factory=list, repr=False)

    @property
    def average_tracking_time(self) -> Optional[float]:
        return (
            sum(self._tracking_times) / len(self._tracking_times)
            if self._tracking_times
            else None
        )

    @property
    def median_tracking_time(self) -> Optional[float]:
        if not self._tracking_times:
            return None
        s = sorted(self._tracking_times)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    @property
    def p95_tracking_time(self) -> Optional[float]:
        if len(self._tracking_times) < 20:
            return None
        s = sorted(self._tracking_times)
        return s[int(len(s) * 0.95)]

    def _record_frame(
        self,
        tracking_time_ms: float,
        active: int,
        created: int,
        removed: int,
    ) -> None:
        self.frames_processed += 1
        self.total_tracks_created += created
        self.total_tracks_removed += removed
        self.current_active_tracks = active
        self.max_active_tracks = max(self.max_active_tracks, active)
        self._tracking_times.append(tracking_time_ms)

    def _reset(self) -> None:
        self.frames_processed = 0
        self.total_tracks_created = 0
        self.total_tracks_removed = 0
        self.current_active_tracks = 0
        self.max_active_tracks = 0
        self._tracking_times.clear()

