"""
Data models for the Entry/Exit Counting module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from src.counting.exceptions import CountingConfigError


class EventType(str, Enum):
    """Semantic type of a line-crossing event."""

    ENTER = "ENTER"
    EXIT = "EXIT"


class TrackSideState(str, Enum):
    """Side classification of a tracked person relative to the virtual line."""

    UNKNOWN = "UNKNOWN"
    OUTSIDE = "OUTSIDE"
    INSIDE = "INSIDE"


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    """A verified directional line-crossing event by a tracked person.

    Attributes
    ----------
    event_type : EventType
        Whether the person entered (ENTER) or exited (EXIT).
    track_id : int
        Temporary monotonic track identifier.
    frame_id : int
        Sequential frame number where the crossing was verified.
    timestamp : float
        Video timestamp in seconds where the crossing was verified.
    """

    event_type: EventType
    track_id: int
    frame_id: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class CountingConfig:
    """Configuration for virtual line geometry and crossing detection.

    Attributes
    ----------
    line_start : tuple[float, float]
        (x, y) start coordinate in original frame pixels.
    line_end : tuple[float, float]
        (x, y) end coordinate in original frame pixels.
    inside_direction : str
        Which half-plane represents INSIDE ("positive" or "negative").
    hysteresis_distance : float
        Perpendicular pixel distance beyond the line required to confirm
        a complete crossing and prevent jitter/oscillation on the boundary.
    """

    line_start: Tuple[float, float] = (0.0, 195.0)
    line_end: Tuple[float, float] = (1520.0, 1080.0)
    inside_direction: str = "positive"
    hysteresis_distance: float = 12.0

    def __post_init__(self) -> None:
        if len(self.line_start) != 2 or len(self.line_end) != 2:
            raise CountingConfigError("line_start and line_end must be 2D points (x, y).")
        if self.line_start == self.line_end:
            raise CountingConfigError("line_start and line_end must not be identical points.")
        if self.inside_direction not in ("positive", "negative"):
            raise CountingConfigError(
                f"inside_direction must be 'positive' or 'negative', got {self.inside_direction!r}"
            )
        if self.hysteresis_distance < 0:
            raise CountingConfigError("hysteresis_distance must be non-negative.")
