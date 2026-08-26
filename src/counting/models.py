"""
Data models for the Entry/Exit Counting module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple

from src.counting.exceptions import CountingConfigError


class EventType(str, Enum):
    """Semantic type of a line-crossing or zone-crossing event."""

    ENTER = "ENTER"
    EXIT = "EXIT"
    ZONE_CROSS_LEFT_TO_RIGHT = "ZONE_CROSS_LEFT_TO_RIGHT"
    ZONE_CROSS_RIGHT_TO_LEFT = "ZONE_CROSS_RIGHT_TO_LEFT"
    ZONE_ENTER_LEFT = "ZONE_ENTER_LEFT"
    ZONE_EXIT_LEFT = "ZONE_EXIT_LEFT"
    ZONE_ENTER_RIGHT = "ZONE_ENTER_RIGHT"
    ZONE_EXIT_RIGHT = "ZONE_EXIT_RIGHT"


class TrackSideState(str, Enum):
    """Side classification of a tracked person relative to the virtual counting line."""

    UNKNOWN = "UNKNOWN"
    OUTSIDE = "OUTSIDE"
    INSIDE = "INSIDE"


class DividerSideState(str, Enum):
    """Zone classification of a tracked person relative to the walkway divider lines."""

    UNKNOWN = "UNKNOWN"
    OUTSIDE = "OUTSIDE"
    LEFT = "LEFT"
    WALKWAY = "WALKWAY"
    RIGHT = "RIGHT"


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    """A verified directional line-crossing or zone-crossing event by a tracked person.

    Attributes
    ----------
    event_type : EventType
        Whether the person entered (ENTER), exited (EXIT), or crossed zones (ZONE_*).
    track_id : int
        Temporary monotonic track identifier.
    frame_id : int
        Sequential frame number where the crossing was verified.
    timestamp : float
        Video timestamp in seconds where the crossing was verified.
    side : str | None
        Associated zone or side ("LEFT", "WALKWAY", "RIGHT") for events.
    """

    event_type: EventType
    track_id: int
    frame_id: int
    timestamp: float
    side: Optional[str] = None



@dataclass(frozen=True, slots=True)
class CountingConfig:
    """Configuration for virtual counting line and dual parallel walkway divider lines.

    Attributes
    ----------
    line_start : tuple[float, float]
        (x, y) start coordinate in original frame pixels.
    line_end : tuple[float, float]
        (x, y) end coordinate in original frame pixels.
    inside_direction : str
        Which half-plane represents INSIDE ("positive" or "negative").
    hysteresis_distance : float
        Perpendicular pixel distance beyond lines required to confirm
        a complete crossing and prevent jitter/oscillation on boundaries.
    perpendicular_line_start : tuple[float, float] | None
        Optional explicit (x, y) start coordinate of the center perpendicular line.
    perpendicular_line_end : tuple[float, float] | None
        Optional explicit (x, y) end coordinate of the center perpendicular line.
    perpendicular_line_length : float
        Default length of perpendicular lines when auto-computed (pixels).
    walkway_width : float
        Width of the central aisle / walkway between Left and Right dividers (pixels).
    divider_left_start : tuple[float, float] | None
        Optional explicit start coordinate of Left Divider line.
    divider_left_end : tuple[float, float] | None
        Optional explicit end coordinate of Left Divider line.
    divider_right_start : tuple[float, float] | None
        Optional explicit start coordinate of Right Divider line.
    divider_right_end : tuple[float, float] | None
        Optional explicit end coordinate of Right Divider line.
    """

    line_start: Tuple[float, float] = (0.0, 80.0)
    line_end: Tuple[float, float] = (1440.0, 1080.0)
    inside_direction: str = "positive"
    hysteresis_distance: float = 12.0
    perpendicular_line_start: Optional[Tuple[float, float]] = None
    perpendicular_line_end: Optional[Tuple[float, float]] = None
    perpendicular_line_length: float = 1700.0
    walkway_width: float = 160.0
    divider_left_start: Optional[Tuple[float, float]] = None
    divider_left_end: Optional[Tuple[float, float]] = None
    divider_right_start: Optional[Tuple[float, float]] = None
    divider_right_end: Optional[Tuple[float, float]] = None

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
        if self.perpendicular_line_length <= 0:
            raise CountingConfigError("perpendicular_line_length must be positive.")
        if self.walkway_width <= 0:
            raise CountingConfigError("walkway_width must be positive.")
        if self.perpendicular_line_start is not None and self.perpendicular_line_end is not None:
            if len(self.perpendicular_line_start) != 2 or len(self.perpendicular_line_end) != 2:
                raise CountingConfigError("perpendicular_line_start and perpendicular_line_end must be 2D points.")
            if self.perpendicular_line_start == self.perpendicular_line_end:
                raise CountingConfigError("perpendicular_line_start and perpendicular_line_end must not be identical.")

    def get_perpendicular_line(
        self, length: Optional[float] = None
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate or return the (start, end) coordinates of the central perpendicular divider line."""
        if self.perpendicular_line_start is not None and self.perpendicular_line_end is not None:
            return (self.perpendicular_line_start, self.perpendicular_line_end)

        x1, y1 = self.line_start
        x2, y2 = self.line_end
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        dx = x2 - x1
        dy = y2 - y1
        line_len = math.hypot(dx, dy) or 1.0

        # Normal vector perpendicular to counting line: (-dy/L, dx/L)
        nx = -dy / line_len
        ny = dx / line_len

        eff_len = length if length is not None else self.perpendicular_line_length
        half_len = eff_len / 2.0

        p1 = (mid_x - nx * half_len, mid_y - ny * half_len)
        p2 = (mid_x + nx * half_len, mid_y + ny * half_len)
        return (p1, p2)

    def get_left_divider_line(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Coordinates (start, end) of the Left Divider line (separates Left Seating from Walkway)."""
        if self.divider_left_start is not None and self.divider_left_end is not None:
            return (self.divider_left_start, self.divider_left_end)

        (c1x, c1y), (c2x, c2y) = self.get_perpendicular_line()
        cdx = c2x - c1x
        cdy = c2y - c1y
        clen = math.hypot(cdx, cdy) or 1.0

        # Offset vector perpendicular to divider line pointing leftward (towards line_start)
        # Normal to divider vector (cdx, cdy): (-cdy/L, cdx/L)
        nx = -cdy / clen
        ny = cdx / clen

        # Ensure normal points toward line_start (negative cross product with counting line)
        (lx1, ly1) = self.line_start
        (lx2, ly2) = self.line_end
        ldx = lx2 - lx1
        ldy = ly2 - ly1
        # Dot product with counting line direction (ldx, ldy)
        dot = nx * ldx + ny * ldy
        if dot > 0:
            nx, ny = -nx, -ny

        half_w = self.walkway_width / 2.0
        p1 = (c1x + nx * half_w, c1y + ny * half_w)
        p2 = (c2x + nx * half_w, c2y + ny * half_w)
        return (p1, p2)

    def get_right_divider_line(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Coordinates (start, end) of the Right Divider line (separates Walkway from Right Seating)."""
        if self.divider_right_start is not None and self.divider_right_end is not None:
            return (self.divider_right_start, self.divider_right_end)

        (c1x, c1y), (c2x, c2y) = self.get_perpendicular_line()
        cdx = c2x - c1x
        cdy = c2y - c1y
        clen = math.hypot(cdx, cdy) or 1.0

        nx = -cdy / clen
        ny = cdx / clen

        (lx1, ly1) = self.line_start
        (lx2, ly2) = self.line_end
        ldx = lx2 - lx1
        ldy = ly2 - ly1
        dot = nx * ldx + ny * ldy
        if dot < 0:
            nx, ny = -nx, -ny

        half_w = self.walkway_width / 2.0
        p1 = (c1x + nx * half_w, c1y + ny * half_w)
        p2 = (c2x + nx * half_w, c2y + ny * half_w)
        return (p1, p2)

    def get_intersection(self) -> Tuple[float, float]:
        """Compute the 2D intersection coordinate (x, y) between the counting line
        and the central perpendicular divider line.
        """
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        (x3, y3), (x4, y4) = self.get_perpendicular_line()

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t = t_num / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    def classify_point_divider_side(self, px: float, py: float) -> str:
        """Classify a 2D point as 'LEFT' or 'RIGHT' relative to the central perpendicular divider line."""
        (x3, y3), (x4, y4) = self.get_perpendicular_line()
        cross = (px - x3) * (y4 - y3) - (py - y3) * (x4 - x3)
        return "LEFT" if cross < 0 else "RIGHT"

    def classify_point_zone(self, px: float, py: float) -> str:
        """Classify a 2D point as 'LEFT', 'WALKWAY', or 'RIGHT' based on the dual parallel divider lines."""
        (lx1, ly1), (lx2, ly2) = self.get_left_divider_line()
        cross_left = (px - lx1) * (ly2 - ly1) - (py - ly1) * (lx2 - lx1)
        if cross_left < 0:
            return "LEFT"

        (rx1, ry1), (rx2, ry2) = self.get_right_divider_line()
        cross_right = (px - rx1) * (ry2 - ry1) - (py - ry1) * (rx2 - rx1)
        if cross_right > 0:
            return "RIGHT"

        return "WALKWAY"


