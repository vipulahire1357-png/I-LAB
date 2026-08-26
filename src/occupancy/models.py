"""
Data models for the Occupancy Manager module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ClassroomState(str, Enum):
    """High-level operational occupancy state of the classroom."""

    OCCUPIED = "OCCUPIED"
    EMPTY = "EMPTY"


@dataclass(frozen=True, slots=True)
class OccupancyState:
    """Snapshot of current classroom occupancy statistics and state.

    Attributes
    ----------
    occupancy_count : int
        Current estimated count of students in the classroom strictly decided by counting line (>= 0).
    occupancy_approximation : int
        Approximation / uncertainty magnitude (0 for ±0, 1 for ±1, 2 for ±2) when occlusion occurs.
    total_entries : int
        Cumulative number of confirmed student entries across counting line since session start.
    total_exits : int
        Cumulative number of confirmed student exits across counting line since session start.
    classroom_state : ClassroomState
        OCCUPIED if occupancy_count > 0, else EMPTY.
    left_area_state : ClassroomState
        Spatial presence in Left Seating Area (OCCUPIED if visible people > 0, else EMPTY).
    right_area_state : ClassroomState
        Spatial presence in Right Seating Area (OCCUPIED if visible people > 0, else EMPTY).
    left_area_count : int
        Number of confirmed people detected spatially in Left Seating Area (static or moving).
    right_area_count : int
        Number of confirmed people detected spatially in Right Seating Area (static or moving).
    left_divider_state : ClassroomState
        Divider line crossing state for Left area (OCCUPIED if divider entries > exits, else EMPTY).
    right_divider_state : ClassroomState
        Divider line crossing state for Right area (OCCUPIED if divider entries > exits, else EMPTY).
    left_divider_count : int
        Net divider crossing accumulator for Left area.
    right_divider_count : int
        Net divider crossing accumulator for Right area.
    walkway_occupancy : int
        Estimated occupant count in central walkway.
    """

    occupancy_count: int
    total_entries: int
    total_exits: int
    classroom_state: ClassroomState
    occupancy_approximation: int = 0
    left_area_state: ClassroomState = ClassroomState.EMPTY
    right_area_state: ClassroomState = ClassroomState.EMPTY
    left_area_count: int = 0
    right_area_count: int = 0
    left_divider_state: ClassroomState = ClassroomState.EMPTY
    right_divider_state: ClassroomState = ClassroomState.EMPTY
    left_divider_count: int = 0
    right_divider_count: int = 0
    left_occupancy: int = 0
    right_occupancy: int = 0
    left_state: ClassroomState = ClassroomState.EMPTY
    right_state: ClassroomState = ClassroomState.EMPTY
    walkway_occupancy: int = 0

    @property
    def is_occupied(self) -> bool:
        """True if the classroom has at least one estimated student."""
        return self.classroom_state is ClassroomState.OCCUPIED

    @property
    def is_empty(self) -> bool:
        """True if the classroom is estimated to be empty."""
        return self.classroom_state is ClassroomState.EMPTY

    @property
    def is_left_area_occupied(self) -> bool:
        """True if the left seating area has at least one detected person (static or moving)."""
        return self.left_area_state is ClassroomState.OCCUPIED

    @property
    def is_right_area_occupied(self) -> bool:
        """True if the right seating area has at least one detected person (static or moving)."""
        return self.right_area_state is ClassroomState.OCCUPIED

    @property
    def is_left_occupied(self) -> bool:
        """True if the left side is occupied."""
        return self.left_state is ClassroomState.OCCUPIED

    @property
    def is_right_occupied(self) -> bool:
        """True if the right side is occupied."""
        return self.right_state is ClassroomState.OCCUPIED

    @property
    def is_walkway_occupied(self) -> bool:
        """True if the walkway has at least one occupant."""
        return self.walkway_occupancy > 0

