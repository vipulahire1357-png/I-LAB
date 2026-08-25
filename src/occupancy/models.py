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
        Current estimated count of students in the classroom (guaranteed >= 0).
    total_entries : int
        Cumulative number of confirmed student entries since session start.
    total_exits : int
        Cumulative number of confirmed student exits since session start.
    classroom_state : ClassroomState
        OCCUPIED if occupancy_count > 0, else EMPTY.
    """

    occupancy_count: int
    total_entries: int
    total_exits: int
    classroom_state: ClassroomState

    @property
    def is_occupied(self) -> bool:
        """True if the classroom has at least one estimated student."""
        return self.classroom_state is ClassroomState.OCCUPIED

    @property
    def is_empty(self) -> bool:
        """True if the classroom is estimated to be empty."""
        return self.classroom_state is ClassroomState.EMPTY
