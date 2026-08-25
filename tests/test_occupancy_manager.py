"""
Unit tests for the OccupancyManager module.
"""

from __future__ import annotations

import pytest

from src.counting.models import CrossingEvent, EventType
from src.occupancy.models import ClassroomState
from src.occupancy.occupancy_manager import OccupancyManager


class TestOccupancyManager:
    def test_initial_state(self):
        mgr = OccupancyManager()
        state = mgr.get_state()

        assert state.occupancy_count == 0
        assert state.total_entries == 0
        assert state.total_exits == 0
        assert state.classroom_state == ClassroomState.EMPTY
        assert state.is_empty is True
        assert state.is_occupied is False

    def test_enter_increments_occupancy(self):
        mgr = OccupancyManager()
        e1 = CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=10, timestamp=0.5)
        state = mgr.apply([e1])

        assert state.occupancy_count == 1
        assert state.total_entries == 1
        assert state.total_exits == 0
        assert state.classroom_state == ClassroomState.OCCUPIED
        assert state.is_occupied is True

    def test_exit_decrements_occupancy(self):
        mgr = OccupancyManager(initial_occupancy=2)
        e1 = CrossingEvent(event_type=EventType.EXIT, track_id=1, frame_id=10, timestamp=0.5)
        state = mgr.apply([e1])

        assert state.occupancy_count == 1
        assert state.total_entries == 0
        assert state.total_exits == 1
        assert state.classroom_state == ClassroomState.OCCUPIED

        # Second exit -> classroom becomes EMPTY
        e2 = CrossingEvent(event_type=EventType.EXIT, track_id=2, frame_id=20, timestamp=1.0)
        state2 = mgr.apply([e2])
        assert state2.occupancy_count == 0
        assert state2.total_exits == 2
        assert state2.classroom_state == ClassroomState.EMPTY

    def test_occupancy_never_becomes_negative(self):
        mgr = OccupancyManager(initial_occupancy=0)
        # Apply 3 exit events even when occupancy is 0
        exits = [
            CrossingEvent(event_type=EventType.EXIT, track_id=i, frame_id=i * 10, timestamp=i * 0.5)
            for i in range(1, 4)
        ]
        state = mgr.apply(exits)

        assert state.occupancy_count == 0  # Invariant: never < 0
        assert state.total_exits == 3
        assert state.classroom_state == ClassroomState.EMPTY

    def test_multiple_mixed_events(self):
        mgr = OccupancyManager()
        events = [
            CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.1),
            CrossingEvent(event_type=EventType.ENTER, track_id=2, frame_id=1, timestamp=0.1),
            CrossingEvent(event_type=EventType.ENTER, track_id=3, frame_id=1, timestamp=0.1),
            CrossingEvent(event_type=EventType.EXIT, track_id=4, frame_id=1, timestamp=0.1),
        ]
        state = mgr.apply(events)

        assert state.total_entries == 3
        assert state.total_exits == 1
        assert state.occupancy_count == 2
        assert state.classroom_state == ClassroomState.OCCUPIED

    def test_reset(self):
        mgr = OccupancyManager()
        mgr.apply([CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.1)])
        mgr.reset(initial_occupancy=5)

        state = mgr.get_state()
        assert state.occupancy_count == 5
        assert state.total_entries == 0
        assert state.total_exits == 0
        assert state.classroom_state == ClassroomState.OCCUPIED
