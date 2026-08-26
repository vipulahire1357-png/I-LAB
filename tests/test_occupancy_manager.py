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

    def test_side_occupancy_state(self):
        mgr = OccupancyManager()
        state = mgr.apply(events=[], left_occupancy=2, right_occupancy=0)

        assert state.left_occupancy == 2
        assert state.right_occupancy == 0
        assert state.left_state == ClassroomState.OCCUPIED
        assert state.right_state == ClassroomState.EMPTY
        assert state.is_left_occupied is True
        assert state.is_right_occupied is False

        # Update next frame
        state2 = mgr.apply(events=[], left_occupancy=0, right_occupancy=3)
        assert state2.left_occupancy == 0
        assert state2.right_occupancy == 3
        assert state2.left_state == ClassroomState.EMPTY
        assert state2.right_state == ClassroomState.OCCUPIED
        assert state2.is_left_occupied is False
        assert state2.is_right_occupied is True

    def test_side_specific_enter_and_exit(self):
        mgr = OccupancyManager()
        e_left = CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.1, side="LEFT")
        e_right = CrossingEvent(event_type=EventType.ENTER, track_id=2, frame_id=1, timestamp=0.1, side="RIGHT")
        state = mgr.apply([e_left, e_right])

        assert state.occupancy_count == 2
        assert state.left_occupancy == 1
        assert state.right_occupancy == 1

        # Exit from left side
        e_exit_left = CrossingEvent(event_type=EventType.EXIT, track_id=1, frame_id=2, timestamp=0.2, side="LEFT")
        state2 = mgr.apply([e_exit_left])
        assert state2.occupancy_count == 1
        assert state2.left_occupancy == 0
        assert state2.right_occupancy == 1

    def test_zone_crossing_events(self):
        mgr = OccupancyManager()
        # 1 person enters on Left
        e_in = CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.1, side="LEFT")
        mgr.apply([e_in])

        # Person inside moves from Left -> Right
        e_cross = CrossingEvent(event_type=EventType.ZONE_CROSS_LEFT_TO_RIGHT, track_id=1, frame_id=2, timestamp=0.2, side="RIGHT")
        state = mgr.apply([e_cross])

        assert state.occupancy_count == 1
        assert state.left_occupancy == 0
        assert state.right_occupancy == 1

        # Person inside moves back from Right -> Left
        e_cross_back = CrossingEvent(event_type=EventType.ZONE_CROSS_RIGHT_TO_LEFT, track_id=1, frame_id=3, timestamp=0.3, side="LEFT")
        state2 = mgr.apply([e_cross_back])

        assert state2.occupancy_count == 1
        assert state2.left_occupancy == 1
        assert state2.right_occupancy == 0

    def test_persistent_occupancy_across_empty_frames(self):
        mgr = OccupancyManager()
        # 2 students enter (1 left, 1 right)
        e1 = CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.1, side="LEFT")
        e2 = CrossingEvent(event_type=EventType.ENTER, track_id=2, frame_id=2, timestamp=0.2, side="RIGHT")
        mgr.apply([e1, e2])

        # Next 50 frames: no new events (students are seated or occluded or out of frame)
        for f in range(3, 53):
            state = mgr.apply(events=[])
            # Counters must remain perfectly persistent
            assert state.occupancy_count == 2
            assert state.left_occupancy == 1
            assert state.right_occupancy == 1
            assert state.classroom_state == ClassroomState.OCCUPIED

        # Finally, student 1 exits
        e_exit = CrossingEvent(event_type=EventType.EXIT, track_id=1, frame_id=54, timestamp=5.4, side="LEFT")
        final_state = mgr.apply([e_exit])
        assert final_state.occupancy_count == 1
        assert final_state.left_occupancy == 0
        assert final_state.right_occupancy == 1

    def test_conservation_law_invariant(self):
        mgr = OccupancyManager()
        # Complex sequence: entries, zone crosses, exits
        events_seq = [
            [CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.1, side="LEFT")],
            [CrossingEvent(event_type=EventType.ENTER, track_id=2, frame_id=2, timestamp=0.2, side="LEFT")],
            [CrossingEvent(event_type=EventType.ENTER, track_id=3, frame_id=3, timestamp=0.3, side="RIGHT")],
            [CrossingEvent(event_type=EventType.ZONE_CROSS_LEFT_TO_RIGHT, track_id=1, frame_id=4, timestamp=0.4, side="RIGHT")],
            [CrossingEvent(event_type=EventType.EXIT, track_id=2, frame_id=5, timestamp=0.5, side="LEFT")],
        ]
        for evs in events_seq:
            state = mgr.apply(evs)
            assert state.left_occupancy + state.right_occupancy <= state.occupancy_count
            assert state.occupancy_count >= 0
            assert state.left_occupancy >= 0
            assert state.right_occupancy >= 0
            assert state.walkway_occupancy >= 0

    def test_walkway_occupancy_accounting(self):
        mgr = OccupancyManager()
        # 1. Student enters into Walkway
        e_enter_walkway = CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.1, side="WALKWAY")
        state1 = mgr.apply([e_enter_walkway])
        assert state1.occupancy_count == 1
        assert state1.left_occupancy == 0
        assert state1.right_occupancy == 0
        assert state1.walkway_occupancy == 1

        # 2. Student moves into Left Seating
        e_enter_left = CrossingEvent(event_type=EventType.ZONE_ENTER_LEFT, track_id=1, frame_id=2, timestamp=0.2, side="LEFT")
        state2 = mgr.apply([e_enter_left])
        assert state2.occupancy_count == 1
        assert state2.left_occupancy == 1
        assert state2.right_occupancy == 0
        assert state2.walkway_occupancy == 0

        # 3. Student moves back to Walkway
        e_exit_left = CrossingEvent(event_type=EventType.ZONE_EXIT_LEFT, track_id=1, frame_id=3, timestamp=0.3, side="WALKWAY")
        state3 = mgr.apply([e_exit_left])
        assert state3.occupancy_count == 1
        assert state3.left_occupancy == 0
        assert state3.right_occupancy == 0
        assert state3.walkway_occupancy == 1

        # 4. Student moves into Right Seating
        e_enter_right = CrossingEvent(event_type=EventType.ZONE_ENTER_RIGHT, track_id=1, frame_id=4, timestamp=0.4, side="RIGHT")
        state4 = mgr.apply([e_enter_right])
        assert state4.occupancy_count == 1
        assert state4.left_occupancy == 0
        assert state4.right_occupancy == 1
        assert state4.walkway_occupancy == 0

        # 5. Student exits room from Right Seating
        e_exit_room = CrossingEvent(event_type=EventType.EXIT, track_id=1, frame_id=5, timestamp=0.5, side="RIGHT")
        state5 = mgr.apply([e_exit_room])
        assert state5.occupancy_count == 0
        assert state5.left_occupancy == 0
        assert state5.right_occupancy == 0
        assert state5.walkway_occupancy == 0

    def test_occlusion_approximation_separated_from_occupancy(self):
        mgr = OccupancyManager()
        # 3 students enter
        events = [
            CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.1),
            CrossingEvent(event_type=EventType.ENTER, track_id=2, frame_id=1, timestamp=0.1),
            CrossingEvent(event_type=EventType.ENTER, track_id=3, frame_id=1, timestamp=0.1),
        ]
        # Occur with moderate occlusion (approximation = 1)
        state1 = mgr.apply(events, approximation=1)
        assert state1.occupancy_count == 3
        assert state1.occupancy_approximation == 1
        assert state1.classroom_state == ClassroomState.OCCUPIED

        # Frame with heavy occlusion (approximation = 2) but no new entries/exits
        state2 = mgr.apply([], approximation=2)
        assert state2.occupancy_count == 3  # Count strictly unaffected
        assert state2.occupancy_approximation == 2

        # Occlusion cleared (approximation = 0)
        state3 = mgr.apply([], approximation=0)
        assert state3.occupancy_count == 3
        assert state3.occupancy_approximation == 0

    def test_decoupled_area_and_divider_detection(self):
        mgr = OccupancyManager()
        # 2 students entered room
        e1 = CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.1)
        e2 = CrossingEvent(event_type=EventType.ENTER, track_id=2, frame_id=1, timestamp=0.1)
        mgr.apply([e1, e2])

        # Track 1 is sitting statically in Left Area (divider didn't catch or not crossed)
        # Track 2 crossed Right Divider
        e_right_div = CrossingEvent(event_type=EventType.ZONE_ENTER_RIGHT, track_id=2, frame_id=2, timestamp=0.2, side="RIGHT")
        state = mgr.apply([e_right_div], left_area_count=1, right_area_count=1)

        # Spatial Area states
        assert state.left_area_count == 1
        assert state.left_area_state == ClassroomState.OCCUPIED
        assert state.is_left_area_occupied is True

        assert state.right_area_count == 1
        assert state.right_area_state == ClassroomState.OCCUPIED
        assert state.is_right_area_occupied is True

        # Divider states
        assert state.left_divider_count == 0
        assert state.left_divider_state == ClassroomState.EMPTY
        assert state.right_divider_count == 1
        assert state.right_divider_state == ClassroomState.OCCUPIED

        # Room occupancy remains strictly governed by counting line
        assert state.occupancy_count == 2





