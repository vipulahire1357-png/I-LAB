"""
Unit tests for the EntryExitCounter module.
"""

from __future__ import annotations

import pytest

from src.counting.entry_exit_counter import EntryExitCounter
from src.counting.exceptions import CountingConfigError
from src.counting.models import CountingConfig, EventType, TrackSideState
from src.tracking.models import Track, TrackingResult, TrackState


def make_track(track_id: int, bbox: tuple[float, float, float, float], state: TrackState = TrackState.CONFIRMED) -> Track:
    return Track(
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        state=state,
        age=1,
        hits=1,
    )


def make_result(frame_id: int, tracks: list[Track], timestamp: float = 0.0) -> TrackingResult:
    return TrackingResult(
        frame_id=frame_id,
        timestamp=timestamp,
        tracks=tuple(tracks),
        tracking_time_ms=1.0,
        active_count=len([t for t in tracks if t.state == TrackState.CONFIRMED]),
    )


class TestEntryExitCounter:
    def test_invalid_config_raises_error(self):
        with pytest.raises(CountingConfigError):
            CountingConfig(line_start=(10, 10), line_end=(10, 10))

        with pytest.raises(CountingConfigError):
            CountingConfig(inside_direction="invalid_direction")

        with pytest.raises(CountingConfigError):
            CountingConfig(hysteresis_distance=-5.0)

    def test_outside_to_inside_generates_enter_and_handles_multiple_crossings(self):
        # Line horizontally at y=100 from x=0 to x=1000, inside is positive (y < 100)
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            hysteresis_distance=10.0,
        )
        counter = EntryExitCounter(config)

        # 1. Track 1 starts OUTSIDE at y2 = 200 (dist = -100 px)
        t1 = make_track(track_id=1, bbox=(50, 150, 100, 200))
        events1 = counter.update(make_result(frame_id=1, tracks=[t1], timestamp=0.1))
        assert len(events1) == 0
        assert counter._track_states[1] == TrackSideState.OUTSIDE

        # 2. Track 1 completely crosses to INSIDE at y2 = 50 (dist = +50 px > 10 px hysteresis)
        t1_moved = make_track(track_id=1, bbox=(50, 10, 100, 50))
        events2 = counter.update(make_result(frame_id=2, tracks=[t1_moved], timestamp=0.2))
        assert len(events2) == 1
        assert events2[0].event_type == EventType.ENTER
        assert events2[0].track_id == 1
        assert counter._track_states[1] == TrackSideState.INSIDE

        # 3. Track 1 stays inside -> 0 duplicate events
        t1_inside = make_track(track_id=1, bbox=(50, 5, 100, 45))
        events3 = counter.update(make_result(frame_id=3, tracks=[t1_inside], timestamp=0.3))
        assert len(events3) == 0

        # 4. Same person later exits back to OUTSIDE at y2 = 180 (dist = -80 px < -10 px)
        t1_exit = make_track(track_id=1, bbox=(50, 130, 100, 180))
        events4 = counter.update(make_result(frame_id=4, tracks=[t1_exit], timestamp=0.4))
        assert len(events4) == 1
        assert events4[0].event_type == EventType.EXIT
        assert events4[0].track_id == 1
        assert counter._track_states[1] == TrackSideState.OUTSIDE

        # 5. Same person re-enters class later at y2 = 40 (dist = +60 px > 10 px)
        t1_reenter = make_track(track_id=1, bbox=(50, 10, 100, 40))
        events5 = counter.update(make_result(frame_id=5, tracks=[t1_reenter], timestamp=0.5))
        assert len(events5) == 1
        assert events5[0].event_type == EventType.ENTER
        assert events5[0].track_id == 1
        assert counter._track_states[1] == TrackSideState.INSIDE

    def test_hysteresis_deadband_prevents_jitter(self):
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            hysteresis_distance=15.0,
        )
        counter = EntryExitCounter(config)

        # Starts outside at y2 = 150 (dist = -50 px)
        t = make_track(track_id=5, bbox=(50, 100, 100, 150))
        counter.update(make_result(frame_id=1, tracks=[t]))

        # Steps close to the line at y2 = 95 (dist = +5 px, which is within the 15px deadband)
        t_near = make_track(track_id=5, bbox=(50, 50, 100, 95))
        events = counter.update(make_result(frame_id=2, tracks=[t_near]))
        # Must not trigger event because crossing is not complete yet (deadband protection)
        assert len(events) == 0

        # Steps completely over the line at y2 = 80 (dist = +20 px > 15px hysteresis)
        t_over = make_track(track_id=5, bbox=(50, 30, 100, 80))
        events2 = counter.update(make_result(frame_id=3, tracks=[t_over]))
        assert len(events2) == 1
        assert events2[0].event_type == EventType.ENTER

    def test_multiple_independent_tracks(self):
        config = CountingConfig(line_start=(0.0, 100.0), line_end=(1000.0, 100.0), inside_direction="positive")
        counter = EntryExitCounter(config)

        # Frame 1: Track 10 is outside, Track 20 is inside
        t10 = make_track(track_id=10, bbox=(50, 150, 100, 200)) # outside
        t20 = make_track(track_id=20, bbox=(200, 10, 250, 50))   # inside
        counter.update(make_result(frame_id=1, tracks=[t10, t20]))

        # Frame 2: Both cross simultaneously!
        t10_in = make_track(track_id=10, bbox=(50, 10, 100, 50))    # enters
        t20_out = make_track(track_id=20, bbox=(200, 150, 250, 200)) # exits
        events = counter.update(make_result(frame_id=2, tracks=[t10_in, t20_out]))

        assert len(events) == 2
        event_types = {e.track_id: e.event_type for e in events}
        assert event_types[10] == EventType.ENTER
        assert event_types[20] == EventType.EXIT

    def test_unconfirmed_tracks_are_ignored(self):
        config = CountingConfig(line_start=(0.0, 100.0), line_end=(1000.0, 100.0))
        counter = EntryExitCounter(config)

        # Lost or unconfirmed track should not trigger state
        t_lost = make_track(track_id=99, bbox=(50, 150, 100, 200), state=TrackState.LOST)
        events = counter.update(make_result(frame_id=1, tracks=[t_lost]))
        assert len(events) == 0
        assert 99 not in counter._track_states

    def test_perpendicular_line_calculation(self):
        # Horizontal counting line from (0, 100) to (1000, 100)
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            perpendicular_line_length=400.0,
        )
        p1, p2 = config.get_perpendicular_line()
        # Midpoint is (500, 100). Perpendicular vector is vertical (x=500, y from 100-200 to 100+200)
        assert p1[0] == pytest.approx(500.0)
        assert p2[0] == pytest.approx(500.0)
        assert abs(p2[1] - p1[1]) == pytest.approx(400.0)

        # Custom explicit perpendicular line
        custom_config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            perpendicular_line_start=(500.0, 0.0),
            perpendicular_line_end=(500.0, 600.0),
        )
        cp1, cp2 = custom_config.get_perpendicular_line()
        assert cp1 == (500.0, 0.0)
        assert cp2 == (500.0, 600.0)

    def test_side_occupancy_classification(self):
        # Horizontal counting line from (0, 100) to (1000, 100), midpoint at x=500
        # Inside is y < 100
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            perpendicular_line_length=400.0,
        )
        counter = EntryExitCounter(config)

        # Track 1 at x=200, y2=50 (LEFT side of divider, INSIDE)
        t1 = make_track(track_id=1, bbox=(150.0, 10.0, 250.0, 50.0))
        # Track 2 at x=800, y2=50 (RIGHT side of divider, INSIDE)
        t2 = make_track(track_id=2, bbox=(750.0, 10.0, 850.0, 50.0))
        # Track 3 at x=100, y2=50 (LEFT side of divider, INSIDE)
        t3 = make_track(track_id=3, bbox=(50.0, 10.0, 150.0, 50.0))

        assert counter.classify_track_side(t1) == "LEFT"
        assert counter.classify_track_side(t2) == "RIGHT"
        assert counter.classify_track_side(t3) == "LEFT"

        result = make_result(frame_id=1, tracks=[t1, t2, t3])
        left_count, right_count = counter.compute_side_occupancy(result)
        assert left_count == 2
        assert right_count == 1


    def test_dual_dot_validation_crossing(self):
        # Line horizontally at y=100 from x=0 to x=1000, inside is positive (y < 100)
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            hysteresis_distance=10.0,
        )
        counter = EntryExitCounter(config)

        # Starts outside at y1=150, y2=250
        t1 = make_track(track_id=1, bbox=(50, 150, 100, 250))
        counter.update(make_result(frame_id=1, tracks=[t1]))

        # Enters: foot at y2=50 (<100), head at y1=10 (<100) -> validated ENTER
        t1_in = make_track(track_id=1, bbox=(50, 10, 100, 50))
        events = counter.update(make_result(frame_id=2, tracks=[t1_in]))
        assert len(events) == 1
        assert events[0].event_type == EventType.ENTER
        assert events[0].side == "LEFT"

    def test_inside_only_side_occupancy(self):
        # Line horizontally at y=100 (y < 100 is INSIDE, y > 100 is OUTSIDE)
        # Divider vertically at x=500 (x < 500 is LEFT, x > 500 is RIGHT)
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            perpendicular_line_start=(500.0, 0.0),
            perpendicular_line_end=(500.0, 1000.0),
        )
        counter = EntryExitCounter(config)

        # Track 1 inside at left: x=200, y2=50 (INSIDE, LEFT)
        t_in_left = make_track(track_id=1, bbox=(150, 10, 250, 50))
        # Track 2 outside at right: x=800, y2=200 (OUTSIDE, RIGHT)
        t_out_right = make_track(track_id=2, bbox=(750, 150, 850, 200))

        counter.update(make_result(frame_id=1, tracks=[t_in_left, t_out_right]))

        left_count, right_count = counter.compute_side_occupancy(
            make_result(frame_id=1, tracks=[t_in_left, t_out_right])
        )
        # Only track 1 is inside, so left=1, right=0 (outside person is excluded!)
        assert left_count == 1
        assert right_count == 0

    def test_occlusion_recovery_across_line(self):
        # Line horizontally at y=100 from x=0 to x=1000, inside is positive (y < 100)
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            hysteresis_distance=10.0,
        )
        counter = EntryExitCounter(config)

        # 1. Track starts outside at y2 = 180 (dist = -80 px)
        t_out = make_track(track_id=1, bbox=(50, 120, 100, 180))
        counter.update(make_result(frame_id=1, tracks=[t_out]))

        # 2. Track is occluded for 3 frames (skipped / LOST)
        counter.update(make_result(frame_id=2, tracks=[]))
        counter.update(make_result(frame_id=3, tracks=[]))

        # 3. Track reappears INSIDE at y2 = 40 (dist = +60 px)
        t_in = make_track(track_id=1, bbox=(50, 10, 100, 40))
        events = counter.update(make_result(frame_id=4, tracks=[t_in]))

        # Should recover and confirm ENTER event despite multi-frame occlusion gap
        assert len(events) == 1
        assert events[0].event_type == EventType.ENTER
        assert events[0].track_id == 1
        assert counter._track_states[1] == TrackSideState.INSIDE

    def test_divider_transitions_ignored_when_outside(self):
        # Line horizontally at y=100 (y < 100 is INSIDE, y > 100 is OUTSIDE)
        # Divider vertically at x=500 (x < 500 is LEFT, x > 500 is RIGHT)
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            perpendicular_line_start=(500.0, 0.0),
            perpendicular_line_end=(500.0, 1000.0),
            hysteresis_distance=10.0,
        )
        counter = EntryExitCounter(config)

        # Track 1 is OUTSIDE on Left (x=200, y2=200)
        t_out_left = make_track(track_id=1, bbox=(150, 150, 250, 200))
        counter.update(make_result(frame_id=1, tracks=[t_out_left]))

        # Track 1 walks OUTSIDE from Left to Right (x=800, y2=200)
        t_out_right = make_track(track_id=1, bbox=(750, 150, 850, 200))
        events = counter.update(make_result(frame_id=2, tracks=[t_out_right]))

        # No ZONE_CROSS events should fire because person is OUTSIDE the classroom!
        assert len(events) == 0
        assert counter._track_states[1] == TrackSideState.OUTSIDE

    def test_aisle_hover_does_not_trigger_zone_flicker(self):
        # Line horizontally at y=100 (y < 100 is INSIDE)
        # Divider vertically at x=500, walkway width 160px (Left divider at x=420, Right divider at x=580)
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            perpendicular_line_start=(500.0, 0.0),
            perpendicular_line_end=(500.0, 1000.0),
            walkway_width=160.0,
            hysteresis_distance=20.0,
        )
        counter = EntryExitCounter(config)

        # Step 1: Enters on LEFT side (x=300, y2=50)
        t_in_left = make_track(track_id=1, bbox=(250, 10, 350, 50))
        t_out = make_track(track_id=1, bbox=(250, 120, 350, 180))
        counter.update(make_result(frame_id=1, tracks=[t_out]))
        events1 = counter.update(make_result(frame_id=2, tracks=[t_in_left]))
        assert len(events1) == 1
        assert events1[0].event_type == EventType.ENTER
        assert events1[0].side == "LEFT"

        # Step 2: Moves near Left divider at x=425 (only +5px past Left Divider at 420, within 20px deadband)
        t_near_divider = make_track(track_id=1, bbox=(400, 10, 450, 50))
        events2 = counter.update(make_result(frame_id=3, tracks=[t_near_divider]))
        # Must NOT trigger zone exit because crossing is not beyond deadband yet
        assert len(events2) == 0

        # Step 3: Completely crosses Left divider into Walkway at x=460 (> 20px deadband)
        t_walkway = make_track(track_id=1, bbox=(435, 10, 485, 50))
        events3 = counter.update(make_result(frame_id=4, tracks=[t_walkway]))
        assert len(events3) == 1
        assert events3[0].event_type == EventType.ZONE_EXIT_LEFT

        # Step 4: Completely crosses Right divider into RIGHT seating at x=610 (> 20px deadband past Right Divider at 580)
        t_right = make_track(track_id=1, bbox=(585, 10, 635, 50))
        events4 = counter.update(make_result(frame_id=5, tracks=[t_right]))
        assert len(events4) == 1
        assert events4[0].event_type == EventType.ZONE_ENTER_RIGHT

    def test_walkway_dual_divider_zone_transitions(self):
        # Horizontal counting line y=100 (y < 100 is INSIDE)
        # Central divider at x=500, walkway width = 100px (Left divider at x=450, Right divider at x=550)
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            perpendicular_line_start=(500.0, 0.0),
            perpendicular_line_end=(500.0, 1000.0),
            walkway_width=100.0,
            hysteresis_distance=10.0,
        )
        counter = EntryExitCounter(config)

        # 1. Enters into central WALKWAY at x=500, y2=50
        t_out = make_track(track_id=1, bbox=(480, 120, 520, 180))
        counter.update(make_result(frame_id=1, tracks=[t_out]))

        t_walkway = make_track(track_id=1, bbox=(480, 10, 520, 50))
        events1 = counter.update(make_result(frame_id=2, tracks=[t_walkway]))
        assert len(events1) == 1
        assert events1[0].event_type == EventType.ENTER
        assert events1[0].side == "WALKWAY"

        # 2. Moves into LEFT seating (x=300, y2=50, past x=450 - 10 = 440)
        t_left = make_track(track_id=1, bbox=(280, 10, 320, 50))
        events2 = counter.update(make_result(frame_id=3, tracks=[t_left]))
        assert any(e.event_type == EventType.ZONE_ENTER_LEFT for e in events2)

        # 3. Moves back into WALKWAY (x=500, y2=50)
        t_walkway2 = make_track(track_id=1, bbox=(480, 10, 520, 50))
        events3 = counter.update(make_result(frame_id=4, tracks=[t_walkway2]))
        assert any(e.event_type == EventType.ZONE_EXIT_LEFT for e in events3)

        # 4. Moves into RIGHT seating (x=700, y2=50, past x=550 + 10 = 560)
        t_right = make_track(track_id=1, bbox=(680, 10, 720, 50))
        events4 = counter.update(make_result(frame_id=5, tracks=[t_right]))
        assert any(e.event_type == EventType.ZONE_ENTER_RIGHT for e in events4)

        # 5. Moves back into WALKWAY (x=500, y2=50)
        events5 = counter.update(make_result(frame_id=6, tracks=[t_walkway2]))
        assert any(e.event_type == EventType.ZONE_EXIT_RIGHT for e in events5)

    def test_compute_occlusion_approximation(self):
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            hysteresis_distance=15.0,
        )
        counter = EntryExitCounter(config)

        # 1 single track near line -> approximation = 0
        t1 = make_track(track_id=1, bbox=(100, 80, 150, 110))
        res1 = make_result(frame_id=1, tracks=[t1])
        assert counter.compute_occlusion_approximation(res1) == 0

        # 2 overlapping tracks near line -> approximation = 1
        t2 = make_track(track_id=2, bbox=(120, 85, 170, 115))
        res2 = make_result(frame_id=2, tracks=[t1, t2])
        assert counter.compute_occlusion_approximation(res2) == 1

        # 3 overlapping tracks near line -> approximation = 2
        t3 = make_track(track_id=3, bbox=(130, 90, 180, 120))
        res3 = make_result(frame_id=3, tracks=[t1, t2, t3])
        assert counter.compute_occlusion_approximation(res3) == 2

    def test_detect_area_occupants_static_people(self):
        config = CountingConfig(
            line_start=(0.0, 100.0),
            line_end=(1000.0, 100.0),
            inside_direction="positive",
            perpendicular_line_start=(500.0, 0.0),
            perpendicular_line_end=(500.0, 1000.0),
            walkway_width=100.0,
        )
        counter = EntryExitCounter(config)

        # 2 people inside: 1 sitting in Left area (x=200, y=50), 1 in Right area (x=800, y=50)
        # 1 person outside in corridor (x=200, y=200)
        t_left = make_track(track_id=10, bbox=(150, 10, 250, 50))
        t_right = make_track(track_id=20, bbox=(750, 10, 850, 50))
        t_out = make_track(track_id=30, bbox=(150, 150, 250, 200))

        res = make_result(frame_id=1, tracks=[t_left, t_right, t_out])
        left_cnt, right_cnt = counter.detect_area_occupants(res)

        assert left_cnt == 1
        assert right_cnt == 1






