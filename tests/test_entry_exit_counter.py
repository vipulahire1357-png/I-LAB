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
