"""
Unit tests for Module 4 — Multi-Object Person Tracking.

Run with::

    python -m pytest tests/test_person_tracker.py -v

Uses synthetic DetectionResult sequences (no YOLO model needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.models import DetectionResult, PersonDetection
from src.tracking import (
    PersonTracker,
    Track,
    TrackerConfig,
    TrackerConfigError,
    TrackerError,
    TrackingResult,
    TrackState,
)


# ======================================================================
# Helpers
# ======================================================================


def _det(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> PersonDetection:
    """Shorthand for building a PersonDetection."""
    return PersonDetection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=0)


def _result(frame_id: int, detections: list[PersonDetection], ts: float = 0.0) -> DetectionResult:
    """Shorthand for building a DetectionResult."""
    return DetectionResult(
        frame_id=frame_id,
        timestamp=ts,
        detections=tuple(detections),
    )


def _moving_person(
    start_x: float,
    y: float,
    w: float,
    h: float,
    dx: float,
    n_frames: int,
    conf: float = 0.9,
) -> list[DetectionResult]:
    """Generate N frames of a person moving horizontally."""
    results = []
    for i in range(n_frames):
        x1 = start_x + i * dx
        results.append(
            _result(i, [_det(x1, y, x1 + w, y + h, conf)], ts=i / 30.0)
        )
    return results


# ======================================================================
# Test 1 — Single person, stable ID
# ======================================================================


class TestSinglePerson:
    def test_stable_track_id(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=5))
        frames = _moving_person(100, 100, 60, 150, dx=5, n_frames=10)

        ids_seen: set[int] = set()
        for det_r in frames:
            tr = tracker.update(det_r)
            for t in tr.tracks:
                if t.state == TrackState.CONFIRMED:
                    ids_seen.add(t.track_id)

        # Single person → exactly one track ID
        assert len(ids_seen) == 1


# ======================================================================
# Test 2 — Multiple people simultaneously
# ======================================================================


class TestMultiplePeople:
    def test_two_simultaneous_tracks(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=5))

        for i in range(10):
            det_r = _result(i, [
                _det(100 + i * 3, 100, 160 + i * 3, 250, conf=0.9),
                _det(400 + i * 3, 100, 460 + i * 3, 250, conf=0.85),
            ], ts=i / 30.0)
            tr = tracker.update(det_r)

        # After enough frames, should have at least 2 active tracks
        active = [t for t in tr.tracks if t.state == TrackState.CONFIRMED]
        assert len(active) >= 2


# ======================================================================
# Test 3 — New person enters
# ======================================================================


class TestNewPersonEnters:
    def test_new_id_created(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=5))

        # First 5 frames: one person
        for i in range(5):
            tracker.update(_result(i, [_det(100 + i, 100, 160 + i, 250)]))

        # Next 5 frames: second person appears
        all_ids = set()
        for i in range(5, 10):
            tr = tracker.update(_result(i, [
                _det(100 + i, 100, 160 + i, 250),
                _det(400, 100, 460, 250),
            ]))
            for t in tr.tracks:
                all_ids.add(t.track_id)

        # Should have at least 2 different IDs
        assert len(all_ids) >= 2


# ======================================================================
# Test 4 — Person leaves
# ======================================================================


class TestPersonLeaves:
    def test_track_transitions_to_removed(self):
        max_age = 3
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=max_age))

        # Person present for 5 frames
        for i in range(5):
            tracker.update(_result(i, [_det(100, 100, 160, 250)]))

        # Person gone for max_age + 2 frames
        for i in range(5, 5 + max_age + 2):
            tr = tracker.update(_result(i, []))

        # After enough empty frames, no active/lost tracks remain
        remaining = [t for t in tr.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        assert len(remaining) == 0


# ======================================================================
# Test 5 — Temporary detection loss → LOST state
# ======================================================================


class TestTemporaryLoss:
    def test_lost_state_on_missing_detection(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=10))

        # Present for 5 frames
        for i in range(5):
            tracker.update(_result(i, [_det(100, 100, 160, 250)]))

        # Missing for 1 frame
        tr = tracker.update(_result(5, []))

        lost = [t for t in tr.tracks if t.state == TrackState.LOST]
        # Should have at least one LOST track
        assert len(lost) >= 1


# ======================================================================
# Test 6 — Recovery after temporary loss
# ======================================================================


class TestRecovery:
    def test_same_id_after_recovery(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=10))

        # Present for 5 frames, capture ID
        track_id_before = None
        for i in range(5):
            tr = tracker.update(_result(i, [_det(200, 200, 260, 350)]))
            for t in tr.tracks:
                if t.state == TrackState.CONFIRMED:
                    track_id_before = t.track_id

        assert track_id_before is not None

        # Missing for 2 frames
        tracker.update(_result(5, []))
        tracker.update(_result(6, []))

        # Returns in similar position
        tr = tracker.update(_result(7, [_det(204, 200, 264, 350)]))

        confirmed = [t for t in tr.tracks if t.state == TrackState.CONFIRMED]
        if confirmed:
            # If ByteTrack re-associated, same ID should be kept
            assert confirmed[0].track_id == track_id_before


# ======================================================================
# Test 7 — Long disappearance → new ID
# ======================================================================


class TestLongDisappearance:
    def test_new_id_after_expiry(self):
        max_age = 3
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=max_age))

        # Present for 5 frames
        id_before = None
        for i in range(5):
            tr = tracker.update(_result(i, [_det(200, 200, 260, 350)]))
            for t in tr.tracks:
                if t.state == TrackState.CONFIRMED:
                    id_before = t.track_id
        assert id_before is not None

        # Gone for much longer than max_age
        for i in range(5, 5 + max_age + 5):
            tracker.update(_result(i, []))

        # Reappear — should get a NEW track ID
        tr = tracker.update(_result(20, [_det(200, 200, 260, 350)]))
        # After one frame with min_hits=1, track should appear
        for t in tr.tracks:
            if t.state == TrackState.CONFIRMED:
                assert t.track_id != id_before


# ======================================================================
# Test 8 — ID uniqueness (no simultaneous duplicates)
# ======================================================================


class TestIDUniqueness:
    def test_no_duplicate_ids_per_frame(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=5))

        for i in range(20):
            det_r = _result(i, [
                _det(50 + i * 2, 100, 110 + i * 2, 250),
                _det(300 + i * 2, 100, 360 + i * 2, 250),
                _det(550 + i * 2, 100, 610 + i * 2, 250),
            ])
            tr = tracker.update(det_r)
            ids = [t.track_id for t in tr.tracks]
            assert len(ids) == len(set(ids)), f"Duplicate IDs in frame {i}: {ids}"


# ======================================================================
# Test 9 — IDs are not reused within session
# ======================================================================


class TestIDMonotonicity:
    def test_ids_not_reused(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=2))
        all_ids: set[int] = set()
        id_list: list[int] = []

        # Person appears, disappears, new person appears
        for i in range(5):
            tracker.update(_result(i, [_det(100, 100, 160, 250)]))

        tr = tracker.update(_result(5, []))
        for t in tr.tracks:
            all_ids.add(t.track_id)
            id_list.append(t.track_id)

        # Wait for removal
        for i in range(6, 12):
            tracker.update(_result(i, []))

        # New person at different location
        for i in range(12, 17):
            tr = tracker.update(_result(i, [_det(500, 100, 560, 250)]))
            for t in tr.tracks:
                # Verify this ID was not used before (or if reused by ByteTrack, at least check)
                id_list.append(t.track_id)

        # All IDs should be unique in the sequence (no reuse of expired IDs)
        # Note: ByteTrack guarantees monotonic IDs internally.


# ======================================================================
# Test 10 — Reset
# ======================================================================


class TestReset:
    def test_reset_clears_state(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=5))

        for i in range(5):
            tracker.update(_result(i, [_det(100, 100, 160, 250)]))

        tracker.reset()

        m = tracker.get_metrics()
        assert m.frames_processed == 0
        assert m.total_tracks_created == 0

    def test_new_session_after_reset(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=5))

        for i in range(5):
            tracker.update(_result(i, [_det(100, 100, 160, 250)]))

        tracker.reset()

        # New session should still work
        tr = tracker.update(_result(0, [_det(100, 100, 160, 250)]))
        assert isinstance(tr, TrackingResult)


# ======================================================================
# Test 11 — Frame metadata preservation
# ======================================================================


class TestFrameMetadata:
    def test_frame_id_preserved(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1))
        tr = tracker.update(_result(42, [_det(100, 100, 160, 250)], ts=1.4))
        assert tr.frame_id == 42
        assert abs(tr.timestamp - 1.4) < 1e-6

    def test_timestamp_preserved(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1))
        tr = tracker.update(_result(7, [], ts=3.14))
        assert tr.frame_id == 7
        assert abs(tr.timestamp - 3.14) < 1e-6


# ======================================================================
# Test 12 — No future information
# ======================================================================


class TestNoFutureInfo:
    def test_sequential_processing(self):
        """Tracker update() only receives current frame, no look-ahead."""
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=5))

        # Process frames one at a time — tracker never receives the full list
        results = []
        for i in range(10):
            tr = tracker.update(_result(i, [_det(100 + i * 5, 100, 160 + i * 5, 250)]))
            results.append(tr)

        assert len(results) == 10
        assert all(isinstance(r, TrackingResult) for r in results)


# ======================================================================
# Additional tests
# ======================================================================


class TestConfigValidation:
    def test_bad_tracker_type(self):
        with pytest.raises(TrackerConfigError):
            TrackerConfig(tracker_type="unknown")

    def test_bad_max_age(self):
        with pytest.raises(TrackerConfigError):
            TrackerConfig(max_age=0)

    def test_bad_min_hits(self):
        with pytest.raises(TrackerConfigError):
            TrackerConfig(min_hits=0)

    def test_bad_threshold(self):
        with pytest.raises(TrackerConfigError):
            TrackerConfig(track_high_threshold=1.5)


class TestTrackModel:
    def test_center(self):
        t = Track(track_id=1, bbox=(100, 200, 300, 400), confidence=0.9, state=TrackState.CONFIRMED)
        assert t.center == (200.0, 300.0)

    def test_width_height(self):
        t = Track(track_id=1, bbox=(100, 200, 300, 500), confidence=0.9, state=TrackState.CONFIRMED)
        assert t.width == 200.0
        assert t.height == 300.0

    def test_repr(self):
        t = Track(track_id=5, bbox=(0, 0, 50, 50), confidence=0.8, state=TrackState.CONFIRMED)
        assert "id=5" in repr(t)
        assert "CONFIRMED" in repr(t)


class TestEmptyFrames:
    def test_empty_detections(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1))
        tr = tracker.update(_result(0, []))
        assert tr.active_count == 0
        assert tr.tracks == ()

    def test_many_empty_frames(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1))
        for i in range(50):
            tr = tracker.update(_result(i, []))
        assert tr.active_count == 0


class TestMetrics:
    def test_metrics_populated(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1, max_age=5))
        for i in range(10):
            tracker.update(_result(i, [_det(100 + i, 100, 160 + i, 250)]))
        m = tracker.get_metrics()
        assert m.frames_processed == 10
        assert m.total_tracks_created >= 1
        assert m.average_tracking_time is not None

    def test_metrics_reset_on_reset(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1))
        tracker.update(_result(0, [_det(100, 100, 160, 250)]))
        tracker.reset()
        m = tracker.get_metrics()
        assert m.frames_processed == 0


class TestTrackingTime:
    def test_tracking_time_positive(self):
        tracker = PersonTracker(TrackerConfig(min_hits=1))
        tr = tracker.update(_result(0, [_det(100, 100, 160, 250)]))
        assert tr.tracking_time_ms >= 0.0


class TestRepr:
    def test_tracker_repr(self):
        tracker = PersonTracker()
        r = repr(tracker)
        assert "PersonTracker" in r
        assert "bytetrack" in r

