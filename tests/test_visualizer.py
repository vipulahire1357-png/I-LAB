"""
Unit tests for the Visualizer module.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.counting.models import CountingConfig, CrossingEvent, EventType
from src.occupancy.models import ClassroomState, OccupancyState
from src.tracking.models import Track, TrackingResult, TrackState
from src.video.models import Frame
from src.visualization.models import VisualizationConfig
from src.visualization.visualizer import Visualizer


class TestVisualizer:
    def test_annotate_returns_ndarray_with_same_shape(self):
        vis = Visualizer()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(frame_id=1, timestamp=0.04, image=image)

        tracks = (
            Track(track_id=1, bbox=(50.0, 50.0, 150.0, 250.0), confidence=0.85, state=TrackState.CONFIRMED),
        )
        tracking_result = TrackingResult(frame_id=1, timestamp=0.04, tracks=tracks, active_count=1)
        occupancy_state = OccupancyState(occupancy_count=1, total_entries=1, total_exits=0, classroom_state=ClassroomState.OCCUPIED)
        events = [CrossingEvent(event_type=EventType.ENTER, track_id=1, frame_id=1, timestamp=0.04)]
        counting_config = CountingConfig(line_start=(0.0, 200.0), line_end=(640.0, 200.0))

        annotated = vis.annotate(
            frame=frame,
            tracking_result=tracking_result,
            occupancy_state=occupancy_state,
            events=events,
            counting_config=counting_config,
            fps=25.0,
        )

        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == (480, 640, 3)
        assert annotated.dtype == np.uint8
        # Ensure the image was modified (annotated) and is not completely black
        assert np.any(annotated > 0)
        # Ensure original frame image was not mutated
        assert np.all(frame.image == 0)

    def test_annotate_empty_frame_safety(self):
        vis = Visualizer()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = Frame(frame_id=0, timestamp=0.0, image=image)
        tracking_result = TrackingResult(frame_id=0, timestamp=0.0, tracks=(), active_count=0)
        occupancy_state = OccupancyState(occupancy_count=0, total_entries=0, total_exits=0, classroom_state=ClassroomState.EMPTY)

        annotated = vis.annotate(
            frame=frame,
            tracking_result=tracking_result,
            occupancy_state=occupancy_state,
            events=[],
        )
        assert annotated.shape == (100, 100, 3)

    def test_annotate_with_perpendicular_line_and_side_occupancy(self):
        vis = Visualizer()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(frame_id=1, timestamp=0.04, image=image)

        tracks = (
            Track(track_id=1, bbox=(50.0, 50.0, 150.0, 250.0), confidence=0.85, state=TrackState.CONFIRMED),
        )
        tracking_result = TrackingResult(frame_id=1, timestamp=0.04, tracks=tracks, active_count=1)
        occupancy_state = OccupancyState(
            occupancy_count=1,
            total_entries=1,
            total_exits=0,
            classroom_state=ClassroomState.OCCUPIED,
            left_occupancy=1,
            right_occupancy=0,
            left_state=ClassroomState.OCCUPIED,
            right_state=ClassroomState.EMPTY,
        )
        counting_config = CountingConfig(
            line_start=(0.0, 200.0),
            line_end=(640.0, 200.0),
            perpendicular_line_start=(320.0, 0.0),
            perpendicular_line_end=(320.0, 480.0),
        )

        annotated = vis.annotate(
            frame=frame,
            tracking_result=tracking_result,
            occupancy_state=occupancy_state,
            events=[],
            counting_config=counting_config,
            fps=25.0,
        )

        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == (480, 640, 3)
        assert np.any(annotated > 0)

