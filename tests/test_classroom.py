"""
Tests for classroom-optimised detection and tracking.

Covers:
- Scene profile configuration
- Far-ROI config validation
- ROI coordinate mapping
- Detection fusion (IoU de-dup)
- Classroom tracker config
- Classroom default factories

Run with::

    python -m pytest tests/test_classroom.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.models import (
    DetectionConfig,
    DetectionResult,
    PersonDetection,
    SceneProfile,
)
from src.detection.person_detector import PersonDetector
from src.detection.exceptions import PersonDetectionError
from src.tracking.models import TrackerConfig
from src.video.models import Frame


# ======================================================================
# SceneProfile enum
# ======================================================================


class TestSceneProfile:
    def test_general_value(self):
        assert SceneProfile.GENERAL.value == "general"

    def test_classroom_value(self):
        assert SceneProfile.CLASSROOM.value == "classroom"


# ======================================================================
# DetectionConfig — scene profile fields
# ======================================================================


class TestDetectionConfigScene:
    def test_default_is_general(self):
        c = DetectionConfig()
        assert c.scene_profile == "general"
        assert c.is_classroom is False

    def test_classroom_profile(self):
        c = DetectionConfig(scene_profile="classroom")
        assert c.is_classroom is True

    def test_invalid_scene_profile(self):
        with pytest.raises(PersonDetectionError):
            DetectionConfig(scene_profile="unknown")

    def test_classroom_far_roi_valid(self):
        c = DetectionConfig(
            scene_profile="classroom",
            classroom_far_roi=(0.0, 0.0, 1.0, 0.45),
        )
        assert c.classroom_far_roi == (0.0, 0.0, 1.0, 0.45)

    def test_classroom_far_roi_invalid_order(self):
        with pytest.raises(PersonDetectionError):
            DetectionConfig(classroom_far_roi=(0.5, 0.0, 0.3, 1.0))  # x1 > x2

    def test_classroom_far_roi_out_of_range(self):
        with pytest.raises(PersonDetectionError):
            DetectionConfig(classroom_far_roi=(0.0, 0.0, 1.5, 0.5))  # x2 > 1.0

    def test_classroom_default_factory(self):
        c = DetectionConfig.classroom_default()
        assert c.is_classroom
        assert c.model_path == "yolo11s.pt"
        assert c.input_size == 1280
        assert c.confidence_threshold == 0.20
        assert c.classroom_far_roi is not None
        assert c.classroom_roi_confidence == 0.15

    def test_classroom_default_roi_valid_range(self):
        c = DetectionConfig.classroom_default()
        rx1, ry1, rx2, ry2 = c.classroom_far_roi
        assert 0.0 <= rx1 < rx2 <= 1.0
        assert 0.0 <= ry1 < ry2 <= 1.0


# ======================================================================
# Detection fusion (IoU dedup)
# ======================================================================


class TestDetectionFusion:
    """Test the internal fusion logic via PersonDetector._fuse_detections."""

    def _make_detector(self):
        # Use general config to avoid loading yolo11s for fusion tests
        return PersonDetector(DetectionConfig(
            model_path="yolo11n.pt",
            confidence_threshold=0.35,
        ))

    def test_no_roi_detections(self):
        det = self._make_detector()
        global_dets = [
            PersonDetection(bbox=(100, 100, 200, 300), confidence=0.9, class_id=0),
        ]
        fused = det._fuse_detections(global_dets, [])
        assert len(fused) == 1
        assert fused[0].confidence == 0.9

    def test_roi_adds_new_detection(self):
        det = self._make_detector()
        global_dets = [
            PersonDetection(bbox=(100, 100, 200, 300), confidence=0.9, class_id=0),
        ]
        roi_dets = [
            PersonDetection(bbox=(500, 50, 600, 200), confidence=0.4, class_id=0),
        ]
        fused = det._fuse_detections(global_dets, roi_dets)
        assert len(fused) == 2

    def test_duplicate_suppression(self):
        det = self._make_detector()
        # Same box detected by both passes
        global_dets = [
            PersonDetection(bbox=(100, 100, 200, 300), confidence=0.8, class_id=0),
        ]
        roi_dets = [
            PersonDetection(bbox=(102, 102, 198, 298), confidence=0.6, class_id=0),
        ]
        fused = det._fuse_detections(global_dets, roi_dets)
        # Should merge into one (IoU high → duplicate)
        assert len(fused) == 1
        # Keep the higher confidence
        assert fused[0].confidence == 0.8

    def test_roi_replaces_when_higher_confidence(self):
        det = self._make_detector()
        global_dets = [
            PersonDetection(bbox=(100, 100, 200, 300), confidence=0.3, class_id=0),
        ]
        roi_dets = [
            PersonDetection(bbox=(101, 101, 199, 299), confidence=0.7, class_id=0),
        ]
        fused = det._fuse_detections(global_dets, roi_dets)
        assert len(fused) == 1
        assert fused[0].confidence == 0.7  # ROI had higher conf

    def test_mixed_new_and_duplicate(self):
        det = self._make_detector()
        global_dets = [
            PersonDetection(bbox=(100, 100, 200, 300), confidence=0.8, class_id=0),
        ]
        roi_dets = [
            # Duplicate of global
            PersonDetection(bbox=(102, 102, 198, 298), confidence=0.5, class_id=0),
            # Genuinely new (no overlap)
            PersonDetection(bbox=(600, 50, 700, 150), confidence=0.35, class_id=0),
        ]
        fused = det._fuse_detections(global_dets, roi_dets)
        assert len(fused) == 2  # 1 merged + 1 new

    def test_fused_sorted_by_confidence(self):
        det = self._make_detector()
        global_dets = [
            PersonDetection(bbox=(100, 100, 200, 300), confidence=0.5, class_id=0),
        ]
        roi_dets = [
            PersonDetection(bbox=(500, 50, 600, 200), confidence=0.9, class_id=0),
        ]
        fused = det._fuse_detections(global_dets, roi_dets)
        assert fused[0].confidence >= fused[-1].confidence


# ======================================================================
# IoU computation
# ======================================================================


class TestIoUComputation:
    def test_identical_boxes(self):
        iou = PersonDetector._compute_iou(
            (100, 100, 200, 200), (100, 100, 200, 200)
        )
        assert abs(iou - 1.0) < 1e-6

    def test_no_overlap(self):
        iou = PersonDetector._compute_iou(
            (0, 0, 50, 50), (100, 100, 150, 150)
        )
        assert iou == 0.0

    def test_partial_overlap(self):
        iou = PersonDetector._compute_iou(
            (0, 0, 100, 100), (50, 50, 150, 150)
        )
        # intersection = 50×50 = 2500, union = 10000+10000-2500 = 17500
        assert abs(iou - 2500 / 17500) < 1e-4

    def test_contained_box(self):
        iou = PersonDetector._compute_iou(
            (0, 0, 200, 200), (50, 50, 150, 150)
        )
        # inner area=10000, outer area=40000, intersection=10000
        # union=40000+10000-10000=40000
        assert abs(iou - 10000 / 40000) < 1e-4


# ======================================================================
# ROI coordinate mapping
# ======================================================================


class TestROICoordinateMapping:
    """Verify that ROI-local coords are correctly mapped to original frame."""

    def _make_classroom_detector(self):
        return PersonDetector(DetectionConfig(
            model_path="yolo11n.pt",  # use nano for speed in tests
            scene_profile="classroom",
            confidence_threshold=0.20,
            input_size=640,
            classroom_far_roi=(0.0, 0.0, 1.0, 0.5),
            classroom_roi_input_size=640,
            classroom_roi_confidence=0.15,
        ))

    def test_roi_mapping_offset(self):
        """Manually verify the offset math."""
        det = self._make_classroom_detector()
        # Simulate: ROI is top half of a 1920×1080 frame → (0, 0, 1920, 540)
        # A detection at ROI-local (100, 50, 200, 150)
        # should map to original (100+0, 50+0, 200+0, 150+0) = (100, 50, 200, 150)
        # because roi_x1=0, roi_y1=0

        # For a different ROI offset: (0.1, 0.2, 0.9, 0.5) on 1000×1000
        # roi_x1=100, roi_y1=200, roi_x2=900, roi_y2=500
        # A detection at ROI-local (50, 30, 150, 180)
        # maps to original (50+100, 30+200, 150+100, 180+200) = (150, 230, 250, 380)

        # We test via the internal method
        class MockResult:
            def __init__(self, xyxy, conf, cls):
                import torch
                _xyxy = torch.tensor(xyxy, dtype=torch.float32)
                _conf = torch.tensor(conf, dtype=torch.float32)
                _cls = torch.tensor(cls, dtype=torch.float32)
                self.boxes = type("B", (), {
                    "xyxy": _xyxy,
                    "conf": _conf,
                    "cls": _cls,
                    "__len__": lambda self_: len(_xyxy),
                })()

        frame = Frame(frame_id=0, timestamp=0.0,
                      image=np.zeros((1000, 1000, 3), dtype=np.uint8))
        mock = MockResult(
            xyxy=[[50.0, 30.0, 150.0, 180.0]],
            conf=[0.7],
            cls=[0.0],
        )
        dets = det._parse_roi_results([mock], frame, 100, 200, 900, 500)
        assert len(dets) == 1
        d = dets[0]
        assert abs(d.x1 - 150.0) < 1.0
        assert abs(d.y1 - 230.0) < 1.0
        assert abs(d.x2 - 250.0) < 1.0
        assert abs(d.y2 - 380.0) < 1.0


# ======================================================================
# Classroom tracker config
# ======================================================================


class TestClassroomTrackerConfig:
    def test_classroom_default_factory(self):
        c = TrackerConfig.classroom_default()
        assert c.tracker_type == "bytetrack"
        assert c.max_age == 60
        assert c.min_hits == 2
        assert c.track_high_threshold == 0.15
        assert c.match_threshold == 0.70

    def test_classroom_default_custom_fps(self):
        c = TrackerConfig.classroom_default(frame_rate=30)
        assert c.frame_rate == 30

    def test_classroom_default_is_valid(self):
        # Should not raise
        c = TrackerConfig.classroom_default()
        assert c.max_age >= 1
        assert c.min_hits >= 1

