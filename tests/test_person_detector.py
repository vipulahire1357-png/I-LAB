"""
Unit tests for Module 3 — Person Detection.

Run with::

    python -m pytest tests/test_person_detector.py -v

Uses real YOLO11n model.  Tests that require a person in the image
generate a synthetic "crowd-like" image or download a public test image.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection import (
    DetectionConfig,
    DetectionResult,
    InvalidFrameError,
    ModelLoadError,
    PersonDetection,
    PersonDetector,
    PersonDetectionError,
)
from src.video.models import Frame


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="session")
def detector() -> PersonDetector:
    """Session-scoped detector so the model is loaded only once."""
    config = DetectionConfig(
        model_path="yolo11n.pt",
        confidence_threshold=0.25,
        device="cpu",
    )
    det = PersonDetector(config)
    det.warmup()
    return det


def _make_frame(
    image: np.ndarray,
    frame_id: int = 0,
    timestamp: float = 0.0,
) -> Frame:
    """Helper to create a Frame from a raw numpy image."""
    return Frame(frame_id=frame_id, timestamp=timestamp, image=image)


def _blank_frame(
    width: int = 640,
    height: int = 480,
    frame_id: int = 0,
) -> Frame:
    """A solid-colour frame with no people."""
    img = np.full((height, width, 3), fill_value=200, dtype=np.uint8)
    return _make_frame(img, frame_id=frame_id)


def _person_like_frame(
    width: int = 640,
    height: int = 480,
    frame_id: int = 0,
) -> Frame:
    """Generate a frame with a simple person-shaped silhouette.

    This is NOT guaranteed to trigger a detection on every run since
    YOLO may or may not recognise a synthetic blob.  Tests that
    absolutely require detections should use a real image or skip.
    """
    import cv2

    img = np.full((height, width, 3), fill_value=220, dtype=np.uint8)

    # Draw a rough person shape: oval head + rectangle body.
    cx, cy = width // 2, height // 3
    # Head
    cv2.ellipse(img, (cx, cy), (25, 35), 0, 0, 360, (80, 60, 50), -1)
    # Body
    cv2.rectangle(img, (cx - 30, cy + 35), (cx + 30, cy + 160), (80, 60, 50), -1)
    # Legs
    cv2.rectangle(img, (cx - 30, cy + 160), (cx - 5, cy + 260), (80, 60, 50), -1)
    cv2.rectangle(img, (cx + 5, cy + 160), (cx + 30, cy + 260), (80, 60, 50), -1)

    return _make_frame(img, frame_id=frame_id)


# ======================================================================
# Test 1 — Model loading
# ======================================================================


class TestModelLoading:
    def test_model_loads(self, detector: PersonDetector):
        assert detector.is_loaded

    def test_device_is_cpu(self, detector: PersonDetector):
        assert detector.config.device == "cpu"

    def test_bad_model_path_raises(self):
        with pytest.raises(ModelLoadError):
            PersonDetector(DetectionConfig(model_path="nonexistent_model_xyz.pt"))


# ======================================================================
# Test 2 — Valid frame returns DetectionResult
# ======================================================================


class TestValidFrame:
    def test_returns_detection_result(self, detector: PersonDetector):
        frame = _blank_frame()
        result = detector.detect(frame)
        assert isinstance(result, DetectionResult)

    def test_result_has_timing(self, detector: PersonDetector):
        frame = _blank_frame()
        result = detector.detect(frame)
        assert result.inference_time_ms >= 0
        assert result.total_time_ms >= 0


# ======================================================================
# Test 3 — No-person frame
# ======================================================================


class TestNoPersonFrame:
    def test_blank_frame_returns_empty_detections(self, detector: PersonDetector):
        frame = _blank_frame()
        result = detector.detect(frame)
        assert result.detections == ()
        assert result.count == 0

    def test_no_exception_on_empty(self, detector: PersonDetector):
        frame = _blank_frame()
        # Should not raise.
        _ = detector.detect(frame)


# ======================================================================
# Test 4 — Person-only filtering
# ======================================================================


class TestPersonFiltering:
    def test_all_detections_are_person_class(self, detector: PersonDetector):
        """Any detection returned must have class_id == person_class_id."""
        frame = _person_like_frame()
        result = detector.detect(frame)
        for det in result.detections:
            assert det.class_id == detector.config.person_class_id


# ======================================================================
# Test 5 — Confidence filtering
# ======================================================================


class TestConfidenceFiltering:
    def test_all_above_threshold(self, detector: PersonDetector):
        frame = _person_like_frame()
        result = detector.detect(frame)
        for det in result.detections:
            assert det.confidence >= detector.config.confidence_threshold

    def test_high_threshold_reduces_detections(self):
        """With a very high threshold, fewer (or zero) detections."""
        strict = PersonDetector(
            DetectionConfig(confidence_threshold=0.99, device="cpu")
        )
        frame = _person_like_frame()
        result = strict.detect(frame)
        # At 0.99 threshold, synthetic frames should yield 0 detections.
        assert result.count == 0


# ======================================================================
# Test 6 — Bounding-box coordinate validity
# ======================================================================


class TestBoundingBoxCoordinates:
    def test_bbox_within_frame(self, detector: PersonDetector):
        w, h = 800, 600
        img = np.full((h, w, 3), 180, dtype=np.uint8)
        frame = _make_frame(img)
        result = detector.detect(frame)
        for det in result.detections:
            assert 0 <= det.x1 < det.x2 <= w, f"x out of range: {det.bbox}"
            assert 0 <= det.y1 < det.y2 <= h, f"y out of range: {det.bbox}"

    def test_bbox_has_positive_area(self, detector: PersonDetector):
        frame = _person_like_frame()
        result = detector.detect(frame)
        for det in result.detections:
            assert det.width > 0
            assert det.height > 0
            assert det.area > 0


# ======================================================================
# Test 7 — Original resolution coordinates
# ======================================================================


class TestOriginalResolutionCoordinates:
    def test_coords_match_original_not_model_input(self, detector: PersonDetector):
        """Frame is 1920×1080 (much larger than model input 640).
        Any returned bbox must be in 1920×1080 space."""
        w, h = 1920, 1080
        img = np.full((h, w, 3), 180, dtype=np.uint8)
        frame = _make_frame(img)
        result = detector.detect(frame)
        for det in result.detections:
            assert det.x2 <= w, f"x2={det.x2} exceeds frame width {w}"
            assert det.y2 <= h, f"y2={det.y2} exceeds frame height {h}"

    def test_small_frame(self, detector: PersonDetector):
        """Frame smaller than model input still returns valid coords."""
        w, h = 320, 240
        img = np.full((h, w, 3), 180, dtype=np.uint8)
        frame = _make_frame(img)
        result = detector.detect(frame)
        for det in result.detections:
            assert det.x2 <= w
            assert det.y2 <= h


# ======================================================================
# Test 8 — Frame metadata preservation
# ======================================================================


class TestFrameMetadataPreservation:
    def test_frame_id_preserved(self, detector: PersonDetector):
        frame = _blank_frame(frame_id=123)
        result = detector.detect(frame)
        assert result.frame_id == 123

    def test_timestamp_preserved(self, detector: PersonDetector):
        frame = _make_frame(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=42,
            timestamp=8.2,
        )
        result = detector.detect(frame)
        assert result.frame_id == 42
        assert abs(result.timestamp - 8.2) < 1e-6


# ======================================================================
# Test 9 — Repeated inference (model not reloaded)
# ======================================================================


class TestRepeatedInference:
    def test_multiple_frames_same_detector(self, detector: PersonDetector):
        results = []
        for i in range(5):
            frame = _blank_frame(frame_id=i)
            r = detector.detect(frame)
            results.append(r)
        assert len(results) == 5
        assert all(isinstance(r, DetectionResult) for r in results)
        # Model should still be loaded.
        assert detector.is_loaded


# ======================================================================
# Test 10 — Invalid input
# ======================================================================


class TestInvalidInput:
    def test_none_frame_raises(self, detector: PersonDetector):
        with pytest.raises(InvalidFrameError):
            detector.detect(None)  # type: ignore[arg-type]

    def test_empty_image_raises(self, detector: PersonDetector):
        frame = Frame(frame_id=0, timestamp=0.0, image=np.array([]))
        with pytest.raises(InvalidFrameError):
            detector.detect(frame)

    def test_1d_image_raises(self, detector: PersonDetector):
        frame = Frame(frame_id=0, timestamp=0.0, image=np.zeros(100))
        with pytest.raises(InvalidFrameError):
            detector.detect(frame)


# ======================================================================
# Additional tests
# ======================================================================


class TestConfigValidation:
    def test_bad_confidence(self):
        with pytest.raises(PersonDetectionError):
            DetectionConfig(confidence_threshold=1.5)

    def test_bad_iou(self):
        with pytest.raises(PersonDetectionError):
            DetectionConfig(iou_threshold=-0.1)

    def test_bad_input_size(self):
        with pytest.raises(PersonDetectionError):
            DetectionConfig(input_size=0)

    def test_empty_model_path(self):
        with pytest.raises(PersonDetectionError):
            DetectionConfig(model_path="")


class TestPersonDetectionModel:
    def test_center_property(self):
        d = PersonDetection(bbox=(100, 200, 300, 400), confidence=0.9, class_id=0)
        cx, cy = d.center
        assert cx == 200.0
        assert cy == 300.0

    def test_width_height(self):
        d = PersonDetection(bbox=(100, 200, 300, 500), confidence=0.9, class_id=0)
        assert d.width == 200.0
        assert d.height == 300.0

    def test_repr(self):
        d = PersonDetection(bbox=(10, 20, 100, 200), confidence=0.85, class_id=0)
        r = repr(d)
        assert "PersonDetection" in r
        assert "0.85" in r


class TestDetectionResultModel:
    def test_count(self):
        r = DetectionResult(
            frame_id=0,
            timestamp=0.0,
            detections=(
                PersonDetection(bbox=(0, 0, 50, 50), confidence=0.9, class_id=0),
                PersonDetection(bbox=(60, 60, 120, 120), confidence=0.8, class_id=0),
            ),
        )
        assert r.count == 2

    def test_repr(self):
        r = DetectionResult(frame_id=5, timestamp=1.5, detections=(), inference_time_ms=12.3)
        assert "frame_id=5" in repr(r)


class TestDetectionOrdering:
    def test_sorted_by_confidence_desc(self, detector: PersonDetector):
        """If any detections are returned, they must be sorted by
        confidence descending."""
        frame = _person_like_frame()
        result = detector.detect(frame)
        confs = [d.confidence for d in result.detections]
        assert confs == sorted(confs, reverse=True)

