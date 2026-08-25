"""
Unit tests for Module 1 — Video Source / Input.

Tests are designed to run with ``pytest`` from the project root::

    python -m pytest tests/ -v

A small synthetic test video is generated on-the-fly using OpenCV so that the
test suite is self-contained (no external fixture files required).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video import (
    FileVideoSource,
    Frame,
    InvalidVideoError,
    VideoMetadata,
    VideoOpenError,
    VideoSourceError,
)


# ======================================================================
# Fixtures
# ======================================================================

# Video parameters for the synthetic test clip.
_TEST_WIDTH = 320
_TEST_HEIGHT = 240
_TEST_FPS = 15.0
_TEST_FRAME_COUNT = 30  # 2 seconds at 15 fps


@pytest.fixture(scope="session")
def valid_video_path(tmp_path_factory) -> Path:
    """Generate a small synthetic MP4 video and return its path."""
    out_dir = tmp_path_factory.mktemp("videos")
    path = out_dir / "test_clip.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, _TEST_FPS, (_TEST_WIDTH, _TEST_HEIGHT))

    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter could not create an MP4 — codec not available")

    for i in range(_TEST_FRAME_COUNT):
        # Deterministic gradient frame so content is verifiable.
        frame = np.full((_TEST_HEIGHT, _TEST_WIDTH, 3), fill_value=i * 8 % 256, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    assert path.exists(), "Synthetic test video was not created"
    return path


@pytest.fixture()
def source(valid_video_path) -> FileVideoSource:
    """Return an *unopened* FileVideoSource pointed at the synthetic clip."""
    return FileVideoSource(valid_video_path)


# ======================================================================
# Test 1 — Valid video opens & metadata populated
# ======================================================================


class TestOpenValidVideo:
    def test_opens_successfully(self, source: FileVideoSource):
        source.open()
        assert source.is_open()
        source.close()

    def test_metadata_populated(self, source: FileVideoSource):
        source.open()
        meta = source.get_metadata()
        assert meta.width == _TEST_WIDTH
        assert meta.height == _TEST_HEIGHT
        assert meta.fps is not None and meta.fps > 0
        assert meta.frame_count is not None and meta.frame_count > 0
        assert meta.source_path is not None
        source.close()

    def test_frames_can_be_read(self, source: FileVideoSource):
        source.open()
        frame = source.read()
        assert frame is not None
        assert isinstance(frame, Frame)
        assert isinstance(frame.image, np.ndarray)
        source.close()


# ======================================================================
# Test 2 — Sequential frame IDs
# ======================================================================


class TestSequentialFrames:
    def test_frame_ids_are_sequential(self, source: FileVideoSource):
        source.open()
        ids: list[int] = []
        while True:
            f = source.read()
            if f is None:
                break
            ids.append(f.frame_id)
        source.close()

        expected = list(range(len(ids)))
        assert ids == expected, f"Frame IDs not sequential: {ids[:10]}…"

    def test_first_frame_is_zero(self, source: FileVideoSource):
        source.open()
        frame = source.read()
        assert frame is not None
        assert frame.frame_id == 0
        source.close()


# ======================================================================
# Test 3 — End of video returns None
# ======================================================================


class TestEndOfVideo:
    def test_returns_none_after_last_frame(self, source: FileVideoSource):
        source.open()
        count = 0
        while source.read() is not None:
            count += 1
        # After exhaustion, subsequent reads must also return None.
        assert source.read() is None
        assert source.read() is None
        assert count > 0
        source.close()


# ======================================================================
# Test 4 — Invalid path raises clear exception
# ======================================================================


class TestInvalidPath:
    def test_nonexistent_file(self):
        s = FileVideoSource("/nonexistent/path/video.mp4")
        with pytest.raises(VideoOpenError):
            s.open()

    def test_directory_instead_of_file(self, tmp_path):
        s = FileVideoSource(tmp_path)
        with pytest.raises(VideoOpenError):
            s.open()


# ======================================================================
# Test 5 — Unsupported / corrupt video
# ======================================================================


class TestCorruptVideo:
    def test_corrupt_file_raises(self, tmp_path):
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"NOT_A_REAL_VIDEO_FILE_CONTENT_1234567890")
        s = FileVideoSource(corrupt)
        with pytest.raises((VideoOpenError, InvalidVideoError)):
            s.open()

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        s = FileVideoSource(empty)
        with pytest.raises((VideoOpenError, InvalidVideoError)):
            s.open()


# ======================================================================
# Test 6 — Context manager
# ======================================================================


class TestContextManager:
    def test_with_block_opens_and_closes(self, valid_video_path):
        with FileVideoSource(valid_video_path) as s:
            assert s.is_open()
            frame = s.read()
            assert frame is not None
        # After exiting the block, source should be closed.
        assert not s.is_open()

    def test_with_block_closes_on_exception(self, valid_video_path):
        try:
            with FileVideoSource(valid_video_path) as s:
                assert s.is_open()
                raise RuntimeError("Deliberate test exception")
        except RuntimeError:
            pass
        assert not s.is_open()


# ======================================================================
# Test 7 — Metadata accuracy
# ======================================================================


class TestMetadata:
    def test_width_height(self, source: FileVideoSource):
        source.open()
        m = source.get_metadata()
        assert m.width == _TEST_WIDTH
        assert m.height == _TEST_HEIGHT
        source.close()

    def test_fps(self, source: FileVideoSource):
        source.open()
        m = source.get_metadata()
        assert m.fps is not None
        assert abs(m.fps - _TEST_FPS) < 1.0  # allow minor rounding
        source.close()

    def test_frame_count(self, source: FileVideoSource):
        source.open()
        m = source.get_metadata()
        assert m.frame_count is not None
        # OpenCV may report ±1 for some codecs; allow a small tolerance.
        assert abs(m.frame_count - _TEST_FRAME_COUNT) <= 2
        source.close()

    def test_duration(self, source: FileVideoSource):
        source.open()
        m = source.get_metadata()
        if m.duration is not None:
            expected = _TEST_FRAME_COUNT / _TEST_FPS
            assert abs(m.duration - expected) < 0.5
        source.close()

    def test_codec_is_string_or_none(self, source: FileVideoSource):
        source.open()
        m = source.get_metadata()
        assert m.codec is None or isinstance(m.codec, str)
        source.close()

    def test_metadata_before_open(self, valid_video_path):
        s = FileVideoSource(valid_video_path)
        m = s.get_metadata()
        assert m.source_path is not None
        # Other fields should be None before opening.
        assert m.width is None

    def test_metadata_is_readonly(self, source: FileVideoSource):
        source.open()
        m = source.get_metadata()
        with pytest.raises(AttributeError):
            m.width = 9999  # type: ignore[misc]
        source.close()


# ======================================================================
# Test 8 — Timestamp monotonicity
# ======================================================================


class TestTimestampMonotonicity:
    def test_timestamps_are_monotonically_non_decreasing(self, source: FileVideoSource):
        source.open()
        prev_ts = -1.0
        count = 0
        while True:
            f = source.read()
            if f is None:
                break
            assert f.timestamp >= prev_ts, (
                f"Timestamp went backwards: frame {f.frame_id} has "
                f"ts={f.timestamp:.4f} but previous was {prev_ts:.4f}"
            )
            prev_ts = f.timestamp
            count += 1
        source.close()
        assert count > 0

    def test_first_timestamp_is_near_zero(self, source: FileVideoSource):
        source.open()
        frame = source.read()
        assert frame is not None
        # First frame should be at or very near 0 seconds.
        assert frame.timestamp < 0.1
        source.close()


# ======================================================================
# Additional edge-case tests
# ======================================================================


class TestReadBeforeOpen:
    def test_read_without_open_raises(self, valid_video_path):
        s = FileVideoSource(valid_video_path)
        with pytest.raises(VideoSourceError):
            s.read()


class TestDoubleClose:
    def test_close_is_idempotent(self, source: FileVideoSource):
        source.open()
        source.close()
        source.close()  # should not raise
        assert not source.is_open()


class TestRepr:
    def test_repr_before_open(self, valid_video_path):
        s = FileVideoSource(valid_video_path)
        r = repr(s)
        assert "FileVideoSource" in r

    def test_repr_after_open(self, source: FileVideoSource):
        source.open()
        r = repr(source)
        assert "is_open=True" in r
        source.close()


class TestFrameRepr:
    def test_frame_repr(self, source: FileVideoSource):
        source.open()
        f = source.read()
        assert f is not None
        r = repr(f)
        assert "frame_id=0" in r
        assert "timestamp=" in r
        assert "shape=" in r
        source.close()

