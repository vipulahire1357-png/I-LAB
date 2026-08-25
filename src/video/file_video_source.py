"""
Concrete VideoSource backed by a local video file (MP4, AVI, MOV, …).

Uses OpenCV's VideoCapture internally but does **not** expose it.
Downstream code should depend only on ``VideoSource``, ``Frame``, and
``VideoMetadata``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from src.video.exceptions import (
    InvalidVideoError,
    VideoOpenError,
    VideoReadError,
    VideoSourceError,
)
from src.video.models import Frame, VideoMetadata
from src.video.video_source import VideoSource


class FileVideoSource(VideoSource):
    """Read frames sequentially from a local video file.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the video file.

    Notes
    -----
    *   OpenCV reports timestamps via ``CAP_PROP_POS_MSEC``.  For constant-
        frame-rate (CFR) files this is accurate.  For variable-frame-rate
        (VFR) files OpenCV may fall back to ``frame_number / fps`` which can
        drift.  This is an OpenCV limitation — not something Module 1 can fix.
    *   ``CAP_PROP_FRAME_COUNT`` is an *estimate* for some codecs (notably
        MJPEG and certain MKV containers).  Treat it as advisory.
    *   Frame images are returned in BGR channel order exactly as decoded by
        OpenCV.  No colour-space conversion, resizing, or normalisation is
        performed.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path: str = str(Path(path).resolve())
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._frame_counter: int = 0
        self._exhausted: bool = False

    # ------------------------------------------------------------------
    # VideoSource interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the video file and validate its metadata.

        Raises
        ------
        VideoOpenError
            If the file does not exist, is not readable, or cannot be opened
            by the video backend.
        InvalidVideoError
            If the opened file has invalid frame dimensions or an
            unrecognisable codec.
        """
        # --- pre-flight checks -------------------------------------------
        if not os.path.exists(self._path):
            raise VideoOpenError(
                f"File does not exist: {self._path}",
                source=self._path,
            )
        if not os.path.isfile(self._path):
            raise VideoOpenError(
                f"Path is not a regular file: {self._path}",
                source=self._path,
            )
        if not os.access(self._path, os.R_OK):
            raise VideoOpenError(
                f"File is not readable: {self._path}",
                source=self._path,
            )

        # --- open via OpenCV ---------------------------------------------
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(
                "OpenCV could not open the video file. "
                "The file may be corrupt or use an unsupported codec.",
                source=self._path,
            )

        # --- extract & validate metadata ---------------------------------
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise InvalidVideoError(
                f"Invalid frame dimensions: {width}×{height}",
                source=self._path,
            )

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps: Optional[float] = raw_fps if raw_fps > 0 else None

        raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count: Optional[int] = raw_count if raw_count > 0 else None

        duration: Optional[float] = None
        if fps is not None and frame_count is not None:
            duration = frame_count / fps

        codec = self._decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))

        self._metadata = VideoMetadata(
            source_path=self._path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
        )
        self._cap = cap
        self._frame_counter = 0
        self._exhausted = False

    def read(self) -> Optional[Frame]:
        """Return the next frame, or ``None`` at end-of-video.

        Raises
        ------
        VideoSourceError
            If the source has not been opened.
        VideoReadError
            If a frame cannot be decoded (corrupted segment).
        """
        if self._cap is None or not self._cap.isOpened():
            raise VideoSourceError(
                "Cannot read: video source is not open.",
                source=self._path,
                operation="read",
            )

        if self._exhausted:
            return None

        ret, image = self._cap.read()

        if not ret:
            # End-of-video (or unrecoverable decode failure on last frame).
            self._exhausted = True
            return None

        if image is None or image.size == 0:
            # Frame was "read" but data is empty — treat as decode error.
            raise VideoReadError(
                "Decoded frame is empty or invalid.",
                source=self._path,
                frame_id=self._frame_counter,
            )

        # Timestamp: prefer OpenCV's millisecond position property.
        # For CFR files this is reliable; for VFR files it may approximate.
        timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        # CAP_PROP_POS_MSEC reports the *next* frame's position after read(),
        # so we grab it *before* incrementing the counter.  However, OpenCV
        # updates the property *after* read().  In practice the value
        # corresponds to the timestamp of the frame just read for most
        # backends.  As a fallback, compute from frame_id / fps.
        if timestamp_ms > 0 or self._frame_counter == 0:
            timestamp = timestamp_ms / 1000.0
        else:
            # Fallback: derive from frame position and FPS.
            fps = (
                self._metadata.fps
                if self._metadata is not None and self._metadata.fps
                else 30.0
            )
            timestamp = self._frame_counter / fps

        frame = Frame(
            frame_id=self._frame_counter,
            timestamp=timestamp,
            image=image,
        )
        self._frame_counter += 1
        return frame

    def get_metadata(self) -> VideoMetadata:
        """Return video metadata.

        If the source has not been opened yet, returns a minimal metadata
        object containing only the source path.
        """
        if self._metadata is not None:
            return self._metadata
        return VideoMetadata(source_path=self._path)

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def close(self) -> None:
        """Release the underlying VideoCapture.  Safe to call repeatedly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._exhausted = True

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FileVideoSource(path={self._path!r}, "
            f"is_open={self.is_open()}, "
            f"frames_read={self._frame_counter})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_fourcc(fourcc_int: int) -> Optional[str]:
        """Decode an integer FourCC code to a 4-character string.

        Returns ``None`` if the result is blank or non-printable.
        """
        if fourcc_int == 0:
            return None
        try:
            codec = struct.pack("<I", fourcc_int & 0xFFFFFFFF).decode(
                "ascii", errors="replace"
            )
            codec = codec.strip("\x00").strip()
            return codec if codec else None
        except Exception:
            return None

