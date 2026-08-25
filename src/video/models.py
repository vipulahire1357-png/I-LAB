"""
Data models for the Video Source module.

These are plain data containers — no business logic, no OpenCV dependency.
Downstream modules should type-hint against Frame and VideoMetadata,
never against cv2 types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """A single decoded video frame.

    Attributes
    ----------
    frame_id : int
        Zero-based sequential frame number.
    timestamp : float
        Seconds from the beginning of the video source.
        Derived from the source's own position data (not from accumulated deltas).
    image : np.ndarray
        The raw decoded image as a NumPy array (H × W × C, BGR channel order
        as produced by the video backend).  Module 1 does **not** resize,
        crop, normalise, or convert the image.
    """

    frame_id: int
    timestamp: float
    image: np.ndarray

    # numpy arrays are not hashable; disable __hash__ inherited from frozen
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        h, w = self.image.shape[:2]
        return (
            f"Frame(frame_id={self.frame_id}, "
            f"timestamp={self.timestamp:.4f}s, "
            f"shape=({h}, {w}, {self.image.shape[2] if self.image.ndim == 3 else 1}))"
        )


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Read-only metadata about a video source.

    Fields that cannot be reliably determined are set to ``None``.

    Attributes
    ----------
    source_path : str
        Filesystem path (or URL/URI for future RTSP sources).
    width : int | None
        Frame width in pixels.
    height : int | None
        Frame height in pixels.
    fps : float | None
        Nominal frames per second reported by the container/codec.
    frame_count : int | None
        Total frame count reported by the container.  May be unreliable for
        some codecs/containers — treat as an *estimate*.
    duration : float | None
        Estimated duration in seconds (``frame_count / fps`` when both are
        available).
    codec : str | None
        FourCC codec identifier (e.g. ``"avc1"``, ``"XVID"``).
    """

    source_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration: Optional[float] = None
    codec: Optional[str] = None

