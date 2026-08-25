"""
Video Source Module (Module 1) — CampusNetra

Provides a clean abstraction for video input sources.
Downstream modules interact only with VideoSource and Frame,
never directly with OpenCV's VideoCapture.

Public API:
    - VideoSource          (abstract interface)
    - FileVideoSource      (local-file implementation)
    - Frame                (decoded frame data model)
    - VideoMetadata        (source metadata data model)
    - VideoSourceError     (base exception)
    - VideoOpenError       (opening / validation failures)
    - VideoReadError       (frame-decode failures)
    - InvalidVideoError    (corrupt / unsupported files)
"""

from src.video.models import Frame, VideoMetadata
from src.video.video_source import VideoSource
from src.video.file_video_source import FileVideoSource
from src.video.exceptions import (
    VideoSourceError,
    VideoOpenError,
    VideoReadError,
    InvalidVideoError,
)

__all__ = [
    "VideoSource",
    "FileVideoSource",
    "Frame",
    "VideoMetadata",
    "VideoSourceError",
    "VideoOpenError",
    "VideoReadError",
    "InvalidVideoError",
]

