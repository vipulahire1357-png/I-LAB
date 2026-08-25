"""
Custom exceptions for the Video Source module.

Hierarchy
---------
VideoSourceError          ← base for all video-source errors
├── VideoOpenError        ← file not found, codec failure, invalid metadata
├── VideoReadError        ← frame-decode failure during reading
└── InvalidVideoError     ← corrupt, truncated, or unsupported file
"""

from __future__ import annotations


class VideoSourceError(Exception):
    """Base exception for all video-source errors.

    Attributes
    ----------
    source : str | None
        Identifier of the video source (e.g. file path, RTSP URL).
    operation : str | None
        The operation that failed (e.g. "open", "read").
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        operation: str | None = None,
    ) -> None:
        self.source = source
        self.operation = operation
        parts: list[str] = []
        if source:
            parts.append(f"[source={source}]")
        if operation:
            parts.append(f"[op={operation}]")
        parts.append(message)
        super().__init__(" ".join(parts))


class VideoOpenError(VideoSourceError):
    """Raised when a video source cannot be opened or validated."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
    ) -> None:
        super().__init__(message, source=source, operation="open")


class VideoReadError(VideoSourceError):
    """Raised when a frame cannot be decoded during reading."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        frame_id: int | None = None,
    ) -> None:
        self.frame_id = frame_id
        detail = message
        if frame_id is not None:
            detail = f"frame_id={frame_id}: {message}"
        super().__init__(detail, source=source, operation="read")


class InvalidVideoError(VideoSourceError):
    """Raised when a file is corrupt, truncated, or in an unsupported format."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
    ) -> None:
        super().__init__(message, source=source, operation="validate")

