"""
Abstract base class for all video sources.

Downstream modules depend on this interface — never on a concrete
implementation or on OpenCV directly.

A future ``RTSPVideoSource`` would subclass ``VideoSource`` and the rest of
the pipeline would remain unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.video.models import Frame, VideoMetadata


class VideoSource(ABC):
    """Generic video-source interface.

    Lifecycle
    ---------
    1. Construct the source (e.g. ``FileVideoSource(path)``).
    2. Call :meth:`open` to validate and prepare the source.
    3. Call :meth:`read` repeatedly to obtain sequential frames.
    4. Call :meth:`close` to release resources (or use as a context manager).

    Context-manager protocol is provided by default so that concrete
    implementations get ``with`` support for free.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Open and validate the video source.

        Raises
        ------
        VideoOpenError
            If the source cannot be opened or validated.
        InvalidVideoError
            If the source file is corrupt or unsupported.
        """

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next sequential frame, or ``None`` at end-of-source.

        Returns
        -------
        Frame | None
            The next decoded frame, or ``None`` when no more frames are
            available.

        Raises
        ------
        VideoReadError
            If a frame cannot be decoded.
        VideoSourceError
            If the source is not open.
        """

    @abstractmethod
    def get_metadata(self) -> VideoMetadata:
        """Return metadata about the video source.

        May be called before or after :meth:`open`; however, metadata will
        be richer after the source has been opened and validated.

        Returns
        -------
        VideoMetadata
        """

    @abstractmethod
    def is_open(self) -> bool:
        """Return ``True`` if the source is currently open and readable."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by the video source.

        Safe to call multiple times.
        """

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(is_open={self.is_open()})"

