"""
Custom exceptions for the Tracking module.

Hierarchy
---------
TrackerError            ← base for all tracker errors
├── TrackerConfigError  ← invalid configuration
└── TrackerUpdateError  ← failure during track update
"""

from __future__ import annotations


class TrackerError(Exception):
    """Base exception for all tracker errors."""

    def __init__(self, message: str, *, operation: str | None = None) -> None:
        self.operation = operation
        parts: list[str] = []
        if operation:
            parts.append(f"[op={operation}]")
        parts.append(message)
        super().__init__(" ".join(parts))


class TrackerConfigError(TrackerError):
    """Raised when tracker configuration is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, operation="configure")


class TrackerUpdateError(TrackerError):
    """Raised when the tracker fails during a frame update."""

    def __init__(self, message: str, *, frame_id: int | None = None) -> None:
        self.frame_id = frame_id
        detail = message
        if frame_id is not None:
            detail = f"frame_id={frame_id}: {message}"
        super().__init__(detail, operation="update")

