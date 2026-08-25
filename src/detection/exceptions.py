"""
Custom exceptions for the Detection module.

Hierarchy
---------
PersonDetectionError          ← base for all detection errors
├── ModelLoadError             ← model file missing, corrupt, or incompatible
├── InferenceError             ← runtime inference failure
└── InvalidFrameError          ← input frame is invalid or unsupported
"""

from __future__ import annotations


class PersonDetectionError(Exception):
    """Base exception for all person-detection errors.

    Attributes
    ----------
    operation : str | None
        The operation that failed (e.g. "load_model", "detect", "warmup").
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
    ) -> None:
        self.operation = operation
        parts: list[str] = []
        if operation:
            parts.append(f"[op={operation}]")
        parts.append(message)
        super().__init__(" ".join(parts))


class ModelLoadError(PersonDetectionError):
    """Raised when the detection model cannot be loaded."""

    def __init__(self, message: str, *, model_path: str | None = None) -> None:
        self.model_path = model_path
        detail = message
        if model_path:
            detail = f"model={model_path}: {message}"
        super().__init__(detail, operation="load_model")


class InferenceError(PersonDetectionError):
    """Raised when inference fails on a valid frame."""

    def __init__(
        self,
        message: str,
        *,
        frame_id: int | None = None,
    ) -> None:
        self.frame_id = frame_id
        detail = message
        if frame_id is not None:
            detail = f"frame_id={frame_id}: {message}"
        super().__init__(detail, operation="detect")


class InvalidFrameError(PersonDetectionError):
    """Raised when the input frame is invalid or unsupported."""

    def __init__(self, message: str) -> None:
        super().__init__(message, operation="validate_frame")

