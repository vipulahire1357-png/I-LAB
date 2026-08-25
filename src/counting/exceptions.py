"""
Custom exceptions for the Entry/Exit Counting module.
"""

from __future__ import annotations


class CountingError(Exception):
    """Base exception for all entry/exit counting errors."""

    def __init__(self, message: str, *, operation: str | None = None) -> None:
        self.operation = operation
        parts: list[str] = []
        if operation:
            parts.append(f"[op={operation}]")
        parts.append(message)
        super().__init__(" ".join(parts))


class CountingConfigError(CountingError):
    """Raised when virtual counting line or counting configuration is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, operation="configure")
