"""Exception types raised by TAME-MT."""

from __future__ import annotations


class TameMTError(Exception):
    """Base class for user-facing TAME-MT errors."""


class AlignmentError(TameMTError):
    """Raised when aligned corpus files have different segment counts."""


class ConfigurationError(TameMTError):
    """Raised when a configuration value is invalid."""


class BackendError(TameMTError):
    """Raised when a requested retrieval backend is unavailable or fails."""


class InputDataError(TameMTError):
    """Raised when corpus inputs are structurally invalid."""
