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


class ApproximationError(TameMTError):
    """Raised when approximate retrieval cannot satisfy a requested guarantee."""


class InputDataError(TameMTError):
    """Raised when corpus inputs are structurally invalid."""


class ArtifactValidationError(InputDataError):
    """Raised when a cached artifact or index bundle fails validation."""


class OutputError(TameMTError):
    """Raised when TAME-MT cannot serialize or write an output artifact."""


class SecurityError(TameMTError):
    """Raised when an artifact is unsafe to load or violates security constraints."""
