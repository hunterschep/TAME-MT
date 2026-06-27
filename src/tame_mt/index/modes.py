from __future__ import annotations

from math import isfinite

from tame_mt.config import IndexConfig
from tame_mt.exceptions import BackendError, ConfigurationError
from tame_mt.native import is_native_available, native_status

from .base import NeighborResult

NATIVE_EXACT_MODE = "native_exact"
NATIVE_FAST_MODE = "native_fast"
NATIVE_MODES = (NATIVE_EXACT_MODE, NATIVE_FAST_MODE)


def resolve_mode(config: IndexConfig) -> str:
    if config.mode == "auto":
        if is_native_available():
            return NATIVE_EXACT_MODE
        status = native_status()
        reason = f": {status.error}" if status.error else ""
        raise BackendError(
            "native Rust backend is required for TAME-MT retrieval but is unavailable"
            f"{reason}. Install a wheel that matches this Python/platform, or rebuild the "
            "editable install with `python -m pip install --force-reinstall --no-deps -e .`."
        )
    if config.mode in NATIVE_MODES:
        return config.mode
    raise BackendError(f"unsupported native index mode: {config.mode}")


def native_mode(resolved_mode: str) -> str:
    if resolved_mode == NATIVE_EXACT_MODE:
        return "exact"
    if resolved_mode == NATIVE_FAST_MODE:
        return "fast"
    raise BackendError(f"unsupported native index mode: {resolved_mode}")


def validate_thresholds(thresholds: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if not thresholds:
        raise ConfigurationError("thresholds must contain at least one value")
    return tuple(validate_unit_threshold(threshold) for threshold in thresholds)


def validate_unit_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ConfigurationError("threshold must be a finite number between 0 and 1")
    parsed = float(threshold)
    if not isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ConfigurationError("threshold must be a finite number between 0 and 1")
    return parsed


def zero_neighbor_if_threshold_zero(
    doc_count: int,
    threshold: float,
) -> NeighborResult | None:
    if threshold <= 0.0 and doc_count > 0:
        return NeighborResult(index=0, score=0.0, exact=False)
    return None


def source_bin_from_exact_score(
    exact: bool,
    score: float,
    far_threshold: float,
    near_threshold: float,
) -> str:
    if exact:
        return "source_exact"
    if score >= near_threshold:
        return "near"
    if score >= far_threshold:
        return "medium"
    return "far"
