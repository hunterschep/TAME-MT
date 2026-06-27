from __future__ import annotations

import sys

from tame_mt.native import native_thread_count
from tame_mt.schema import PerformanceMetadata


def build_performance_metadata(
    *,
    backend: str,
    index_reused: bool,
    timings_sec: dict[str, float | None] | None = None,
) -> PerformanceMetadata:
    return PerformanceMetadata(
        backend=backend,
        threads=native_thread_count(),
        index_reused=index_reused,
        timings_sec=dict(timings_sec or {}),
        memory={"peak_rss_mb": current_peak_rss_mb()},
    )


def current_peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows and minimal platforms
        return None

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (OSError, ValueError):  # pragma: no cover - defensive platform guard
        return None
    raw_peak = float(usage.ru_maxrss)
    if raw_peak <= 0:
        return None
    if sys.platform == "darwin":
        return raw_peak / (1024.0 * 1024.0)
    return raw_peak / 1024.0
