from __future__ import annotations

from collections.abc import Iterable


def exact_pair_key(normalized_source: str, normalized_target: str) -> str:
    """Return an unambiguous exact-match key for a normalized source/target pair."""

    return f"{len(normalized_source)}\0{normalized_source}{normalized_target}"


def build_exact_pair_keys(
    normalized_sources: Iterable[str],
    normalized_targets: Iterable[str],
) -> set[str]:
    return {
        exact_pair_key(source, target)
        for source, target in zip(normalized_sources, normalized_targets, strict=True)
    }
