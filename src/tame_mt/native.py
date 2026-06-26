from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class NativeStatus:
    available: bool
    version: str | None
    error: str | None


def native_status() -> NativeStatus:
    try:
        module = import_module("tame_mt._native")
    except Exception as exc:  # pragma: no cover - depends on install mode
        return NativeStatus(available=False, version=None, error=str(exc))

    version = str(module.native_version())
    return NativeStatus(available=True, version=version, error=None)


def is_native_available() -> bool:
    return native_status().available


def build_native_index(
    normalized_lines: list[str],
    ngram_orders: tuple[int, ...],
    mode: str,
    candidate_gram_limit: int,
    posting_limit: int,
    max_candidates: int,
    rerank_limit: int,
) -> Any:
    module = import_module("tame_mt._native")
    return module.NativeNgramIndex(
        normalized_lines,
        list(ngram_orders),
        mode,
        candidate_gram_limit,
        posting_limit,
        max_candidates,
        rerank_limit,
    )


def native_index_to_bytes(native_index: Any) -> bytes:
    return bytes(native_index.to_bytes())


def native_index_from_bytes(data: bytes) -> Any:
    module = import_module("tame_mt._native")
    return module.NativeNgramIndex.from_bytes(data)
