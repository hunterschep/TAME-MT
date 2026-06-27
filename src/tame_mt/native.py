from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from tame_mt.version import __version__


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

    try:
        version = _module_version(module)
    except Exception as exc:  # pragma: no cover - defensive against broken extensions
        return NativeStatus(
            available=False,
            version=None,
            error=f"native backend version probe failed: {exc}",
        )
    if version != __version__:
        return NativeStatus(
            available=False,
            version=version,
            error=(
                f"native backend version {version} does not match "
                f"Python package version {__version__}"
            ),
        )
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
    module = _load_native_module()
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
    module = _load_native_module()
    return module.NativeNgramIndex.from_bytes(data)


def _load_native_module() -> Any:
    module = import_module("tame_mt._native")
    version = _module_version(module)
    if version != __version__:
        raise RuntimeError(
            f"native backend version {version} does not match Python package version {__version__}"
        )
    return module


def _module_version(module: Any) -> str:
    return str(module.native_version())
