from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tame_mt.config import IndexConfig, NormalizationConfig, SimilarityConfig
from tame_mt.exceptions import BackendError
from tame_mt.native import (
    build_native_index,
    is_native_available,
    native_index_to_bytes,
    native_status,
)
from tame_mt.normalize import normalize_text


@dataclass(frozen=True, slots=True)
class NeighborResult:
    index: int | None
    score: float
    exact: bool = False


@dataclass(frozen=True, slots=True)
class IndexBackendInfo:
    name: str
    native: bool
    exact: bool
    requested_mode: str
    resolved_mode: str


class NgramInvertedIndex:
    """Rust-backed nearest-neighbor index exposed through a small Python wrapper."""

    def __init__(
        self,
        normalized_lines: list[str],
        norm_config: NormalizationConfig,
        sim_config: SimilarityConfig,
        index_config: IndexConfig,
        resolved_mode: str,
        backend_info: IndexBackendInfo,
        native_index: Any,
        doc_count: int | None = None,
    ) -> None:
        self.normalized_lines = normalized_lines
        self.norm_config = norm_config
        self.sim_config = sim_config
        self.index_config = index_config
        self.resolved_mode = resolved_mode
        self.backend_info = backend_info
        self._native_index = native_index
        self.doc_count = doc_count if doc_count is not None else len(normalized_lines)

    @classmethod
    def build(
        cls,
        lines: list[str],
        norm_config: NormalizationConfig | None = None,
        sim_config: SimilarityConfig | None = None,
        index_config: IndexConfig | None = None,
    ) -> NgramInvertedIndex:
        norm_config = norm_config or NormalizationConfig()
        sim_config = sim_config or SimilarityConfig()
        index_config = index_config or IndexConfig()
        resolved_mode = _resolve_mode(index_config)
        normalized_lines = [normalize_text(line, norm_config) for line in lines]

        try:
            native_index = build_native_index(
                normalized_lines=normalized_lines,
                ngram_orders=sim_config.ngram_orders,
                mode=_native_mode(resolved_mode),
                candidate_gram_limit=index_config.candidate_gram_limit,
                posting_limit=index_config.posting_limit,
                max_candidates=index_config.max_candidates,
                rerank_limit=index_config.rerank_limit,
            )
        except Exception as exc:
            status = native_status()
            message = (
                "native Rust backend is unavailable"
                if status.error
                else "native Rust backend failed to build the index"
            )
            raise BackendError(f"{message}: {exc}") from exc

        return cls(
            normalized_lines=normalized_lines,
            norm_config=norm_config,
            sim_config=sim_config,
            index_config=index_config,
            resolved_mode=resolved_mode,
            backend_info=IndexBackendInfo(
                name=resolved_mode,
                native=True,
                exact=resolved_mode == "native_exact",
                requested_mode=index_config.mode,
                resolved_mode=resolved_mode,
            ),
            native_index=native_index,
            doc_count=len(normalized_lines),
        )

    @classmethod
    def from_native(
        cls,
        native_index: Any,
        norm_config: NormalizationConfig,
        sim_config: SimilarityConfig,
        index_config: IndexConfig,
        resolved_mode: str,
        lines: list[str] | None = None,
        normalized_lines: list[str] | None = None,
    ) -> NgramInvertedIndex:
        if resolved_mode not in {"native_exact", "native_fast"}:
            raise BackendError(
                f"native index wrapper requires native_exact/native_fast, got {resolved_mode!r}"
            )

        doc_count = int(native_index.doc_count())
        if normalized_lines is None:
            normalized_lines = (
                [normalize_text(line, norm_config) for line in lines] if lines is not None else []
            )
        if normalized_lines and len(normalized_lines) != doc_count:
            raise BackendError(
                "native index document count does not match supplied normalized lines"
            )
        if lines is not None and len(lines) != doc_count:
            raise BackendError("native index document count does not match supplied raw lines")

        return cls(
            normalized_lines=normalized_lines,
            norm_config=norm_config,
            sim_config=sim_config,
            index_config=index_config,
            resolved_mode=resolved_mode,
            backend_info=IndexBackendInfo(
                name=resolved_mode,
                native=True,
                exact=resolved_mode == "native_exact",
                requested_mode=index_config.mode,
                resolved_mode=resolved_mode,
            ),
            native_index=native_index,
            doc_count=doc_count,
        )

    def query_best(self, text: str) -> NeighborResult:
        results = self.query_topk(text, 1)
        if results:
            return results[0]
        return NeighborResult(index=None, score=0.0, exact=False)

    def query_topk(self, text: str, k: int) -> list[NeighborResult]:
        if k <= 0:
            return []
        return self._native_query_topk_normalized(normalize_text(text, self.norm_config), k)

    def batch_query_topk(self, texts: list[str], k: int) -> list[list[NeighborResult]]:
        if k <= 0:
            return [[] for _ in texts]
        normalized = [normalize_text(text, self.norm_config) for text in texts]
        return self.batch_query_topk_normalized(normalized, k)

    def batch_query_topk_normalized(
        self, normalized_texts: list[str], k: int
    ) -> list[list[NeighborResult]]:
        if k <= 0:
            return [[] for _ in normalized_texts]
        native_rows = self._native_index.batch_query_topk(normalized_texts, k)
        return [
            [
                NeighborResult(index=item[0], score=float(item[1]), exact=bool(item[2]))
                for item in row
            ]
            for row in native_rows
        ]

    def score_candidate(self, text: str, index: int) -> float:
        if index < 0 or index >= self.doc_count:
            raise IndexError(f"candidate index out of range: {index}")
        query_norm = normalize_text(text, self.norm_config)
        return float(self._native_index.score_candidate(query_norm, index))

    def score_candidates(self, text: str, indices: list[int]) -> dict[int, float]:
        if not indices:
            return {}
        query_norm = normalize_text(text, self.norm_config)
        return {
            int(index): float(score)
            for index, score in self._native_index.score_candidates(query_norm, indices)
        }

    def best_pair_candidate(
        self,
        target_index: NgramInvertedIndex,
        source_text: str,
        ref_texts: list[str],
        indices: list[int],
    ) -> NeighborResult:
        source_norm = normalize_text(source_text, self.norm_config)
        target_norms = [normalize_text(ref, target_index.norm_config) for ref in ref_texts]
        item = self._native_index.best_pair_candidate(
            target_index._native_index,
            source_norm,
            target_norms,
            indices,
        )
        return NeighborResult(index=item[0], score=float(item[1]), exact=bool(item[2]))

    def batch_best_pair_candidates(
        self,
        target_index: NgramInvertedIndex,
        source_texts: list[str],
        ref_texts_by_segment: list[list[str]],
        candidate_indices_by_segment: list[list[int]],
    ) -> list[NeighborResult]:
        source_norms = [normalize_text(text, self.norm_config) for text in source_texts]
        target_norms_by_segment = [
            [normalize_text(ref, target_index.norm_config) for ref in ref_texts]
            for ref_texts in ref_texts_by_segment
        ]
        return self.batch_best_pair_candidates_normalized(
            target_index,
            source_norms,
            target_norms_by_segment,
            candidate_indices_by_segment,
        )

    def batch_best_pair_candidates_normalized(
        self,
        target_index: NgramInvertedIndex,
        source_norms: list[str],
        target_norms_by_segment: list[list[str]],
        candidate_indices_by_segment: list[list[int]],
    ) -> list[NeighborResult]:
        native_rows = self._native_index.batch_best_pair_candidates(
            target_index._native_index,
            source_norms,
            target_norms_by_segment,
            candidate_indices_by_segment,
        )
        return [
            NeighborResult(index=item[0], score=float(item[1]), exact=bool(item[2]))
            for item in native_rows
        ]

    def contains_exact_normalized(self, normalized_text: str) -> bool:
        return bool(self._native_index.contains_exact(normalized_text))

    def release_python_normalized_lines(self) -> bool:
        """Drop Python normalized text copies after native build/persistence."""

        if not self.normalized_lines:
            return False
        self.normalized_lines = []
        return True

    def native_bytes(self) -> bytes:
        return native_index_to_bytes(self._native_index)

    def normalized(self, text: str) -> str:
        return normalize_text(text, self.norm_config)

    def _native_query_topk_normalized(self, query_norm: str, k: int) -> list[NeighborResult]:
        return [
            NeighborResult(index=item[0], score=float(item[1]), exact=bool(item[2]))
            for item in self._native_index.query_topk(query_norm, k)
        ]


def _resolve_mode(config: IndexConfig) -> str:
    if config.mode == "auto":
        if is_native_available():
            return "native_exact"
        status = native_status()
        reason = f": {status.error}" if status.error else ""
        raise BackendError(
            "native Rust backend is required for TAME-MT retrieval but is unavailable"
            f"{reason}. Install a wheel that matches this Python/platform, or rebuild the "
            "editable install with `python -m pip install --force-reinstall --no-deps -e .`."
        )
    if config.mode in {"native_exact", "native_fast"}:
        return config.mode
    raise BackendError(f"unsupported native index mode: {config.mode}")


def _native_mode(resolved_mode: str) -> str:
    if resolved_mode == "native_exact":
        return "exact"
    if resolved_mode == "native_fast":
        return "fast"
    raise BackendError(f"unsupported native index mode: {resolved_mode}")
