from __future__ import annotations

from typing import Any

from tame_mt.config import IndexConfig, NormalizationConfig, SimilarityConfig
from tame_mt.exceptions import ApproximationError, BackendError, ConfigurationError
from tame_mt.native import (
    build_native_index,
    native_index_to_bytes,
    native_status,
)
from tame_mt.normalize import normalize_text

from .base import IndexBackendInfo, NeighborResult
from .modes import (
    native_mode,
    resolve_mode,
    source_bin_from_exact_score,
    validate_thresholds,
    validate_unit_threshold,
    zero_neighbor_if_threshold_zero,
)


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
        *,
        keep_normalized_lines: bool = True,
    ) -> NgramInvertedIndex:
        norm_config = norm_config or NormalizationConfig()
        sim_config = sim_config or SimilarityConfig()
        index_config = index_config or IndexConfig()
        resolved_mode = resolve_mode(index_config)
        normalized_lines = [normalize_text(line, norm_config) for line in lines]
        doc_count = len(normalized_lines)

        try:
            native_index = build_native_index(
                normalized_lines=normalized_lines,
                ngram_orders=sim_config.ngram_orders,
                mode=native_mode(resolved_mode),
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
            normalized_lines=normalized_lines if keep_normalized_lines else [],
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

    def best_above(self, text: str, threshold: float) -> NeighborResult | None:
        return self.batch_best_above([text], threshold)[0]

    def batch_best_above(
        self,
        texts: list[str],
        threshold: float,
    ) -> list[NeighborResult | None]:
        normalized = [normalize_text(text, self.norm_config) for text in texts]
        return self.batch_best_above_normalized(normalized, threshold)

    def batch_best_above_normalized(
        self,
        normalized_texts: list[str],
        threshold: float,
    ) -> list[NeighborResult | None]:
        threshold = validate_unit_threshold(threshold)
        self._require_exact_threshold_backend()
        tops = self.batch_query_topk_normalized(normalized_texts, 1)
        results: list[NeighborResult | None] = []
        for top in tops:
            best = top[0] if top else zero_neighbor_if_threshold_zero(self.doc_count, threshold)
            results.append(best if best is not None and best.score >= threshold else None)
        return results

    def batch_threshold_flags(
        self,
        texts: list[str],
        thresholds: list[float] | tuple[float, ...],
    ) -> list[dict[float, bool]]:
        normalized = [normalize_text(text, self.norm_config) for text in texts]
        return self.batch_threshold_flags_normalized(normalized, thresholds)

    def batch_threshold_flags_normalized(
        self,
        normalized_texts: list[str],
        thresholds: list[float] | tuple[float, ...],
    ) -> list[dict[float, bool]]:
        parsed_thresholds = validate_thresholds(thresholds)
        self._require_exact_threshold_backend()
        tops = self.batch_query_topk_normalized(normalized_texts, 1)
        rows: list[dict[float, bool]] = []
        for top in tops:
            score = top[0].score if top else 0.0
            rows.append({threshold: score >= threshold for threshold in parsed_thresholds})
        return rows

    def batch_source_bins_exact(
        self,
        texts: list[str],
        *,
        far_threshold: float,
        near_threshold: float,
    ) -> list[str]:
        normalized = [normalize_text(text, self.norm_config) for text in texts]
        return self.batch_source_bins_exact_normalized(
            normalized,
            far_threshold=far_threshold,
            near_threshold=near_threshold,
        )

    def batch_source_bins_exact_normalized(
        self,
        normalized_texts: list[str],
        *,
        far_threshold: float,
        near_threshold: float,
    ) -> list[str]:
        far_threshold = validate_unit_threshold(far_threshold)
        near_threshold = validate_unit_threshold(near_threshold)
        if far_threshold > near_threshold:
            raise ConfigurationError("far_threshold must be no larger than near_threshold")
        self._require_exact_threshold_backend()
        tops = self.batch_query_topk_normalized(normalized_texts, 1)
        bins: list[str] = []
        for top in tops:
            best = top[0] if top else None
            score = best.score if best is not None else 0.0
            exact = best.exact if best is not None else False
            bins.append(source_bin_from_exact_score(exact, score, far_threshold, near_threshold))
        return bins

    def score_candidate(self, text: str, index: int) -> float:
        if index < 0 or index >= self.doc_count:
            raise IndexError(f"candidate index out of range: {index}")
        query_norm = normalize_text(text, self.norm_config)
        return float(self._native_index.score_candidate(query_norm, index))

    def score_candidates(self, text: str, indices: list[int]) -> dict[int, float]:
        if not indices:
            return {}
        query_norm = normalize_text(text, self.norm_config)
        return self.score_candidates_normalized(query_norm, indices)

    def score_candidates_normalized(
        self, normalized_text: str, indices: list[int]
    ) -> dict[int, float]:
        if not indices:
            return {}
        return {
            int(index): float(score)
            for index, score in self._native_index.score_candidates(normalized_text, indices)
        }

    def pair_threshold_flags_exact(
        self,
        target_index: NgramInvertedIndex,
        source_text: str,
        ref_texts: list[str],
        thresholds: list[float] | tuple[float, ...],
    ) -> dict[str, bool]:
        source_norm = normalize_text(source_text, self.norm_config)
        target_norms = [normalize_text(ref, target_index.norm_config) for ref in ref_texts]
        return self.pair_threshold_flags_exact_normalized(
            target_index,
            source_norm,
            target_norms,
            thresholds,
        )

    def pair_threshold_flags_exact_normalized(
        self,
        target_index: NgramInvertedIndex,
        source_norm: str,
        target_norms: list[str],
        thresholds: list[float] | tuple[float, ...],
    ) -> dict[str, bool]:
        parsed_thresholds = validate_thresholds(thresholds)
        self._require_exact_threshold_backend()
        target_index._require_exact_threshold_backend()
        if self.doc_count != target_index.doc_count:
            raise BackendError("source and target indexes must have the same document count")
        flags = {f"{threshold:.2f}": False for threshold in parsed_thresholds}
        if not target_norms or self.doc_count == 0:
            return flags

        doc_indices = list(range(self.doc_count))
        min_threshold = min(parsed_thresholds)
        source_scores = self.score_candidates_normalized(source_norm, doc_indices)
        source_candidates = [
            index for index, score in source_scores.items() if score >= min_threshold
        ]
        if not source_candidates:
            return flags

        for target_norm in target_norms:
            target_scores = target_index.score_candidates_normalized(target_norm, source_candidates)
            for index in source_candidates:
                pair_score = min(source_scores[index], target_scores.get(index, 0.0))
                for threshold in parsed_thresholds:
                    if pair_score >= threshold:
                        flags[f"{threshold:.2f}"] = True
            if all(flags.values()):
                break
        return flags

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

    def _require_exact_threshold_backend(self) -> None:
        if not self.backend_info.exact:
            raise ApproximationError(
                "exact threshold flags require native_exact; approximate native_fast "
                "candidate search can miss neighbors above the threshold"
            )
