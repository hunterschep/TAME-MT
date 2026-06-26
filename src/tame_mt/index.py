from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from tame_mt.config import IndexConfig, NormalizationConfig, SimilarityConfig
from tame_mt.exceptions import BackendError
from tame_mt.native import build_native_index, is_native_available, native_status
from tame_mt.ngrams import char_ngrams
from tame_mt.normalize import normalize_text
from tame_mt.similarity import jaccard


@dataclass(frozen=True)
class NeighborResult:
    index: int | None
    score: float
    exact: bool = False


@dataclass(frozen=True)
class IndexBackendInfo:
    name: str
    native: bool
    exact: bool
    requested_mode: str
    resolved_mode: str


class NgramInvertedIndex:
    def __init__(
        self,
        normalized_lines: list[str],
        gram_sets: list[frozenset[str]] | None,
        postings: dict[str, list[int]] | None,
        exact_map: dict[str, list[int]],
        norm_config: NormalizationConfig,
        sim_config: SimilarityConfig,
        index_config: IndexConfig,
        resolved_mode: str,
        backend_info: IndexBackendInfo,
        native_index: Any | None = None,
    ) -> None:
        self.normalized_lines = normalized_lines
        self.gram_sets = gram_sets
        self.gram_counts = [len(grams) for grams in gram_sets] if gram_sets is not None else []
        self.postings = postings
        self.exact_map = exact_map
        self.norm_config = norm_config
        self.sim_config = sim_config
        self.index_config = index_config
        self.resolved_mode = resolved_mode
        self.backend_info = backend_info
        self._native_index = native_index

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
        resolved_mode = _resolve_mode(index_config, len(lines))
        normalized_lines = [normalize_text(line, norm_config) for line in lines]
        exact_map: dict[str, list[int]] = defaultdict(list)
        for idx, line in enumerate(normalized_lines):
            exact_map[line].append(idx)

        if resolved_mode.startswith("native_"):
            try:
                native_mode = "exact" if resolved_mode == "native_exact" else "fast"
                native_index = build_native_index(
                    normalized_lines=normalized_lines,
                    ngram_orders=sim_config.ngram_orders,
                    mode=native_mode,
                    candidate_gram_limit=index_config.candidate_gram_limit,
                    posting_limit=index_config.posting_limit,
                    max_candidates=index_config.max_candidates,
                    rerank_limit=index_config.rerank_limit,
                )
            except Exception as exc:
                status = native_status()
                message = (
                    "native backend is unavailable"
                    if status.error
                    else "native backend failed to build the index"
                )
                raise BackendError(f"{message}: {exc}") from exc

            return cls(
                normalized_lines=normalized_lines,
                gram_sets=None,
                postings=None,
                exact_map={line: indices for line, indices in exact_map.items()},
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
            )

        gram_sets = [char_ngrams(line, sim_config.ngram_orders) for line in normalized_lines]
        postings: dict[str, list[int]] = defaultdict(list)
        for idx, grams in enumerate(gram_sets):
            for gram in grams:
                postings[gram].append(idx)

        return cls(
            normalized_lines=normalized_lines,
            gram_sets=gram_sets,
            postings={gram: sorted(indices) for gram, indices in postings.items()},
            exact_map={line: indices for line, indices in exact_map.items()},
            norm_config=norm_config,
            sim_config=sim_config,
            index_config=index_config,
            resolved_mode=resolved_mode,
            backend_info=IndexBackendInfo(
                name=resolved_mode,
                native=False,
                exact=resolved_mode == "python_exact",
                requested_mode=index_config.mode,
                resolved_mode=resolved_mode,
            ),
        )

    def query_best(self, text: str) -> NeighborResult:
        results = self.query_topk(text, 1)
        if results:
            return results[0]
        return NeighborResult(index=None, score=0.0, exact=False)

    def batch_query_topk(self, texts: list[str], k: int) -> list[list[NeighborResult]]:
        if k <= 0:
            return [[] for _ in texts]
        if self._native_index is None:
            return [self.query_topk(text, k) for text in texts]

        normalized = [normalize_text(text, self.norm_config) for text in texts]
        native_rows = self._native_index.batch_query_topk(normalized, k)
        return [
            [
                NeighborResult(index=item[0], score=float(item[1]), exact=bool(item[2]))
                for item in row
            ]
            for row in native_rows
        ]

    def query_topk(self, text: str, k: int) -> list[NeighborResult]:
        if k <= 0:
            return []
        query_norm = normalize_text(text, self.norm_config)
        if self._native_index is not None:
            return [
                NeighborResult(index=item[0], score=float(item[1]), exact=bool(item[2]))
                for item in self._native_index.query_topk(query_norm, k)
            ]

        query_grams = char_ngrams(query_norm, self.sim_config.ngram_orders)

        if not query_grams:
            exact_indices = self.exact_map.get(query_norm)
            if exact_indices:
                return [NeighborResult(index=exact_indices[0], score=1.0, exact=True)]
            return []

        exact_indices = self.exact_map.get(query_norm)
        if exact_indices:
            return [NeighborResult(index=exact_indices[0], score=1.0, exact=True)]

        if self.resolved_mode == "python_fast":
            return self._query_topk_fast(query_norm, query_grams, k)
        return self._query_topk_exact(query_norm, query_grams, k)

    def _query_topk_exact(
        self,
        query_norm: str,
        query_grams: frozenset[str],
        k: int,
    ) -> list[NeighborResult]:
        if self.postings is None:
            raise BackendError("python postings are not available for this index")
        intersection_counts: Counter[int] = Counter()
        for gram in query_grams:
            for idx in self.postings.get(gram, []):
                intersection_counts[idx] += 1
        return self._rank_candidates(query_norm, query_grams, intersection_counts, k)

    def _query_topk_fast(
        self,
        query_norm: str,
        query_grams: frozenset[str],
        k: int,
    ) -> list[NeighborResult]:
        if self.postings is None:
            raise BackendError("python postings are not available for this index")
        ranked_grams = sorted(
            ((len(self.postings.get(gram, [])), gram) for gram in query_grams),
            key=lambda item: (item[0], item[1]),
        )
        selected = [
            gram
            for posting_count, gram in ranked_grams
            if 0 < posting_count <= self.index_config.posting_limit
        ][: self.index_config.candidate_gram_limit]
        if not selected:
            selected = [gram for posting_count, gram in ranked_grams if posting_count > 0][
                : self.index_config.candidate_gram_limit
            ]

        intersection_counts: Counter[int] = Counter()
        for gram in selected:
            for idx in self.postings.get(gram, [])[: self.index_config.posting_limit]:
                intersection_counts[idx] += 1
                if len(intersection_counts) >= self.index_config.max_candidates:
                    break
            if len(intersection_counts) >= self.index_config.max_candidates:
                break
        if len(intersection_counts) > self.index_config.rerank_limit:
            intersection_counts = Counter(
                dict(
                    sorted(
                        intersection_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[: self.index_config.rerank_limit]
                )
            )
        return self._rank_candidates(query_norm, query_grams, intersection_counts, k)

    def _rank_candidates(
        self,
        query_norm: str,
        query_grams: frozenset[str],
        intersection_counts: Counter[int],
        k: int,
    ) -> list[NeighborResult]:
        if self.gram_sets is None:
            raise BackendError("python gram sets are not available for this index")
        results: list[NeighborResult] = []
        for idx, inter in intersection_counts.items():
            candidate_grams = self.gram_sets[idx]
            inter = len(query_grams & candidate_grams)
            union = len(query_grams) + self.gram_counts[idx] - inter
            score = inter / union if union else 1.0
            if score > 0:
                results.append(
                    NeighborResult(
                        index=idx,
                        score=score,
                        exact=self.normalized_lines[idx] == query_norm,
                    )
                )

        results.sort(
            key=lambda result: (-result.score, result.index if result.index is not None else 10**18)
        )
        return results[:k]

    def score_candidate(self, text: str, index: int) -> float:
        if index < 0 or index >= len(self.normalized_lines):
            raise IndexError(f"candidate index out of range: {index}")
        query_norm = normalize_text(text, self.norm_config)
        if self._native_index is not None:
            return float(self._native_index.score_candidate(query_norm, index))
        if self.gram_sets is None:
            raise BackendError("python gram sets are not available for this index")
        query_grams = char_ngrams(query_norm, self.sim_config.ngram_orders)
        return jaccard(query_grams, self.gram_sets[index])

    def score_candidates(self, text: str, indices: list[int]) -> dict[int, float]:
        if not indices:
            return {}
        query_norm = normalize_text(text, self.norm_config)
        if self._native_index is not None:
            return {
                int(index): float(score)
                for index, score in self._native_index.score_candidates(query_norm, indices)
            }
        if self.gram_sets is None:
            raise BackendError("python gram sets are not available for this index")
        query_grams = char_ngrams(query_norm, self.sim_config.ngram_orders)
        scores: dict[int, float] = {}
        for index in indices:
            if index < 0 or index >= len(self.gram_sets):
                raise IndexError(f"candidate index out of range: {index}")
            scores[index] = jaccard(query_grams, self.gram_sets[index])
        return scores

    def normalized(self, text: str) -> str:
        return normalize_text(text, self.norm_config)


def _resolve_mode(config: IndexConfig, num_lines: int) -> str:
    if config.mode == "inverted_exact":
        return "python_exact"
    if config.mode == "inverted_fast":
        return "python_fast"
    if config.mode == "auto":
        exact_mode = num_lines <= config.auto_exact_cutoff
        if is_native_available():
            return "native_exact" if exact_mode else "native_fast"
        return "python_exact" if exact_mode else "python_fast"
    return config.mode
