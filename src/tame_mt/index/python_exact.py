from __future__ import annotations

from tame_mt.config import IndexConfig, NormalizationConfig, SimilarityConfig
from tame_mt.ngrams import char_ngrams
from tame_mt.normalize import normalize_text
from tame_mt.similarity import jaccard

from .base import IndexBackendInfo, NeighborResult


class PythonExactSimilarityIndex:
    """Reference exact Jaccard index used for parity tests and debugging."""

    def __init__(
        self,
        lines: list[str],
        norm_config: NormalizationConfig | None = None,
        sim_config: SimilarityConfig | None = None,
        index_config: IndexConfig | None = None,
    ) -> None:
        self.norm_config = norm_config or NormalizationConfig()
        self.sim_config = sim_config or SimilarityConfig()
        self.index_config = index_config or IndexConfig()
        self.normalized_lines = [normalize_text(line, self.norm_config) for line in lines]
        self._doc_grams = [
            char_ngrams(normalized, self.sim_config.ngram_orders)
            for normalized in self.normalized_lines
        ]
        self.doc_count = len(self.normalized_lines)
        self.backend_info = IndexBackendInfo(
            name="python_exact",
            native=False,
            exact=True,
            requested_mode="python_exact",
            resolved_mode="python_exact",
        )

    def query_best(self, text: str) -> NeighborResult:
        results = self.query_topk(text, 1)
        if results:
            return results[0]
        return NeighborResult(index=None, score=0.0, exact=False)

    def query_topk(self, text: str, k: int) -> list[NeighborResult]:
        if k <= 0:
            return []
        query_norm = normalize_text(text, self.norm_config)
        query_grams = char_ngrams(query_norm, self.sim_config.ngram_orders)
        results: list[NeighborResult] = []
        for index, doc_grams in enumerate(self._doc_grams):
            exact = query_norm == self.normalized_lines[index]
            score = 1.0 if exact else jaccard(query_grams, doc_grams)
            if score > 0.0:
                results.append(NeighborResult(index=index, score=score, exact=exact))
        results.sort(key=lambda item: (-item.score, item.index if item.index is not None else -1))
        if results and results[0].exact:
            return [results[0]]
        return results[:k]

    def batch_query_topk(self, texts: list[str], k: int) -> list[list[NeighborResult]]:
        return [self.query_topk(text, k) for text in texts]

    def score_candidate(self, text: str, index: int) -> float:
        if index < 0 or index >= self.doc_count:
            raise IndexError(f"candidate index out of range: {index}")
        query_norm = normalize_text(text, self.norm_config)
        if query_norm == self.normalized_lines[index]:
            return 1.0
        query_grams = char_ngrams(query_norm, self.sim_config.ngram_orders)
        return jaccard(query_grams, self._doc_grams[index])

    def score_candidates(self, text: str, indices: list[int]) -> dict[int, float]:
        return {index: self.score_candidate(text, index) for index in indices}
