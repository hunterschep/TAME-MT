from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from tame_mt.config import NormalizationConfig, SimilarityConfig
from tame_mt.ngrams import char_ngrams
from tame_mt.normalize import normalize_text
from tame_mt.similarity import jaccard


@dataclass(frozen=True)
class NeighborResult:
    index: int | None
    score: float
    exact: bool = False


class NgramInvertedIndex:
    def __init__(
        self,
        normalized_lines: list[str],
        gram_sets: list[frozenset[str]],
        postings: dict[str, list[int]],
        exact_map: dict[str, list[int]],
        norm_config: NormalizationConfig,
        sim_config: SimilarityConfig,
    ) -> None:
        self.normalized_lines = normalized_lines
        self.gram_sets = gram_sets
        self.gram_counts = [len(grams) for grams in gram_sets]
        self.postings = postings
        self.exact_map = exact_map
        self.norm_config = norm_config
        self.sim_config = sim_config

    @classmethod
    def build(
        cls,
        lines: list[str],
        norm_config: NormalizationConfig | None = None,
        sim_config: SimilarityConfig | None = None,
    ) -> "NgramInvertedIndex":
        norm_config = norm_config or NormalizationConfig()
        sim_config = sim_config or SimilarityConfig()
        normalized_lines = [normalize_text(line, norm_config) for line in lines]
        gram_sets = [char_ngrams(line, sim_config.ngram_orders) for line in normalized_lines]
        postings: dict[str, list[int]] = defaultdict(list)
        exact_map: dict[str, list[int]] = defaultdict(list)

        for idx, (line, grams) in enumerate(zip(normalized_lines, gram_sets)):
            exact_map[line].append(idx)
            for gram in grams:
                postings[gram].append(idx)

        return cls(
            normalized_lines=normalized_lines,
            gram_sets=gram_sets,
            postings={gram: sorted(indices) for gram, indices in postings.items()},
            exact_map={line: indices for line, indices in exact_map.items()},
            norm_config=norm_config,
            sim_config=sim_config,
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

        if not query_grams:
            exact_indices = self.exact_map.get(query_norm)
            if exact_indices:
                return [NeighborResult(index=exact_indices[0], score=1.0, exact=True)]
            return []

        intersection_counts: Counter[int] = Counter()
        for gram in query_grams:
            for idx in self.postings.get(gram, []):
                intersection_counts[idx] += 1

        results: list[NeighborResult] = []
        for idx, inter in intersection_counts.items():
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

        results.sort(key=lambda result: (-result.score, result.index if result.index is not None else 10**18))
        return results[:k]

    def score_candidate(self, text: str, index: int) -> float:
        if index < 0 or index >= len(self.gram_sets):
            raise IndexError(f"candidate index out of range: {index}")
        query_norm = normalize_text(text, self.norm_config)
        query_grams = char_ngrams(query_norm, self.sim_config.ngram_orders)
        return jaccard(query_grams, self.gram_sets[index])

    def normalized(self, text: str) -> str:
        return normalize_text(text, self.norm_config)
