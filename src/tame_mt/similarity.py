from __future__ import annotations

from tame_mt.config import NormalizationConfig, SimilarityConfig
from tame_mt.ngrams import char_ngrams
from tame_mt.normalize import normalize_text


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def text_similarity(
    left: str,
    right: str,
    norm_config: NormalizationConfig | None = None,
    sim_config: SimilarityConfig | None = None,
) -> float:
    norm_config = norm_config or NormalizationConfig()
    sim_config = sim_config or SimilarityConfig()
    left_grams = char_ngrams(normalize_text(left, norm_config), sim_config.ngram_orders)
    right_grams = char_ngrams(normalize_text(right, norm_config), sim_config.ngram_orders)
    return jaccard(left_grams, right_grams)
