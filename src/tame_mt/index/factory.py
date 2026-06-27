from __future__ import annotations

from tame_mt.config import IndexConfig, NormalizationConfig, SimilarityConfig

from .native import NgramInvertedIndex


def build_similarity_index(
    lines: list[str],
    norm_config: NormalizationConfig | None = None,
    sim_config: SimilarityConfig | None = None,
    index_config: IndexConfig | None = None,
) -> NgramInvertedIndex:
    return NgramInvertedIndex.build(
        lines,
        norm_config=norm_config,
        sim_config=sim_config,
        index_config=index_config,
    )
