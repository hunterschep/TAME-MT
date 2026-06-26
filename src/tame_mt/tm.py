from __future__ import annotations

from tame_mt.config import ScoreConfig
from tame_mt.schema import SegmentExposure, SegmentTMResult


def build_tm_hypotheses(
    train_tgt: list[str],
    exposures: list[SegmentExposure],
    config: ScoreConfig,
) -> tuple[list[str], list[SegmentTMResult]]:
    hyps: list[str] = []
    results: list[SegmentTMResult] = []
    for segment in exposures:
        source_index = segment.source_nn_index
        if source_index is None:
            if config.tm.zero_policy == "nearest" and train_tgt:
                tm_hyp = train_tgt[0]
                tm_index = 0
            else:
                tm_hyp = ""
                tm_index = None
        else:
            tm_hyp = train_tgt[source_index]
            tm_index = source_index
        hyps.append(tm_hyp)
        results.append(
            SegmentTMResult(
                index=segment.index,
                tm_hyp=tm_hyp,
                tm_source_index=tm_index,
                tm_source_similarity=segment.source_exposure,
            )
        )
    return hyps, results
