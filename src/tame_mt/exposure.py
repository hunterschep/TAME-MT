from __future__ import annotations

from statistics import mean, median
from typing import Iterable

import numpy as np

from tame_mt.bins import assign_bin_values
from tame_mt.config import ScoreConfig
from tame_mt.index import NeighborResult, NgramInvertedIndex
from tame_mt.normalize import normalize_text
from tame_mt.schema import ExposureSummary, SegmentExposure


def compute_exposure(
    train_src: list[str],
    train_tgt: list[str] | None,
    test_src: list[str],
    refs: list[list[str]] | None,
    config: ScoreConfig,
) -> list[SegmentExposure]:
    source_index = NgramInvertedIndex.build(
        train_src,
        norm_config=config.normalization,
        sim_config=config.similarity,
    )
    target_index = (
        NgramInvertedIndex.build(train_tgt, config.normalization, config.similarity)
        if train_tgt is not None
        else None
    )
    exact_source_set = set(source_index.normalized_lines)
    exact_target_set = set(target_index.normalized_lines) if target_index else None
    exact_pair_set = _build_exact_pair_set(train_src, train_tgt, config) if train_tgt else None

    exposures: list[SegmentExposure] = []
    for idx, source_text in enumerate(test_src):
        src_nn = source_index.query_best(source_text)
        norm_source = normalize_text(source_text, config.normalization)
        source_exact = norm_source in exact_source_set

        ref_texts = [ref[idx] for ref in refs] if refs is not None else []
        target_nn = _best_target_neighbor(ref_texts, target_index) if target_index else None
        target_exact = (
            any(normalize_text(ref, config.normalization) in exact_target_set for ref in ref_texts)
            if exact_target_set is not None
            else None
        )
        pair_exact = (
            any(
                (norm_source, normalize_text(ref, config.normalization)) in exact_pair_set
                for ref in ref_texts
            )
            if exact_pair_set is not None
            else None
        )
        pair_nn = (
            _compute_pair_neighbor(source_text, ref_texts, source_index, target_index, config)
            if target_index is not None and ref_texts
            else None
        )

        bin_name = assign_bin_values(source_exact, src_nn.score, config.bins)
        exposures.append(
            SegmentExposure(
                index=idx,
                source_exposure=src_nn.score,
                source_nn_index=src_nn.index,
                source_exact=source_exact,
                target_exposure=target_nn.score if target_nn else None,
                target_nn_index=target_nn.index if target_nn else None,
                target_exact=target_exact,
                pair_exposure=pair_nn.score if pair_nn else None,
                pair_nn_index=pair_nn.index if pair_nn else None,
                pair_exact=pair_exact,
                bin=bin_name,
            )
        )
    return exposures


def summarize_exposures(exposures: list[SegmentExposure], config: ScoreConfig) -> ExposureSummary:
    return ExposureSummary(
        source=_summarize_side(
            [segment.source_exposure for segment in exposures],
            [segment.source_exact for segment in exposures],
            config.bins.leak_thresholds,
        ),
        target=(
            _summarize_side(
                [segment.target_exposure for segment in exposures if segment.target_exposure is not None],
                [bool(segment.target_exact) for segment in exposures if segment.target_exact is not None],
                config.bins.leak_thresholds,
            )
            if any(segment.target_exposure is not None for segment in exposures)
            else None
        ),
        pair=(
            _summarize_side(
                [segment.pair_exposure for segment in exposures if segment.pair_exposure is not None],
                [bool(segment.pair_exact) for segment in exposures if segment.pair_exact is not None],
                config.bins.leak_thresholds,
            )
            if any(segment.pair_exposure is not None for segment in exposures)
            else None
        ),
    )


def _summarize_side(
    scores: list[float],
    exact_flags: list[bool],
    thresholds: tuple[float, ...],
) -> dict[str, object]:
    if not scores:
        return {
            "mean": None,
            "median": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
            "exact_overlap": None,
            "at_threshold": {f"{threshold:.2f}": None for threshold in thresholds},
        }
    arr = np.array(scores, dtype=float)
    return {
        "mean": float(mean(scores)),
        "median": float(median(scores)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(max(scores)),
        "exact_overlap": sum(exact_flags) / len(exact_flags) if exact_flags else None,
        "at_threshold": {
            f"{threshold:.2f}": sum(score >= threshold for score in scores) / len(scores)
            for threshold in thresholds
        },
    }


def _best_target_neighbor(
    ref_texts: list[str],
    target_index: NgramInvertedIndex | None,
) -> NeighborResult | None:
    if target_index is None or not ref_texts:
        return None
    best = NeighborResult(index=None, score=0.0, exact=False)
    for ref in ref_texts:
        result = target_index.query_best(ref)
        best_index = best.index if best.index is not None else 10**18
        result_index = result.index if result.index is not None else 10**18
        if result.score > best.score or (result.score == best.score and result_index < best_index):
            best = result
    return best


def _compute_pair_neighbor(
    source_text: str,
    ref_texts: list[str],
    source_index: NgramInvertedIndex,
    target_index: NgramInvertedIndex | None,
    config: ScoreConfig,
) -> NeighborResult:
    if target_index is None:
        return NeighborResult(index=None, score=0.0, exact=False)

    candidates = _candidate_indices(source_index.query_topk(source_text, config.index.topk))
    for ref in ref_texts:
        candidates.update(_candidate_indices(target_index.query_topk(ref, config.index.topk)))

    best = NeighborResult(index=None, score=0.0, exact=False)
    for candidate in sorted(candidates):
        src_sim = source_index.score_candidate(source_text, candidate)
        tgt_sim = max(target_index.score_candidate(ref, candidate) for ref in ref_texts)
        pair_sim = min(src_sim, tgt_sim)
        if pair_sim > best.score:
            best = NeighborResult(index=candidate, score=pair_sim, exact=pair_sim == 1.0)
    return best


def _candidate_indices(results: Iterable[NeighborResult]) -> set[int]:
    return {result.index for result in results if result.index is not None}


def _build_exact_pair_set(
    train_src: list[str],
    train_tgt: list[str] | None,
    config: ScoreConfig,
) -> set[tuple[str, str]]:
    if train_tgt is None:
        return set()
    return {
        (
            normalize_text(source, config.normalization),
            normalize_text(target, config.normalization),
        )
        for source, target in zip(train_src, train_tgt)
    }
