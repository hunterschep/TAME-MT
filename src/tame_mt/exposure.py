from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean, median

from tame_mt.bins import assign_bin_values
from tame_mt.config import ScoreConfig
from tame_mt.index import IndexBackendInfo, NeighborResult, NgramInvertedIndex
from tame_mt.normalize import normalize_text
from tame_mt.schema import ExposureSummary, SegmentExposure


@dataclass(frozen=True)
class ExposureComputation:
    segments: list[SegmentExposure]
    backend: IndexBackendInfo


def compute_exposure(
    train_src: list[str],
    train_tgt: list[str] | None,
    test_src: list[str],
    refs: list[list[str]] | None,
    config: ScoreConfig,
) -> list[SegmentExposure]:
    return compute_exposure_result(train_src, train_tgt, test_src, refs, config).segments


def compute_exposure_result(
    train_src: list[str],
    train_tgt: list[str] | None,
    test_src: list[str],
    refs: list[list[str]] | None,
    config: ScoreConfig,
) -> ExposureComputation:
    source_index = NgramInvertedIndex.build(
        train_src,
        norm_config=config.normalization,
        sim_config=config.similarity,
        index_config=config.index,
    )
    target_index = (
        NgramInvertedIndex.build(train_tgt, config.normalization, config.similarity, config.index)
        if train_tgt is not None
        else None
    )
    exact_source_set = set(source_index.normalized_lines)
    exact_target_set = set(target_index.normalized_lines) if target_index else None
    exact_pair_set = _build_exact_pair_set(train_src, train_tgt, config) if train_tgt else None

    retrieval_k = max(1, config.index.topk)
    source_tops_by_segment = source_index.batch_query_topk(test_src, retrieval_k)
    target_tops_by_ref = (
        [target_index.batch_query_topk(ref, retrieval_k) for ref in refs]
        if target_index is not None and refs is not None
        else []
    )

    exposures: list[SegmentExposure] = []
    for idx, source_text in enumerate(test_src):
        source_top = source_tops_by_segment[idx]
        src_nn = source_top[0] if source_top else NeighborResult(index=None, score=0.0, exact=False)
        norm_source = normalize_text(source_text, config.normalization)
        source_exact = norm_source in exact_source_set

        ref_texts = [ref[idx] for ref in refs] if refs is not None else []
        target_tops = [tops_by_ref[idx] for tops_by_ref in target_tops_by_ref]
        target_nn = _best_target_neighbor(target_tops) if target_index else None
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
            _compute_pair_neighbor(
                source_text=source_text,
                ref_texts=ref_texts,
                source_top=source_top,
                target_tops=target_tops,
                source_index=source_index,
                target_index=target_index,
            )
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
    return ExposureComputation(segments=exposures, backend=source_index.backend_info)


def summarize_exposures(exposures: list[SegmentExposure], config: ScoreConfig) -> ExposureSummary:
    return ExposureSummary(
        source=_summarize_side(
            [segment.source_exposure for segment in exposures],
            [segment.source_exact for segment in exposures],
            config.bins.leak_thresholds,
        ),
        target=(
            _summarize_side(
                [
                    segment.target_exposure
                    for segment in exposures
                    if segment.target_exposure is not None
                ],
                [
                    bool(segment.target_exact)
                    for segment in exposures
                    if segment.target_exact is not None
                ],
                config.bins.leak_thresholds,
            )
            if any(segment.target_exposure is not None for segment in exposures)
            else None
        ),
        pair=(
            _summarize_side(
                [
                    segment.pair_exposure
                    for segment in exposures
                    if segment.pair_exposure is not None
                ],
                [
                    bool(segment.pair_exact)
                    for segment in exposures
                    if segment.pair_exact is not None
                ],
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
    return {
        "mean": float(mean(scores)),
        "median": float(median(scores)),
        "p05": _percentile(scores, 5),
        "p25": _percentile(scores, 25),
        "p75": _percentile(scores, 75),
        "p95": _percentile(scores, 95),
        "max": float(max(scores)),
        "exact_overlap": sum(exact_flags) / len(exact_flags) if exact_flags else None,
        "at_threshold": {
            f"{threshold:.2f}": sum(score >= threshold for score in scores) / len(scores)
            for threshold in thresholds
        },
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _best_target_neighbor(target_tops: list[list[NeighborResult]]) -> NeighborResult | None:
    if not target_tops:
        return None
    best = NeighborResult(index=None, score=0.0, exact=False)
    for top_results in target_tops:
        if not top_results:
            continue
        result = top_results[0]
        best_index = best.index if best.index is not None else 10**18
        result_index = result.index if result.index is not None else 10**18
        if result.score > best.score or (result.score == best.score and result_index < best_index):
            best = result
    return best


def _compute_pair_neighbor(
    source_text: str,
    ref_texts: list[str],
    source_top: list[NeighborResult],
    target_tops: list[list[NeighborResult]],
    source_index: NgramInvertedIndex,
    target_index: NgramInvertedIndex | None,
) -> NeighborResult:
    if target_index is None:
        return NeighborResult(index=None, score=0.0, exact=False)

    candidates = _candidate_indices(source_top)
    for top_results in target_tops:
        candidates.update(_candidate_indices(top_results))

    best = NeighborResult(index=None, score=0.0, exact=False)
    sorted_candidates = sorted(candidates)
    source_scores = source_index.score_candidates(source_text, sorted_candidates)
    target_scores_by_ref = [
        target_index.score_candidates(ref, sorted_candidates) for ref in ref_texts
    ]
    for candidate in sorted_candidates:
        src_sim = source_scores[candidate]
        tgt_sim = max(target_scores[candidate] for target_scores in target_scores_by_ref)
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
        for source, target in zip(train_src, train_tgt, strict=True)
    }
