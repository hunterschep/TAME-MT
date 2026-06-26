from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean, median

from tame_mt.bins import assign_bin_values
from tame_mt.config import ScoreConfig
from tame_mt.exact import build_exact_pair_keys, exact_pair_key
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
    source_index: NgramInvertedIndex | None = None,
    target_index: NgramInvertedIndex | None = None,
    exact_pair_keys: set[str] | None = None,
) -> ExposureComputation:
    if source_index is None:
        source_index = NgramInvertedIndex.build(
            train_src,
            norm_config=config.normalization,
            sim_config=config.similarity,
            index_config=config.index,
        )
    if target_index is None and train_tgt is not None:
        target_index = NgramInvertedIndex.build(
            train_tgt,
            norm_config=config.normalization,
            sim_config=config.similarity,
            index_config=config.index,
        )
    if train_tgt is not None and exact_pair_keys is None:
        exact_pair_keys = _build_exact_pair_keys(
            train_src, train_tgt, source_index, target_index, config
        )

    retrieval_k = max(1, config.index.topk)
    source_tops_by_segment = source_index.batch_query_topk(test_src, retrieval_k)
    target_tops_by_ref = (
        [target_index.batch_query_topk(ref, retrieval_k) for ref in refs]
        if target_index is not None and refs is not None
        else []
    )
    pair_neighbors_by_segment = (
        _batch_pair_neighbors(
            test_src=test_src,
            refs=refs,
            source_tops_by_segment=source_tops_by_segment,
            target_tops_by_ref=target_tops_by_ref,
            source_index=source_index,
            target_index=target_index,
        )
        if target_index is not None
        and refs is not None
        and source_index.supports_native_pair_candidates(target_index)
        else None
    )

    exposures: list[SegmentExposure] = []
    for idx, source_text in enumerate(test_src):
        source_top = source_tops_by_segment[idx]
        src_nn = source_top[0] if source_top else NeighborResult(index=None, score=0.0, exact=False)
        norm_source = normalize_text(source_text, config.normalization)
        source_exact = source_index.contains_exact_normalized(norm_source)

        ref_texts = [ref[idx] for ref in refs] if refs is not None else []
        target_tops = [tops_by_ref[idx] for tops_by_ref in target_tops_by_ref]
        target_nn = _best_target_neighbor(target_tops) if target_index else None
        target_exact = (
            any(
                target_index.contains_exact_normalized(normalize_text(ref, config.normalization))
                for ref in ref_texts
            )
            if target_index is not None
            else None
        )
        pair_exact = (
            any(
                exact_pair_key(norm_source, normalize_text(ref, config.normalization))
                in exact_pair_keys
                for ref in ref_texts
            )
            if exact_pair_keys is not None
            else None
        )
        pair_nn = (
            pair_neighbors_by_segment[idx]
            if pair_neighbors_by_segment is not None
            else _compute_pair_neighbor(
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

    sorted_candidates = sorted(candidates)
    native_best = source_index.best_pair_candidate(
        target_index,
        source_text,
        ref_texts,
        sorted_candidates,
    )
    if native_best is not None:
        return native_best

    best = NeighborResult(index=None, score=0.0, exact=False)
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


def _batch_pair_neighbors(
    test_src: list[str],
    refs: list[list[str]],
    source_tops_by_segment: list[list[NeighborResult]],
    target_tops_by_ref: list[list[list[NeighborResult]]],
    source_index: NgramInvertedIndex,
    target_index: NgramInvertedIndex,
) -> list[NeighborResult] | None:
    ref_texts_by_segment = [[ref[idx] for ref in refs] for idx in range(len(test_src))]
    candidate_indices_by_segment: list[list[int]] = []
    for idx, source_top in enumerate(source_tops_by_segment):
        candidates = _candidate_indices(source_top)
        for tops_by_ref in target_tops_by_ref:
            candidates.update(_candidate_indices(tops_by_ref[idx]))
        candidate_indices_by_segment.append(sorted(candidates))
    return source_index.batch_best_pair_candidates(
        target_index,
        test_src,
        ref_texts_by_segment,
        candidate_indices_by_segment,
    )


def _candidate_indices(results: Iterable[NeighborResult]) -> set[int]:
    return {result.index for result in results if result.index is not None}


def _build_exact_pair_keys(
    train_src: list[str],
    train_tgt: list[str] | None,
    source_index: NgramInvertedIndex,
    target_index: NgramInvertedIndex | None,
    config: ScoreConfig,
) -> set[str]:
    if train_tgt is None:
        return set()
    if target_index is not None and source_index.normalized_lines and target_index.normalized_lines:
        return build_exact_pair_keys(source_index.normalized_lines, target_index.normalized_lines)
    return build_exact_pair_keys(
        (normalize_text(source, config.normalization) for source in train_src),
        (normalize_text(target, config.normalization) for target in train_tgt),
    )
