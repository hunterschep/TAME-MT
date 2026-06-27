from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass

from tame_mt.bins import assign_bin_values
from tame_mt.config import ScoreConfig
from tame_mt.exact import build_exact_pair_keys, exact_pair_key
from tame_mt.index import IndexBackendInfo, NeighborResult, NgramInvertedIndex
from tame_mt.normalize import normalize_text
from tame_mt.schema import ExposureSummary, SegmentExposure


@dataclass(frozen=True, slots=True)
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
    source_index.release_python_normalized_lines()
    if target_index is not None:
        target_index.release_python_normalized_lines()

    needs_pair_candidates = target_index is not None and refs is not None
    retrieval_k = max(1, config.index.topk if needs_pair_candidates else 1)
    normalized_test_src = [source_index.normalized(source) for source in test_src]
    source_tops_by_segment = source_index.batch_query_topk_normalized(
        normalized_test_src, retrieval_k
    )
    normalized_refs_by_ref = (
        [[target_index.normalized(text) for text in ref] for ref in refs]
        if target_index is not None and refs is not None
        else []
    )
    target_tops_by_ref = (
        [
            target_index.batch_query_topk_normalized(normalized_ref, retrieval_k)
            for normalized_ref in normalized_refs_by_ref
        ]
        if target_index is not None
        else []
    )
    pair_neighbors_by_segment = (
        _batch_pair_neighbors(
            normalized_test_src=normalized_test_src,
            normalized_refs_by_ref=normalized_refs_by_ref,
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
        source_exact = src_nn.exact

        ref_texts = [ref[idx] for ref in refs] if refs is not None else []
        target_tops = [tops_by_ref[idx] for tops_by_ref in target_tops_by_ref]
        target_nn, target_ref_index = (
            _best_target_neighbor(target_tops) if target_index else (None, None)
        )
        target_exact = _has_exact_neighbor(target_tops) if target_index is not None else None
        pair_exact = (
            any(
                exact_pair_key(normalized_test_src[idx], normalized_refs_by_ref[ref_idx][idx])
                in exact_pair_keys
                for ref_idx in range(len(ref_texts))
            )
            if exact_pair_keys is not None and refs is not None
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
        pair_ref_index: int | None = None
        if (
            target_index is not None
            and pair_nn is not None
            and pair_nn.index is not None
            and normalized_refs_by_ref
        ):
            pair_ref_index = (
                0
                if len(normalized_refs_by_ref) == 1
                else _best_pair_ref_index(
                    target_index=target_index,
                    ref_texts=[ref[idx] for ref in normalized_refs_by_ref],
                    candidate_index=pair_nn.index,
                    pair_score=pair_nn.score,
                )
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
                target_ref_index=target_ref_index,
                pair_ref_index=pair_ref_index,
            )
        )
    return ExposureComputation(segments=exposures, backend=source_index.backend_info)


def summarize_exposures(exposures: list[SegmentExposure], config: ScoreConfig) -> ExposureSummary:
    source_scores: list[float] = []
    source_exact_flags: list[bool] = []
    target_scores: list[float] = []
    target_exact_flags: list[bool] = []
    pair_scores: list[float] = []
    pair_exact_flags: list[bool] = []

    for segment in exposures:
        source_scores.append(segment.source_exposure)
        source_exact_flags.append(segment.source_exact)
        if segment.target_exposure is not None:
            target_scores.append(segment.target_exposure)
        if segment.target_exact is not None:
            target_exact_flags.append(bool(segment.target_exact))
        if segment.pair_exposure is not None:
            pair_scores.append(segment.pair_exposure)
        if segment.pair_exact is not None:
            pair_exact_flags.append(bool(segment.pair_exact))

    return ExposureSummary(
        source=_summarize_side(source_scores, source_exact_flags, config.bins.leak_thresholds),
        target=(
            _summarize_side(target_scores, target_exact_flags, config.bins.leak_thresholds)
            if target_scores
            else None
        ),
        pair=(
            _summarize_side(pair_scores, pair_exact_flags, config.bins.leak_thresholds)
            if pair_scores
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
    sorted_scores = sorted(scores)
    score_count = len(sorted_scores)
    return {
        "mean": float(sum(sorted_scores) / score_count),
        "median": _percentile_sorted(sorted_scores, 50),
        "p05": _percentile_sorted(sorted_scores, 5),
        "p25": _percentile_sorted(sorted_scores, 25),
        "p75": _percentile_sorted(sorted_scores, 75),
        "p95": _percentile_sorted(sorted_scores, 95),
        "max": float(sorted_scores[-1]),
        "exact_overlap": sum(exact_flags) / len(exact_flags) if exact_flags else None,
        "at_threshold": {
            f"{threshold:.2f}": (score_count - bisect_left(sorted_scores, threshold)) / score_count
            for threshold in thresholds
        },
    }


def _percentile_sorted(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty list")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _best_target_neighbor(
    target_tops: list[list[NeighborResult]],
) -> tuple[NeighborResult | None, int | None]:
    if not target_tops:
        return None, None
    best = NeighborResult(index=None, score=0.0, exact=False)
    best_ref_index: int | None = None
    for ref_index, top_results in enumerate(target_tops):
        if not top_results:
            continue
        result = top_results[0]
        best_index = best.index if best.index is not None else 10**18
        result_index = result.index if result.index is not None else 10**18
        best_ref_tiebreak = best_ref_index if best_ref_index is not None else 10**18
        if (
            result.score > best.score
            or (result.score == best.score and result_index < best_index)
            or (
                result.score == best.score
                and result_index == best_index
                and ref_index < best_ref_tiebreak
            )
        ):
            best = result
            best_ref_index = ref_index if result.index is not None else None
    return best, best_ref_index


def _has_exact_neighbor(target_tops: list[list[NeighborResult]]) -> bool:
    return any(top_results and top_results[0].exact for top_results in target_tops)


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


def _best_pair_ref_index(
    target_index: NgramInvertedIndex,
    ref_texts: list[str],
    candidate_index: int,
    pair_score: float,
) -> int | None:
    if not ref_texts:
        return None
    best_ref_index: int | None = None
    best_target_score = -1.0
    for ref_index, ref_text in enumerate(ref_texts):
        target_score = target_index.score_candidate(ref_text, candidate_index)
        if target_score + 1e-12 >= pair_score:
            return ref_index
        if target_score > best_target_score:
            best_ref_index = ref_index
            best_target_score = target_score
    return best_ref_index


def _batch_pair_neighbors(
    normalized_test_src: list[str],
    normalized_refs_by_ref: list[list[str]],
    source_tops_by_segment: list[list[NeighborResult]],
    target_tops_by_ref: list[list[list[NeighborResult]]],
    source_index: NgramInvertedIndex,
    target_index: NgramInvertedIndex,
) -> list[NeighborResult] | None:
    ref_texts_by_segment = [
        [ref[idx] for ref in normalized_refs_by_ref] for idx in range(len(normalized_test_src))
    ]
    candidate_indices_by_segment: list[list[int]] = []
    for idx, source_top in enumerate(source_tops_by_segment):
        candidates = _candidate_indices(source_top)
        for tops_by_ref in target_tops_by_ref:
            candidates.update(_candidate_indices(tops_by_ref[idx]))
        candidate_indices_by_segment.append(sorted(candidates))
    return source_index.batch_best_pair_candidates_normalized(
        target_index,
        normalized_test_src,
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
