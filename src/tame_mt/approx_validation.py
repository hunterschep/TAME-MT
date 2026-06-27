from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar

from tame_mt.config import RetrievalConfig, ScoreConfig
from tame_mt.exposure import compute_exposure_result
from tame_mt.schema import SegmentExposure, SegmentTMResult
from tame_mt.scoring import score_metrics
from tame_mt.tm import build_tm_hypotheses

MIN_SOURCE_TOP1_AGREEMENT = 0.95
MIN_SOURCE_BIN_AGREEMENT = 0.99
MAX_SOURCE_MEAN_ABS_ERROR = 0.05
MAX_SOURCE_P95_ABS_ERROR = 0.15
MIN_TARGET_TOP1_AGREEMENT = 0.95
MIN_PAIR_THRESHOLD_AGREEMENT = 0.99
MAX_TM_BLEU_ABS_DELTA = 1.0


class _Indexed(Protocol):
    index: int


T = TypeVar("T", bound=_Indexed)


@dataclass(frozen=True, slots=True)
class ApproxValidation:
    payload: dict[str, Any]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def validate_approximate_run(
    *,
    train_src: list[str],
    train_tgt: list[str] | None,
    test_src: list[str],
    refs: list[list[str]] | None,
    approx_exposures: list[SegmentExposure],
    approx_tm_results: list[SegmentTMResult],
    config: ScoreConfig,
    sample_size: int,
    seed: int,
    exact_mode: str = "native_exact",
) -> ApproxValidation:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not test_src:
        raise ValueError("test_src must contain at least one segment")

    sampled_indices = _sample_indices(len(test_src), sample_size, seed)
    exact_config = replace(
        config,
        index=replace(config.index, mode=exact_mode),
        retrieval=RetrievalConfig(mode="exact", allow_approximate=False),
    )
    sample_test_src = [test_src[index] for index in sampled_indices]
    sample_refs = (
        [[ref[index] for index in sampled_indices] for ref in refs] if refs is not None else None
    )
    exact_result = compute_exposure_result(
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=sample_test_src,
        refs=sample_refs,
        config=exact_config,
    )

    approx_by_index = _items_by_index(approx_exposures, len(test_src), "approx exposures")
    source_top1_matches = 0
    source_bin_matches = 0
    source_abs_errors: list[float] = []
    target_total = 0
    target_top1_matches = 0
    pair_threshold_totals = {threshold: 0 for threshold in config.bins.leak_thresholds}
    pair_threshold_matches = {threshold: 0 for threshold in config.bins.leak_thresholds}

    for local_index, original_index in enumerate(sampled_indices):
        exact_segment = exact_result.segments[local_index]
        approx_segment = approx_by_index[original_index]

        if approx_segment.source_nn_index == exact_segment.source_nn_index:
            source_top1_matches += 1
        if approx_segment.bin == exact_segment.bin:
            source_bin_matches += 1
        source_abs_errors.append(
            abs(float(approx_segment.source_exposure) - float(exact_segment.source_exposure))
        )

        if exact_segment.target_exposure is not None and approx_segment.target_exposure is not None:
            target_total += 1
            if (
                approx_segment.target_nn_index == exact_segment.target_nn_index
                and approx_segment.target_ref_index == exact_segment.target_ref_index
            ):
                target_top1_matches += 1

        if exact_segment.pair_exposure is not None and approx_segment.pair_exposure is not None:
            for threshold in config.bins.leak_thresholds:
                exact_flag = exact_segment.pair_exposure >= threshold
                approx_flag = approx_segment.pair_exposure >= threshold
                pair_threshold_totals[threshold] += 1
                if exact_flag == approx_flag:
                    pair_threshold_matches[threshold] += 1

    source_top1_agreement = source_top1_matches / len(sampled_indices)
    source_bin_agreement = source_bin_matches / len(sampled_indices)
    source_mean_abs_error = sum(source_abs_errors) / len(source_abs_errors)
    source_p95_abs_error = _percentile(source_abs_errors, 95)
    target_top1_agreement = target_top1_matches / target_total if target_total > 0 else None
    pair_threshold_agreement = {
        f"{threshold:.2f}": (
            pair_threshold_matches[threshold] / pair_threshold_totals[threshold]
            if pair_threshold_totals[threshold] > 0
            else None
        )
        for threshold in config.bins.leak_thresholds
    }
    sample_approx_tm_results: list[SegmentTMResult] = []
    if train_tgt is not None and sample_refs is not None:
        approx_tm_by_index = _items_by_index(approx_tm_results, len(test_src), "approx TM results")
        sample_approx_tm_results = [approx_tm_by_index[index] for index in sampled_indices]
    tm_bleu_abs_delta = _tm_bleu_abs_delta(
        train_tgt=train_tgt,
        exact_exposures=exact_result.segments,
        approx_tm_results=sample_approx_tm_results,
        sample_refs=sample_refs,
        config=config,
    )

    failures = _validation_failures(
        source_top1_agreement=source_top1_agreement,
        source_bin_agreement=source_bin_agreement,
        source_mean_abs_error=source_mean_abs_error,
        source_p95_abs_error=source_p95_abs_error,
        target_top1_agreement=target_top1_agreement,
        pair_threshold_agreement=pair_threshold_agreement,
        tm_bleu_abs_delta=tm_bleu_abs_delta,
    )
    payload: dict[str, Any] = {
        "sample_size": len(sampled_indices),
        "requested_sample_size": sample_size,
        "seed": seed,
        "exact_mode": exact_mode,
        "sample_indices": sampled_indices,
        "source_top1_agreement": source_top1_agreement,
        "source_bin_agreement": source_bin_agreement,
        "source_mean_abs_error": source_mean_abs_error,
        "source_p95_abs_error": source_p95_abs_error,
        "target_top1_agreement": target_top1_agreement,
        "pair_threshold_agreement": pair_threshold_agreement,
        "tm_bleu_abs_delta_on_sample": tm_bleu_abs_delta,
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "min_source_top1_agreement": MIN_SOURCE_TOP1_AGREEMENT,
            "min_source_bin_agreement": MIN_SOURCE_BIN_AGREEMENT,
            "max_source_mean_abs_error": MAX_SOURCE_MEAN_ABS_ERROR,
            "max_source_p95_abs_error": MAX_SOURCE_P95_ABS_ERROR,
            "min_target_top1_agreement": MIN_TARGET_TOP1_AGREEMENT,
            "min_pair_threshold_agreement": MIN_PAIR_THRESHOLD_AGREEMENT,
            "max_tm_bleu_abs_delta": MAX_TM_BLEU_ABS_DELTA,
        },
    }
    return ApproxValidation(payload=payload, failures=failures)


def _sample_indices(total: int, sample_size: int, seed: int) -> list[int]:
    count = min(sample_size, total)
    if count == total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), count))


def _items_by_index(items: list[T], expected_total: int, label: str) -> dict[int, T]:
    by_index: dict[int, T] = {}
    for item in items:
        index = item.index
        if not isinstance(index, int):
            raise ValueError(f"{label} contains an item without an integer index")
        if index in by_index:
            raise ValueError(f"{label} contains duplicate index {index}")
        by_index[index] = item
    missing = [index for index in range(expected_total) if index not in by_index]
    if missing:
        preview = ", ".join(str(index) for index in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"{label} is missing index {preview}{suffix}")
    return by_index


def _tm_bleu_abs_delta(
    *,
    train_tgt: list[str] | None,
    exact_exposures: list[SegmentExposure],
    approx_tm_results: list[SegmentTMResult],
    sample_refs: list[list[str]] | None,
    config: ScoreConfig,
) -> float | None:
    if train_tgt is None or sample_refs is None:
        return None
    exact_tm_hyp, _ = build_tm_hypotheses(train_tgt, exact_exposures, config)
    approx_tm_hyp = [result.tm_hyp for result in approx_tm_results]
    bleu_config = replace(config, metrics=("bleu",))
    exact_bleu = score_metrics(exact_tm_hyp, sample_refs, bleu_config)["bleu"]
    approx_bleu = score_metrics(approx_tm_hyp, sample_refs, bleu_config)["bleu"]
    if exact_bleu is None or approx_bleu is None:
        return None
    return abs(approx_bleu - exact_bleu)


def _validation_failures(
    *,
    source_top1_agreement: float,
    source_bin_agreement: float,
    source_mean_abs_error: float,
    source_p95_abs_error: float,
    target_top1_agreement: float | None,
    pair_threshold_agreement: dict[str, float | None],
    tm_bleu_abs_delta: float | None,
) -> list[str]:
    failures: list[str] = []
    if source_top1_agreement < MIN_SOURCE_TOP1_AGREEMENT:
        failures.append(
            f"source_top1_agreement {source_top1_agreement:.4f} < {MIN_SOURCE_TOP1_AGREEMENT:.4f}"
        )
    if source_bin_agreement < MIN_SOURCE_BIN_AGREEMENT:
        failures.append(
            f"source_bin_agreement {source_bin_agreement:.4f} < {MIN_SOURCE_BIN_AGREEMENT:.4f}"
        )
    if source_mean_abs_error > MAX_SOURCE_MEAN_ABS_ERROR:
        failures.append(
            f"source_mean_abs_error {source_mean_abs_error:.4f} > {MAX_SOURCE_MEAN_ABS_ERROR:.4f}"
        )
    if source_p95_abs_error > MAX_SOURCE_P95_ABS_ERROR:
        failures.append(
            f"source_p95_abs_error {source_p95_abs_error:.4f} > {MAX_SOURCE_P95_ABS_ERROR:.4f}"
        )
    if target_top1_agreement is not None and target_top1_agreement < MIN_TARGET_TOP1_AGREEMENT:
        failures.append(
            f"target_top1_agreement {target_top1_agreement:.4f} < {MIN_TARGET_TOP1_AGREEMENT:.4f}"
        )
    for threshold, agreement in pair_threshold_agreement.items():
        if agreement is not None and agreement < MIN_PAIR_THRESHOLD_AGREEMENT:
            failures.append(
                f"pair_threshold_agreement[{threshold}] {agreement:.4f} "
                f"< {MIN_PAIR_THRESHOLD_AGREEMENT:.4f}"
            )
    if tm_bleu_abs_delta is not None and tm_bleu_abs_delta > MAX_TM_BLEU_ABS_DELTA:
        failures.append(
            f"tm_bleu_abs_delta_on_sample {tm_bleu_abs_delta:.4f} > {MAX_TM_BLEU_ABS_DELTA:.4f}"
        )
    return failures


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
