from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sacrebleu
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

from tame_mt.config import MetricConfig


def score_bleu(hyps: list[str], refs: list[list[str]], config: MetricConfig) -> float:
    score = sacrebleu.corpus_bleu(
        hyps,
        refs,
        tokenize=config.bleu_tokenize,
        lowercase=config.bleu_lowercase,
    )
    return float(score.score)


def score_chrf(hyps: list[str], refs: list[list[str]], config: MetricConfig) -> float:
    score = sacrebleu.corpus_chrf(
        hyps,
        refs,
        word_order=config.chrf_word_order,
    )
    return float(score.score)


def score_sacre_metric_groups(
    metric: str,
    hyps: list[str],
    refs: list[list[str]],
    groups: Mapping[str, Sequence[int]],
    config: MetricConfig,
) -> dict[str, float | None]:
    """Score one SacreBLEU metric for multiple index groups in one stats pass."""

    return score_sacre_metric_groups_for_systems(
        metric,
        {"system": hyps},
        refs,
        groups,
        config,
    )["system"]


def score_sacre_metric_groups_for_systems(
    metric: str,
    systems: Mapping[str, list[str]],
    refs: list[list[str]],
    groups: Mapping[str, Sequence[int]],
    config: MetricConfig,
) -> dict[str, dict[str, float | None]]:
    """Score one SacreBLEU metric for multiple systems and groups.

    The SacreBLEU metric object owns a preprocessed reference cache, so this
    avoids rebuilding reference n-grams for the system and TM baseline.
    """

    scorer = _build_sacre_metric(metric, config, refs)
    if not _supports_segment_stats(scorer):
        return _score_metric_groups_public(metric, systems, refs, groups, config)

    results: dict[str, dict[str, float | None]] = {}
    try:
        for system_name, hyps in systems.items():
            segment_stats = scorer._extract_corpus_statistics(hyps, None)
            results[system_name] = _aggregate_groups(scorer, segment_stats, groups)
        return results
    except (AttributeError, TypeError):
        return _score_metric_groups_public(metric, systems, refs, groups, config)


def _aggregate_groups(
    scorer: Any,
    segment_stats: list[Any],
    groups: Mapping[str, Sequence[int]],
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for name, indices in groups.items():
        if not indices:
            results[name] = None
            continue
        group_stats = (
            segment_stats
            if _is_full_span(indices, len(segment_stats))
            else [segment_stats[index] for index in indices]
        )
        results[name] = float(scorer._aggregate_and_compute(group_stats).score)
    return results


def _is_full_span(indices: Sequence[int], length: int) -> bool:
    if len(indices) != length:
        return False
    if isinstance(indices, range):
        return indices.start == 0 and indices.stop == length and indices.step == 1
    return all(index == position for position, index in enumerate(indices))


def _supports_segment_stats(scorer: Any) -> bool:
    return callable(getattr(scorer, "_extract_corpus_statistics", None)) and callable(
        getattr(scorer, "_aggregate_and_compute", None)
    )


def _score_metric_groups_public(
    metric: str,
    systems: Mapping[str, list[str]],
    refs: list[list[str]],
    groups: Mapping[str, Sequence[int]],
    config: MetricConfig,
) -> dict[str, dict[str, float | None]]:
    return {
        system_name: _score_public_groups(metric, hyps, refs, groups, config)
        for system_name, hyps in systems.items()
    }


def _score_public_groups(
    metric: str,
    hyps: list[str],
    refs: list[list[str]],
    groups: Mapping[str, Sequence[int]],
    config: MetricConfig,
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for name, indices in groups.items():
        if not indices:
            results[name] = None
            continue
        subset_hyps = [hyps[index] for index in indices]
        subset_refs = [[ref[index] for index in indices] for ref in refs]
        results[name] = _score_public_metric(metric, subset_hyps, subset_refs, config)
    return results


def _score_public_metric(
    metric: str,
    hyps: list[str],
    refs: list[list[str]],
    config: MetricConfig,
) -> float:
    if metric == "bleu":
        return score_bleu(hyps, refs, config)
    if metric == "chrf":
        return score_chrf(hyps, refs, config)
    raise ValueError(f"unsupported metric: {metric}")


def _build_sacre_metric(
    metric: str,
    config: MetricConfig,
    refs: list[list[str]] | None = None,
) -> Any:
    if metric == "bleu":
        return BLEU(
            tokenize=config.bleu_tokenize,
            lowercase=config.bleu_lowercase,
            smooth_method="exp",
            smooth_value=None,
            effective_order=False,
            references=refs,
        )
    if metric == "chrf":
        return CHRF(word_order=config.chrf_word_order, whitespace=False, references=refs)
    raise ValueError(f"unsupported metric: {metric}")
