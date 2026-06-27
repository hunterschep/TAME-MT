from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sacrebleu
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

from tame_mt.config import MetricConfig


class PreparedSacreMetricGroupScorer:
    """Reusable SacreBLEU metric scorer for one metric, reference set, and group map."""

    def __init__(
        self,
        metric: str,
        refs: list[list[str]],
        groups: Mapping[str, Sequence[int]],
        config: MetricConfig,
    ) -> None:
        self.metric = metric
        self.refs = refs
        self.groups = tuple(groups.items())
        self.config = config
        self._scorer = _build_sacre_metric(metric, config, refs)
        self._use_segment_stats = _supports_segment_stats(self._scorer)

    def score_systems(self, systems: Mapping[str, list[str]]) -> dict[str, dict[str, float | None]]:
        if not self._use_segment_stats:
            return _score_metric_groups_public(
                self.metric,
                systems,
                self.refs,
                dict(self.groups),
                self.config,
            )

        results: dict[str, dict[str, float | None]] = {}
        try:
            for system_name, hyps in systems.items():
                segment_stats = self._scorer._extract_corpus_statistics(hyps, None)
                results[system_name] = _aggregate_groups(self._scorer, segment_stats, self.groups)
            return results
        except (AttributeError, TypeError):
            self._use_segment_stats = False
            return _score_metric_groups_public(
                self.metric,
                systems,
                self.refs,
                dict(self.groups),
                self.config,
            )


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

    return PreparedSacreMetricGroupScorer(metric, refs, groups, config).score_systems(systems)


def _aggregate_groups(
    scorer: Any,
    segment_stats: list[Any],
    groups: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for name, indices in groups:
        if not indices:
            results[name] = None
            continue
        group_stats = _sum_segment_stats(segment_stats, indices)
        results[name] = float(scorer._compute_score_from_stats(group_stats).score)
    return results


def _sum_segment_stats(segment_stats: list[Any], indices: Sequence[int]) -> Any:
    if len(indices) == 1:
        return segment_stats[indices[0]]

    is_full_span = _is_full_span(indices, len(segment_stats))
    first_stats = segment_stats[0] if is_full_span else segment_stats[indices[0]]
    stat_count = len(first_stats)
    total = [type(first_stats[0])(0.0)] * stat_count
    if is_full_span:
        for stats in segment_stats:
            for stat_idx in range(stat_count):
                total[stat_idx] += stats[stat_idx]
    else:
        for segment_idx in indices:
            stats = segment_stats[segment_idx]
            for stat_idx in range(stat_count):
                total[stat_idx] += stats[stat_idx]
    return total


def _is_full_span(indices: Sequence[int], length: int) -> bool:
    if len(indices) != length:
        return False
    if isinstance(indices, range):
        return indices.start == 0 and indices.stop == length and indices.step == 1
    return all(index == position for position, index in enumerate(indices))


def _supports_segment_stats(scorer: Any) -> bool:
    return callable(getattr(scorer, "_extract_corpus_statistics", None)) and callable(
        getattr(scorer, "_compute_score_from_stats", None)
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
