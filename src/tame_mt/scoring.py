from __future__ import annotations

from collections.abc import Mapping, Sequence

from tame_mt.config import ScoreConfig
from tame_mt.metrics.sacre import (
    score_bleu,
    score_chrf,
    score_sacre_metric_groups,
    score_sacre_metric_groups_for_systems,
)

GroupedScores = dict[str, dict[str, float | None]]
SystemGroupedScores = dict[str, GroupedScores]


def score_metrics(
    hyps: list[str],
    refs: list[list[str]],
    config_or_metrics: ScoreConfig | tuple[str, ...],
) -> dict[str, float | None]:
    if isinstance(config_or_metrics, ScoreConfig):
        config = config_or_metrics
        metrics = config.metrics
    else:
        config = ScoreConfig(metrics=tuple(config_or_metrics))
        metrics = config.metrics

    if not hyps:
        return {metric: None for metric in metrics}

    results: dict[str, float | None] = {}
    for metric in metrics:
        metric_key = metric.lower()
        if metric_key == "bleu":
            results[metric_key] = score_bleu(hyps, refs, config.metric)
        elif metric_key == "chrf":
            results[metric_key] = score_chrf(hyps, refs, config.metric)
        else:
            raise ValueError(f"unsupported metric: {metric}")
    return results


def score_metrics_by_groups(
    hyps: list[str] | None,
    refs: list[list[str]] | None,
    groups: Mapping[str, Sequence[int]],
    config: ScoreConfig,
) -> GroupedScores:
    results = _empty_group_scores(groups, config.metrics)
    if hyps is None or refs is None or not hyps:
        return results

    for metric in config.metrics:
        metric_results = score_sacre_metric_groups(metric, hyps, refs, groups, config.metric)
        for group_name, score in metric_results.items():
            results[group_name][metric] = score
    return results


def score_systems_by_groups(
    systems: Mapping[str, list[str] | None],
    refs: list[list[str]] | None,
    groups: Mapping[str, Sequence[int]],
    config: ScoreConfig,
) -> SystemGroupedScores:
    results: SystemGroupedScores = {
        system_name: _empty_group_scores(groups, config.metrics) for system_name in systems
    }
    if refs is None:
        return results

    active_systems = {
        system_name: hyps for system_name, hyps in systems.items() if hyps is not None and hyps
    }
    if not active_systems:
        return results

    for metric in config.metrics:
        metric_results = score_sacre_metric_groups_for_systems(
            metric,
            active_systems,
            refs,
            groups,
            config.metric,
        )
        for system_name, group_scores in metric_results.items():
            for group_name, score in group_scores.items():
                results[system_name][group_name][metric] = score
    return results


def delta_scores(
    system_scores: dict[str, float | None],
    tm_scores: dict[str, float | None],
    metrics: tuple[str, ...],
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for metric in metrics:
        key = metric.lower()
        system_score = system_scores.get(key)
        tm_score = tm_scores.get(key)
        results[key] = (
            system_score - tm_score if system_score is not None and tm_score is not None else None
        )
    return results


def _empty_group_scores(
    groups: Mapping[str, Sequence[int]],
    metrics: tuple[str, ...],
) -> GroupedScores:
    return {group_name: {metric: None for metric in metrics} for group_name in groups}
