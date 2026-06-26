from __future__ import annotations

from tame_mt.config import ScoreConfig
from tame_mt.metrics.sacre import score_bleu, score_chrf


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
