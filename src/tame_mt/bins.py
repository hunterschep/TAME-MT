from __future__ import annotations

from statistics import mean

from tame_mt.config import BinConfig, ScoreConfig
from tame_mt.schema import BinReport, SegmentExposure
from tame_mt.scoring import delta_scores, score_metrics

BIN_ORDER = ("source_exact", "near", "medium", "far")


def assign_bin(segment_exposure: SegmentExposure, config: BinConfig) -> str:
    if segment_exposure.source_exact:
        return "source_exact"
    if segment_exposure.source_exposure >= config.near_threshold:
        return "near"
    if segment_exposure.source_exposure >= config.far_threshold:
        return "medium"
    return "far"


def assign_bin_values(source_exact: bool, source_exposure: float, config: BinConfig) -> str:
    if source_exact:
        return "source_exact"
    if source_exposure >= config.near_threshold:
        return "near"
    if source_exposure >= config.far_threshold:
        return "medium"
    return "far"


def score_bins(
    hyp: list[str] | None,
    tm_hyp: list[str] | None,
    refs: list[list[str]] | None,
    exposures: list[SegmentExposure],
    config: ScoreConfig,
) -> list[BinReport]:
    num_test = len(exposures)
    reports: list[BinReport] = []
    for name in BIN_ORDER:
        segments = [segment for segment in exposures if segment.bin == name]
        indices = [segment.index for segment in segments]
        subset_refs = _subset_refs(refs, indices) if refs is not None else None
        system_scores = _score_subset(hyp, subset_refs, indices, config)
        tm_scores = _score_subset(tm_hyp, subset_refs, indices, config)
        reports.append(
            BinReport(
                name=name,
                count=len(indices),
                percentage=(len(indices) / num_test) if num_test else 0.0,
                mean_source_exposure=(
                    mean(segment.source_exposure for segment in segments) if segments else None
                ),
                system_scores=system_scores,
                tm_scores=tm_scores,
                delta_scores=delta_scores(system_scores, tm_scores, config.metrics),
            )
        )
    return reports


def compute_generalization_gap(
    bin_reports: list[BinReport],
    metrics: tuple[str, ...],
) -> dict[str, float | None]:
    by_name = {report.name: report for report in bin_reports}
    near = by_name.get("near")
    far = by_name.get("far")
    gap: dict[str, float | None] = {}
    for metric in metrics:
        if not near or not far:
            gap[metric] = None
            continue
        near_score = near.system_scores.get(metric)
        far_score = far.system_scores.get(metric)
        gap[metric] = (
            near_score - far_score if near_score is not None and far_score is not None else None
        )
    return gap


def _subset_refs(refs: list[list[str]], indices: list[int]) -> list[list[str]]:
    return [[ref[idx] for idx in indices] for ref in refs]


def _score_subset(
    hyps: list[str] | None,
    refs: list[list[str]] | None,
    indices: list[int],
    config: ScoreConfig,
) -> dict[str, float | None]:
    if hyps is None or refs is None or not indices:
        return {metric: None for metric in config.metrics}
    return score_metrics([hyps[idx] for idx in indices], refs, config)
