from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from tame_mt.config import BinConfig, ScoreConfig
from tame_mt.schema import BinReport, SegmentExposure
from tame_mt.scoring import delta_scores, score_systems_by_groups

BIN_ORDER = ("source_exact", "near", "medium", "far")
ALL_GROUP = "__all__"


@dataclass(frozen=True)
class BinScoringResult:
    system_scores: dict[str, float | None]
    tm_scores: dict[str, float | None]
    bin_reports: list[BinReport]


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
    return score_corpus_and_bins(hyp, tm_hyp, refs, exposures, config).bin_reports


def score_corpus_and_bins(
    hyp: list[str] | None,
    tm_hyp: list[str] | None,
    refs: list[list[str]] | None,
    exposures: list[SegmentExposure],
    config: ScoreConfig,
) -> BinScoringResult:
    num_test = len(exposures)
    segments_by_bin = {
        name: [segment for segment in exposures if segment.bin == name] for name in BIN_ORDER
    }
    groups = {
        ALL_GROUP: [segment.index for segment in exposures],
        **{
            name: [segment.index for segment in segments]
            for name, segments in segments_by_bin.items()
        },
    }
    grouped_scores = score_systems_by_groups(
        {"system": hyp, "tm": tm_hyp},
        refs,
        groups,
        config,
    )
    system_group_scores = grouped_scores["system"]
    tm_group_scores = grouped_scores["tm"]

    reports: list[BinReport] = []
    for name in BIN_ORDER:
        segments = segments_by_bin[name]
        system_scores = system_group_scores[name]
        tm_scores = tm_group_scores[name]
        reports.append(
            BinReport(
                name=name,
                count=len(segments),
                percentage=(len(segments) / num_test) if num_test else 0.0,
                mean_source_exposure=(
                    mean(segment.source_exposure for segment in segments) if segments else None
                ),
                system_scores=system_scores,
                tm_scores=tm_scores,
                delta_scores=delta_scores(system_scores, tm_scores, config.metrics),
            )
        )
    return BinScoringResult(
        system_scores=system_group_scores[ALL_GROUP],
        tm_scores=tm_group_scores[ALL_GROUP],
        bin_reports=reports,
    )


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
