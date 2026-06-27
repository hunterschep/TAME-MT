from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tame_mt.config import BinConfig, ScoreConfig
from tame_mt.schema import BinReport, SegmentExposure
from tame_mt.scoring import delta_scores, score_systems_by_groups

BIN_ORDER = ("source_exact", "near", "medium", "far")
ALL_GROUP = "__all__"
TM_GROUP = "__tame_tm__"


@dataclass(frozen=True)
class BinScoringResult:
    system_scores: dict[str, float | None]
    tm_scores: dict[str, float | None]
    bin_reports: list[BinReport]


@dataclass(frozen=True)
class MultiBinScoringResult:
    system_scores: dict[str, dict[str, float | None]]
    tm_scores: dict[str, float | None]
    bin_reports: dict[str, list[BinReport]]


@dataclass(frozen=True)
class BinGroupIndex:
    groups: dict[str, Sequence[int]]
    counts: dict[str, int]
    percentages: dict[str, float]
    mean_source_exposure: dict[str, float | None]


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
    bin_index = _build_bin_group_index(exposures)
    grouped_scores = score_systems_by_groups(
        {"system": hyp, "tm": tm_hyp},
        refs,
        bin_index.groups,
        config,
    )
    system_group_scores = grouped_scores["system"]
    tm_group_scores = grouped_scores["tm"]
    return BinScoringResult(
        system_scores=system_group_scores[ALL_GROUP],
        tm_scores=tm_group_scores[ALL_GROUP],
        bin_reports=_build_bin_reports(
            bin_index,
            system_group_scores,
            tm_group_scores,
            config,
        ),
    )


def score_many_corpus_and_bins(
    systems: Mapping[str, list[str] | None],
    tm_hyp: list[str] | None,
    refs: list[list[str]] | None,
    exposures: list[SegmentExposure],
    config: ScoreConfig,
) -> MultiBinScoringResult:
    if TM_GROUP in systems:
        raise ValueError(f"{TM_GROUP!r} is reserved for internal TM scoring")
    bin_index = _build_bin_group_index(exposures)
    grouped_scores = score_systems_by_groups(
        {**systems, TM_GROUP: tm_hyp},
        refs,
        bin_index.groups,
        config,
    )
    tm_group_scores = grouped_scores[TM_GROUP]
    return MultiBinScoringResult(
        system_scores={
            system_name: grouped_scores[system_name][ALL_GROUP] for system_name in systems
        },
        tm_scores=tm_group_scores[ALL_GROUP],
        bin_reports={
            system_name: _build_bin_reports(
                bin_index,
                grouped_scores[system_name],
                tm_group_scores,
                config,
            )
            for system_name in systems
        },
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


def _build_bin_group_index(exposures: list[SegmentExposure]) -> BinGroupIndex:
    num_test = len(exposures)
    bin_groups: dict[str, list[int]] = {name: [] for name in BIN_ORDER}
    source_sums = {name: 0.0 for name in BIN_ORDER}
    aligned_indices = True
    for expected_index, segment in enumerate(exposures):
        if segment.index != expected_index:
            aligned_indices = False
        if segment.bin in bin_groups:
            bin_groups[segment.bin].append(segment.index)
            source_sums[segment.bin] += segment.source_exposure
    all_indices = range(num_test) if aligned_indices else [segment.index for segment in exposures]
    counts = {name: len(indices) for name, indices in bin_groups.items()}
    return BinGroupIndex(
        groups={ALL_GROUP: all_indices, **bin_groups},
        counts=counts,
        percentages={
            name: (count / num_test) if num_test else 0.0 for name, count in counts.items()
        },
        mean_source_exposure={
            name: source_sums[name] / count if count else None for name, count in counts.items()
        },
    )


def _build_bin_reports(
    bin_index: BinGroupIndex,
    system_group_scores: dict[str, dict[str, float | None]],
    tm_group_scores: dict[str, dict[str, float | None]],
    config: ScoreConfig,
) -> list[BinReport]:
    reports: list[BinReport] = []
    for name in BIN_ORDER:
        system_scores = system_group_scores[name]
        tm_scores = tm_group_scores[name]
        reports.append(
            BinReport(
                name=name,
                count=bin_index.counts[name],
                percentage=bin_index.percentages[name],
                mean_source_exposure=bin_index.mean_source_exposure[name],
                system_scores=system_scores,
                tm_scores=tm_scores,
                delta_scores=delta_scores(system_scores, tm_scores, config.metrics),
            )
        )
    return reports
