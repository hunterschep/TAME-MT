from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tame_mt.artifacts import validate_segment_artifacts
from tame_mt.bins import (
    ALL_GROUP,
    TM_GROUP,
    _build_bin_group_index,
    _build_bin_reports,
    compute_generalization_gap,
    score_corpus_and_bins,
)
from tame_mt.config import ScoreConfig
from tame_mt.exceptions import ConfigurationError, InputDataError
from tame_mt.exposure import compute_exposure_result, summarize_exposures
from tame_mt.index import NgramInvertedIndex
from tame_mt.io import read_lines, validate_corpus_inputs, validate_equal_lengths
from tame_mt.persistence import IndexBundle
from tame_mt.report import build_signature, config_to_dict
from tame_mt.schema import BinReport, ExposureSummary, SegmentExposure, SegmentTMResult, TameReport
from tame_mt.scoring import PreparedGroupScorer, delta_scores
from tame_mt.tm import build_tm_hypotheses
from tame_mt.version import __version__
from tame_mt.warnings import generate_warnings


@dataclass
class EvaluationResult:
    """Full evaluation result, including report and optional artifacts."""

    report: TameReport
    exposures: list[SegmentExposure]
    tm_hyp: list[str]
    tm_results: list[SegmentTMResult]


class CachedSegmentScorer:
    """Prepared scorer for repeated scoring from cached segment diagnostics."""

    def __init__(
        self,
        config: ScoreConfig,
        exposures: list[SegmentExposure],
        tm_results: list[SegmentTMResult],
        refs: list[list[str]],
        num_train: int,
    ) -> None:
        if not exposures:
            raise InputDataError("segment artifact must contain at least one segment")
        if not refs:
            raise InputDataError("refs must contain at least one reference")
        if num_train <= 0:
            raise InputDataError("num_train must be positive")

        self.config = config
        self.num_train = num_train
        self.exposures, self.tm_results = validate_segment_artifacts(exposures, tm_results)
        self.refs = [list(ref) for ref in refs]
        self.tm_hyp = [result.tm_hyp for result in self.tm_results]
        for ref_idx, ref in enumerate(self.refs):
            validate_equal_lengths("segments", self.exposures, f"ref[{ref_idx}]", ref)
        validate_equal_lengths("segments", self.exposures, "tm_hyp", self.tm_hyp)

        self.exposure_summary = summarize_exposures(self.exposures, self.config)
        self._bin_index = _build_bin_group_index(self.exposures)
        self._group_scorer = PreparedGroupScorer(self.refs, self._bin_index.groups, self.config)
        self._tm_group_scores = self._group_scorer.score_systems({TM_GROUP: self.tm_hyp})[TM_GROUP]
        self.tm_scores = self._tm_group_scores[ALL_GROUP]

    def score(self, hyp: list[str], *, system_name: str = "system") -> TameReport:
        return self.score_many({system_name: hyp})[system_name]

    def score_many(self, systems: Mapping[str, list[str]]) -> dict[str, TameReport]:
        if not systems:
            raise InputDataError("systems must contain at least one hypothesis")
        if TM_GROUP in systems:
            raise InputDataError(f"system name {TM_GROUP!r} is reserved")
        for system_name, hyp in systems.items():
            if not system_name:
                raise InputDataError("system names must be non-empty")
            validate_equal_lengths("segments", self.exposures, f"hyp[{system_name}]", hyp)

        grouped_scores = self._group_scorer.score_systems(systems)
        reports: dict[str, TameReport] = {}
        for system_name, system_group_scores in grouped_scores.items():
            system_scores = system_group_scores[ALL_GROUP]
            deltas = delta_scores(system_scores, self.tm_scores, self.config.metrics)
            bin_reports = _build_bin_reports(
                self._bin_index,
                system_group_scores,
                self._tm_group_scores,
                self.config,
            )
            gen_gap = compute_generalization_gap(bin_reports, self.config.metrics)
            warnings = generate_warnings(
                exposure=self.exposure_summary,
                system_scores=system_scores,
                tm_scores=self.tm_scores,
                bin_reports=bin_reports,
                generalization_gap=gen_gap,
                config=self.config,
                num_test=len(self.exposures),
            )
            reports[system_name] = _build_cached_report(
                config=self.config,
                num_train=self.num_train,
                num_test=len(self.exposures),
                num_refs=len(self.refs),
                system_scores=system_scores,
                tm_scores=self.tm_scores,
                deltas=deltas,
                exposure_summary=self.exposure_summary,
                bin_reports=bin_reports,
                generalization_gap=gen_gap,
                warnings=warnings,
            )
        return reports


class TameScorer:
    """Compute TAME-MT reports from files or in-memory corpora."""

    def __init__(self, config: ScoreConfig | None = None) -> None:
        self.config = config or ScoreConfig()

    def score_files(
        self,
        train_src: str,
        train_tgt: str,
        test_src: str,
        refs: list[str],
        hyp: str,
    ) -> TameReport:
        return self.evaluate_files(train_src, train_tgt, test_src, refs, hyp).report

    def audit_files(
        self,
        train_src: str,
        train_tgt: str | None,
        test_src: str,
        refs: list[str] | None,
    ) -> TameReport:
        return self.evaluate_files(train_src, train_tgt, test_src, refs, hyp=None).report

    def evaluate_files(
        self,
        train_src: str,
        train_tgt: str | None,
        test_src: str,
        refs: list[str] | None = None,
        hyp: str | None = None,
    ) -> EvaluationResult:
        return self.evaluate_corpus(
            train_src=read_lines(train_src),
            train_tgt=read_lines(train_tgt) if train_tgt is not None else None,
            test_src=read_lines(test_src),
            refs=[read_lines(ref) for ref in refs] if refs else None,
            hyp=read_lines(hyp) if hyp is not None else None,
        )

    def score_corpus(
        self,
        train_src: list[str],
        train_tgt: list[str],
        test_src: list[str],
        refs: list[list[str]],
        hyp: list[str],
    ) -> TameReport:
        return self.evaluate_corpus(train_src, train_tgt, test_src, refs, hyp).report

    def score_from_artifacts(
        self,
        exposures: list[SegmentExposure],
        tm_results: list[SegmentTMResult],
        refs: list[list[str]],
        hyp: list[str],
        num_train: int,
    ) -> TameReport:
        return self.prepare_from_artifacts(
            exposures=exposures,
            tm_results=tm_results,
            refs=refs,
            num_train=num_train,
        ).score(hyp)

    def score_many_from_artifacts(
        self,
        exposures: list[SegmentExposure],
        tm_results: list[SegmentTMResult],
        refs: list[list[str]],
        systems: Mapping[str, list[str]],
        num_train: int,
    ) -> dict[str, TameReport]:
        return self.prepare_from_artifacts(
            exposures=exposures,
            tm_results=tm_results,
            refs=refs,
            num_train=num_train,
        ).score_many(systems)

    def prepare_from_artifacts(
        self,
        exposures: list[SegmentExposure],
        tm_results: list[SegmentTMResult],
        refs: list[list[str]],
        num_train: int,
    ) -> CachedSegmentScorer:
        return CachedSegmentScorer(
            config=self.config,
            exposures=exposures,
            tm_results=tm_results,
            refs=refs,
            num_train=num_train,
        )

    def evaluate_index_bundle(
        self,
        bundle: IndexBundle,
        test_src: list[str],
        refs: list[list[str]] | None,
        hyp: list[str] | None = None,
    ) -> EvaluationResult:
        self._validate_bundle_config(bundle)
        validate_corpus_inputs(
            train_src=bundle.train_src,
            train_tgt=bundle.train_tgt,
            test_src=test_src,
            refs=refs,
            hyp=hyp,
        )
        if bundle.train_tgt is None and hyp is not None:
            raise InputDataError(
                "indexed train.tgt is required for full scoring because the TM baseline needs "
                "targets"
            )
        return self._evaluate_validated(
            train_src=bundle.train_src,
            train_tgt=bundle.train_tgt,
            test_src=test_src,
            refs=refs,
            hyp=hyp,
            source_index=bundle.source_index,
            target_index=bundle.target_index,
            exact_pair_keys=bundle.exact_pair_keys,
            index_reused=True,
        )

    def _validate_bundle_config(self, bundle: IndexBundle) -> None:
        if bundle.source_index.norm_config != self.config.normalization:
            raise ConfigurationError("index bundle normalization does not match scorer config")
        if bundle.source_index.sim_config != self.config.similarity:
            raise ConfigurationError("index bundle similarity does not match scorer config")
        if bundle.source_index.index_config != self.config.index:
            raise ConfigurationError("index bundle retrieval settings do not match scorer config")
        if bundle.target_index is not None:
            if bundle.target_index.norm_config != self.config.normalization:
                raise ConfigurationError("target index normalization does not match scorer config")
            if bundle.target_index.sim_config != self.config.similarity:
                raise ConfigurationError("target index similarity does not match scorer config")
            if bundle.target_index.index_config != self.config.index:
                raise ConfigurationError(
                    "target index retrieval settings do not match scorer config"
                )

    def evaluate_corpus(
        self,
        train_src: list[str],
        train_tgt: list[str] | None,
        test_src: list[str],
        refs: list[list[str]] | None,
        hyp: list[str] | None = None,
    ) -> EvaluationResult:
        validate_corpus_inputs(
            train_src=train_src,
            train_tgt=train_tgt,
            test_src=test_src,
            refs=refs,
            hyp=hyp,
        )
        if train_tgt is None and hyp is not None:
            raise InputDataError(
                "train.tgt is required for full scoring because the TM baseline needs targets"
            )
        return self._evaluate_validated(
            train_src=train_src,
            train_tgt=train_tgt,
            test_src=test_src,
            refs=refs,
            hyp=hyp,
            source_index=None,
            target_index=None,
            exact_pair_keys=None,
            index_reused=False,
        )

    def _evaluate_validated(
        self,
        train_src: list[str],
        train_tgt: list[str] | None,
        test_src: list[str],
        refs: list[list[str]] | None,
        hyp: list[str] | None,
        source_index: NgramInvertedIndex | None,
        target_index: NgramInvertedIndex | None,
        exact_pair_keys: set[str] | None,
        index_reused: bool,
    ) -> EvaluationResult:
        exposure_result = compute_exposure_result(
            train_src=train_src,
            train_tgt=train_tgt,
            test_src=test_src,
            refs=refs,
            config=self.config,
            source_index=source_index,
            target_index=target_index,
            exact_pair_keys=exact_pair_keys,
        )
        exposures = exposure_result.segments
        tm_hyp: list[str] = []
        tm_results: list[SegmentTMResult] = []
        if train_tgt is not None:
            tm_hyp, tm_results = build_tm_hypotheses(train_tgt, exposures, self.config)

        scored_bins = score_corpus_and_bins(
            hyp,
            tm_hyp if tm_hyp else None,
            refs,
            exposures,
            self.config,
        )
        system_scores = scored_bins.system_scores
        tm_scores = scored_bins.tm_scores
        deltas: dict[str, float | None] = (
            delta_scores(system_scores, tm_scores, self.config.metrics)
            if hyp is not None
            else _empty_metric_scores(self.config)
        )
        bin_reports = scored_bins.bin_reports
        gen_gap = compute_generalization_gap(bin_reports, self.config.metrics)
        exposure_summary = summarize_exposures(exposures, self.config)
        warnings = generate_warnings(
            exposure=exposure_summary,
            system_scores=system_scores,
            tm_scores=tm_scores,
            bin_reports=bin_reports,
            generalization_gap=gen_gap,
            config=self.config,
            num_test=len(test_src),
        )
        report = TameReport(
            tame_version=__version__,
            signature=build_signature(self.config, backend_name=exposure_result.backend.name),
            num_train=len(train_src),
            num_test=len(test_src),
            num_refs=len(refs) if refs else 0,
            config=config_to_dict(self.config),
            backend={
                "name": exposure_result.backend.name,
                "native": exposure_result.backend.native,
                "exact": exposure_result.backend.exact,
                "requested_mode": exposure_result.backend.requested_mode,
                "resolved_mode": exposure_result.backend.resolved_mode,
                "index_reused": index_reused,
            },
            system_scores=system_scores,
            tm_scores=tm_scores,
            delta_scores=deltas,
            exposure=exposure_summary,
            bins=bin_reports,
            generalization_gap=gen_gap,
            warnings=warnings,
        )
        return EvaluationResult(
            report=report,
            exposures=exposures,
            tm_hyp=tm_hyp,
            tm_results=tm_results,
        )


def score(
    train_src_path: str,
    train_tgt_path: str,
    test_src_path: str,
    ref_paths: list[str],
    hyp_path: str,
    config: ScoreConfig | None = None,
) -> TameReport:
    return TameScorer(config).score_files(
        train_src=train_src_path,
        train_tgt=train_tgt_path,
        test_src=test_src_path,
        refs=ref_paths,
        hyp=hyp_path,
    )


def audit(
    train_src_path: str,
    train_tgt_path: str | None,
    test_src_path: str,
    ref_paths: list[str] | None,
    config: ScoreConfig | None = None,
) -> TameReport:
    return TameScorer(config).audit_files(
        train_src=train_src_path,
        train_tgt=train_tgt_path,
        test_src=test_src_path,
        refs=ref_paths,
    )


def _empty_metric_scores(config: ScoreConfig) -> dict[str, float | None]:
    return {metric: None for metric in config.metrics}


def _build_cached_report(
    *,
    config: ScoreConfig,
    num_train: int,
    num_test: int,
    num_refs: int,
    system_scores: dict[str, float | None],
    tm_scores: dict[str, float | None],
    deltas: dict[str, float | None],
    exposure_summary: ExposureSummary,
    bin_reports: list[BinReport],
    generalization_gap: dict[str, float | None],
    warnings: list[str],
) -> TameReport:
    return TameReport(
        tame_version=__version__,
        signature=build_signature(config, backend_name="cached_segments"),
        num_train=num_train,
        num_test=num_test,
        num_refs=num_refs,
        config=config_to_dict(config),
        backend={
            "name": "cached_segments",
            "native": False,
            "exact": False,
            "requested_mode": config.index.mode,
            "resolved_mode": "cached_segments",
            "index_reused": True,
        },
        system_scores=system_scores,
        tm_scores=tm_scores,
        delta_scores=deltas,
        exposure=exposure_summary,
        bins=bin_reports,
        generalization_gap=generalization_gap,
        warnings=warnings,
    )
