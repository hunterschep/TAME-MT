from __future__ import annotations

from dataclasses import dataclass

from tame_mt.bins import compute_generalization_gap, score_bins
from tame_mt.config import ScoreConfig
from tame_mt.exceptions import ConfigurationError, InputDataError
from tame_mt.exposure import compute_exposure_result, summarize_exposures
from tame_mt.index import NgramInvertedIndex
from tame_mt.io import read_lines, validate_corpus_inputs, validate_equal_lengths
from tame_mt.persistence import IndexBundle
from tame_mt.report import build_signature, config_to_dict
from tame_mt.schema import SegmentExposure, SegmentTMResult, TameReport
from tame_mt.scoring import delta_scores, score_metrics
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
        if not exposures:
            raise InputDataError("segment artifact must contain at least one segment")
        tm_hyp = [result.tm_hyp for result in tm_results]
        validate_equal_lengths("segments", exposures, "tm_results", tm_results)
        for ref_idx, ref in enumerate(refs):
            validate_equal_lengths("segments", exposures, f"ref[{ref_idx}]", ref)
        validate_equal_lengths("segments", exposures, "hyp", hyp)
        validate_equal_lengths("segments", exposures, "tm_hyp", tm_hyp)

        system_scores: dict[str, float | None] = score_metrics(hyp, refs, self.config)
        tm_scores: dict[str, float | None] = score_metrics(tm_hyp, refs, self.config)
        deltas = delta_scores(system_scores, tm_scores, self.config.metrics)
        bin_reports = score_bins(hyp, tm_hyp, refs, exposures, self.config)
        gen_gap = compute_generalization_gap(bin_reports, self.config.metrics)
        exposure_summary = summarize_exposures(exposures, self.config)
        warnings = generate_warnings(
            exposure=exposure_summary,
            system_scores=system_scores,
            tm_scores=tm_scores,
            bin_reports=bin_reports,
            generalization_gap=gen_gap,
            config=self.config,
            num_test=len(exposures),
        )
        return TameReport(
            tame_version=__version__,
            signature=build_signature(self.config, backend_name="cached_segments"),
            num_train=num_train,
            num_test=len(exposures),
            num_refs=len(refs),
            config=config_to_dict(self.config),
            backend={
                "name": "cached_segments",
                "native": False,
                "exact": False,
                "requested_mode": self.config.index.mode,
                "resolved_mode": "cached_segments",
                "index_reused": True,
            },
            system_scores=system_scores,
            tm_scores=tm_scores,
            delta_scores=deltas,
            exposure=exposure_summary,
            bins=bin_reports,
            generalization_gap=gen_gap,
            warnings=warnings,
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

        system_scores: dict[str, float | None] = (
            score_metrics(hyp, refs, self.config)
            if hyp is not None and refs is not None
            else _empty_metric_scores(self.config)
        )
        tm_scores: dict[str, float | None] = (
            score_metrics(tm_hyp, refs, self.config)
            if tm_hyp and refs is not None
            else _empty_metric_scores(self.config)
        )
        deltas: dict[str, float | None] = (
            delta_scores(system_scores, tm_scores, self.config.metrics)
            if hyp is not None
            else _empty_metric_scores(self.config)
        )
        bin_reports = score_bins(hyp, tm_hyp if tm_hyp else None, refs, exposures, self.config)
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
