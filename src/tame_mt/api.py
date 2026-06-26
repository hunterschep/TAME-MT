from __future__ import annotations

from dataclasses import dataclass

from tame_mt.bins import compute_generalization_gap, score_bins
from tame_mt.config import ScoreConfig
from tame_mt.exposure import compute_exposure, summarize_exposures
from tame_mt.io import read_lines, validate_parallel_lengths
from tame_mt.report import build_signature, config_to_dict
from tame_mt.schema import SegmentExposure, SegmentTMResult, TameReport
from tame_mt.scoring import delta_scores, score_metrics
from tame_mt.tm import build_tm_hypotheses
from tame_mt.version import __version__
from tame_mt.warnings import generate_warnings


@dataclass
class EvaluationResult:
    report: TameReport
    exposures: list[SegmentExposure]
    tm_hyp: list[str]
    tm_results: list[SegmentTMResult]


class TameScorer:
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

    def evaluate_corpus(
        self,
        train_src: list[str],
        train_tgt: list[str] | None,
        test_src: list[str],
        refs: list[list[str]] | None,
        hyp: list[str] | None = None,
    ) -> EvaluationResult:
        validate_parallel_lengths(train_src=train_src, train_tgt=train_tgt, test_src=test_src, refs=refs, hyp=hyp)
        if hyp is not None and refs is None:
            raise ValueError("refs are required when hyp is provided")

        exposures = compute_exposure(train_src, train_tgt, test_src, refs, self.config)
        tm_hyp: list[str] = []
        tm_results: list[SegmentTMResult] = []
        if train_tgt is not None:
            tm_hyp, tm_results = build_tm_hypotheses(train_tgt, exposures, self.config)

        system_scores = (
            score_metrics(hyp, refs, self.config)
            if hyp is not None and refs is not None
            else {metric: None for metric in self.config.metrics}
        )
        tm_scores = (
            score_metrics(tm_hyp, refs, self.config)
            if tm_hyp and refs is not None
            else {metric: None for metric in self.config.metrics}
        )
        deltas = (
            delta_scores(system_scores, tm_scores, self.config.metrics)
            if hyp is not None
            else {metric: None for metric in self.config.metrics}
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
            signature=build_signature(self.config),
            num_train=len(train_src),
            num_test=len(test_src),
            num_refs=len(refs) if refs else 0,
            config=config_to_dict(self.config),
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
