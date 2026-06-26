from pathlib import Path

import pytest

from tame_mt import TameScorer
from tame_mt.exceptions import InputDataError
from tame_mt.schema import SegmentExposure, SegmentTMResult

FIXTURES = Path(__file__).parent / "fixtures"


def test_score_files_produces_report() -> None:
    report = TameScorer().score_files(
        train_src=str(FIXTURES / "train.src"),
        train_tgt=str(FIXTURES / "train.tgt"),
        test_src=str(FIXTURES / "test.src"),
        refs=[str(FIXTURES / "test.ref")],
        hyp=str(FIXTURES / "hyp.out"),
    )
    assert report.num_train == 4
    assert report.num_test == 4
    assert report.exposure.source["max"] == 1.0
    assert report.tm_scores["bleu"] is not None
    assert report.signature.startswith("tame-mt|v:0.1.0|")


def test_audit_without_system_scores_does_not_warn_about_gengap() -> None:
    report = TameScorer().audit_files(
        train_src=str(FIXTURES / "train.src"),
        train_tgt=str(FIXTURES / "train.tgt"),
        test_src=str(FIXTURES / "test.src"),
        refs=[str(FIXTURES / "test.ref")],
    )
    assert not any("GenGap cannot be computed" in warning for warning in report.warnings)


def test_score_from_artifacts_sorts_reordered_segments() -> None:
    exposures = [
        _segment(1, 0.1, "far"),
        _segment(0, 1.0, "source_exact"),
    ]
    tm_results = [
        SegmentTMResult(index=1, tm_hyp="bad", tm_source_index=1, tm_source_similarity=0.1),
        SegmentTMResult(index=0, tm_hyp="good", tm_source_index=0, tm_source_similarity=1.0),
    ]

    report = TameScorer().score_from_artifacts(
        exposures=exposures,
        tm_results=tm_results,
        refs=[["good", "bad"]],
        hyp=["good", "bad"],
        num_train=2,
    )

    assert report.bins[0].name == "source_exact"
    assert report.bins[0].mean_source_exposure == 1.0
    assert report.bins[-1].name == "far"
    assert report.bins[-1].mean_source_exposure == 0.1


def test_score_from_artifacts_rejects_missing_segment_index() -> None:
    exposures = [_segment(1, 0.1, "far")]
    tm_results = [
        SegmentTMResult(index=1, tm_hyp="bad", tm_source_index=1, tm_source_similarity=0.1)
    ]

    with pytest.raises(InputDataError, match="contiguous"):
        TameScorer().score_from_artifacts(
            exposures=exposures,
            tm_results=tm_results,
            refs=[["bad"]],
            hyp=["bad"],
            num_train=2,
        )


def test_score_from_artifacts_rejects_empty_refs() -> None:
    with pytest.raises(InputDataError, match="at least one reference"):
        TameScorer().score_from_artifacts(
            exposures=[_segment(0, 0.1, "far")],
            tm_results=[
                SegmentTMResult(index=0, tm_hyp="bad", tm_source_index=0, tm_source_similarity=0.1)
            ],
            refs=[],
            hyp=["bad"],
            num_train=1,
        )


def test_score_from_artifacts_rejects_non_positive_num_train() -> None:
    with pytest.raises(InputDataError, match="num_train must be positive"):
        TameScorer().score_from_artifacts(
            exposures=[_segment(0, 0.1, "far")],
            tm_results=[
                SegmentTMResult(index=0, tm_hyp="bad", tm_source_index=0, tm_source_similarity=0.1)
            ],
            refs=[["bad"]],
            hyp=["bad"],
            num_train=0,
        )


def _segment(index: int, exposure: float, bin_name: str) -> SegmentExposure:
    return SegmentExposure(
        index=index,
        source_exposure=exposure,
        source_nn_index=index,
        source_exact=bin_name == "source_exact",
        target_exposure=None,
        target_nn_index=None,
        target_exact=None,
        pair_exposure=None,
        pair_nn_index=None,
        pair_exact=None,
        bin=bin_name,
    )
