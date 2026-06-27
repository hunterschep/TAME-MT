from pathlib import Path

import pytest

import tame_mt
from tame_mt import (
    BinConfig,
    CachedSegmentScorer,
    MetricConfig,
    ScoreConfig,
    SegmentExposure,
    SegmentTMResult,
    TameScorer,
    read_segment_jsonl,
    read_segment_metadata,
    segment_metadata_path,
    validate_segment_metadata,
)
from tame_mt.exceptions import InputDataError

FIXTURES = Path(__file__).parent / "fixtures"


def test_public_api_exports_cached_scoring_types() -> None:
    assert CachedSegmentScorer.__name__ == "CachedSegmentScorer"
    assert MetricConfig.__name__ == "MetricConfig"
    assert SegmentExposure.__name__ == "SegmentExposure"
    assert SegmentTMResult.__name__ == "SegmentTMResult"
    assert read_segment_jsonl.__name__ == "read_segment_jsonl"
    assert read_segment_metadata.__name__ == "read_segment_metadata"
    assert segment_metadata_path.__name__ == "segment_metadata_path"
    assert validate_segment_metadata.__name__ == "validate_segment_metadata"
    for name in (
        "CachedSegmentScorer",
        "MetricConfig",
        "SegmentExposure",
        "SegmentTMResult",
        "read_segment_jsonl",
        "read_segment_metadata",
        "segment_metadata_path",
        "validate_segment_metadata",
    ):
        assert name in tame_mt.__all__


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


def test_score_many_from_artifacts_matches_single_system_reports() -> None:
    exposures = [
        _segment(0, 1.0, "source_exact"),
        _segment(1, 0.1, "far"),
    ]
    tm_results = [
        SegmentTMResult(index=0, tm_hyp="good", tm_source_index=0, tm_source_similarity=1.0),
        SegmentTMResult(index=1, tm_hyp="bad", tm_source_index=1, tm_source_similarity=0.1),
    ]
    refs = [["good", "bad"]]
    systems = {
        "baseline": ["good", "bad"],
        "variant": ["good", "different"],
    }

    scorer = TameScorer()
    batch_reports = scorer.score_many_from_artifacts(
        exposures=exposures,
        tm_results=tm_results,
        refs=refs,
        systems=systems,
        num_train=2,
    )
    single_report = scorer.score_from_artifacts(
        exposures=exposures,
        tm_results=tm_results,
        refs=refs,
        hyp=systems["baseline"],
        num_train=2,
    )

    assert set(batch_reports) == {"baseline", "variant"}
    assert batch_reports["baseline"].to_dict() == single_report.to_dict()
    assert (
        batch_reports["variant"].system_scores["chrf"]
        != batch_reports["baseline"].system_scores["chrf"]
    )


def test_score_from_artifacts_records_artifact_backend_provenance() -> None:
    report = TameScorer().score_from_artifacts(
        exposures=[_segment(0, 1.0, "source_exact")],
        tm_results=[
            SegmentTMResult(index=0, tm_hyp="good", tm_source_index=0, tm_source_similarity=1.0)
        ],
        refs=[["good"]],
        hyp=["good"],
        num_train=1,
        artifact_backend={
            "name": "native_exact",
            "native": True,
            "exact": True,
        },
    )

    assert report.backend["name"] == "cached_segments"
    assert report.backend["artifact_backend"] == {
        "name": "native_exact",
        "native": True,
        "exact": True,
    }


def test_prepare_from_artifacts_reuses_cached_setup_for_later_systems() -> None:
    exposures = [
        _segment(0, 1.0, "source_exact"),
        _segment(1, 0.1, "far"),
    ]
    tm_results = [
        SegmentTMResult(index=0, tm_hyp="good", tm_source_index=0, tm_source_similarity=1.0),
        SegmentTMResult(index=1, tm_hyp="bad", tm_source_index=1, tm_source_similarity=0.1),
    ]
    refs = [["good", "bad"]]
    scorer = TameScorer()

    cached = scorer.prepare_from_artifacts(
        exposures=exposures,
        tm_results=tm_results,
        refs=refs,
        num_train=2,
    )
    direct_report = scorer.score_from_artifacts(
        exposures=exposures,
        tm_results=tm_results,
        refs=[["good", "bad"]],
        hyp=["good", "bad"],
        num_train=2,
    )
    refs[0][0] = "mutated after prepare"
    exposures[0].source_exposure = 0.0
    tm_results[0].tm_hyp = "mutated after prepare"

    prepared_report = cached.score(["good", "bad"])
    batch_reports = cached.score_many(
        {
            "baseline": ["good", "bad"],
            "variant": ["good", "different"],
        }
    )

    assert prepared_report.to_dict() == direct_report.to_dict()
    assert batch_reports["baseline"].to_dict() == direct_report.to_dict()
    assert batch_reports["variant"].system_scores["chrf"] != direct_report.system_scores["chrf"]


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


def test_score_from_artifacts_rejects_bin_threshold_mismatch() -> None:
    scorer = TameScorer(
        ScoreConfig(
            bins=BinConfig(
                far_threshold=0.40,
                near_threshold=0.70,
            )
        )
    )

    with pytest.raises(InputDataError, match="cached segment bin mismatch"):
        scorer.score_from_artifacts(
            exposures=[_segment(0, 0.35, "medium")],
            tm_results=[
                SegmentTMResult(index=0, tm_hyp="bad", tm_source_index=0, tm_source_similarity=0.35)
            ],
            refs=[["bad"]],
            hyp=["bad"],
            num_train=1,
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
