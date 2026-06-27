import pytest

from tame_mt.approx_validation import validate_approximate_run
from tame_mt.config import RetrievalConfig, ScoreConfig
from tame_mt.exposure import compute_exposure_result
from tame_mt.schema import SegmentExposure
from tame_mt.tm import build_tm_hypotheses


def test_validate_approximate_run_reports_exact_agreement() -> None:
    pytest.importorskip("tame_mt._native")
    config = ScoreConfig(retrieval=RetrievalConfig(mode="approx", allow_approximate=True))
    train_src = ["god created the heaven", "hello how are you", "we went to market"]
    train_tgt = ["dios creo el cielo", "hola como estas", "fuimos al mercado"]
    test_src = ["god created the heaven", "hello my friend"]
    refs = [["dios creo el cielo", "hola mi amigo"]]
    approx_result = compute_exposure_result(
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        refs=refs,
        config=config,
    )
    _, approx_tm_results = build_tm_hypotheses(train_tgt, approx_result.segments, config)

    validation = validate_approximate_run(
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        refs=refs,
        approx_exposures=approx_result.segments,
        approx_tm_results=approx_tm_results,
        config=config,
        sample_size=10,
        seed=7,
    )

    assert validation.passed is True
    assert validation.payload["sample_size"] == 2
    assert validation.payload["requested_sample_size"] == 10
    assert validation.payload["source_top1_agreement"] == 1.0
    assert validation.payload["target_top1_agreement"] == 1.0
    assert validation.payload["tm_bleu_abs_delta_on_sample"] == 0.0


def test_validate_approximate_run_rejects_missing_exposure_index() -> None:
    pytest.importorskip("tame_mt._native")
    config = ScoreConfig(retrieval=RetrievalConfig(mode="approx", allow_approximate=True))
    segment = SegmentExposure(
        index=1,
        source_exposure=0.0,
        source_nn_index=None,
        source_exact=False,
        target_exposure=None,
        target_nn_index=None,
        target_exact=None,
        pair_exposure=None,
        pair_nn_index=None,
        pair_exact=None,
        bin="far",
    )

    with pytest.raises(ValueError, match="approx exposures is missing index 0"):
        validate_approximate_run(
            train_src=["train"],
            train_tgt=None,
            test_src=["test"],
            refs=None,
            approx_exposures=[segment],
            approx_tm_results=[],
            config=config,
            sample_size=1,
            seed=7,
        )


def test_validate_approximate_run_supports_source_only_validation() -> None:
    pytest.importorskip("tame_mt._native")
    config = ScoreConfig(retrieval=RetrievalConfig(mode="approx", allow_approximate=True))
    train_src = ["alpha beta", "gamma delta"]
    test_src = ["alpha beta", "epsilon zeta"]
    approx_result = compute_exposure_result(
        train_src=train_src,
        train_tgt=None,
        test_src=test_src,
        refs=None,
        config=config,
    )

    validation = validate_approximate_run(
        train_src=train_src,
        train_tgt=None,
        test_src=test_src,
        refs=None,
        approx_exposures=approx_result.segments,
        approx_tm_results=[],
        config=config,
        sample_size=2,
        seed=7,
    )

    assert validation.passed is True
    assert validation.payload["target_top1_agreement"] is None
    assert validation.payload["tm_bleu_abs_delta_on_sample"] is None
