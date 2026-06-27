from tame_mt.config import IndexConfig, ScoreConfig
from tame_mt.exposure import compute_exposure, summarize_exposures
from tame_mt.schema import SegmentExposure


def test_normalized_exact_overlap_ignores_whitespace() -> None:
    exposures = compute_exposure(
        train_src=["hello world"],
        train_tgt=["hola mundo"],
        test_src=[" hello   world "],
        refs=[["hola mundo"]],
        config=ScoreConfig(),
    )
    assert exposures[0].source_exact is True
    assert exposures[0].source_exposure == 1.0


def test_pair_exposure_requires_same_pair_similarity() -> None:
    exposures = compute_exposure(
        train_src=["abcde", "vwxyz"],
        train_tgt=["klmno", "pqrst"],
        test_src=["abcde", "11111"],
        refs=[["pqrst", "klmno"]],
        config=ScoreConfig(),
    )
    assert exposures[0].source_exposure == 1.0
    assert exposures[0].target_exposure == 1.0
    assert exposures[0].pair_exposure < 1.0
    assert exposures[1].source_exposure == 0.0
    assert exposures[1].target_exposure == 1.0
    assert exposures[1].pair_exposure == 0.0


def test_exact_pair_overlap_is_exact() -> None:
    exposures = compute_exposure(
        train_src=["a b c"],
        train_tgt=["x y z"],
        test_src=["a b c"],
        refs=[["x y z"]],
        config=ScoreConfig(),
    )
    assert exposures[0].pair_exact is True
    assert exposures[0].pair_exposure == 1.0


def test_pair_fields_are_absent_without_references() -> None:
    exposures = compute_exposure(
        train_src=["a b c"],
        train_tgt=["x y z"],
        test_src=["a b c"],
        refs=None,
        config=ScoreConfig(),
    )
    assert exposures[0].pair_exact is None
    assert exposures[0].pair_exposure is None


def test_multi_ref_exposure_records_best_reference_indices() -> None:
    exposures = compute_exposure(
        train_src=["shared source"],
        train_tgt=["matching target"],
        test_src=["shared source"],
        refs=[["unrelated target"], ["matching target"]],
        config=ScoreConfig(),
    )

    assert exposures[0].target_exposure == 1.0
    assert exposures[0].target_ref_index == 1
    assert exposures[0].pair_exposure == 1.0
    assert exposures[0].pair_ref_index == 1


def test_chunked_exposure_matches_large_batch_results() -> None:
    train_src = [
        "shared source",
        "near source alpha",
        "near source beta",
        "unrelated source",
    ]
    train_tgt = [
        "matching target",
        "alpha target",
        "beta target",
        "unrelated target",
    ]
    test_src = [
        "shared source",
        "near source alpah",
        "near source bet",
        "new source",
        "unrelated source",
    ]
    refs = [
        [
            "unrelated target",
            "alpha target",
            "beta target",
            "new target",
            "unrelated target",
        ],
        [
            "matching target",
            "other alpha",
            "other beta",
            "other new",
            "other unrelated",
        ],
    ]

    one_at_a_time = compute_exposure(
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        refs=refs,
        config=ScoreConfig(index=IndexConfig(batch_size=1)),
    )
    large_batch = compute_exposure(
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        refs=refs,
        config=ScoreConfig(index=IndexConfig(batch_size=100)),
    )

    assert one_at_a_time == large_batch


def test_exposure_summary_uses_stable_percentile_and_threshold_semantics() -> None:
    exposures = [
        _segment(0, 0.0, False),
        _segment(1, 0.25, False),
        _segment(2, 0.50, False),
        _segment(3, 0.75, True),
        _segment(4, 1.0, True),
    ]

    summary = summarize_exposures(exposures, ScoreConfig())

    assert summary.source["mean"] == 0.5
    assert summary.source["median"] == 0.5
    assert summary.source["p05"] == 0.05
    assert summary.source["p25"] == 0.25
    assert summary.source["p75"] == 0.75
    assert summary.source["p95"] == 0.95
    assert summary.source["max"] == 1.0
    assert summary.source["exact_overlap"] == 0.4
    assert summary.source["at_threshold"]["0.70"] == 0.4
    assert summary.source["at_threshold"]["0.85"] == 0.2
    assert summary.source["at_threshold"]["0.95"] == 0.2


def _segment(index: int, source_exposure: float, source_exact: bool) -> SegmentExposure:
    return SegmentExposure(
        index=index,
        source_exposure=source_exposure,
        source_nn_index=index if source_exact else None,
        source_exact=source_exact,
        target_exposure=None,
        target_nn_index=None,
        target_exact=None,
        pair_exposure=None,
        pair_nn_index=None,
        pair_exact=None,
        bin="source_exact" if source_exact else "far",
    )
