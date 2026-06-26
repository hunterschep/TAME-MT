from tame_mt.config import ScoreConfig
from tame_mt.exposure import compute_exposure


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
