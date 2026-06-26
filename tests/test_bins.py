from tame_mt.bins import assign_bin
from tame_mt.config import ScoreConfig
from tame_mt.schema import SegmentExposure


def _segment(source_exposure: float, source_exact: bool = False) -> SegmentExposure:
    return SegmentExposure(
        index=0,
        source_exposure=source_exposure,
        source_nn_index=None,
        source_exact=source_exact,
        target_exposure=None,
        target_nn_index=None,
        target_exact=None,
        pair_exposure=None,
        pair_nn_index=None,
        pair_exact=None,
        bin="",
    )


def test_assign_bin_defaults() -> None:
    config = ScoreConfig().bins
    assert assign_bin(_segment(1.0, source_exact=True), config) == "source_exact"
    assert assign_bin(_segment(0.80), config) == "near"
    assert assign_bin(_segment(0.50), config) == "medium"
    assert assign_bin(_segment(0.10), config) == "far"
