from tame_mt.config import ScoreConfig
from tame_mt.schema import SegmentExposure
from tame_mt.tm import build_tm_hypotheses


def test_tm_uses_source_neighbor_target() -> None:
    exposure = SegmentExposure(0, 0.8, 1, False, None, None, None, None, None, None, "near")
    hyps, results = build_tm_hypotheses(["a", "b"], [exposure], ScoreConfig())
    assert hyps == ["b"]
    assert results[0].tm_source_index == 1


def test_tm_zero_policy_empty() -> None:
    exposure = SegmentExposure(0, 0.0, None, False, None, None, None, None, None, None, "far")
    hyps, results = build_tm_hypotheses(["a"], [exposure], ScoreConfig())
    assert hyps == [""]
    assert results[0].tm_source_index is None
