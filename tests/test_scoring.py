import pytest

from tame_mt.config import ScoreConfig
from tame_mt.metrics import sacre
from tame_mt.scoring import score_metrics, score_metrics_by_groups, score_systems_by_groups


def test_sacrebleu_metrics_return_scores() -> None:
    scores = score_metrics(
        ["hello world", "good day"],
        [["hello world", "good day"]],
        ScoreConfig(),
    )
    assert scores["bleu"] is not None
    assert scores["chrf"] is not None


def test_grouped_scores_match_subset_scores() -> None:
    config = ScoreConfig()
    hyps = ["hello world", "good day", "fresh sentence"]
    refs = [["hello world", "good morning", "fresh sentence"]]
    groups = {"all": [0, 1, 2], "first": [0], "rest": [1, 2], "empty": []}

    grouped = score_metrics_by_groups(hyps, refs, groups, config)

    assert grouped["all"] == score_metrics(hyps, refs, config)
    assert grouped["first"] == score_metrics([hyps[0]], [[refs[0][0]]], config)
    assert grouped["rest"] == score_metrics(hyps[1:], [refs[0][1:]], config)
    assert grouped["empty"] == {"bleu": None, "chrf": None}


def test_multi_system_grouped_scores_reuse_same_semantics() -> None:
    config = ScoreConfig()
    refs = [["hello world", "good morning", "fresh sentence"]]
    systems = {
        "system": ["hello world", "good day", "fresh sentence"],
        "tm": ["hello", "good morning", "fresh"],
        "missing": None,
    }
    groups = {"all": [0, 1, 2], "tail": [1, 2]}

    grouped = score_systems_by_groups(systems, refs, groups, config)

    assert grouped["system"]["all"] == score_metrics(systems["system"], refs, config)
    assert grouped["tm"]["tail"] == score_metrics(
        ["good morning", "fresh"],
        [["good morning", "fresh sentence"]],
        config,
    )
    assert grouped["missing"]["all"] == {"bleu": None, "chrf": None}


def test_supported_sacrebleu_version_exposes_segment_stats_acceleration() -> None:
    scorer = sacre.PreparedSacreMetricGroupScorer(
        "bleu",
        refs=[["hello world", "good morning"]],
        groups={"all": [0, 1]},
        config=ScoreConfig().metric,
    )

    assert scorer._use_segment_stats is True


def test_grouped_scores_fall_back_when_sacrebleu_stats_api_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ScoreConfig()
    hyps = ["hello world", "good day", "fresh sentence"]
    refs = [["hello world", "good morning", "fresh sentence"]]
    groups = {"all": [0, 1, 2], "tail": [1, 2], "empty": []}

    monkeypatch.setattr(sacre, "_build_sacre_metric", lambda *args, **kwargs: object())

    grouped = score_metrics_by_groups(hyps, refs, groups, config)

    assert grouped["all"] == score_metrics(hyps, refs, config)
    assert grouped["tail"] == score_metrics(hyps[1:], [refs[0][1:]], config)
    assert grouped["empty"] == {"bleu": None, "chrf": None}


def test_grouped_scores_fall_back_when_sacrebleu_stats_api_drifts_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ScoreConfig(metrics=("bleu",))
    hyps = ["hello world", "good day", "fresh sentence"]
    refs = [["hello world", "good morning", "fresh sentence"]]
    groups = {"all": [0, 1, 2], "tail": [1, 2]}
    real_builder = sacre._build_sacre_metric

    class FlakyMetric:
        def __init__(self) -> None:
            self._real = real_builder("bleu", config.metric, refs)
            self._extract_calls = 0

        def _extract_corpus_statistics(self, *args: object) -> object:
            self._extract_calls += 1
            if self._extract_calls > 2:
                raise RuntimeError("simulated SacreBLEU stats API drift")
            return self._real._extract_corpus_statistics(*args)

        def _compute_score_from_stats(self, *args: object) -> object:
            return self._real._compute_score_from_stats(*args)

    monkeypatch.setattr(sacre, "_build_sacre_metric", lambda *args, **kwargs: FlakyMetric())

    grouped = score_metrics_by_groups(hyps, refs, groups, config)

    assert grouped["all"] == score_metrics(hyps, refs, config)
    assert grouped["tail"] == score_metrics(hyps[1:], [refs[0][1:]], config)
