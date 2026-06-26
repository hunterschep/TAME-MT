from tame_mt.config import ScoreConfig
from tame_mt.scoring import score_metrics


def test_sacrebleu_metrics_return_scores() -> None:
    scores = score_metrics(
        ["hello world", "good day"],
        [["hello world", "good day"]],
        ScoreConfig(),
    )
    assert scores["bleu"] is not None
    assert scores["chrf"] is not None
