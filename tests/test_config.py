import pytest

from tame_mt import BinConfig, IndexConfig, ScoreConfig, SimilarityConfig, TMConfig
from tame_mt.exceptions import ConfigurationError


def test_config_rejects_invalid_metric() -> None:
    with pytest.raises(ConfigurationError, match="unsupported metrics"):
        ScoreConfig(metrics=("bleu", "meteor"))


def test_config_normalizes_metric_case() -> None:
    assert ScoreConfig(metrics=("BLEU", "chrF")).metrics == ("bleu", "chrf")


def test_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ConfigurationError, match="far_threshold"):
        BinConfig(far_threshold=0.8, near_threshold=0.7)


def test_config_rejects_invalid_index_and_tm_options() -> None:
    with pytest.raises(ConfigurationError, match="positive"):
        IndexConfig(topk=0)
    with pytest.raises(ConfigurationError, match="rerank_limit"):
        IndexConfig(max_candidates=100, rerank_limit=101)
    with pytest.raises(ConfigurationError, match="zero_policy"):
        TMConfig(zero_policy="random")
    with pytest.raises(ConfigurationError, match="positive"):
        SimilarityConfig(ngram_orders=(0,))
