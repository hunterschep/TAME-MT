import pytest

from tame_mt import BinConfig, IndexConfig, ScoreConfig, SimilarityConfig, TMConfig
from tame_mt.config import MetricConfig, parse_float_tuple, parse_int_tuple
from tame_mt.exceptions import ConfigurationError


def test_config_rejects_invalid_metric() -> None:
    with pytest.raises(ConfigurationError, match="unsupported metrics"):
        ScoreConfig(metrics=("bleu", "meteor"))


def test_config_normalizes_metric_case() -> None:
    assert ScoreConfig(metrics=("BLEU", "chrF")).metrics == ("bleu", "chrf")


def test_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ConfigurationError, match="far_threshold"):
        BinConfig(far_threshold=0.8, near_threshold=0.7)
    with pytest.raises(ConfigurationError, match="finite"):
        BinConfig(far_threshold=float("nan"))
    with pytest.raises(ConfigurationError, match="finite"):
        BinConfig(near_threshold=float("inf"))
    with pytest.raises(ConfigurationError, match="finite"):
        BinConfig(leak_thresholds=(0.7, float("-inf")))
    with pytest.raises(ConfigurationError, match="far_threshold"):
        BinConfig(far_threshold=1.01, near_threshold=1.0)
    with pytest.raises(ConfigurationError, match="near_threshold"):
        BinConfig(near_threshold=1.01)
    with pytest.raises(ConfigurationError, match="leak_thresholds"):
        BinConfig(leak_thresholds=(0.7, 1.01))
    with pytest.raises(ConfigurationError, match="leak_thresholds"):
        BinConfig(leak_thresholds=())
    with pytest.raises(ConfigurationError, match="far_threshold"):
        BinConfig(far_threshold=True, near_threshold=1.0)
    with pytest.raises(ConfigurationError, match="far_threshold"):
        BinConfig(far_threshold="0.5")  # type: ignore[arg-type]


def test_parse_float_tuple_rejects_non_finite_values() -> None:
    with pytest.raises(ConfigurationError, match="finite"):
        parse_float_tuple("0.70,nan")


def test_tuple_parsers_reject_empty_components() -> None:
    with pytest.raises(ConfigurationError, match="comma-separated list of floats"):
        parse_float_tuple(",")
    with pytest.raises(ConfigurationError, match="comma-separated list of floats"):
        parse_float_tuple("0.70,,0.85")
    with pytest.raises(ConfigurationError, match="comma-separated list of integers"):
        parse_int_tuple("3,,5")


def test_config_rejects_invalid_index_and_tm_options() -> None:
    with pytest.raises(ConfigurationError, match="positive"):
        IndexConfig(topk=0)
    with pytest.raises(ConfigurationError, match="topk"):
        IndexConfig(topk=True)
    with pytest.raises(ConfigurationError, match="topk"):
        IndexConfig(topk=1.5)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="auto_exact_cutoff"):
        IndexConfig(auto_exact_cutoff=False)
    with pytest.raises(ConfigurationError, match="candidate_gram_limit"):
        IndexConfig(candidate_gram_limit=1.5)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="rerank_limit"):
        IndexConfig(max_candidates=100, rerank_limit=101)
    with pytest.raises(ConfigurationError, match="zero_policy"):
        TMConfig(zero_policy="random")
    with pytest.raises(ConfigurationError, match="positive"):
        SimilarityConfig(ngram_orders=(0,))
    with pytest.raises(ConfigurationError, match="ngram_orders"):
        SimilarityConfig(ngram_orders=(True,))
    with pytest.raises(ConfigurationError, match="ngram_orders"):
        SimilarityConfig(ngram_orders=(3.5,))  # type: ignore[arg-type]


def test_config_rejects_invalid_integer_typed_options() -> None:
    with pytest.raises(ConfigurationError, match="min_bin_size_warning"):
        BinConfig(min_bin_size_warning=False)
    with pytest.raises(ConfigurationError, match="min_bin_size_warning"):
        BinConfig(min_bin_size_warning=1.5)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="chrf_word_order"):
        MetricConfig(chrf_word_order=False)
    with pytest.raises(ConfigurationError, match="chrf_word_order"):
        MetricConfig(chrf_word_order=1.5)  # type: ignore[arg-type]
