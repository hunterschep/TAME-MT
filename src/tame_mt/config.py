from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from tame_mt.exceptions import ConfigurationError

SUPPORTED_METRICS = ("bleu", "chrf")


@dataclass(frozen=True)
class NormalizationConfig:
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC"
    strip: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False
    strip_diacritics: bool = False
    normalize_punctuation: bool = False


@dataclass(frozen=True)
class SimilarityConfig:
    ngram_orders: tuple[int, ...] = (3, 4, 5)
    similarity: str = "jaccard_set"

    def __post_init__(self) -> None:
        if not self.ngram_orders:
            raise ConfigurationError("ngram_orders must contain at least one order")
        if any(order <= 0 for order in self.ngram_orders):
            raise ConfigurationError("ngram_orders must be positive integers")
        if self.similarity != "jaccard_set":
            raise ConfigurationError("only jaccard_set similarity is supported in v0.1")


@dataclass(frozen=True)
class IndexConfig:
    mode: str = "auto"
    topk: int = 50
    auto_exact_cutoff: int = 5_000
    candidate_gram_limit: int = 8
    posting_limit: int = 500
    max_candidates: int = 3_000
    rerank_limit: int = 1_000

    def __post_init__(self) -> None:
        valid_modes = {
            "auto",
            "inverted_exact",
            "inverted_fast",
            "python_exact",
            "python_fast",
            "native_exact",
            "native_fast",
        }
        if self.mode not in valid_modes:
            raise ConfigurationError(
                "index mode must be one of: auto, inverted_exact, inverted_fast, "
                "python_exact, python_fast, native_exact, native_fast"
            )
        if self.topk <= 0:
            raise ConfigurationError("pair/top-k value must be positive")
        if self.auto_exact_cutoff < 0:
            raise ConfigurationError("auto_exact_cutoff must be non-negative")
        if self.candidate_gram_limit <= 0:
            raise ConfigurationError("candidate_gram_limit must be positive")
        if self.posting_limit <= 0:
            raise ConfigurationError("posting_limit must be positive")
        if self.max_candidates <= 0:
            raise ConfigurationError("max_candidates must be positive")
        if self.rerank_limit <= 0:
            raise ConfigurationError("rerank_limit must be positive")
        if self.rerank_limit > self.max_candidates:
            raise ConfigurationError("rerank_limit must be no larger than max_candidates")


@dataclass(frozen=True)
class BinConfig:
    far_threshold: float = 0.30
    near_threshold: float = 0.70
    leak_thresholds: tuple[float, ...] = (0.70, 0.85, 0.95)
    min_bin_size_warning: int = 30

    def __post_init__(self) -> None:
        _require_finite("far_threshold", self.far_threshold)
        _require_finite("near_threshold", self.near_threshold)
        for threshold in self.leak_thresholds:
            _require_finite("leak_thresholds", threshold)
        if self.far_threshold < 0 or self.near_threshold < 0:
            raise ConfigurationError("bin thresholds must be non-negative")
        if self.far_threshold > self.near_threshold:
            raise ConfigurationError("far_threshold must be no larger than near_threshold")
        if any(threshold < 0 for threshold in self.leak_thresholds):
            raise ConfigurationError("leak thresholds must be non-negative")
        if self.min_bin_size_warning < 0:
            raise ConfigurationError("min_bin_size_warning must be non-negative")


@dataclass(frozen=True)
class TMConfig:
    zero_policy: str = "empty"

    def __post_init__(self) -> None:
        if self.zero_policy not in {"empty", "nearest"}:
            raise ConfigurationError("tm zero_policy must be 'empty' or 'nearest'")


@dataclass(frozen=True)
class MetricConfig:
    bleu_tokenize: str = "13a"
    bleu_lowercase: bool = False
    chrf_word_order: int = 2

    def __post_init__(self) -> None:
        if self.chrf_word_order < 0:
            raise ConfigurationError("chrf_word_order must be non-negative")


@dataclass(frozen=True)
class ScoreConfig:
    metrics: tuple[str, ...] = ("bleu", "chrf")
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    bins: BinConfig = field(default_factory=BinConfig)
    tm: TMConfig = field(default_factory=TMConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)

    def __post_init__(self) -> None:
        normalized_metrics = tuple(metric.lower() for metric in self.metrics)
        if not normalized_metrics:
            raise ConfigurationError("at least one metric must be selected")
        unsupported = sorted(set(normalized_metrics) - set(SUPPORTED_METRICS))
        if unsupported:
            raise ConfigurationError(f"unsupported metrics: {', '.join(unsupported)}")
        object.__setattr__(self, "metrics", normalized_metrics)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    if not value:
        raise ConfigurationError("expected a comma-separated list of floats")
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ConfigurationError(f"invalid float list: {value!r}") from exc
    for item in parsed:
        _require_finite("float list", item)
    return parsed


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value:
        raise ConfigurationError("expected a comma-separated list of integers")
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ConfigurationError(f"invalid integer list: {value!r}") from exc
    if not parsed or any(order <= 0 for order in parsed):
        raise ConfigurationError("ngram orders must be positive integers")
    return parsed


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ConfigurationError(f"{name} must be a finite number")
