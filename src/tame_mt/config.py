from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Literal

from tame_mt.exceptions import ConfigurationError

SUPPORTED_METRICS = ("bleu", "chrf")


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC"
    strip: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False
    strip_diacritics: bool = False
    normalize_punctuation: bool = False

    def __post_init__(self) -> None:
        if self.unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ConfigurationError("unicode_form must be one of: NFC, NFD, NFKC, NFKD")
        _require_bool("strip", self.strip)
        _require_bool("collapse_whitespace", self.collapse_whitespace)
        _require_bool("lowercase", self.lowercase)
        _require_bool("strip_diacritics", self.strip_diacritics)
        _require_bool("normalize_punctuation", self.normalize_punctuation)


@dataclass(frozen=True, slots=True)
class SimilarityConfig:
    ngram_orders: tuple[int, ...] = (3, 4, 5)
    similarity: str = "jaccard_set"

    def __post_init__(self) -> None:
        if not isinstance(self.ngram_orders, tuple):
            raise ConfigurationError("ngram_orders must be a tuple of positive integers")
        if not self.ngram_orders:
            raise ConfigurationError("ngram_orders must contain at least one order")
        for order in self.ngram_orders:
            _require_positive_int("ngram_orders", order)
        if self.similarity != "jaccard_set":
            raise ConfigurationError("only jaccard_set similarity is supported in v0.1")


@dataclass(frozen=True, slots=True)
class IndexConfig:
    mode: str = "auto"
    topk: int = 50
    batch_size: int = 8_192
    auto_exact_cutoff: int = 5_000
    candidate_gram_limit: int = 8
    posting_limit: int = 500
    max_candidates: int = 3_000
    rerank_limit: int = 1_000

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise ConfigurationError("index mode must be a string")
        valid_modes = {
            "auto",
            "native_exact",
            "native_fast",
        }
        if self.mode not in valid_modes:
            raise ConfigurationError("index mode must be one of: auto, native_exact, native_fast")
        _require_positive_int("topk", self.topk)
        _require_positive_int("batch_size", self.batch_size)
        _require_non_negative_int("auto_exact_cutoff", self.auto_exact_cutoff)
        _require_positive_int("candidate_gram_limit", self.candidate_gram_limit)
        _require_positive_int("posting_limit", self.posting_limit)
        _require_positive_int("max_candidates", self.max_candidates)
        _require_positive_int("rerank_limit", self.rerank_limit)
        if self.rerank_limit > self.max_candidates:
            raise ConfigurationError("rerank_limit must be no larger than max_candidates")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    mode: str = "exact"
    allow_approximate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise ConfigurationError("retrieval mode must be a string")
        valid_modes = {"exact", "guarded", "approx"}
        if self.mode not in valid_modes:
            raise ConfigurationError("retrieval mode must be one of: exact, guarded, approx")
        _require_bool("allow_approximate", self.allow_approximate)
        if self.mode == "approx" and not self.allow_approximate:
            raise ConfigurationError(
                "approximate retrieval must be explicitly enabled with allow_approximate=True"
            )


@dataclass(frozen=True, slots=True)
class BinConfig:
    far_threshold: float = 0.30
    near_threshold: float = 0.70
    leak_thresholds: tuple[float, ...] = (0.70, 0.85, 0.95)
    min_bin_size_warning: int = 30

    def __post_init__(self) -> None:
        _require_unit_interval("far_threshold", self.far_threshold)
        _require_unit_interval("near_threshold", self.near_threshold)
        if not isinstance(self.leak_thresholds, tuple):
            raise ConfigurationError("leak_thresholds must be a tuple of thresholds")
        if not self.leak_thresholds:
            raise ConfigurationError("leak_thresholds must contain at least one threshold")
        for threshold in self.leak_thresholds:
            _require_unit_interval("leak_thresholds", threshold)
        if self.far_threshold > self.near_threshold:
            raise ConfigurationError("far_threshold must be no larger than near_threshold")
        _require_non_negative_int("min_bin_size_warning", self.min_bin_size_warning)


@dataclass(frozen=True, slots=True)
class TMConfig:
    zero_policy: str = "empty"

    def __post_init__(self) -> None:
        if not isinstance(self.zero_policy, str):
            raise ConfigurationError("tm zero_policy must be 'empty' or 'nearest'")
        if self.zero_policy not in {"empty", "nearest"}:
            raise ConfigurationError("tm zero_policy must be 'empty' or 'nearest'")


@dataclass(frozen=True, slots=True)
class MetricConfig:
    bleu_tokenize: str = "13a"
    bleu_lowercase: bool = False
    chrf_word_order: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.bleu_tokenize, str):
            raise ConfigurationError("bleu_tokenize must be a string")
        _require_bool("bleu_lowercase", self.bleu_lowercase)
        _require_non_negative_int("chrf_word_order", self.chrf_word_order)


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    metrics: tuple[str, ...] = ("bleu", "chrf")
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    bins: BinConfig = field(default_factory=BinConfig)
    tm: TMConfig = field(default_factory=TMConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)

    def __post_init__(self) -> None:
        _require_config_type("normalization", self.normalization, NormalizationConfig)
        _require_config_type("similarity", self.similarity, SimilarityConfig)
        _require_config_type("index", self.index, IndexConfig)
        _require_config_type("retrieval", self.retrieval, RetrievalConfig)
        _require_config_type("bins", self.bins, BinConfig)
        _require_config_type("tm", self.tm, TMConfig)
        _require_config_type("metric", self.metric, MetricConfig)
        _normalize_retrieval_index_pair(self)
        if isinstance(self.metrics, str) or not isinstance(self.metrics, Sequence):
            raise ConfigurationError("metrics must be a sequence of metric names")
        normalized_metrics: list[str] = []
        for metric in self.metrics:
            if not isinstance(metric, str):
                raise ConfigurationError("metrics must contain only metric names")
            normalized_metrics.append(metric.lower())
        normalized_metrics_tuple = tuple(normalized_metrics)
        if not normalized_metrics_tuple:
            raise ConfigurationError("at least one metric must be selected")
        unsupported = sorted(set(normalized_metrics_tuple) - set(SUPPORTED_METRICS))
        if unsupported:
            raise ConfigurationError(f"unsupported metrics: {', '.join(unsupported)}")
        duplicates = _find_duplicates(normalized_metrics_tuple)
        if duplicates:
            raise ConfigurationError(f"duplicate metrics are not allowed: {', '.join(duplicates)}")
        object.__setattr__(self, "metrics", normalized_metrics_tuple)


def _normalize_retrieval_index_pair(config: ScoreConfig) -> None:
    if config.retrieval.mode == "approx":
        if config.index.mode == "auto":
            object.__setattr__(config, "index", replace(config.index, mode="native_fast"))
        elif config.index.mode == "native_exact":
            raise ConfigurationError(
                "retrieval mode 'approx' requires an approximate backend such as native_fast"
            )
        return

    if config.index.mode == "native_fast":
        raise ConfigurationError(
            f"retrieval mode {config.retrieval.mode!r} cannot use approximate backend "
            f"{config.index.mode!r}; use RetrievalConfig(mode='approx', "
            "allow_approximate=True) for approximate scoring"
        )


def parse_float_tuple(value: str) -> tuple[float, ...]:
    if not value:
        raise ConfigurationError("expected a comma-separated list of floats")
    parts = tuple(part.strip() for part in value.split(","))
    if any(part == "" for part in parts):
        raise ConfigurationError("expected a comma-separated list of floats")
    try:
        parsed = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ConfigurationError(f"invalid float list: {value!r}") from exc
    for item in parsed:
        _require_finite("float list", item)
    return parsed


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value:
        raise ConfigurationError("expected a comma-separated list of integers")
    parts = tuple(part.strip() for part in value.split(","))
    if any(part == "" for part in parts):
        raise ConfigurationError("expected a comma-separated list of integers")
    try:
        parsed = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ConfigurationError(f"invalid integer list: {value!r}") from exc
    if not parsed or any(order <= 0 for order in parsed):
        raise ConfigurationError("ngram orders must be positive integers")
    return parsed


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be a finite number")
    if not isfinite(value):
        raise ConfigurationError(f"{name} must be a finite number")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")


def _require_unit_interval(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0 or value > 1:
        raise ConfigurationError(f"{name} must be between 0 and 1")


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    return value


def _require_positive_int(name: str, value: object) -> None:
    parsed = _require_int(name, value)
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be positive")


def _require_non_negative_int(name: str, value: object) -> None:
    parsed = _require_int(name, value)
    if parsed < 0:
        raise ConfigurationError(f"{name} must be non-negative")


def _require_config_type(name: str, value: object, expected_type: type[object]) -> None:
    if not isinstance(value, expected_type):
        raise ConfigurationError(f"{name} must be a {expected_type.__name__}")


def _find_duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
