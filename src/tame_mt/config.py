from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NormalizationConfig:
    unicode_form: str = "NFKC"
    strip: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False
    strip_diacritics: bool = False
    normalize_punctuation: bool = False


@dataclass(frozen=True)
class SimilarityConfig:
    ngram_orders: tuple[int, ...] = (3, 4, 5)
    similarity: str = "jaccard_set"


@dataclass(frozen=True)
class IndexConfig:
    mode: str = "inverted_exact"
    topk: int = 50


@dataclass(frozen=True)
class BinConfig:
    far_threshold: float = 0.30
    near_threshold: float = 0.70
    leak_thresholds: tuple[float, ...] = (0.70, 0.85, 0.95)
    min_bin_size_warning: int = 30


@dataclass(frozen=True)
class TMConfig:
    zero_policy: str = "empty"


@dataclass(frozen=True)
class MetricConfig:
    bleu_tokenize: str = "13a"
    bleu_lowercase: bool = False
    chrf_word_order: int = 2


@dataclass(frozen=True)
class ScoreConfig:
    metrics: tuple[str, ...] = ("bleu", "chrf")
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    bins: BinConfig = field(default_factory=BinConfig)
    tm: TMConfig = field(default_factory=TMConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    if not value:
        raise ValueError("expected a comma-separated list of floats")
    try:
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"invalid float list: {value!r}") from exc


def parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value:
        raise ValueError("expected a comma-separated list of integers")
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"invalid integer list: {value!r}") from exc
    if not parsed or any(order <= 0 for order in parsed):
        raise ValueError("ngram orders must be positive integers")
    return parsed
