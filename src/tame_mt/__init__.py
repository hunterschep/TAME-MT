"""Public API for TAME-MT."""

from tame_mt.api import CachedSegmentScorer, TameScorer, audit, score
from tame_mt.artifacts import read_segment_jsonl
from tame_mt.config import (
    BinConfig,
    IndexConfig,
    MetricConfig,
    NormalizationConfig,
    ScoreConfig,
    SimilarityConfig,
    TMConfig,
)
from tame_mt.persistence import (
    IndexBundle,
    inspect_index_bundle,
    load_index_bundle,
    save_index_bundle,
)
from tame_mt.schema import SegmentExposure, SegmentTMResult, TameReport
from tame_mt.version import __version__

__all__ = [
    "__version__",
    "audit",
    "BinConfig",
    "CachedSegmentScorer",
    "IndexConfig",
    "IndexBundle",
    "inspect_index_bundle",
    "load_index_bundle",
    "MetricConfig",
    "NormalizationConfig",
    "read_segment_jsonl",
    "score",
    "ScoreConfig",
    "save_index_bundle",
    "SegmentExposure",
    "SegmentTMResult",
    "SimilarityConfig",
    "TameReport",
    "TameScorer",
    "TMConfig",
]
