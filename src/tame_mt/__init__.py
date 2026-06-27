"""Public API for TAME-MT."""

from tame_mt.api import CachedSegmentScorer, TameScorer, audit, score
from tame_mt.artifacts import (
    CachedArtifact,
    load_cached_artifact,
    read_segment_jsonl,
    read_segment_metadata,
    segment_metadata_path,
    validate_segment_metadata,
)
from tame_mt.config import (
    BinConfig,
    IndexConfig,
    MetricConfig,
    NormalizationConfig,
    PairConfig,
    RetrievalConfig,
    ScoreConfig,
    SimilarityConfig,
    TMConfig,
)
from tame_mt.persistence import (
    IndexBundle,
    IndexVerification,
    inspect_index_bundle,
    load_index_bundle,
    save_index_bundle,
    verify_index_bundle,
)
from tame_mt.schema import SegmentExposure, SegmentTMResult, TameReport
from tame_mt.version import __version__

__all__ = [
    "__version__",
    "audit",
    "BinConfig",
    "CachedArtifact",
    "CachedSegmentScorer",
    "IndexConfig",
    "IndexBundle",
    "IndexVerification",
    "inspect_index_bundle",
    "load_cached_artifact",
    "load_index_bundle",
    "MetricConfig",
    "NormalizationConfig",
    "PairConfig",
    "read_segment_jsonl",
    "read_segment_metadata",
    "RetrievalConfig",
    "score",
    "ScoreConfig",
    "save_index_bundle",
    "SegmentExposure",
    "SegmentTMResult",
    "segment_metadata_path",
    "SimilarityConfig",
    "TameReport",
    "TameScorer",
    "TMConfig",
    "validate_segment_metadata",
    "verify_index_bundle",
]
