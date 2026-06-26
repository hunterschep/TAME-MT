"""Public API for TAME-MT."""

from tame_mt.api import TameScorer, audit, score
from tame_mt.config import (
    BinConfig,
    IndexConfig,
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
from tame_mt.schema import TameReport
from tame_mt.version import __version__

__all__ = [
    "__version__",
    "audit",
    "BinConfig",
    "IndexConfig",
    "IndexBundle",
    "inspect_index_bundle",
    "load_index_bundle",
    "NormalizationConfig",
    "score",
    "ScoreConfig",
    "save_index_bundle",
    "SimilarityConfig",
    "TameReport",
    "TameScorer",
    "TMConfig",
]
