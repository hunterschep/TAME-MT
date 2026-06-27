from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class NeighborResult:
    index: int | None
    score: float
    exact: bool = False


@dataclass(frozen=True, slots=True)
class IndexBackendInfo:
    name: str
    native: bool
    exact: bool
    requested_mode: str
    resolved_mode: str


class SimilarityIndex(Protocol):
    backend_info: IndexBackendInfo
    doc_count: int

    def query_best(self, text: str) -> NeighborResult: ...

    def query_topk(self, text: str, k: int) -> list[NeighborResult]: ...

    def batch_query_topk(self, texts: list[str], k: int) -> list[list[NeighborResult]]: ...

    def score_candidate(self, text: str, index: int) -> float: ...


class ExactSimilarityIndex(SimilarityIndex, Protocol):
    pass


class ApproxSimilarityIndex(SimilarityIndex, Protocol):
    pass


class ThresholdSimilarityIndex(ExactSimilarityIndex, Protocol):
    def batch_best_above(
        self,
        texts: list[str],
        threshold: float,
    ) -> list[NeighborResult | None]: ...

    def batch_threshold_flags(
        self,
        texts: list[str],
        thresholds: list[float] | tuple[float, ...],
    ) -> list[dict[float, bool]]: ...
