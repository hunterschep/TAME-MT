from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SegmentExposure:
    index: int
    source_exposure: float
    source_nn_index: int | None
    source_exact: bool
    target_exposure: float | None
    target_nn_index: int | None
    target_exact: bool | None
    pair_exposure: float | None
    pair_nn_index: int | None
    pair_exact: bool | None
    bin: str


@dataclass
class SegmentTMResult:
    index: int
    tm_hyp: str
    tm_source_index: int | None
    tm_source_similarity: float


@dataclass
class ExposureSummary:
    source: dict[str, Any]
    target: dict[str, Any] | None
    pair: dict[str, Any] | None


@dataclass
class BinReport:
    name: str
    count: int
    percentage: float
    mean_source_exposure: float | None
    system_scores: dict[str, float | None]
    tm_scores: dict[str, float | None]
    delta_scores: dict[str, float | None]


@dataclass
class TameReport:
    tame_version: str
    signature: str
    num_train: int
    num_test: int
    num_refs: int
    config: dict[str, Any]
    backend: dict[str, Any]
    system_scores: dict[str, float | None]
    tm_scores: dict[str, float | None]
    delta_scores: dict[str, float | None]
    exposure: ExposureSummary
    bins: list[BinReport]
    generalization_gap: dict[str, float | None]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "tame_version": self.tame_version,
            "signature": self.signature,
            "data": {
                "num_train": self.num_train,
                "num_test": self.num_test,
                "num_refs": self.num_refs,
            },
            "config": self.config,
            "backend": self.backend,
            "quality": {
                "system": self.system_scores,
                "tm": self.tm_scores,
                "delta_tm": self.delta_scores,
            },
            "exposure": asdict(self.exposure),
            "bins": [
                {
                    "name": item.name,
                    "count": item.count,
                    "percentage": item.percentage,
                    "mean_source_exposure": item.mean_source_exposure,
                    "system": item.system_scores,
                    "tm": item.tm_scores,
                    "delta_tm": item.delta_scores,
                }
                for item in self.bins
            ],
            "generalization_gap": self.generalization_gap,
            "warnings": self.warnings,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)
