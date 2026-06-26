from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tame_mt.exceptions import InputDataError
from tame_mt.schema import SegmentExposure, SegmentTMResult


def read_segment_jsonl(path: str | Path) -> tuple[list[SegmentExposure], list[SegmentTMResult]]:
    exposures: list[SegmentExposure] = []
    tm_results: list[SegmentTMResult] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputDataError(f"segment JSONL line {line_number} is invalid JSON") from exc
            if not isinstance(payload, dict):
                raise InputDataError(f"segment JSONL line {line_number} is not an object")
            exposures.append(_segment_exposure_from_payload(payload, line_number))
            tm_results.append(_tm_result_from_payload(payload, line_number))
    return exposures, tm_results


def _segment_exposure_from_payload(payload: dict[str, Any], line_number: int) -> SegmentExposure:
    try:
        return SegmentExposure(
            index=int(payload["index"]),
            source_exposure=float(payload["source_exposure"]),
            source_nn_index=_optional_int(payload.get("source_nn_index")),
            source_exact=bool(payload["source_exact"]),
            target_exposure=_optional_float(payload.get("target_exposure")),
            target_nn_index=_optional_int(payload.get("target_nn_index")),
            target_exact=_optional_bool(payload.get("target_exact")),
            pair_exposure=_optional_float(payload.get("pair_exposure")),
            pair_nn_index=_optional_int(payload.get("pair_nn_index")),
            pair_exact=_optional_bool(payload.get("pair_exact")),
            bin=str(payload["bin"]),
        )
    except KeyError as exc:
        raise InputDataError(f"segment JSONL line {line_number} is missing field {exc}") from exc


def _tm_result_from_payload(payload: dict[str, Any], line_number: int) -> SegmentTMResult:
    try:
        return SegmentTMResult(
            index=int(payload["index"]),
            tm_hyp=str(payload.get("tm_hyp", "")),
            tm_source_index=_optional_int(payload.get("tm_source_index")),
            tm_source_similarity=float(payload.get("tm_source_similarity") or 0.0),
        )
    except KeyError as exc:
        raise InputDataError(f"segment JSONL line {line_number} is missing field {exc}") from exc


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str | bytes | bytearray | int | float):
        return int(value)
    raise InputDataError(f"expected int-compatible value, got {type(value).__name__}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str | bytes | bytearray | int | float):
        return float(value)
    raise InputDataError(f"expected float-compatible value, got {type(value).__name__}")


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)
