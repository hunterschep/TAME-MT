from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, TypeVar

from tame_mt.exceptions import InputDataError
from tame_mt.io import open_text
from tame_mt.json_utils import strict_json_loads
from tame_mt.schema import SegmentExposure, SegmentTMResult

SegmentRow = TypeVar("SegmentRow", SegmentExposure, SegmentTMResult)


def read_segment_jsonl(path: str | Path) -> tuple[list[SegmentExposure], list[SegmentTMResult]]:
    exposures: list[SegmentExposure] = []
    tm_results: list[SegmentTMResult] = []
    input_path = Path(path)
    try:
        with open_text(input_path, "r") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = strict_json_loads(line)
                except ValueError as exc:
                    raise InputDataError(
                        f"segment JSONL line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(payload, dict):
                    raise InputDataError(f"segment JSONL line {line_number} is not an object")
                exposures.append(_segment_exposure_from_payload(payload, line_number))
                tm_results.append(_tm_result_from_payload(payload, line_number))
    except UnicodeDecodeError as exc:
        raise InputDataError(f"{input_path} is not valid UTF-8 text") from exc
    return validate_segment_artifacts(exposures, tm_results)


def validate_segment_artifacts(
    exposures: list[SegmentExposure],
    tm_results: list[SegmentTMResult],
) -> tuple[list[SegmentExposure], list[SegmentTMResult]]:
    if len(exposures) != len(tm_results):
        raise InputDataError(
            "segment artifact is inconsistent: exposure and TM rows have different counts"
        )
    if _is_aligned_contiguous(exposures, tm_results):
        return exposures, tm_results

    exposure_by_index = _unique_by_index(exposures, "exposure")
    tm_by_index = _unique_by_index(tm_results, "tm_result")
    exposure_indices = set(exposure_by_index)
    tm_indices = set(tm_by_index)
    if exposure_indices != tm_indices:
        missing_tm = sorted(exposure_indices - tm_indices)
        missing_exposure = sorted(tm_indices - exposure_indices)
        details: list[str] = []
        if missing_tm:
            details.append(f"missing TM rows for indices {_preview_indices(missing_tm)}")
        if missing_exposure:
            details.append(
                f"missing exposure rows for indices {_preview_indices(missing_exposure)}"
            )
        raise InputDataError("segment artifact index mismatch: " + "; ".join(details))

    expected = set(range(len(exposures)))
    if exposure_indices != expected:
        missing = sorted(expected - exposure_indices)
        unexpected = sorted(exposure_indices - expected)
        details = []
        if missing:
            details.append(f"missing indices {_preview_indices(missing)}")
        if unexpected:
            details.append(f"unexpected indices {_preview_indices(unexpected)}")
        raise InputDataError(
            "segment artifact indices must be contiguous from 0 to "
            f"{len(exposures) - 1}: " + "; ".join(details)
        )

    ordered_indices = sorted(exposure_indices)
    return (
        [exposure_by_index[index] for index in ordered_indices],
        [tm_by_index[index] for index in ordered_indices],
    )


def _is_aligned_contiguous(
    exposures: list[SegmentExposure],
    tm_results: list[SegmentTMResult],
) -> bool:
    return all(
        exposure.index == expected and tm_result.index == expected
        for expected, (exposure, tm_result) in enumerate(zip(exposures, tm_results, strict=True))
    )


def _segment_exposure_from_payload(payload: dict[str, Any], line_number: int) -> SegmentExposure:
    try:
        return SegmentExposure(
            index=_required_int(payload["index"]),
            source_exposure=_required_float(payload["source_exposure"]),
            source_nn_index=_optional_int(payload.get("source_nn_index")),
            source_exact=_required_bool(payload["source_exact"]),
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
    except InputDataError as exc:
        raise InputDataError(f"segment JSONL line {line_number}: {exc}") from exc


def _tm_result_from_payload(payload: dict[str, Any], line_number: int) -> SegmentTMResult:
    try:
        tm_source_similarity = _optional_float(payload.get("tm_source_similarity"))
        return SegmentTMResult(
            index=_required_int(payload["index"]),
            tm_hyp=str(payload.get("tm_hyp", "")),
            tm_source_index=_optional_int(payload.get("tm_source_index")),
            tm_source_similarity=tm_source_similarity if tm_source_similarity is not None else 0.0,
        )
    except KeyError as exc:
        raise InputDataError(f"segment JSONL line {line_number} is missing field {exc}") from exc
    except InputDataError as exc:
        raise InputDataError(f"segment JSONL line {line_number}: {exc}") from exc


def _unique_by_index(
    rows: list[SegmentRow],
    label: str,
) -> dict[int, SegmentRow]:
    by_index: dict[int, SegmentRow] = {}
    for row in rows:
        if row.index < 0:
            raise InputDataError(f"segment artifact {label} index must be non-negative")
        if row.index in by_index:
            raise InputDataError(f"segment artifact has duplicate {label} index {row.index}")
        by_index[row.index] = row
    return by_index


def _preview_indices(indices: list[int]) -> str:
    preview = ", ".join(str(index) for index in indices[:5])
    if len(indices) > 5:
        return f"{preview}, ..."
    return preview


def _required_int(value: object) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise InputDataError("expected int-compatible value, got null")
    return parsed


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise InputDataError("expected int-compatible value, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise InputDataError(f"expected int-compatible value, got {value!r}") from exc
    raise InputDataError(f"expected int-compatible value, got {type(value).__name__}")


def _required_float(value: object) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        raise InputDataError("expected float-compatible value, got null")
    return parsed


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise InputDataError("expected float-compatible value, got bool")
    if isinstance(value, str | int | float):
        try:
            parsed = float(value)
        except ValueError as exc:
            raise InputDataError(f"expected float-compatible value, got {value!r}") from exc
        if not isfinite(parsed):
            raise InputDataError("expected finite float-compatible value")
        return parsed
    raise InputDataError(f"expected float-compatible value, got {type(value).__name__}")


def _required_bool(value: object) -> bool:
    parsed = _optional_bool(value)
    if parsed is None:
        raise InputDataError("expected bool value, got null")
    return parsed


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise InputDataError(f"expected bool-compatible value, got {type(value).__name__}")
