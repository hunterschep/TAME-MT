from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, TypeVar

from tame_mt.config import ScoreConfig
from tame_mt.exceptions import ConfigurationError, InputDataError
from tame_mt.io import open_text
from tame_mt.json_utils import strict_json_loads
from tame_mt.report import SEGMENT_METADATA_SUFFIX, config_to_dict
from tame_mt.schema import SegmentExposure, SegmentTMResult

SegmentRow = TypeVar("SegmentRow", SegmentExposure, SegmentTMResult)
VALID_SEGMENT_BINS = frozenset({"source_exact", "near", "medium", "far"})
SEGMENT_METADATA_SCHEMA_VERSION = "0.1"


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


def read_segment_metadata(path: str | Path) -> dict[str, Any] | None:
    metadata_path = segment_metadata_path(path)
    if not metadata_path.exists():
        return None
    try:
        with open_text(metadata_path, "r") as handle:
            payload = strict_json_loads(handle.read())
    except UnicodeDecodeError as exc:
        raise InputDataError(f"{metadata_path} is not valid UTF-8 text") from exc
    except ValueError as exc:
        raise InputDataError(f"segment metadata {metadata_path} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise InputDataError(f"segment metadata {metadata_path} must be a JSON object")
    return payload


def segment_metadata_path(path: str | Path) -> Path:
    return Path(f"{path}{SEGMENT_METADATA_SUFFIX}")


def validate_segment_metadata(
    metadata: dict[str, Any],
    *,
    config: ScoreConfig,
    num_train: int,
    num_test: int,
    num_refs: int,
) -> None:
    if metadata.get("schema_version") != SEGMENT_METADATA_SCHEMA_VERSION:
        raise InputDataError(
            f"segment metadata schema_version must be {SEGMENT_METADATA_SCHEMA_VERSION!r}"
        )
    if metadata.get("artifact") != "segment_jsonl":
        raise InputDataError("segment metadata artifact must be 'segment_jsonl'")
    signature = metadata.get("signature")
    if not isinstance(signature, str) or not signature:
        raise InputDataError("segment metadata signature must be a non-empty string")
    _metadata_object(metadata, "backend")
    data = _metadata_object(metadata, "data")
    _require_metadata_int(data, "num_train", num_train)
    _require_metadata_int(data, "num_test", num_test)
    _require_metadata_int(data, "num_refs", num_refs)

    saved_config = _metadata_object(metadata, "config")
    current_config = config_to_dict(config)
    for key in ("normalization", "similarity", "index", "pair", "tm"):
        if saved_config.get(key) != current_config[key]:
            raise ConfigurationError(
                f"segment metadata {key} config does not match current scorer config"
            )

    saved_bins = _metadata_object(saved_config, "bins")
    current_bins = current_config["bins"]
    for key in ("far_threshold", "near_threshold"):
        if saved_bins.get(key) != current_bins[key]:
            raise ConfigurationError(
                f"segment metadata bins.{key} does not match current scorer config"
            )


def _metadata_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise InputDataError(f"segment metadata field {key!r} must be an object")
    return value


def _require_metadata_int(payload: dict[str, Any], key: str, expected: int) -> None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputDataError(f"segment metadata data.{key} must be an integer")
    if value != expected:
        raise ConfigurationError(
            f"segment metadata data.{key}={value} does not match current value {expected}"
        )


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
            bin=_required_bin(payload["bin"]),
            target_ref_index=_optional_int(payload.get("target_ref_index")),
            pair_ref_index=_optional_int(payload.get("pair_ref_index")),
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
            tm_hyp=_optional_str(payload.get("tm_hyp")) or "",
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


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise InputDataError(f"expected string value, got {type(value).__name__}")


def _required_bin(value: object) -> str:
    parsed = _optional_str(value)
    if parsed is None:
        raise InputDataError("expected bin string, got null")
    if parsed not in VALID_SEGMENT_BINS:
        raise InputDataError(f"unknown segment bin {parsed!r}")
    return parsed
