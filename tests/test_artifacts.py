import json
import random
from pathlib import Path

import pytest

from tame_mt.artifacts import (
    read_segment_jsonl,
    read_segment_metadata,
    segment_metadata_path,
    validate_segment_artifacts,
    validate_segment_metadata,
)
from tame_mt.config import ScoreConfig
from tame_mt.exceptions import ConfigurationError, InputDataError
from tame_mt.report import config_to_dict
from tame_mt.schema import SegmentExposure, SegmentTMResult


def _payload(index: int) -> dict[str, object]:
    return {
        "index": index,
        "source_exposure": 1.0 if index == 0 else 0.2,
        "source_nn_index": index,
        "source_exact": index == 0,
        "target_exposure": None,
        "target_nn_index": None,
        "target_exact": None,
        "pair_exposure": None,
        "pair_nn_index": None,
        "pair_exact": None,
        "bin": "source_exact" if index == 0 else "far",
        "tm_hyp": f"tm {index}",
        "tm_source_index": index,
        "tm_source_similarity": 1.0 if index == 0 else 0.2,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metadata(config: ScoreConfig | None = None) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "artifact": "segment_jsonl",
        "signature": "tame-mt|test",
        "data": {"num_train": 4, "num_test": 1, "num_refs": 1},
        "config": config_to_dict(config or ScoreConfig()),
        "backend": {"name": "native_exact"},
    }


def test_validate_segment_artifacts_keeps_aligned_rows() -> None:
    exposures = [
        SegmentExposure(0, 1.0, 0, True, None, None, None, None, None, None, "source_exact")
    ]
    tm_results = [SegmentTMResult(0, "tm 0", 0, 1.0)]

    validated_exposures, validated_tm_results = validate_segment_artifacts(exposures, tm_results)

    assert validated_exposures is exposures
    assert validated_tm_results is tm_results


def test_read_segment_jsonl_sorts_valid_reordered_rows(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    _write_jsonl(path, [_payload(1), _payload(0)])

    exposures, tm_results = read_segment_jsonl(path)

    assert [segment.index for segment in exposures] == [0, 1]
    assert [result.tm_hyp for result in tm_results] == ["tm 0", "tm 1"]


def test_read_segment_jsonl_accepts_seeded_fuzzed_scalar_encodings(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    rng = random.Random(17)
    rows = []
    for index in range(25):
        row = _payload(index)
        row["index"] = str(index) if index % 2 else index
        row["source_exposure"] = "1.0" if index == 0 else f"0.{index % 9 + 1}"
        row["source_exact"] = "true" if index == 0 else "false"
        row["target_exposure"] = None if index % 3 else 0
        row["target_exact"] = None if index % 3 else 0
        row["tm_source_similarity"] = str(row["source_exposure"])
        rows.append(row)
    rng.shuffle(rows)
    _write_jsonl(path, rows)

    exposures, tm_results = read_segment_jsonl(path)

    assert [segment.index for segment in exposures] == list(range(25))
    assert [result.index for result in tm_results] == list(range(25))
    assert exposures[0].source_exact is True
    assert exposures[-1].source_exact is False


def test_read_segment_jsonl_rejects_duplicate_indices(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    _write_jsonl(path, [_payload(0), _payload(0)])

    with pytest.raises(InputDataError, match="duplicate exposure index 0"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_parses_false_string_as_false(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["source_exact"] = "false"
    _write_jsonl(path, [row])

    exposures, _ = read_segment_jsonl(path)

    assert exposures[0].source_exact is False


def test_read_segment_jsonl_parses_optional_reference_indices(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["target_ref_index"] = 2
    row["pair_ref_index"] = 1
    _write_jsonl(path, [row])

    exposures, _ = read_segment_jsonl(path)

    assert exposures[0].target_ref_index == 2
    assert exposures[0].pair_ref_index == 1


def test_read_segment_jsonl_defaults_missing_reference_indices(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    _write_jsonl(path, [_payload(0)])

    exposures, _ = read_segment_jsonl(path)

    assert exposures[0].target_ref_index is None
    assert exposures[0].pair_ref_index is None


def test_read_segment_jsonl_rejects_non_bool_string(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["source_exact"] = "not really"
    _write_jsonl(path, [row])

    with pytest.raises(InputDataError, match="expected bool-compatible"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_rejects_unknown_bin(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["bin"] = "close-ish"
    _write_jsonl(path, [row])

    with pytest.raises(InputDataError, match="unknown segment bin"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_rejects_non_string_tm_hyp(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["tm_hyp"] = ["not", "a", "string"]
    _write_jsonl(path, [row])

    with pytest.raises(InputDataError, match="expected string value"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_reports_invalid_required_float(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["source_exposure"] = "not-a-float"
    _write_jsonl(path, [row])

    with pytest.raises(InputDataError, match="line 1: expected float-compatible"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_reports_invalid_optional_float(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["tm_source_similarity"] = "not-a-float"
    _write_jsonl(path, [row])

    with pytest.raises(InputDataError, match="line 1: expected float-compatible"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_rejects_non_finite_float(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["source_exposure"] = "nan"
    _write_jsonl(path, [row])

    with pytest.raises(InputDataError, match="finite float"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_rejects_non_standard_json_constant(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = json.dumps(_payload(0), ensure_ascii=False)[:-1] + ', "ignored": NaN}\n'
    path.write_text(row, encoding="utf-8")

    with pytest.raises(InputDataError, match="invalid JSON"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_rejects_duplicate_json_key(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = json.dumps(_payload(0), ensure_ascii=False)[:-1] + ', "index": 1}\n'
    path.write_text(row, encoding="utf-8")

    with pytest.raises(InputDataError, match="invalid JSON"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_reports_invalid_required_index(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["index"] = None
    _write_jsonl(path, [row])

    with pytest.raises(InputDataError, match="line 1: expected int-compatible"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    path.write_bytes(b"\xff")

    with pytest.raises(InputDataError, match="not valid UTF-8"):
        read_segment_jsonl(path)


def test_read_segment_jsonl_rejects_invalid_gzip(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl.gz"
    path.write_bytes(b"not gzip")

    with pytest.raises(InputDataError, match="not a valid gzip file"):
        read_segment_jsonl(path)


def test_read_segment_metadata_reads_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    metadata_path = segment_metadata_path(path)
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")

    metadata = read_segment_metadata(path)

    assert metadata is not None
    assert metadata["artifact"] == "segment_jsonl"


def test_read_segment_metadata_returns_none_without_sidecar(tmp_path: Path) -> None:
    assert read_segment_metadata(tmp_path / "segments.jsonl") is None


def test_validate_segment_metadata_rejects_schema_mismatch() -> None:
    metadata = _metadata()
    metadata["schema_version"] = "9"

    with pytest.raises(InputDataError, match="schema_version"):
        validate_segment_metadata(
            metadata,
            config=ScoreConfig(),
            num_train=4,
            num_test=1,
            num_refs=1,
        )


def test_validate_segment_metadata_rejects_missing_backend() -> None:
    metadata = _metadata()
    metadata.pop("backend")

    with pytest.raises(InputDataError, match="backend"):
        validate_segment_metadata(
            metadata,
            config=ScoreConfig(),
            num_train=4,
            num_test=1,
            num_refs=1,
        )


def test_validate_segment_metadata_rejects_missing_signature() -> None:
    metadata = _metadata()
    metadata.pop("signature")

    with pytest.raises(InputDataError, match="signature"):
        validate_segment_metadata(
            metadata,
            config=ScoreConfig(),
            num_train=4,
            num_test=1,
            num_refs=1,
        )


def test_validate_segment_metadata_rejects_pair_config_mismatch() -> None:
    metadata = _metadata()
    config = metadata["config"]
    assert isinstance(config, dict)
    pair = config["pair"]
    assert isinstance(pair, dict)
    pair["pair_k"] = 10

    with pytest.raises(ConfigurationError, match="pair config"):
        validate_segment_metadata(
            metadata,
            config=ScoreConfig(),
            num_train=4,
            num_test=1,
            num_refs=1,
        )
