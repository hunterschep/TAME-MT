import json
from pathlib import Path

import pytest

from tame_mt.artifacts import read_segment_jsonl
from tame_mt.exceptions import InputDataError


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


def test_read_segment_jsonl_sorts_valid_reordered_rows(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    _write_jsonl(path, [_payload(1), _payload(0)])

    exposures, tm_results = read_segment_jsonl(path)

    assert [segment.index for segment in exposures] == [0, 1]
    assert [result.tm_hyp for result in tm_results] == ["tm 0", "tm 1"]


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


def test_read_segment_jsonl_rejects_non_bool_string(tmp_path: Path) -> None:
    path = tmp_path / "segments.jsonl"
    row = _payload(0)
    row["source_exact"] = "not really"
    _write_jsonl(path, [row])

    with pytest.raises(InputDataError, match="expected bool-compatible"):
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
