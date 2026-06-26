from pathlib import Path

import pytest

from tame_mt.exceptions import AlignmentError, InputDataError
from tame_mt.io import read_lines, validate_corpus_inputs, write_lines


def test_read_and_write_lines_preserve_empty_segments(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "lines.txt"
    write_lines(out, ["a", "", "b"])
    assert read_lines(out) == ["a", "", "b"]


def test_read_and_write_lines_support_gzip(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "lines.txt.gz"
    write_lines(out, ["a", "", "b"])

    assert read_lines(out) == ["a", "", "b"]


def test_read_lines_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")

    with pytest.raises(InputDataError, match="not valid UTF-8"):
        read_lines(path)


def test_validate_corpus_inputs_rejects_empty_train_or_test() -> None:
    with pytest.raises(InputDataError, match="train.src"):
        validate_corpus_inputs([], [], ["x"], [["y"]], ["z"])
    with pytest.raises(InputDataError, match="test.src"):
        validate_corpus_inputs(["x"], ["y"], [], [[]], [])
    with pytest.raises(InputDataError, match="refs are required"):
        validate_corpus_inputs(["x"], ["y"], ["x"], [], ["z"])


def test_validate_corpus_inputs_rejects_misalignment() -> None:
    with pytest.raises(AlignmentError, match="train.src"):
        validate_corpus_inputs(["x"], [], ["x"], [["y"]], ["z"])
