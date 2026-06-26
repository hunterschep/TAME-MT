from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tame_mt.exceptions import AlignmentError, InputDataError


def read_lines(path: str | Path) -> list[str]:
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            return [line.rstrip("\n\r") for line in handle]
    except UnicodeDecodeError as exc:
        raise InputDataError(f"{input_path} is not valid UTF-8 text") from exc


def write_lines(path: str | Path, lines: list[str]) -> None:
    output_path = Path(path)
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def ensure_parent_dir(path: str | Path) -> None:
    parent = Path(path).parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def validate_equal_lengths(
    label_a: str,
    values_a: Sequence[object],
    label_b: str,
    values_b: Sequence[object],
) -> None:
    if len(values_a) != len(values_b):
        raise AlignmentError(
            f"misaligned input files: {label_a} has {len(values_a)} lines but "
            f"{label_b} has {len(values_b)} lines"
        )


def validate_parallel_lengths(
    train_src: list[str] | None = None,
    train_tgt: list[str] | None = None,
    test_src: list[str] | None = None,
    refs: list[list[str]] | None = None,
    hyp: list[str] | None = None,
) -> None:
    if train_src is not None and train_tgt is not None:
        validate_equal_lengths("train.src", train_src, "train.tgt", train_tgt)
    if test_src is not None and refs:
        for ref_idx, ref in enumerate(refs):
            validate_equal_lengths("test.src", test_src, f"ref[{ref_idx}]", ref)
    if test_src is not None and hyp is not None:
        validate_equal_lengths("test.src", test_src, "hyp", hyp)


def validate_corpus_inputs(
    train_src: list[str],
    train_tgt: list[str] | None,
    test_src: list[str],
    refs: list[list[str]] | None,
    hyp: list[str] | None,
) -> None:
    if not train_src:
        raise InputDataError("train.src must contain at least one segment")
    if not test_src:
        raise InputDataError("test.src must contain at least one segment")
    if hyp is not None and not refs:
        raise InputDataError("refs are required when hyp is provided")
    validate_parallel_lengths(
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        refs=refs,
        hyp=hyp,
    )
