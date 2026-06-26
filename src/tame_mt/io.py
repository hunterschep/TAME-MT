from __future__ import annotations

from pathlib import Path


def read_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n\r") for line in handle]


def write_lines(path: str | Path, lines: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def validate_equal_lengths(label_a: str, values_a: list[object], label_b: str, values_b: list[object]) -> None:
    if len(values_a) != len(values_b):
        raise ValueError(
            f"misaligned input files: {label_a} has {len(values_a)} lines but "
            f"{label_b} has {len(values_b)} lines"
        )


def validate_parallel_lengths(train_src: list[str] | None = None, train_tgt: list[str] | None = None,
                              test_src: list[str] | None = None, refs: list[list[str]] | None = None,
                              hyp: list[str] | None = None) -> None:
    if train_src is not None and train_tgt is not None:
        validate_equal_lengths("train.src", train_src, "train.tgt", train_tgt)
    if test_src is not None and refs:
        for ref_idx, ref in enumerate(refs):
            validate_equal_lengths("test.src", test_src, f"ref[{ref_idx}]", ref)
    if test_src is not None and hyp is not None:
        validate_equal_lengths("test.src", test_src, "hyp", hyp)
