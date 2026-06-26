#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    fixtures = project / "tests" / "fixtures"
    with tempfile.TemporaryDirectory(prefix="tame-mt-wheel-smoke-") as tmpdir:
        tmp = Path(tmpdir)
        gz_inputs = _write_gzip_inputs(fixtures, tmp)

        _run(["doctor"])
        _run(
            [
                "score",
                "--train-src",
                str(gz_inputs["train.src"]),
                "--train-tgt",
                str(gz_inputs["train.tgt"]),
                "--test-src",
                str(gz_inputs["test.src"]),
                "--ref",
                str(gz_inputs["test.ref"]),
                "--hyp",
                str(gz_inputs["hyp.out"]),
                "--json-out",
                str(tmp / "fresh.json.gz"),
                "--segment-out",
                str(tmp / "segments.jsonl.gz"),
                "--tm-out",
                str(tmp / "tm.out.gz"),
                "--quiet",
            ]
        )
        _run(
            [
                "index",
                "build",
                "--train-src",
                str(gz_inputs["train.src"]),
                "--train-tgt",
                str(gz_inputs["train.tgt"]),
                "--out",
                str(tmp / "train.tameidx"),
                "--quiet",
            ]
        )
        _run(
            [
                "score",
                "--index",
                str(tmp / "train.tameidx"),
                "--test-src",
                str(gz_inputs["test.src"]),
                "--ref",
                str(gz_inputs["test.ref"]),
                "--hyp",
                str(gz_inputs["hyp.out"]),
                "--json-out",
                str(tmp / "indexed.json.gz"),
                "--quiet",
            ]
        )
        _run(
            [
                "score-cached",
                "--segment-in",
                str(tmp / "segments.jsonl.gz"),
                "--ref",
                str(gz_inputs["test.ref"]),
                "--hyp",
                str(gz_inputs["hyp.out"]),
                "--num-train",
                "4",
                "--json-out",
                str(tmp / "cached.json.gz"),
                "--quiet",
            ]
        )
        _run(
            [
                "tm-baseline",
                "--train-src",
                str(gz_inputs["train.src"]),
                "--train-tgt",
                str(gz_inputs["train.tgt"]),
                "--test-src",
                str(gz_inputs["test.src"]),
                "--out",
                str(tmp / "tm_baseline.out.gz"),
                "--metadata-out",
                str(tmp / "tm_metadata.jsonl.gz"),
            ]
        )

        fresh = _read_json(tmp / "fresh.json.gz")
        indexed = _read_json(tmp / "indexed.json.gz")
        cached = _read_json(tmp / "cached.json.gz")
        if fresh["quality"] != indexed["quality"] or fresh["exposure"] != indexed["exposure"]:
            raise SystemExit("indexed score drifted from fresh score")
        if fresh["quality"] != cached["quality"] or fresh["exposure"] != cached["exposure"]:
            raise SystemExit("cached score drifted from fresh score")
        if indexed["backend"]["index_reused"] is not True:
            raise SystemExit("indexed score did not report index reuse")
        if cached["backend"]["resolved_mode"] != "cached_segments":
            raise SystemExit("cached score did not report cached_segments backend")
        if len(_read_gzip_lines(tmp / "tm_baseline.out.gz")) != 4:
            raise SystemExit("TM baseline output is not aligned")
        if len(_read_gzip_lines(tmp / "tm_metadata.jsonl.gz")) != 4:
            raise SystemExit("TM metadata output is not aligned")
    return 0


def _write_gzip_inputs(fixtures: Path, tmp: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in ("train.src", "train.tgt", "test.src", "test.ref", "hyp.out"):
        output = tmp / f"{name}.gz"
        output.write_bytes(gzip.compress((fixtures / name).read_bytes()))
        paths[name] = output
    return paths


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "tame_mt", *args], check=True)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads("\n".join(_read_gzip_lines(path)))


def _read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read().splitlines()


if __name__ == "__main__":
    raise SystemExit(main())
