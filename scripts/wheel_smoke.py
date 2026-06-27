#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tame_mt import CachedSegmentScorer, TameScorer, read_segment_jsonl


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    fixtures = project / "tests" / "fixtures"
    with tempfile.TemporaryDirectory(prefix="tame-mt-wheel-smoke-") as tmpdir:
        tmp = Path(tmpdir)
        gz_inputs = _write_gzip_inputs(fixtures, tmp)
        variant_hyp = tmp / "variant.out.gz"
        variant_hyp.write_bytes(gzip.compress(b"hola mundo\nbuenos dias\nhasta luego\ndistinto\n"))

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
                "score-cached-batch",
                "--segment-in",
                str(tmp / "segments.jsonl.gz"),
                "--ref",
                str(gz_inputs["test.ref"]),
                "--system",
                f"baseline={gz_inputs['hyp.out']}",
                "--system",
                f"variant={variant_hyp}",
                "--num-train",
                "4",
                "--json-out-dir",
                str(tmp / "batch_reports"),
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
        batch_baseline = _read_json(tmp / "batch_reports" / "baseline.json")
        batch_variant = _read_json(tmp / "batch_reports" / "variant.json")
        exposures, tm_results = read_segment_jsonl(tmp / "segments.jsonl.gz")
        prepared = TameScorer().prepare_from_artifacts(
            exposures=exposures,
            tm_results=tm_results,
            refs=[_read_gzip_lines(gz_inputs["test.ref"])],
            num_train=4,
        )
        if not isinstance(prepared, CachedSegmentScorer):
            raise SystemExit("prepared cached scorer did not use the public API type")
        prepared_report = prepared.score(_read_gzip_lines(gz_inputs["hyp.out"]))
        prepared_batch = prepared.score_many(
            {
                "baseline": _read_gzip_lines(gz_inputs["hyp.out"]),
                "variant": _read_gzip_lines(variant_hyp),
            }
        )
        if fresh["quality"] != indexed["quality"] or fresh["exposure"] != indexed["exposure"]:
            raise SystemExit("indexed score drifted from fresh score")
        if fresh["quality"] != cached["quality"] or fresh["exposure"] != cached["exposure"]:
            raise SystemExit("cached score drifted from fresh score")
        if prepared_report.to_dict()["quality"] != cached["quality"]:
            raise SystemExit("prepared cached API drifted from single cached score")
        if prepared_batch["baseline"].to_dict()["quality"] != cached["quality"]:
            raise SystemExit("prepared cached batch API drifted from single cached score")
        if batch_baseline["quality"] != cached["quality"]:
            raise SystemExit("batch cached baseline drifted from single cached score")
        if batch_variant["quality"]["system"] == batch_baseline["quality"]["system"]:
            raise SystemExit("batch cached variant did not produce distinct system scores")
        if indexed["backend"]["index_reused"] is not True:
            raise SystemExit("indexed score did not report index reuse")
        if cached["backend"]["resolved_mode"] != "cached_segments":
            raise SystemExit("cached score did not report cached_segments backend")
        if batch_baseline["backend"]["resolved_mode"] != "cached_segments":
            raise SystemExit("batch cached score did not report cached_segments backend")
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
    if path.suffix == ".gz":
        return json.loads("\n".join(_read_gzip_lines(path)))
    return json.loads(path.read_text(encoding="utf-8"))


def _read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read().splitlines()


if __name__ == "__main__":
    raise SystemExit(main())
