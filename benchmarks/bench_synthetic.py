#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import tempfile
import time
from pathlib import Path

from tame_mt import IndexConfig, ScoreConfig, TameScorer, load_index_bundle, save_index_bundle
from tame_mt.exceptions import TameMTError
from tame_mt.json_utils import strict_json_dumps
from tame_mt.native import native_status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a synthetic TAME-MT performance smoke benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--small", action="store_true", help="use CI-friendly corpus sizes")
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--index-mode", default="auto")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--staged", action="store_true", help="also benchmark index reuse stages")
    parser.add_argument("--max-index-build-seconds", type=float, default=None)
    parser.add_argument(
        "--max-indexed-seconds",
        type=float,
        default=None,
        help="staged mode threshold for .tameidx load plus indexed audit time",
    )
    parser.add_argument("--max-cached-seconds", type=float, default=None)
    parser.add_argument(
        "--max-index-bytes",
        type=int,
        default=None,
        help="staged mode threshold for persisted .tameidx bundle size",
    )
    parser.add_argument("--assert-thresholds", action="store_true")
    args = parser.parse_args()

    train_size = 2_000 if args.small else args.train_size
    test_size = 200 if args.small else args.test_size
    max_seconds = args.max_seconds
    if max_seconds is None:
        max_seconds = 8.0 if args.small else 60.0

    train_src, train_tgt, test_src, refs = make_corpus(train_size, test_size)
    config = ScoreConfig(index=IndexConfig(mode=args.index_mode))
    scorer = TameScorer(config)

    started = time.perf_counter()
    report = scorer.evaluate_corpus(train_src, train_tgt, test_src, [refs], hyp=None).report
    elapsed = time.perf_counter() - started

    payload = {
        "train_size": train_size,
        "test_size": test_size,
        "seconds": elapsed,
        "backend": report.backend,
        "signature": report.signature,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "native": native_status().__dict__,
    }
    if args.staged:
        payload["stages"] = run_staged_benchmark(
            train_src=train_src,
            train_tgt=train_tgt,
            test_src=test_src,
            refs=refs,
            config=config,
        )
    print(strict_json_dumps(payload, indent=2, sort_keys=True))

    if args.assert_thresholds and elapsed > max_seconds:
        raise SystemExit(
            f"synthetic benchmark exceeded threshold: {elapsed:.2f}s > {max_seconds:.2f}s"
        )
    if args.assert_thresholds and args.staged:
        assert_stage_thresholds(
            payload["stages"],
            small=args.small,
            max_index_build_seconds=args.max_index_build_seconds,
            max_indexed_seconds=args.max_indexed_seconds,
            max_cached_seconds=args.max_cached_seconds,
            max_index_bytes=args.max_index_bytes,
        )
    return 0


def run_staged_benchmark(
    train_src: list[str],
    train_tgt: list[str],
    test_src: list[str],
    refs: list[str],
    config: ScoreConfig,
) -> dict[str, float | int | str]:
    scorer = TameScorer(config)
    with tempfile.TemporaryDirectory(prefix="tame-mt-bench-") as tmpdir:
        index_path = Path(tmpdir) / "train.tameidx"

        started = time.perf_counter()
        try:
            save_index_bundle(index_path, train_src, train_tgt, config)
        except TameMTError as exc:
            raise SystemExit(f"staged benchmark requires native index persistence: {exc}") from exc
        index_build_seconds = time.perf_counter() - started

        started = time.perf_counter()
        bundle = load_index_bundle(index_path, config)
        index_load_seconds = time.perf_counter() - started

        started = time.perf_counter()
        indexed_result = scorer.evaluate_index_bundle(bundle, test_src, [refs], hyp=None)
        indexed_audit_seconds = time.perf_counter() - started
        indexed_total_seconds = index_load_seconds + indexed_audit_seconds

        started = time.perf_counter()
        cached_report = scorer.score_from_artifacts(
            exposures=indexed_result.exposures,
            tm_results=indexed_result.tm_results,
            refs=[refs],
            hyp=refs,
            num_train=len(train_src),
        )
        cached_score_seconds = time.perf_counter() - started

        if cached_report.exposure != indexed_result.report.exposure:
            raise SystemExit("cached scoring exposure summary drifted from indexed audit")

        return {
            "index_build_seconds": index_build_seconds,
            "index_load_seconds": index_load_seconds,
            "indexed_audit_seconds": indexed_audit_seconds,
            "indexed_total_seconds": indexed_total_seconds,
            "cached_score_seconds": cached_score_seconds,
            "index_bytes": index_path.stat().st_size,
            "indexed_backend": indexed_result.report.backend["name"],
            "cached_backend": cached_report.backend["name"],
        }


def assert_stage_thresholds(
    stages: object,
    *,
    small: bool,
    max_index_build_seconds: float | None,
    max_indexed_seconds: float | None,
    max_cached_seconds: float | None,
    max_index_bytes: int | None,
) -> None:
    if not isinstance(stages, dict):
        raise SystemExit("staged benchmark payload is missing stage timings")
    thresholds = {
        "index_build_seconds": (
            max_index_build_seconds
            if max_index_build_seconds is not None
            else (8.0 if small else 60.0)
        ),
        "indexed_total_seconds": (
            max_indexed_seconds if max_indexed_seconds is not None else (4.0 if small else 15.0)
        ),
        "cached_score_seconds": (
            max_cached_seconds if max_cached_seconds is not None else (3.0 if small else 10.0)
        ),
    }
    for key, threshold in thresholds.items():
        value = stages[key]
        if not isinstance(value, int | float):
            raise SystemExit(f"staged benchmark field {key} is not numeric")
        if value > threshold:
            raise SystemExit(
                f"staged benchmark {key} exceeded threshold: {value:.2f}s > {threshold:.2f}s"
            )
    if max_index_bytes is not None:
        value = stages["index_bytes"]
        if not isinstance(value, int):
            raise SystemExit("staged benchmark field index_bytes is not an integer")
        if value > max_index_bytes:
            raise SystemExit(
                f"staged benchmark index_bytes exceeded threshold: {value} > {max_index_bytes}"
            )


def make_corpus(
    train_size: int, test_size: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    train_src = [
        f"domain {idx % 97} source sentence {idx:06d} topic {idx % 17} with shared template"
        for idx in range(train_size)
    ]
    train_tgt = [
        f"domain {idx % 97} target sentence {idx:06d} topic {idx % 17} translated template"
        for idx in range(train_size)
    ]
    test_src: list[str] = []
    refs: list[str] = []
    stride = max(1, train_size // max(1, test_size))
    for idx in range(test_size):
        train_idx = (idx * stride) % train_size
        if idx % 5 == 0:
            test_src.append(train_src[train_idx])
            refs.append(train_tgt[train_idx])
        else:
            test_src.append(f"heldout domain {idx % 101} source sample {idx:06d} topic {idx % 23}")
            refs.append(f"heldout domain {idx % 101} target sample {idx:06d} topic {idx % 23}")
    return train_src, train_tgt, test_src, refs


if __name__ == "__main__":
    raise SystemExit(main())
