#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import time

from tame_mt import IndexConfig, ScoreConfig, TameScorer
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
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.assert_thresholds and elapsed > max_seconds:
        raise SystemExit(
            f"synthetic benchmark exceeded threshold: {elapsed:.2f}s > {max_seconds:.2f}s"
        )
    return 0


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
