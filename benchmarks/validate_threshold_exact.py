#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass

from tame_mt.config import BinConfig, IndexConfig, parse_float_tuple
from tame_mt.exceptions import ApproximationError
from tame_mt.index import NgramInvertedIndex
from tame_mt.json_utils import strict_json_dumps
from tame_mt.native import native_status


@dataclass(frozen=True, slots=True)
class ThresholdCase:
    name: str
    train: list[str]
    queries: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate exact threshold flags against exact top-1 retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-size", type=int, default=3_000)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--thresholds", default="0.30,0.70,0.85,0.95")
    parser.add_argument("--far-threshold", type=float, default=0.30)
    parser.add_argument("--near-threshold", type=float, default=0.70)
    parser.add_argument("--require-native", action="store_true")
    args = parser.parse_args()

    status = native_status()
    if not status.available:
        message = f"native backend is required for threshold validation: {status.error}"
        if args.require_native:
            raise SystemExit(message)
        print(message)
        return 0

    thresholds = parse_float_tuple(args.thresholds)
    bins = BinConfig(
        far_threshold=args.far_threshold,
        near_threshold=args.near_threshold,
        leak_thresholds=thresholds,
    )
    cases = [
        evaluate_case(case, thresholds=thresholds, bins=bins)
        for case in build_cases(args.train_size, args.test_size)
    ]
    payload = {
        "backend": "native_exact",
        "native_available": status.available,
        "thresholds": list(thresholds),
        "cases": cases,
        "pair_case": evaluate_pair_case(thresholds=thresholds),
        "approximate_backend_rejected": _approximate_backend_rejected(),
    }
    print(strict_json_dumps(payload, indent=2, sort_keys=True))

    failures: list[str] = []
    for case in cases:
        if case["false_negatives"] != 0:
            failures.append(f"{case['name']}: {case['false_negatives']} threshold false negatives")
        if case["false_positives"] != 0:
            failures.append(f"{case['name']}: {case['false_positives']} threshold false positives")
        if case["bin_mismatches"] != 0:
            failures.append(f"{case['name']}: {case['bin_mismatches']} source-bin mismatches")
    pair_case = payload["pair_case"]
    if not isinstance(pair_case, dict):
        raise SystemExit("internal error: malformed pair case payload")
    if pair_case["false_negatives"] != 0:
        failures.append(f"pair_threshold: {pair_case['false_negatives']} false negatives")
    if pair_case["false_positives"] != 0:
        failures.append(f"pair_threshold: {pair_case['false_positives']} false positives")
    if not payload["approximate_backend_rejected"]:
        failures.append("native_fast did not reject exact threshold API")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


def evaluate_case(
    case: ThresholdCase,
    *,
    thresholds: tuple[float, ...],
    bins: BinConfig,
) -> dict[str, object]:
    index = NgramInvertedIndex.build(case.train, index_config=IndexConfig(mode="native_exact"))
    exact_tops = [index.query_best(query) for query in case.queries]
    flags = index.batch_threshold_flags(case.queries, thresholds)
    source_bins = index.batch_source_bins_exact(
        case.queries,
        far_threshold=bins.far_threshold,
        near_threshold=bins.near_threshold,
    )

    false_negatives = 0
    false_positives = 0
    for top, flag_row in zip(exact_tops, flags, strict=True):
        for threshold in thresholds:
            expected = top.score >= threshold
            observed = flag_row[threshold]
            false_negatives += int(expected and not observed)
            false_positives += int(observed and not expected)

    bin_mismatches = sum(
        _source_bin(top.exact, top.score, bins) != observed
        for top, observed in zip(exact_tops, source_bins, strict=True)
    )
    return {
        "name": case.name,
        "train_size": len(case.train),
        "query_size": len(case.queries),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "bin_mismatches": bin_mismatches,
    }


def evaluate_pair_case(*, thresholds: tuple[float, ...]) -> dict[str, object]:
    source_index = NgramInvertedIndex.build(
        [
            "shared segment with many common words and token one",
            "shared segment with many common words and token two",
            "unrelated source text",
        ],
        index_config=IndexConfig(mode="native_exact"),
    )
    target_index = NgramInvertedIndex.build(
        [
            "unrelated target text",
            "the quick brown fox jumps over the lazy cat",
            "the quick brown fox jumps over the lazy dog",
        ],
        index_config=IndexConfig(mode="native_exact"),
    )
    observed = source_index.pair_threshold_flags_exact(
        target_index,
        "shared segment with many common words and token one",
        ["the quick brown fox jumps over the lazy dog"],
        thresholds,
    )
    expected = {"0.30": True, "0.70": True, "0.85": True, "0.95": False}
    expected = {key: value for key, value in expected.items() if key in observed}
    return {
        "name": "same_index_pair_threshold",
        "false_negatives": sum(int(expected[key] and not observed[key]) for key in expected),
        "false_positives": sum(int(observed[key] and not expected[key]) for key in expected),
        "observed": observed,
        "expected": expected,
    }


def build_cases(train_size: int, test_size: int) -> list[ThresholdCase]:
    return [
        _threshold_boundary_case(train_size, test_size),
        _template_common_gram_case(max(1_000, train_size // 2), max(120, test_size // 2)),
        _short_string_case(max(500, train_size // 4), max(100, test_size // 3)),
    ]


def _threshold_boundary_case(train_size: int, test_size: int) -> ThresholdCase:
    train = [
        f"threshold source segment {idx:05d} family {idx % 37} marker {idx % 11}"
        for idx in range(train_size)
    ]
    queries = [
        train[(idx * 17) % train_size]
        if idx % 5 == 0
        else f"threshold source segment heldout {idx:05d} family {idx % 37} marker {idx % 11}"
        for idx in range(test_size)
    ]
    return ThresholdCase("threshold_boundary", train, queries)


def _template_common_gram_case(train_size: int, test_size: int) -> ThresholdCase:
    train = [
        " ".join(
            [
                "shared legal boilerplate translation memory segment",
                f"jurisdiction{idx % 17}",
                f"clause{idx:05d}",
                f"revision{(idx * 29) % 997}",
            ]
        )
        for idx in range(train_size)
    ]
    queries = [
        train[(idx * 13) % train_size]
        if idx % 4 == 0
        else " ".join(
            [
                "shared legal boilerplate translation memory segment",
                f"jurisdiction{idx % 17}",
                f"clause-heldout-{idx:05d}",
                f"revision{(idx * 29) % 997}",
            ]
        )
        for idx in range(test_size)
    ]
    return ThresholdCase("template_common_grams", train, queries)


def _short_string_case(train_size: int, test_size: int) -> ThresholdCase:
    alphabet = ["abc", "abd", "abe", "abf", "xyz", "xya", "xyb"]
    train = [f"{alphabet[idx % len(alphabet)]}{idx % 31:02d}" for idx in range(train_size)]
    queries = [
        train[(idx * 19) % train_size]
        if idx % 3 == 0
        else f"{alphabet[idx % len(alphabet)]}{(idx + 1) % 31:02d}"
        for idx in range(test_size)
    ]
    return ThresholdCase("short_strings", train, queries)


def _approximate_backend_rejected() -> bool:
    index = NgramInvertedIndex.build(
        ["alpha beta gamma", "alpha beta delta"],
        index_config=IndexConfig(mode="native_fast"),
    )
    try:
        index.batch_threshold_flags(["alpha beta"], (0.70,))
    except ApproximationError:
        return True
    return False


def _source_bin(exact: bool, score: float, config: BinConfig) -> str:
    if exact:
        return "source_exact"
    if score >= config.near_threshold:
        return "near"
    if score >= config.far_threshold:
        return "medium"
    return "far"


if __name__ == "__main__":
    raise SystemExit(main())
