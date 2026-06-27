#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass

from tame_mt.config import IndexConfig
from tame_mt.index import NgramInvertedIndex
from tame_mt.json_utils import strict_json_dumps
from tame_mt.native import native_status


@dataclass(frozen=True, slots=True)
class RecallCase:
    name: str
    train: list[str]
    queries: list[str]
    exact_queries: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate approximate fast retrieval against exact retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-size", type=int, default=3_000)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--top1-agreement", type=float, default=0.92)
    parser.add_argument("--mean-score-gap", type=float, default=0.03)
    parser.add_argument("--p95-score-gap", type=float, default=0.15)
    parser.add_argument("--require-native", action="store_true")
    args = parser.parse_args()

    status = native_status()
    if not status.available:
        raise SystemExit(f"native backend is required for this validation: {status.error}")
    exact_mode = "native_exact"
    fast_mode = "native_fast"

    payload = {
        "exact_mode": exact_mode,
        "fast_mode": fast_mode,
        "native_available": status.available,
        "cases": [
            evaluate_case(case, exact_mode=exact_mode, fast_mode=fast_mode)
            for case in build_cases(args.train_size, args.test_size)
        ],
    }
    print(strict_json_dumps(payload, indent=2, sort_keys=True))

    failures: list[str] = []
    for case in payload["cases"]:
        if not isinstance(case, dict):
            raise SystemExit("internal error: malformed case payload")
        if case["exact_match_recall"] != 1.0:
            failures.append(
                f"{case['name']}: exact_match_recall {case['exact_match_recall']:.4f} < 1.0"
            )
        if case["top1_agreement"] < args.top1_agreement:
            failures.append(
                f"{case['name']}: top1_agreement {case['top1_agreement']:.4f} "
                f"< {args.top1_agreement:.4f}"
            )
        if case["mean_score_gap"] > args.mean_score_gap:
            failures.append(
                f"{case['name']}: mean_score_gap {case['mean_score_gap']:.4f} "
                f"> {args.mean_score_gap:.4f}"
            )
        if case["p95_score_gap"] > args.p95_score_gap:
            failures.append(
                f"{case['name']}: p95_score_gap {case['p95_score_gap']:.4f} "
                f"> {args.p95_score_gap:.4f}"
            )
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


def evaluate_case(case: RecallCase, *, exact_mode: str, fast_mode: str) -> dict[str, object]:
    exact = NgramInvertedIndex.build(case.train, index_config=IndexConfig(mode=exact_mode))
    fast = NgramInvertedIndex.build(case.train, index_config=IndexConfig(mode=fast_mode))
    agreements = 0
    gaps: list[float] = []
    for query in case.queries:
        exact_top = exact.query_best(query)
        fast_top = fast.query_best(query)
        if exact_top.index == fast_top.index:
            agreements += 1
        gaps.append(max(0.0, exact_top.score - fast_top.score))

    exact_hits = 0
    for query in case.exact_queries:
        result = fast.query_best(query)
        if result.exact and result.score == 1.0:
            exact_hits += 1

    return {
        "name": case.name,
        "train_size": len(case.train),
        "query_size": len(case.queries),
        "exact_query_size": len(case.exact_queries),
        "exact_match_recall": exact_hits / len(case.exact_queries),
        "top1_agreement": agreements / len(case.queries),
        "mean_score_gap": statistics.fmean(gaps),
        "p95_score_gap": _quantile(gaps, 0.95),
        "max_score_gap": max(gaps),
    }


def build_cases(train_size: int, test_size: int) -> list[RecallCase]:
    return [
        _domain_template_case(train_size, test_size),
        _multilingual_case(max(800, train_size // 3), max(120, test_size // 3)),
        _lexical_family_case(max(1_200, train_size // 2), max(160, test_size // 2)),
        _duplicate_heavy_case(max(1_500, train_size // 2), max(180, test_size // 2)),
        _noisy_perturbation_case(max(1_500, train_size // 2), max(180, test_size // 2)),
    ]


def _domain_template_case(train_size: int, test_size: int) -> RecallCase:
    train = [
        f"domain {idx % 97} source sentence {idx:06d} topic {idx % 17} shared template"
        for idx in range(train_size)
    ]
    queries = [
        train[(idx * 13) % train_size]
        if idx % 4 == 0
        else f"domain {idx % 97} source sentence heldout {idx:06d} topic {idx % 17} template"
        for idx in range(test_size)
    ]
    exact_queries = [train[(idx * 19) % train_size] for idx in range(max(1, test_size // 5))]
    return RecallCase("domain_template", train, queries, exact_queries)


def _multilingual_case(train_size: int, test_size: int) -> RecallCase:
    stems = [
        "mañana será otro día",
        "測試 句子 主題",
        "नमस्ते दुनिया विषय",
        "bonjour le monde sujet",
        "alpha beta gamma topic",
    ]
    train = [f"{stems[idx % len(stems)]} {idx:05d} cluster {idx % 31}" for idx in range(train_size)]
    queries = [
        train[(idx * 7) % train_size]
        if idx % 5 == 0
        else f"{stems[idx % len(stems)]} heldout {idx:05d} cluster {idx % 31}"
        for idx in range(test_size)
    ]
    exact_queries = [train[(idx * 23) % train_size] for idx in range(max(1, test_size // 5))]
    return RecallCase("multilingual", train, queries, exact_queries)


def _lexical_family_case(train_size: int, test_size: int) -> RecallCase:
    train = [
        " ".join(
            [
                f"family{idx % 41}",
                f"branch{idx % 29}",
                f"leaf{idx:05d}",
                f"signal{(idx * 17) % 997}",
                f"tail{idx % 13}",
            ]
        )
        for idx in range(train_size)
    ]
    queries = [
        train[(idx * 11) % train_size]
        if idx % 6 == 0
        else " ".join(
            [
                f"family{idx % 41}",
                f"branch{idx % 29}",
                f"leaf-heldout-{idx:05d}",
                f"signal{(idx * 17) % 997}",
                f"tail{idx % 13}",
            ]
        )
        for idx in range(test_size)
    ]
    exact_queries = [train[(idx * 31) % train_size] for idx in range(max(1, test_size // 5))]
    return RecallCase("lexical_family", train, queries, exact_queries)


def _duplicate_heavy_case(train_size: int, test_size: int) -> RecallCase:
    boilerplates = [
        "copyright notice reusable translation memory segment",
        "common website navigation reusable translation memory segment",
        "product catalog reusable translation memory segment",
    ]
    train = [
        " ".join(
            [
                boilerplates[idx % len(boilerplates)],
                f"region{idx % 23}",
                f"sku{idx:06d}",
                f"variant{(idx * 37) % 997}",
            ]
        )
        for idx in range(train_size)
    ]
    queries = [
        train[(idx * 17) % train_size]
        if idx % 4 == 0
        else " ".join(
            [
                boilerplates[idx % len(boilerplates)],
                f"region{idx % 23}",
                f"sku-heldout-{idx:06d}",
                f"variant{(idx * 37) % 997}",
            ]
        )
        for idx in range(test_size)
    ]
    exact_queries = [train[(idx * 29) % train_size] for idx in range(max(1, test_size // 5))]
    return RecallCase("duplicate_heavy", train, queries, exact_queries)


def _noisy_perturbation_case(train_size: int, test_size: int) -> RecallCase:
    rng = random.Random(17)
    domains = ["medical", "legal", "support", "finance", "education"]
    verbs = ["confirm", "review", "translate", "archive", "publish", "compare"]
    objects = ["record", "form", "message", "notice", "article", "dataset"]
    modifiers = ["urgent", "draft", "regional", "final", "manual", "automated"]
    train = [
        " ".join(
            [
                domains[idx % len(domains)],
                verbs[(idx * 3) % len(verbs)],
                objects[(idx * 5) % len(objects)],
                modifiers[(idx * 7) % len(modifiers)],
                f"case{idx:05d}",
                f"checksum{(idx * 104729) % 999983}",
            ]
        )
        for idx in range(train_size)
    ]
    queries: list[str] = []
    for idx in range(test_size):
        source = train[(idx * 41) % train_size]
        if idx % 5 == 0:
            queries.append(source)
        else:
            queries.append(_perturb_sentence(source, rng))
    exact_queries = [train[(idx * 43) % train_size] for idx in range(max(1, test_size // 5))]
    return RecallCase("noisy_perturbation", train, queries, exact_queries)


def _perturb_sentence(sentence: str, rng: random.Random) -> str:
    tokens = sentence.split()
    if len(tokens) < 3:
        return sentence
    token_index = rng.randrange(1, len(tokens))
    token = tokens[token_index]
    if len(token) <= 3:
        tokens[token_index] = f"{token}x"
    else:
        cut = rng.randrange(1, len(token) - 1)
        tokens[token_index] = f"{token[:cut]}x{token[cut + 1 :]}"
    return " ".join(tokens)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
