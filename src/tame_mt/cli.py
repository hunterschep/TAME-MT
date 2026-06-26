from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tame_mt.api import TameScorer
from tame_mt.config import (
    BinConfig,
    IndexConfig,
    MetricConfig,
    NormalizationConfig,
    ScoreConfig,
    SimilarityConfig,
    TMConfig,
    parse_float_tuple,
    parse_int_tuple,
)
from tame_mt.io import read_lines, write_lines
from tame_mt.report import render_text_report, write_json_report, write_segment_jsonl
from tame_mt.version import __version__


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return args.handler(args)
    except ValueError as exc:
        print(f"tame-mt: error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tame-mt",
        description="Training-aware machine translation evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"tame-mt {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    score_parser = subparsers.add_parser("score", help="run full train-aware MT scoring")
    _add_config_args(score_parser)
    _add_segment_args(score_parser)
    score_parser.add_argument("--train-src", required=True)
    score_parser.add_argument("--train-tgt", required=True)
    score_parser.add_argument("--test-src", required=True)
    score_parser.add_argument("--ref", action="append", required=True)
    score_parser.add_argument("--hyp", required=True)
    score_parser.add_argument("--json-out")
    score_parser.add_argument("--segment-out")
    score_parser.add_argument("--tm-out")
    score_parser.add_argument("--quiet", action="store_true")
    score_parser.add_argument("--verbose", action="store_true")
    score_parser.set_defaults(handler=run_score)

    audit_parser = subparsers.add_parser("audit", help="audit train-test exposure without system outputs")
    _add_config_args(audit_parser)
    _add_segment_args(audit_parser)
    audit_parser.add_argument("--train-src", required=True)
    audit_parser.add_argument("--train-tgt")
    audit_parser.add_argument("--test-src", required=True)
    audit_parser.add_argument("--ref", action="append")
    audit_parser.add_argument("--json-out")
    audit_parser.add_argument("--segment-out")
    audit_parser.add_argument("--quiet", action="store_true")
    audit_parser.add_argument("--verbose", action="store_true")
    audit_parser.set_defaults(handler=run_audit)

    tm_parser = subparsers.add_parser("tm-baseline", help="write nearest-neighbor TM hypotheses")
    _add_config_args(tm_parser)
    tm_parser.add_argument("--train-src", required=True)
    tm_parser.add_argument("--train-tgt", required=True)
    tm_parser.add_argument("--test-src", required=True)
    tm_parser.add_argument("--out", required=True)
    tm_parser.add_argument("--metadata-out")
    tm_parser.set_defaults(handler=run_tm_baseline)
    return parser


def run_score(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    train_src = read_lines(args.train_src)
    train_tgt = read_lines(args.train_tgt)
    test_src = read_lines(args.test_src)
    refs = [read_lines(path) for path in args.ref]
    hyp = read_lines(args.hyp)

    scorer = TameScorer(config)
    result = scorer.evaluate_corpus(train_src, train_tgt, test_src, refs, hyp)
    if args.tm_out:
        write_lines(args.tm_out, result.tm_hyp)
    if args.json_out:
        write_json_report(args.json_out, result.report)
    if args.segment_out:
        _warn_neighbor_text(args)
        write_segment_jsonl(
            args.segment_out,
            result.exposures,
            result.tm_results,
            train_src=train_src,
            train_tgt=train_tgt,
            test_src=test_src,
            refs=refs,
            hyp=hyp,
            include_neighbor_text=args.include_neighbor_text,
            include_source_text=args.include_source_text,
            include_reference_text=args.include_reference_text,
            include_hyp_text=args.include_hyp_text,
        )
    if not args.quiet:
        print(render_text_report(result.report))
    return 0


def run_audit(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    train_src = read_lines(args.train_src)
    train_tgt = read_lines(args.train_tgt) if args.train_tgt else None
    test_src = read_lines(args.test_src)
    refs = [read_lines(path) for path in args.ref] if args.ref else None

    scorer = TameScorer(config)
    result = scorer.evaluate_corpus(train_src, train_tgt, test_src, refs, hyp=None)
    if args.json_out:
        write_json_report(args.json_out, result.report)
    if args.segment_out:
        _warn_neighbor_text(args)
        write_segment_jsonl(
            args.segment_out,
            result.exposures,
            result.tm_results,
            train_src=train_src,
            train_tgt=train_tgt,
            test_src=test_src,
            refs=refs,
            hyp=None,
            include_neighbor_text=args.include_neighbor_text,
            include_source_text=args.include_source_text,
            include_reference_text=args.include_reference_text,
            include_hyp_text=False,
        )
    if not args.quiet:
        print(render_text_report(result.report))
    return 0


def run_tm_baseline(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    train_src = read_lines(args.train_src)
    train_tgt = read_lines(args.train_tgt)
    test_src = read_lines(args.test_src)

    scorer = TameScorer(config)
    result = scorer.evaluate_corpus(train_src, train_tgt, test_src, refs=None, hyp=None)
    write_lines(args.out, result.tm_hyp)
    if args.metadata_out:
        with Path(args.metadata_out).open("w", encoding="utf-8") as handle:
            for item in result.tm_results:
                handle.write(
                    json.dumps(
                        {
                            "index": item.index,
                            "tm_source_index": item.tm_source_index,
                            "tm_source_similarity": item.tm_source_similarity,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return 0


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metrics", nargs="+", default=["bleu", "chrf"], choices=["bleu", "chrf"])
    parser.add_argument("--ngram-orders", default="3,4,5")
    parser.add_argument("--far-threshold", type=float, default=0.30)
    parser.add_argument("--near-threshold", type=float, default=0.70)
    parser.add_argument("--leak-thresholds", default="0.70,0.85,0.95")
    parser.add_argument("--pair-k", type=int, default=50)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip-diacritics", action="store_true")
    parser.add_argument("--normalize-punctuation", action="store_true")
    parser.add_argument("--bleu-tokenize", default="13a")
    parser.add_argument("--bleu-lowercase", action="store_true")
    parser.add_argument("--chrf-word-order", type=int, default=2)


def _add_segment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--include-neighbor-text", action="store_true")
    parser.add_argument("--include-source-text", action="store_true")
    parser.add_argument("--include-reference-text", action="store_true")
    parser.add_argument("--include-hyp-text", action="store_true")


def _config_from_args(args: argparse.Namespace) -> ScoreConfig:
    ngram_orders = parse_int_tuple(args.ngram_orders)
    leak_thresholds = parse_float_tuple(args.leak_thresholds)
    if args.pair_k <= 0:
        raise ValueError("--pair-k must be positive")
    if args.far_threshold < 0 or args.near_threshold < 0 or args.far_threshold > args.near_threshold:
        raise ValueError("--far-threshold must be non-negative and no larger than --near-threshold")
    return ScoreConfig(
        metrics=tuple(metric.lower() for metric in args.metrics),
        normalization=NormalizationConfig(
            lowercase=args.lowercase,
            strip_diacritics=args.strip_diacritics,
            normalize_punctuation=args.normalize_punctuation,
        ),
        similarity=SimilarityConfig(ngram_orders=ngram_orders),
        index=IndexConfig(topk=args.pair_k),
        bins=BinConfig(
            far_threshold=args.far_threshold,
            near_threshold=args.near_threshold,
            leak_thresholds=leak_thresholds,
        ),
        tm=TMConfig(),
        metric=MetricConfig(
            bleu_tokenize=args.bleu_tokenize,
            bleu_lowercase=args.bleu_lowercase,
            chrf_word_order=args.chrf_word_order,
        ),
    )


def _warn_neighbor_text(args: argparse.Namespace) -> None:
    if args.include_neighbor_text:
        print(
            "Warning: --include-neighbor-text may write raw training text to the segment report.",
            file=sys.stderr,
        )
