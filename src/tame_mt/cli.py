from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import sacrebleu

from tame_mt.api import TameScorer
from tame_mt.artifacts import read_segment_jsonl
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
from tame_mt.exceptions import TameMTError
from tame_mt.io import ensure_parent_dir, open_text, read_lines, write_lines
from tame_mt.native import native_status
from tame_mt.persistence import inspect_index_bundle, load_index_bundle, save_index_bundle
from tame_mt.report import render_text_report, write_json_report, write_segment_jsonl
from tame_mt.version import __version__


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        handler = cast(Callable[[argparse.Namespace], int], args.handler)
        return handler(args)
    except TameMTError as exc:
        print(f"tame-mt: error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"tame-mt: file error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("tame-mt: interrupted", file=sys.stderr)
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tame-mt",
        description="Training-aware machine translation evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"tame-mt {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="show install, backend, and dependency status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    doctor_parser.set_defaults(handler=run_doctor)

    score_parser = subparsers.add_parser(
        "score",
        help="run full train-aware MT scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_args(score_parser)
    _add_segment_args(score_parser, include_hyp=True)
    score_parser.add_argument("--index", help="load a reusable training index bundle")
    score_parser.add_argument("--train-src", help="training source text file")
    score_parser.add_argument("--train-tgt", help="training target text file")
    score_parser.add_argument("--test-src", required=True, help="test source text file")
    score_parser.add_argument(
        "--ref", action="append", required=True, help="reference file; repeat for multi-ref"
    )
    score_parser.add_argument("--hyp", required=True, help="system hypothesis text file")
    score_parser.add_argument("--json-out", help="write the full JSON report")
    score_parser.add_argument("--segment-out", help="write per-segment JSONL diagnostics")
    score_parser.add_argument("--tm-out", help="write translation-memory baseline hypotheses")
    score_parser.add_argument(
        "--quiet", action="store_true", help="suppress human-readable stdout report"
    )
    score_parser.add_argument(
        "--verbose", action="store_true", help="reserved for future progress reporting"
    )
    score_parser.set_defaults(handler=run_score)

    audit_parser = subparsers.add_parser(
        "audit",
        help="audit train-test exposure without system outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_args(audit_parser)
    _add_segment_args(audit_parser, include_hyp=False)
    audit_parser.add_argument("--index", help="load a reusable training index bundle")
    audit_parser.add_argument("--train-src", help="training source text file")
    audit_parser.add_argument("--train-tgt", help="training target text file")
    audit_parser.add_argument("--test-src", required=True, help="test source text file")
    audit_parser.add_argument("--ref", action="append", help="reference file; repeat for multi-ref")
    audit_parser.add_argument("--json-out", help="write the full JSON report")
    audit_parser.add_argument("--segment-out", help="write per-segment JSONL diagnostics")
    audit_parser.add_argument(
        "--quiet", action="store_true", help="suppress human-readable stdout report"
    )
    audit_parser.add_argument(
        "--verbose", action="store_true", help="reserved for future progress reporting"
    )
    audit_parser.set_defaults(handler=run_audit)

    cached_parser = subparsers.add_parser(
        "score-cached",
        help="score a hypothesis using cached segment diagnostics from a prior audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_args(cached_parser)
    cached_parser.add_argument("--segment-in", required=True, help="segment JSONL from audit/score")
    cached_parser.add_argument("--ref", action="append", required=True, help="reference file")
    cached_parser.add_argument("--hyp", required=True, help="system hypothesis text file")
    cached_parser.add_argument(
        "--num-train", type=int, required=True, help="training segment count"
    )
    cached_parser.add_argument("--json-out", help="write the full JSON report")
    cached_parser.add_argument(
        "--quiet", action="store_true", help="suppress human-readable stdout report"
    )
    cached_parser.set_defaults(handler=run_score_cached)

    index_parser = subparsers.add_parser(
        "index",
        help="build or inspect reusable training index bundles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    index_subparsers = index_parser.add_subparsers(dest="index_command", required=True)
    index_build_parser = index_subparsers.add_parser(
        "build",
        help="build a reusable native training index bundle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_args(index_build_parser)
    index_build_parser.add_argument("--train-src", required=True, help="training source text file")
    index_build_parser.add_argument("--train-tgt", help="training target text file")
    index_build_parser.add_argument("--out", required=True, help="write index bundle")
    index_build_parser.add_argument(
        "--quiet", action="store_true", help="suppress index-build summary"
    )
    index_build_parser.set_defaults(handler=run_index_build)

    index_inspect_parser = index_subparsers.add_parser(
        "inspect",
        help="print index bundle metadata without loading native indexes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    index_inspect_parser.add_argument("path", help="index bundle to inspect")
    index_inspect_parser.set_defaults(handler=run_index_inspect)

    tm_parser = subparsers.add_parser(
        "tm-baseline",
        help="write nearest-neighbor TM hypotheses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_args(tm_parser)
    tm_parser.add_argument("--train-src", required=True, help="training source text file")
    tm_parser.add_argument("--train-tgt", required=True, help="training target text file")
    tm_parser.add_argument("--test-src", required=True, help="test source text file")
    tm_parser.add_argument(
        "--out", required=True, help="write translation-memory baseline hypotheses"
    )
    tm_parser.add_argument("--metadata-out", help="write JSONL nearest-neighbor metadata")
    tm_parser.set_defaults(handler=run_tm_baseline)
    return parser


def run_doctor(args: argparse.Namespace) -> int:
    _ = args
    status = native_status()
    lines = [
        f"TAME-MT: {__version__}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"SacreBLEU: {sacrebleu.__version__}",
    ]
    if status.available:
        lines.extend(
            [
                "Native backend: available",
                f"Native backend version: {status.version}",
                "Default backend: auto -> native_exact/native_fast",
            ]
        )
    else:
        lines.extend(
            [
                "Native backend: unavailable",
                f"Reason: {status.error}",
                "Default backend: auto -> python_exact/python_fast fallback",
            ]
        )
    print("\n".join(lines))
    return 0


def run_score(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    test_src = read_lines(args.test_src)
    refs = [read_lines(path) for path in args.ref]
    hyp = read_lines(args.hyp)

    scorer = TameScorer(config)
    if args.index:
        bundle = load_index_bundle(args.index, config)
        if bundle.train_tgt is None:
            raise TameMTError("indexed train.tgt is required for score mode")
        train_src = bundle.train_src
        train_tgt = bundle.train_tgt
        result = scorer.evaluate_index_bundle(bundle, test_src, refs, hyp)
    else:
        train_src_path = _required_arg(args, "train_src", "--train-src")
        train_tgt_path = _required_arg(args, "train_tgt", "--train-tgt")
        train_src = read_lines(train_src_path)
        train_tgt = read_lines(train_tgt_path)
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
    test_src = read_lines(args.test_src)
    refs = [read_lines(path) for path in args.ref] if args.ref else None

    scorer = TameScorer(config)
    if args.index:
        bundle = load_index_bundle(args.index, config)
        train_src = bundle.train_src
        train_tgt = bundle.train_tgt
        result = scorer.evaluate_index_bundle(bundle, test_src, refs, hyp=None)
    else:
        train_src_path = _required_arg(args, "train_src", "--train-src")
        train_src = read_lines(train_src_path)
        train_tgt = read_lines(args.train_tgt) if args.train_tgt else None
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


def run_score_cached(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    exposures, tm_results = read_segment_jsonl(args.segment_in)
    refs = [read_lines(path) for path in args.ref]
    hyp = read_lines(args.hyp)
    scorer = TameScorer(config)
    report = scorer.score_from_artifacts(
        exposures=exposures,
        tm_results=tm_results,
        refs=refs,
        hyp=hyp,
        num_train=args.num_train,
    )
    if args.json_out:
        write_json_report(args.json_out, report)
    if not args.quiet:
        print(render_text_report(report))
    return 0


def run_index_build(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    train_src = read_lines(args.train_src)
    train_tgt = read_lines(args.train_tgt) if args.train_tgt else None
    bundle = save_index_bundle(args.out, train_src, train_tgt, config)
    if not args.quiet:
        print(
            "\n".join(
                [
                    f"Index bundle:    {args.out}",
                    f"Train segments:  {len(bundle.train_src):,}",
                    f"Source backend:  {bundle.source_index.backend_info.resolved_mode}",
                    "Target backend:  "
                    + (
                        bundle.target_index.backend_info.resolved_mode
                        if bundle.target_index is not None
                        else "not included"
                    ),
                    "Privacy:         stores raw training text and normalized exact-match keys",
                ]
            )
        )
    return 0


def run_index_inspect(args: argparse.Namespace) -> int:
    manifest = inspect_index_bundle(args.path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
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
        metadata_path = Path(args.metadata_out)
        ensure_parent_dir(metadata_path)
        with open_text(metadata_path, "w") as handle:
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
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["bleu", "chrf"],
        help="metrics to report; accepts space or comma separation",
    )
    parser.add_argument(
        "--ngram-orders", default="3,4,5", help="comma-separated character n-gram orders"
    )
    parser.add_argument(
        "--far-threshold", type=float, default=0.30, help="upper bound for source-far examples"
    )
    parser.add_argument(
        "--near-threshold", type=float, default=0.70, help="lower bound for source-near examples"
    )
    parser.add_argument(
        "--leak-thresholds", default="0.70,0.85,0.95", help="comma-separated exposure thresholds"
    )
    parser.add_argument(
        "--pair-k", type=int, default=50, help="top-k source/target candidates for pair reranking"
    )
    parser.add_argument(
        "--index-mode",
        choices=[
            "auto",
            "inverted_exact",
            "inverted_fast",
            "python_exact",
            "python_fast",
            "native_exact",
            "native_fast",
        ],
        default="auto",
        help="nearest-neighbor retrieval mode",
    )
    parser.add_argument(
        "--auto-exact-cutoff",
        type=int,
        default=5_000,
        help="auto mode uses exact retrieval up to this training size",
    )
    parser.add_argument(
        "--candidate-gram-limit",
        type=int,
        default=8,
        help="fast mode: rare query grams used for candidate generation",
    )
    parser.add_argument(
        "--posting-limit",
        type=int,
        default=500,
        help="fast mode: maximum posting entries read per selected query gram",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=3_000,
        help="fast mode: maximum candidate segments generated per query",
    )
    parser.add_argument(
        "--rerank-limit",
        type=int,
        default=1_000,
        help="fast mode: approximate candidates kept for exact Jaccard reranking",
    )
    parser.add_argument(
        "--min-bin-size-warning", type=int, default=30, help="warn when a bin has fewer segments"
    )
    parser.add_argument(
        "--tm-zero-policy",
        choices=["empty", "nearest"],
        default="empty",
        help="TM output when no source grams overlap",
    )
    parser.add_argument(
        "--lowercase", action="store_true", help="case-fold text before similarity computation"
    )
    parser.add_argument(
        "--strip-diacritics",
        action="store_true",
        help="strip diacritics before similarity computation",
    )
    parser.add_argument(
        "--normalize-punctuation",
        action="store_true",
        help="normalize common punctuation before similarity computation",
    )
    parser.add_argument("--bleu-tokenize", default="13a", help="SacreBLEU tokenization setting")
    parser.add_argument(
        "--bleu-lowercase", action="store_true", help="lowercase for SacreBLEU BLEU scoring"
    )
    parser.add_argument(
        "--chrf-word-order", type=int, default=2, help="SacreBLEU chrF word-order setting"
    )


def _add_segment_args(parser: argparse.ArgumentParser, *, include_hyp: bool) -> None:
    parser.add_argument(
        "--include-neighbor-text",
        action="store_true",
        help="include raw nearest-neighbor training text in JSONL",
    )
    parser.add_argument(
        "--include-source-text", action="store_true", help="include raw test source text in JSONL"
    )
    parser.add_argument(
        "--include-reference-text", action="store_true", help="include raw reference text in JSONL"
    )
    if include_hyp:
        parser.add_argument(
            "--include-hyp-text",
            action="store_true",
            help="include raw system hypothesis text in JSONL",
        )


def _config_from_args(args: argparse.Namespace) -> ScoreConfig:
    ngram_orders = parse_int_tuple(args.ngram_orders)
    leak_thresholds = parse_float_tuple(args.leak_thresholds)
    metrics = _parse_metrics(args.metrics)
    return ScoreConfig(
        metrics=metrics,
        normalization=NormalizationConfig(
            lowercase=args.lowercase,
            strip_diacritics=args.strip_diacritics,
            normalize_punctuation=args.normalize_punctuation,
        ),
        similarity=SimilarityConfig(ngram_orders=ngram_orders),
        index=IndexConfig(
            mode=args.index_mode,
            topk=args.pair_k,
            auto_exact_cutoff=args.auto_exact_cutoff,
            candidate_gram_limit=args.candidate_gram_limit,
            posting_limit=args.posting_limit,
            max_candidates=args.max_candidates,
            rerank_limit=args.rerank_limit,
        ),
        bins=BinConfig(
            far_threshold=args.far_threshold,
            near_threshold=args.near_threshold,
            leak_thresholds=leak_thresholds,
            min_bin_size_warning=args.min_bin_size_warning,
        ),
        tm=TMConfig(zero_policy=args.tm_zero_policy),
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


def _required_arg(args: argparse.Namespace, attr: str, flag: str) -> str:
    value = getattr(args, attr)
    if value is None:
        raise TameMTError(f"{flag} is required unless --index is provided")
    return str(value)


def _parse_metrics(values: list[str]) -> tuple[str, ...]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip().lower() for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(parsed))
