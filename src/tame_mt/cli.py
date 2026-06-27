from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import cast

import sacrebleu

from tame_mt.api import EvaluationResult, TameScorer
from tame_mt.approx_validation import validate_approximate_run
from tame_mt.artifacts import (
    build_segment_fingerprints,
    read_segment_jsonl,
    read_segment_metadata,
    validate_segment_metadata,
)
from tame_mt.config import (
    BinConfig,
    IndexConfig,
    MetricConfig,
    NormalizationConfig,
    RetrievalConfig,
    ScoreConfig,
    SimilarityConfig,
    TMConfig,
    parse_float_tuple,
    parse_int_tuple,
)
from tame_mt.exceptions import TameMTError
from tame_mt.io import ensure_parent_dir, open_text, read_lines, write_lines
from tame_mt.json_utils import strict_json_dumps
from tame_mt.native import native_status, native_thread_count
from tame_mt.performance import build_performance_metadata
from tame_mt.persistence import (
    MAX_BUNDLE_LOAD_BYTES,
    inspect_index_bundle,
    load_index_bundle,
    save_index_bundle,
    verify_index_bundle,
)
from tame_mt.report import (
    render_text_report,
    segment_metadata_path,
    write_json_report,
    write_segment_jsonl,
    write_segment_metadata,
)
from tame_mt.schema import SegmentExposure, SegmentTMResult, TameReport
from tame_mt.version import __version__


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        handler = cast(Callable[[argparse.Namespace], int], args.handler)
        _start_profile(args)
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
    _add_index_load_args(score_parser)
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
        "--verbose", action="store_true", help="write stage timing details to stderr"
    )
    _add_profile_arg(score_parser)
    score_parser.set_defaults(handler=run_score)

    audit_parser = subparsers.add_parser(
        "audit",
        help="audit train-test exposure without system outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_args(audit_parser)
    _add_segment_args(audit_parser, include_hyp=False)
    audit_parser.add_argument("--index", help="load a reusable training index bundle")
    _add_index_load_args(audit_parser)
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
        "--verbose", action="store_true", help="write stage timing details to stderr"
    )
    _add_profile_arg(audit_parser)
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
    cached_parser.add_argument("--num-train", type=int, help="training segment count")
    _add_cached_validation_args(cached_parser)
    cached_parser.add_argument("--json-out", help="write the full JSON report")
    cached_parser.add_argument(
        "--quiet", action="store_true", help="suppress human-readable stdout report"
    )
    cached_parser.add_argument(
        "--verbose", action="store_true", help="write stage timing details to stderr"
    )
    _add_profile_arg(cached_parser)
    cached_parser.set_defaults(handler=run_score_cached)

    cached_batch_parser = subparsers.add_parser(
        "score-cached-batch",
        help="score multiple hypotheses using one cached segment diagnostic file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_args(cached_batch_parser)
    cached_batch_parser.add_argument(
        "--segment-in", required=True, help="segment JSONL from audit/score"
    )
    cached_batch_parser.add_argument("--ref", action="append", required=True, help="reference file")
    cached_batch_parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="named system hypothesis file; repeat for multiple systems",
    )
    cached_batch_parser.add_argument("--num-train", type=int, help="training segment count")
    _add_cached_validation_args(cached_batch_parser)
    cached_batch_parser.add_argument(
        "--json-out-dir", required=True, help="directory for per-system JSON reports"
    )
    cached_batch_parser.add_argument(
        "--quiet", action="store_true", help="suppress human-readable stdout summary"
    )
    cached_batch_parser.add_argument(
        "--verbose", action="store_true", help="write stage timing details to stderr"
    )
    _add_profile_arg(cached_batch_parser)
    cached_batch_parser.set_defaults(handler=run_score_cached_batch)

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
    index_build_parser.add_argument(
        "--verbose", action="store_true", help="write stage timing details to stderr"
    )
    _add_profile_arg(index_build_parser)
    index_build_parser.set_defaults(handler=run_index_build)

    index_inspect_parser = index_subparsers.add_parser(
        "inspect",
        help="print index bundle metadata without loading native indexes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    index_inspect_parser.add_argument("path", help="index bundle to inspect")
    index_inspect_parser.set_defaults(handler=run_index_inspect)

    index_verify_parser = index_subparsers.add_parser(
        "verify",
        help="verify index bundle shape, hashes, and native-index invariants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    index_verify_parser.add_argument("path", help="index bundle to verify")
    index_verify_parser.add_argument("--train-src", help="optional training source file to verify")
    index_verify_parser.add_argument("--train-tgt", help="optional training target file to verify")
    _add_index_load_args(index_verify_parser)
    index_verify_parser.add_argument(
        "--json", action="store_true", help="print machine-readable verification summary"
    )
    index_verify_parser.set_defaults(handler=run_index_verify)

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
    tm_parser.add_argument(
        "--verbose", action="store_true", help="write stage timing details to stderr"
    )
    _add_profile_arg(tm_parser)
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
                f"Rayon threads: {native_thread_count()}",
                "Default retrieval: exact -> native_exact",
            ]
        )
    else:
        lines.extend(
            [
                "Native backend: unavailable",
                f"Reason: {status.error}",
                "Default retrieval: exact -> error (native backend required)",
                "Repair editable installs: python -m pip install --force-reinstall --no-deps -e .",
            ]
        )
    print("\n".join(lines))
    return 0


def run_score(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    with _timed_step(args, "read evaluation inputs"):
        test_src = read_lines(args.test_src)
        refs = [read_lines(path) for path in args.ref]
        hyp = read_lines(args.hyp)

    scorer = TameScorer(config)
    if args.index:
        with _timed_step(args, "load index bundle"):
            bundle = load_index_bundle(
                args.index,
                config,
                max_load_bytes=args.max_index_load_bytes,
            )
        if bundle.train_tgt is None:
            raise TameMTError("indexed train.tgt is required for score mode")
        train_src = bundle.train_src
        train_tgt = bundle.train_tgt
        with _timed_step(args, "evaluate indexed corpus"):
            result = scorer.evaluate_index_bundle(bundle, test_src, refs, hyp)
    else:
        train_src_path = _required_arg(args, "train_src", "--train-src")
        train_tgt_path = _required_arg(args, "train_tgt", "--train-tgt")
        with _timed_step(args, "read training inputs"):
            train_src = read_lines(train_src_path)
            train_tgt = read_lines(train_tgt_path)
        with _timed_step(args, "evaluate corpus"):
            result = scorer.evaluate_corpus(train_src, train_tgt, test_src, refs, hyp)
    _maybe_validate_approximate_run(
        args,
        config=config,
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        refs=refs,
        result=result,
    )
    _attach_report_performance(args, result.report)
    with _timed_step(args, "write outputs"):
        if args.tm_out:
            write_lines(args.tm_out, result.tm_hyp)
        if args.json_out:
            write_json_report(args.json_out, result.report)
        if args.segment_out:
            _warn_neighbor_text(args)
            include_tm_text = not args.no_tm_text_in_segments
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
                include_tm_text=include_tm_text,
            )
            write_segment_metadata(
                segment_metadata_path(args.segment_out),
                result.report,
                fingerprints=build_segment_fingerprints(
                    config,
                    train_src=train_src,
                    train_tgt=train_tgt,
                    test_src=test_src,
                    refs=refs,
                    tm_results=result.tm_results,
                ),
                tm_text_included=include_tm_text,
                neighbor_text_included=args.include_neighbor_text,
            )
    _write_profile_json(args, [result.report])
    if not args.quiet:
        print(render_text_report(result.report))
    return 0


def run_audit(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    with _timed_step(args, "read evaluation inputs"):
        test_src = read_lines(args.test_src)
        refs = [read_lines(path) for path in args.ref] if args.ref else None

    scorer = TameScorer(config)
    if args.index:
        with _timed_step(args, "load index bundle"):
            bundle = load_index_bundle(
                args.index,
                config,
                max_load_bytes=args.max_index_load_bytes,
            )
        train_src = bundle.train_src
        train_tgt = bundle.train_tgt
        with _timed_step(args, "evaluate indexed corpus"):
            result = scorer.evaluate_index_bundle(bundle, test_src, refs, hyp=None)
    else:
        train_src_path = _required_arg(args, "train_src", "--train-src")
        with _timed_step(args, "read training inputs"):
            train_src = read_lines(train_src_path)
            train_tgt = read_lines(args.train_tgt) if args.train_tgt else None
        with _timed_step(args, "evaluate corpus"):
            result = scorer.evaluate_corpus(train_src, train_tgt, test_src, refs, hyp=None)
    _maybe_validate_approximate_run(
        args,
        config=config,
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        refs=refs,
        result=result,
    )
    _attach_report_performance(args, result.report)
    with _timed_step(args, "write outputs"):
        if args.json_out:
            write_json_report(args.json_out, result.report)
        if args.segment_out:
            _warn_neighbor_text(args)
            include_tm_text = not args.no_tm_text_in_segments
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
                include_tm_text=include_tm_text,
            )
            write_segment_metadata(
                segment_metadata_path(args.segment_out),
                result.report,
                fingerprints=build_segment_fingerprints(
                    config,
                    train_src=train_src,
                    train_tgt=train_tgt,
                    test_src=test_src,
                    refs=refs,
                    tm_results=result.tm_results,
                ),
                tm_text_included=include_tm_text,
                neighbor_text_included=args.include_neighbor_text,
            )
    _write_profile_json(args, [result.report])
    if not args.quiet:
        print(render_text_report(result.report))
    return 0


def run_score_cached(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    _reject_cached_approx_validation(args)
    with _timed_step(args, "read cached inputs"):
        exposures, tm_results = read_segment_jsonl(args.segment_in)
        metadata = read_segment_metadata(args.segment_in)
        refs = [read_lines(path) for path in args.ref]
        hyp = read_lines(args.hyp)
        num_train = _resolve_cached_num_train(args.num_train, metadata, args)
        _validate_cached_metadata(
            metadata,
            config=config,
            num_train=num_train,
            exposures=exposures,
            tm_results=tm_results,
            refs=refs,
            args=args,
        )
        artifact_backend = _artifact_backend_from_metadata(metadata)
    scorer = TameScorer(config)
    with _timed_step(args, "score cached hypothesis"):
        report = scorer.score_from_artifacts(
            exposures=exposures,
            tm_results=tm_results,
            refs=refs,
            hyp=hyp,
            num_train=num_train,
            artifact_backend=artifact_backend,
        )
    _attach_report_performance(args, report)
    with _timed_step(args, "write outputs"):
        if args.json_out:
            write_json_report(args.json_out, report)
    _write_profile_json(args, [report])
    if not args.quiet:
        print(render_text_report(report))
    return 0


def run_score_cached_batch(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    _reject_cached_approx_validation(args)
    with _timed_step(args, "read cached inputs"):
        exposures, tm_results = read_segment_jsonl(args.segment_in)
        metadata = read_segment_metadata(args.segment_in)
        refs = [read_lines(path) for path in args.ref]
        systems = _read_system_specs(args.system)
        num_train = _resolve_cached_num_train(args.num_train, metadata, args)
        _validate_cached_metadata(
            metadata,
            config=config,
            num_train=num_train,
            exposures=exposures,
            tm_results=tm_results,
            refs=refs,
            args=args,
        )
        artifact_backend = _artifact_backend_from_metadata(metadata)
    scorer = TameScorer(config)
    with _timed_step(args, "score cached systems"):
        reports = scorer.score_many_from_artifacts(
            exposures=exposures,
            tm_results=tm_results,
            refs=refs,
            systems=systems,
            num_train=num_train,
            artifact_backend=artifact_backend,
        )
        for report in reports.values():
            _attach_report_performance(args, report)
    with _timed_step(args, "write outputs"):
        output_paths = _write_batch_reports(args.json_out_dir, reports)
    _write_profile_json(args, list(reports.values()))
    if not args.quiet:
        print("\n".join(f"{system_name}\t{path}" for system_name, path in output_paths.items()))
    return 0


def run_index_build(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    with _timed_step(args, "read training inputs"):
        train_src = read_lines(args.train_src)
        train_tgt = read_lines(args.train_tgt) if args.train_tgt else None
    with _timed_step(args, "build index bundle"):
        bundle = save_index_bundle(args.out, train_src, train_tgt, config)
    _write_profile_json(
        args,
        [],
        backend=bundle.source_index.backend_info.name,
        index_reused=False,
    )
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
    print(strict_json_dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def run_index_verify(args: argparse.Namespace) -> int:
    train_src = read_lines(args.train_src) if args.train_src else None
    train_tgt = read_lines(args.train_tgt) if args.train_tgt else None
    verification = verify_index_bundle(
        args.path,
        train_src=train_src,
        train_tgt=train_tgt,
        max_load_bytes=args.max_index_load_bytes,
    )
    payload = verification.to_dict()
    if args.json:
        print(strict_json_dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            "\n".join(
                [
                    f"Index bundle verified: {args.path}",
                    f"Train segments:        {payload['num_train']:,}",
                    f"Has target:            {payload['has_target']}",
                    "Source backend:        "
                    + str(payload["source_backend"].get("resolved_mode", "unknown")),
                    "Target backend:        "
                    + (
                        str(payload["target_backend"].get("resolved_mode", "none"))
                        if isinstance(payload.get("target_backend"), dict)
                        else "none"
                    ),
                    f"Members checked:       {len(verification.checked_members)}",
                    f"Hashes checked:        {len(verification.checked_hashes)}",
                    f"Native indexes checked:{len(verification.checked_native_indexes)}",
                ]
            )
        )
    return 0


def run_tm_baseline(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    with _timed_step(args, "read inputs"):
        train_src = read_lines(args.train_src)
        train_tgt = read_lines(args.train_tgt)
        test_src = read_lines(args.test_src)

    scorer = TameScorer(config)
    with _timed_step(args, "evaluate tm baseline"):
        result = scorer.evaluate_corpus(train_src, train_tgt, test_src, refs=None, hyp=None)
    _maybe_validate_approximate_run(
        args,
        config=config,
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        refs=None,
        result=result,
    )
    _attach_report_performance(args, result.report)
    with _timed_step(args, "write outputs"):
        write_lines(args.out, result.tm_hyp)
        if args.metadata_out:
            metadata_path = Path(args.metadata_out)
            ensure_parent_dir(metadata_path)
            with open_text(metadata_path, "w") as handle:
                for item in result.tm_results:
                    handle.write(
                        strict_json_dumps(
                            {
                                "index": item.index,
                                "tm_source_index": item.tm_source_index,
                                "tm_source_similarity": item.tm_source_similarity,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    _write_profile_json(args, [result.report])
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
        "--batch-size",
        type=int,
        default=8_192,
        help="evaluation segments processed per native retrieval batch",
    )
    parser.add_argument(
        "--retrieval",
        choices=["exact", "guarded", "approx"],
        default="exact",
        help=(
            "metric retrieval semantics; exact is the paper-facing default, approx requires "
            "--allow-approximate"
        ),
    )
    parser.add_argument(
        "--allow-approximate",
        action="store_true",
        help="allow approximate retrieval semantics for speed-critical exploratory runs",
    )
    parser.add_argument(
        "--validate-approx-sample",
        type=int,
        default=0,
        help="sample this many segments and compare approximate retrieval with exact retrieval",
    )
    parser.add_argument(
        "--validate-approx-seed",
        type=int,
        default=13,
        help="random seed for --validate-approx-sample",
    )
    parser.add_argument(
        "--validate-approx-exact-mode",
        choices=["native_exact"],
        default="native_exact",
        help="exact backend used for approximate validation",
    )
    parser.add_argument(
        "--allow-approx-validation-failure",
        action="store_true",
        help="write approximate validation failures as warnings instead of exiting nonzero",
    )
    parser.add_argument(
        "--index-mode",
        choices=["auto", "native_exact", "native_fast"],
        default="auto",
        help=(
            "Rust nearest-neighbor mode; use native_fast only with --retrieval approx "
            "--allow-approximate"
        ),
    )
    parser.add_argument(
        "--auto-exact-cutoff",
        type=int,
        default=5_000,
        help="reserved compatibility setting; default auto retrieval is exact",
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
    parser.add_argument(
        "--no-tm-text-in-segments",
        action="store_true",
        help=(
            "omit TM baseline hypotheses from segment JSONL; privacy-safer diagnostics, "
            "but score-cached cannot use the artifact"
        ),
    )
    if include_hyp:
        parser.add_argument(
            "--include-hyp-text",
            action="store_true",
            help="include raw system hypothesis text in JSONL",
        )


def _add_cached_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-unsafe-no-metadata",
        action="store_true",
        help="allow legacy segment JSONL without sidecar metadata; disables hash validation",
    )
    parser.add_argument(
        "--allow-reference-hash-mismatch",
        action="store_true",
        help=(
            "allow cached scoring with different references of the same length; target and "
            "pair exposure may be stale"
        ),
    )


def _add_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile-json",
        help="write structured performance metadata and command timing JSON",
    )


def _add_index_load_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-index-load-bytes",
        type=int,
        default=MAX_BUNDLE_LOAD_BYTES,
        help=(
            "maximum uncompressed bytes to load from --index; raise only for trusted bundles "
            "on machines with enough memory"
        ),
    )


def _config_from_args(args: argparse.Namespace) -> ScoreConfig:
    ngram_orders = parse_int_tuple(args.ngram_orders)
    leak_thresholds = parse_float_tuple(args.leak_thresholds)
    metrics = _parse_metrics(args.metrics)
    index_mode = args.index_mode
    if args.retrieval == "approx" and index_mode == "auto":
        index_mode = "native_fast"
    return ScoreConfig(
        metrics=metrics,
        normalization=NormalizationConfig(
            lowercase=args.lowercase,
            strip_diacritics=args.strip_diacritics,
            normalize_punctuation=args.normalize_punctuation,
        ),
        similarity=SimilarityConfig(ngram_orders=ngram_orders),
        index=IndexConfig(
            mode=index_mode,
            topk=args.pair_k,
            batch_size=args.batch_size,
            auto_exact_cutoff=args.auto_exact_cutoff,
            candidate_gram_limit=args.candidate_gram_limit,
            posting_limit=args.posting_limit,
            max_candidates=args.max_candidates,
            rerank_limit=args.rerank_limit,
        ),
        retrieval=RetrievalConfig(
            mode=args.retrieval,
            allow_approximate=args.allow_approximate,
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


def _start_profile(args: argparse.Namespace) -> None:
    args._tame_profile_started = perf_counter()
    args._tame_timings = {}


@contextmanager
def _timed_step(args: argparse.Namespace, label: str) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    except Exception:
        elapsed = perf_counter() - started
        _record_timing(args, label, elapsed)
        _emit_timing(args, label, elapsed, failed=True)
        raise
    elapsed = perf_counter() - started
    _record_timing(args, label, elapsed)
    _emit_timing(args, label, elapsed, failed=False)


def _record_timing(args: argparse.Namespace, label: str, seconds: float) -> None:
    timings = getattr(args, "_tame_timings", None)
    if not isinstance(timings, dict):
        return
    key = _timing_key(label)
    existing = timings.get(key, 0.0)
    timings[key] = float(existing) + float(seconds)


def _timing_key(label: str) -> str:
    return "_".join(part for part in label.strip().lower().replace("-", " ").split() if part)


def _profile_timings(args: argparse.Namespace, *, include_total: bool) -> dict[str, float | None]:
    timings = getattr(args, "_tame_timings", {})
    result: dict[str, float | None] = {
        str(key): round(float(value), 6)
        for key, value in sorted(timings.items())
        if isinstance(value, int | float)
    }
    if include_total:
        started = getattr(args, "_tame_profile_started", None)
        if isinstance(started, int | float):
            result["total"] = round(perf_counter() - float(started), 6)
    return result


def _attach_report_performance(args: argparse.Namespace, report: TameReport) -> None:
    report.performance = build_performance_metadata(
        backend=str(report.backend.get("name", "unknown")),
        index_reused=bool(report.backend.get("index_reused")),
        timings_sec=_profile_timings(args, include_total=True),
    )


def _write_profile_json(
    args: argparse.Namespace,
    reports: list[TameReport],
    *,
    backend: str | None = None,
    index_reused: bool | None = None,
) -> None:
    profile_path = getattr(args, "profile_json", None)
    if not profile_path:
        return
    profile_backend = backend or (str(reports[0].backend.get("name", "unknown")) if reports else "")
    profile_index_reused = (
        index_reused
        if index_reused is not None
        else bool(reports[0].backend.get("index_reused"))
        if reports
        else False
    )
    performance = build_performance_metadata(
        backend=profile_backend or "unknown",
        index_reused=profile_index_reused,
        timings_sec=_profile_timings(args, include_total=True),
    )
    payload = {
        "schema_version": "0.1",
        "tame_version": __version__,
        "command": getattr(args, "command", None),
        "performance": asdict(performance),
        "reports": [_report_profile_summary(report) for report in reports],
    }
    output_path = Path(str(profile_path))
    ensure_parent_dir(output_path)
    with open_text(output_path, "w") as handle:
        handle.write(strict_json_dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _report_profile_summary(report: TameReport) -> dict[str, object]:
    return {
        "signature": report.signature,
        "data": {
            "num_train": report.num_train,
            "num_test": report.num_test,
            "num_refs": report.num_refs,
        },
        "backend": report.backend,
        "retrieval": asdict(report.retrieval),
    }


def _maybe_validate_approximate_run(
    args: argparse.Namespace,
    *,
    config: ScoreConfig,
    train_src: list[str],
    train_tgt: list[str] | None,
    test_src: list[str],
    refs: list[list[str]] | None,
    result: EvaluationResult,
) -> None:
    sample_size = int(getattr(args, "validate_approx_sample", 0) or 0)
    if sample_size <= 0:
        return
    if config.retrieval.mode != "approx":
        raise TameMTError("--validate-approx-sample requires --retrieval approx")

    try:
        with _timed_step(args, "validate approximate retrieval"):
            validation = validate_approximate_run(
                train_src=train_src,
                train_tgt=train_tgt,
                test_src=test_src,
                refs=refs,
                approx_exposures=result.exposures,
                approx_tm_results=result.tm_results,
                config=config,
                sample_size=sample_size,
                seed=int(args.validate_approx_seed),
                exact_mode=str(args.validate_approx_exact_mode),
            )
    except ValueError as exc:
        raise TameMTError(f"approximate validation failed: {exc}") from exc

    result.report.approx_validation = validation.payload
    if validation.failures:
        message = "approximate validation failed: " + "; ".join(validation.failures)
        if getattr(args, "allow_approx_validation_failure", False):
            result.report.warnings.append(message)
        else:
            raise TameMTError(message)


def _reject_cached_approx_validation(args: argparse.Namespace) -> None:
    if int(getattr(args, "validate_approx_sample", 0) or 0) > 0:
        raise TameMTError(
            "--validate-approx-sample cannot be used with cached scoring; validate "
            "approximate retrieval when creating the segment diagnostics"
        )


def _emit_timing(
    args: argparse.Namespace,
    label: str,
    seconds: float,
    *,
    failed: bool,
) -> None:
    if not getattr(args, "verbose", False):
        return
    status = "failed after" if failed else "completed in"
    print(f"tame-mt: {label} {status} {seconds:.3f}s", file=sys.stderr)


def _required_arg(args: argparse.Namespace, attr: str, flag: str) -> str:
    value = getattr(args, attr)
    if value is None:
        raise TameMTError(f"{flag} is required unless --index is provided")
    return str(value)


def _validate_num_train_arg(num_train: int) -> None:
    if num_train <= 0:
        raise TameMTError("num_train must be positive")


def _resolve_cached_num_train(
    num_train: int | None,
    metadata: dict[str, object] | None,
    args: argparse.Namespace,
) -> int:
    if metadata is None:
        if not args.allow_unsafe_no_metadata:
            raise TameMTError(
                "segment metadata sidecar is required for cached scoring; pass "
                "--allow-unsafe-no-metadata only for legacy artifacts you trust"
            )
        if num_train is None:
            raise TameMTError("--num-train is required when segment metadata is missing")
        _validate_num_train_arg(num_train)
        return num_train

    metadata_num_train = _metadata_num_train(metadata)
    if num_train is not None:
        _validate_num_train_arg(num_train)
        if num_train != metadata_num_train:
            raise TameMTError(
                f"--num-train {num_train} does not match segment metadata num_train "
                f"{metadata_num_train}"
            )
    _validate_num_train_arg(metadata_num_train)
    return metadata_num_train


def _metadata_num_train(metadata: dict[str, object]) -> int:
    data = metadata.get("data")
    if not isinstance(data, dict):
        raise TameMTError("segment metadata data field is required")
    value = data.get("num_train")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TameMTError("segment metadata data.num_train must be an integer")
    return value


def _validate_cached_metadata(
    metadata: dict[str, object] | None,
    *,
    config: ScoreConfig,
    num_train: int,
    exposures: list[SegmentExposure],
    tm_results: list[SegmentTMResult],
    refs: list[list[str]],
    args: argparse.Namespace,
) -> None:
    if metadata is None:
        return
    validate_segment_metadata(
        metadata,
        config=config,
        num_train=num_train,
        num_test=len(exposures),
        num_refs=len(refs),
        refs=refs,
        tm_results=tm_results,
        allow_reference_hash_mismatch=args.allow_reference_hash_mismatch,
        require_tm_text=True,
    )
    if args.allow_reference_hash_mismatch:
        print(
            "Warning: --allow-reference-hash-mismatch may reuse stale target/pair exposure.",
            file=sys.stderr,
        )


def _artifact_backend_from_metadata(metadata: dict[str, object] | None) -> dict[str, object] | None:
    if metadata is None:
        return None
    backend = metadata.get("backend")
    if isinstance(backend, dict):
        return dict(backend)
    return None


def _parse_metrics(values: list[str]) -> tuple[str, ...]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip().lower() for part in value.split(",") if part.strip())
    return tuple(parsed)


def _read_system_specs(specs: list[str]) -> dict[str, list[str]]:
    systems: dict[str, list[str]] = {}
    for spec in specs:
        name, path = _parse_system_spec(spec)
        if name in systems:
            raise TameMTError(f"duplicate system name: {name}")
        systems[name] = read_lines(path)
    return systems


def _parse_system_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise TameMTError("--system must be formatted as NAME=PATH")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise TameMTError("--system name must be non-empty")
    if not path:
        raise TameMTError("--system path must be non-empty")
    return name, path


def _write_batch_reports(
    output_dir: str,
    reports: dict[str, TameReport],
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths_by_system: dict[str, str] = {}
    used_paths: dict[Path, str] = {}
    for system_name, report in reports.items():
        report_path = output_path / f"{_safe_report_stem(system_name)}.json"
        previous_system = used_paths.get(report_path)
        if previous_system is not None:
            raise TameMTError(
                "system names produce the same report filename: "
                f"{previous_system!r} and {system_name!r}"
            )
        used_paths[report_path] = system_name
        write_json_report(report_path, report)
        paths_by_system[system_name] = str(report_path)
    return paths_by_system


def _safe_report_stem(system_name: str) -> str:
    stem = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in system_name.strip()
    ).strip("._")
    if not stem:
        raise TameMTError(f"system name {system_name!r} does not produce a safe filename")
    return stem
