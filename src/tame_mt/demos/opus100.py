#!/usr/bin/env python3
"""Run a reproducible TAME-MT audit demo on public OPUS-100 corpora.

The script downloads OPUS-100 language-pair tarballs, prepares capped train/test
files, runs TAME-MT audit mode through the Python API, and writes summary files.

The default caps are intentionally modest so the demo can run on a laptop. Do
not treat capped-subset numbers as final benchmark claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from tame_mt import IndexConfig, RetrievalConfig, ScoreConfig, TameScorer
from tame_mt.approx_validation import validate_approximate_run
from tame_mt.io import write_lines
from tame_mt.native import configure_native_threads, native_status
from tame_mt.report import write_json_report
from tame_mt.schema import SCHEMA_VERSION
from tame_mt.version import __version__

DEFAULT_PAIRS = ("de-en", "en-hi", "en-tr", "ar-en")
OPUS100_BASE_URL = "https://object.pouta.csc.fi/OPUS-100/v1.0"
DEFAULT_DOWNLOAD_RETRIES = 3
DEFAULT_DOWNLOAD_TIMEOUT = 60.0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DemoTier:
    pairs: tuple[str, ...]
    train_limit: int
    test_limit: int
    description: str


TIERS = {
    "quick": DemoTier(
        pairs=("de-en",),
        train_limit=10_000,
        test_limit=500,
        description="one-pair laptop smoke demo",
    ),
    "standard": DemoTier(
        pairs=DEFAULT_PAIRS,
        train_limit=50_000,
        test_limit=2_000,
        description="four-pair release demo",
    ),
    "paper": DemoTier(
        pairs=DEFAULT_PAIRS,
        train_limit=1_000_000,
        test_limit=2_000,
        description="larger configurable paper-prep demo",
    ),
}


class DemoError(RuntimeError):
    """Raised for user-facing public-corpus demo failures."""


@dataclass(frozen=True)
class PairData:
    pair: str
    src_lang: str
    tgt_lang: str
    train_src: list[str]
    train_tgt: list[str]
    test_src: list[str]
    test_ref: list[str]


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a TAME-MT audit demo on OPUS-100 public corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    return parser


def add_arguments(parser: argparse.ArgumentParser) -> None:
    tier_group = parser.add_mutually_exclusive_group()
    tier_group.add_argument(
        "--quick",
        action="store_const",
        const="quick",
        dest="tier",
        help="10k train / 500 test / one pair",
    )
    tier_group.add_argument(
        "--standard",
        action="store_const",
        const="standard",
        dest="tier",
        help="50k train / 2k test / four pairs",
    )
    tier_group.add_argument(
        "--paper",
        action="store_const",
        const="paper",
        dest="tier",
        help="larger configurable paper-prep tier",
    )
    parser.add_argument("--pair", action="append", dest="pairs", help="OPUS-100 pair, e.g. de-en")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--retrieval", choices=["exact", "guarded", "approx"], default="exact")
    parser.add_argument("--allow-approximate", action="store_true")
    parser.add_argument("--index-mode", default="auto")
    parser.add_argument("--threads", default="auto")
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--validate-approx-sample", type=int, default=0)
    parser.add_argument("--validate-approx-seed", type=int, default=13)
    parser.add_argument(
        "--validate-approx-exact-mode", choices=["native_exact"], default="native_exact"
    )
    parser.add_argument("--allow-approx-validation-failure", action="store_true")
    parser.add_argument("--profile-json", help="write machine, command, and per-pair timing JSON")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="directory for committed-style CSV and Markdown summaries",
    )
    parser.add_argument(
        "--summary-prefix",
        help="summary basename without extension; defaults to '<tier>.summary'",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=DEFAULT_DOWNLOAD_RETRIES,
        help="download attempts per OPUS-100 pair",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=DEFAULT_DOWNLOAD_TIMEOUT,
        help="per-request download timeout in seconds",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_demo(args, command=[sys.executable, *sys.argv])


def run_demo(args: argparse.Namespace, *, command: list[str] | None = None) -> int:
    resolve_tier_defaults(args)
    args.command = command or ["tame-mt", "demo", "opus100"]
    configure_demo_threads(args.threads)
    status = native_status()
    if args.require_native and not status.available:
        raise SystemExit(f"native backend is required: {status.error}")
    if args.validate_approx_sample > 0 and args.retrieval != "approx":
        raise SystemExit("--validate-approx-sample requires --retrieval approx")

    pairs = tuple(args.pairs)
    output_dir = Path(args.output_dir)
    downloads_dir = output_dir / "downloads"
    prepared_dir = output_dir / "prepared"
    reports_dir = output_dir / "reports"
    summary_dir = Path(args.summary_dir)
    for directory in (downloads_dir, prepared_dir, reports_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    index_mode = args.index_mode
    if args.retrieval == "approx" and index_mode == "auto":
        index_mode = "native_fast"
    scorer = TameScorer(
        ScoreConfig(
            index=IndexConfig(mode=index_mode),
            retrieval=RetrievalConfig(
                mode=args.retrieval,
                allow_approximate=args.allow_approximate,
            ),
        )
    )
    for pair in pairs:
        print(f"Preparing {pair}...")
        try:
            archive = download_pair(
                pair,
                downloads_dir,
                retries=args.download_retries,
                timeout=args.download_timeout,
            )
            data = load_pair_data(archive, pair, args.train_limit, args.test_limit)
        except DemoError as exc:
            raise SystemExit(f"failed to prepare {pair}: {exc}") from exc
        write_prepared_files(data, prepared_dir)
        audit_start = time.perf_counter()
        result = scorer.evaluate_corpus(
            train_src=data.train_src,
            train_tgt=data.train_tgt,
            test_src=data.test_src,
            refs=[data.test_ref],
            hyp=None,
        )
        validation_payload = None
        if args.validate_approx_sample > 0:
            validation = validate_approximate_run(
                train_src=data.train_src,
                train_tgt=data.train_tgt,
                test_src=data.test_src,
                refs=[data.test_ref],
                approx_exposures=result.exposures,
                approx_tm_results=result.tm_results,
                config=scorer.config,
                sample_size=args.validate_approx_sample,
                seed=args.validate_approx_seed,
                exact_mode=args.validate_approx_exact_mode,
            )
            result.report.approx_validation = validation.payload
            validation_payload = validation.payload
            if validation.failures:
                message = "approximate validation failed: " + "; ".join(validation.failures)
                if args.allow_approx_validation_failure:
                    result.report.warnings.append(message)
                else:
                    raise SystemExit(message)
        audit_seconds = time.perf_counter() - audit_start
        report_path = reports_dir / f"{pair}.audit.json"
        write_json_report(report_path, result.report)
        rows.append(summary_row(data, result.report.to_dict(), audit_seconds))
        profiles.append(
            {
                "pair": pair,
                "audit_seconds": audit_seconds,
                "report_path": str(report_path),
                "approx_validation": validation_payload,
                "signature": result.report.signature,
                "backend": result.report.backend,
                "retrieval": asdict(result.report.retrieval),
                "performance": asdict(result.report.performance),
            }
        )

    summary_base = summary_dir / args.summary_prefix
    summary_csv = summary_artifact_path(summary_base, "csv")
    summary_json = summary_artifact_path(summary_base, "json")
    summary_md = summary_artifact_path(summary_base, "md")
    write_summary_csv(summary_csv, rows)
    write_summary_json(summary_json, rows, args=args)
    write_summary_markdown(
        summary_md,
        rows,
        args.train_limit,
        args.test_limit,
        tier=args.tier,
    )
    if args.profile_json:
        write_profile_json(Path(args.profile_json), args=args, profiles=profiles)
    print(f"Wrote {summary_md}")
    return 0


def resolve_tier_defaults(args: argparse.Namespace) -> None:
    tier_name = args.tier or "standard"
    tier = TIERS[tier_name]
    args.tier = tier_name
    if not args.pairs:
        args.pairs = list(tier.pairs)
    if args.train_limit is None:
        args.train_limit = tier.train_limit
    if args.test_limit is None:
        args.test_limit = tier.test_limit
    if args.train_limit <= 0:
        raise SystemExit("--train-limit must be a positive integer")
    if args.test_limit <= 0:
        raise SystemExit("--test-limit must be a positive integer")
    if args.download_retries <= 0:
        raise SystemExit("--download-retries must be a positive integer")
    if args.download_timeout <= 0:
        raise SystemExit("--download-timeout must be positive")
    if not args.output_dir:
        args.output_dir = f"demo_runs/opus100_{tier_name}"
    if not args.summary_dir:
        args.summary_dir = str(Path(args.output_dir) / "summary")
    if not args.summary_prefix:
        args.summary_prefix = f"{tier_name}.summary"


def summary_artifact_path(summary_base: Path, extension: str) -> Path:
    return summary_base.parent / f"{summary_base.name}.{extension}"


def configure_demo_threads(value: str) -> None:
    if value == "auto":
        return
    try:
        requested = int(value)
    except ValueError as exc:
        raise SystemExit("--threads must be 'auto' or a positive integer") from exc
    if requested <= 0:
        raise SystemExit("--threads must be 'auto' or a positive integer")
    try:
        configure_native_threads(requested)
    except Exception as exc:
        raise SystemExit(f"failed to configure native threads: {exc}") from exc


def download_pair(pair: str, downloads_dir: Path, *, retries: int, timeout: float) -> Path:
    archive = downloads_dir / f"opus-100-corpus-{pair}-v1.0.tar.gz"
    if archive.exists() and is_complete_tar(archive):
        return archive
    if archive.exists():
        archive.unlink()
    url = f"{OPUS100_BASE_URL}/opus-100-corpus-{pair}-v1.0.tar.gz"
    partial = archive.with_suffix(archive.suffix + ".part")
    if partial.exists():
        partial.unlink()
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        print(f"Downloading {url} (attempt {attempt}/{retries})")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": f"TAME-MT/{__version__}"})
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                partial.open("wb") as handle,
            ):
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
            if not is_complete_tar(partial):
                raise DemoError("downloaded archive is incomplete or not a tar file")
            partial.replace(archive)
            return archive
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise DemoError(
                    f"OPUS-100 pair {pair!r} was not found at {url}; check the pair name"
                ) from exc
            last_error = f"HTTP {exc.code}: {exc.reason}"
        except (urllib.error.URLError, TimeoutError, OSError, DemoError) as exc:
            last_error = str(exc)
        partial.unlink(missing_ok=True)
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 10))
    raise DemoError(f"could not download {url} after {retries} attempts: {last_error}")


def is_complete_tar(path: Path) -> bool:
    try:
        with tarfile.open(path) as tar:
            tar.getmembers()
    except (tarfile.TarError, EOFError, OSError):
        return False
    return True


def load_pair_data(archive: Path, pair: str, train_limit: int, test_limit: int) -> PairData:
    if "-" not in pair:
        raise DemoError(f"pair must look like 'src-tgt', got {pair!r}")
    src_lang, tgt_lang = pair.split("-", maxsplit=1)
    prefix = f"opus-100-corpus/v1.0/supervised/{pair}/opus.{pair}"
    try:
        with tarfile.open(archive) as tar:
            train_src = read_member_lines(tar, f"{prefix}-train.{src_lang}", train_limit)
            train_tgt = read_member_lines(tar, f"{prefix}-train.{tgt_lang}", train_limit)
            test_src = read_member_lines(tar, f"{prefix}-test.{src_lang}", test_limit)
            test_ref = read_member_lines(tar, f"{prefix}-test.{tgt_lang}", test_limit)
    except tarfile.TarError as exc:
        raise DemoError(f"{archive} is not a readable tar archive") from exc
    if not train_src or not train_tgt or not test_src or not test_ref:
        raise DemoError(
            f"pair {pair!r} did not yield non-empty capped train/test files "
            f"with train_limit={train_limit} and test_limit={test_limit}"
        )
    return PairData(
        pair=pair,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        test_ref=test_ref,
    )


def read_member_lines(tar: tarfile.TarFile, member_name: str, limit: int) -> list[str]:
    try:
        member = tar.getmember(member_name)
    except KeyError as exc:
        raise DemoError(f"archive member not found: {member_name}") from exc
    handle = tar.extractfile(member)
    if handle is None:
        raise DemoError(f"could not read archive member {member_name}")
    lines: list[str] = []
    for idx, raw_line in enumerate(handle):
        if idx >= limit:
            break
        lines.append(raw_line.decode("utf-8").rstrip("\n\r"))
    return lines


def write_prepared_files(data: PairData, prepared_dir: Path) -> None:
    pair_dir = prepared_dir / data.pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    write_lines(pair_dir / "train.src", data.train_src)
    write_lines(pair_dir / "train.tgt", data.train_tgt)
    write_lines(pair_dir / "test.src", data.test_src)
    write_lines(pair_dir / "test.ref", data.test_ref)


def summary_row(
    data: PairData,
    report: dict[str, object],
    audit_seconds: float,
) -> dict[str, object]:
    quality = report["quality"]
    exposure = report["exposure"]
    backend = report["backend"]
    retrieval = report["retrieval"]
    warnings = report["warnings"]
    bins_payload = report["bins"]
    assert isinstance(quality, dict)
    assert isinstance(exposure, dict)
    assert isinstance(backend, dict)
    assert isinstance(retrieval, dict)
    assert isinstance(warnings, list)
    assert isinstance(bins_payload, list)
    bins = {
        str(item["name"]): item for item in (cast(dict[str, object], row) for row in bins_payload)
    }
    source = exposure["source"]
    pair = exposure["pair"]
    tm = quality["tm"]
    assert isinstance(source, dict)
    assert isinstance(pair, dict)
    assert isinstance(tm, dict)
    source_thresholds = source["at_threshold"]
    pair_thresholds = pair["at_threshold"]
    assert isinstance(source_thresholds, dict)
    assert isinstance(pair_thresholds, dict)
    far = bins["far"]
    near = bins["near"]
    assert isinstance(far, dict)
    assert isinstance(near, dict)
    return {
        "pair": data.pair,
        "direction": f"{data.src_lang}->{data.tgt_lang}",
        "train_used": len(data.train_src),
        "test_used": len(data.test_src),
        "backend": backend["name"],
        "backend_exact": backend["exact"],
        "retrieval": retrieval["mode"],
        "source_exposure_mode": retrieval["source_exposure_mode"],
        "pair_exposure_mode": retrieval["pair_exposure_mode"],
        "tm_retrieval_exact": retrieval["tm_retrieval_exact"],
        "audit_seconds": audit_seconds,
        "tm_bleu": tm["bleu"],
        "tm_chrf": tm["chrf"],
        "mean_source_exposure": source["mean"],
        "source_near_dup_at_085": source_thresholds["0.85"],
        "exact_source_overlap": source["exact_overlap"],
        "pair_leak_topk_at_085": pair_thresholds["0.85"],
        "exact_pair_overlap": pair["exact_overlap"],
        "near_count": near["count"],
        "far_count": far["count"],
        "far_pct": far["percentage"],
        "warning_count": len(warnings),
        "warnings": warnings,
        "signature": report["signature"],
    }


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(
    path: Path,
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tame_version": __version__,
        "command": command_from_args(args),
        "machine": machine_profile(),
        "settings": {
            "tier": args.tier,
            "tier_description": TIERS[args.tier].description,
            "pairs": args.pairs or list(DEFAULT_PAIRS),
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
            "retrieval": args.retrieval,
            "allow_approximate": args.allow_approximate,
            "index_mode": args.index_mode,
            "threads": args.threads,
            "validate_approx_sample": args.validate_approx_sample,
            "validate_approx_seed": args.validate_approx_seed,
            "download_retries": args.download_retries,
            "download_timeout": args.download_timeout,
        },
        "rows": rows,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_summary_markdown(
    path: Path,
    rows: list[dict[str, object]],
    train_limit: int,
    test_limit: int,
    *,
    tier: str,
) -> None:
    lines = [
        "# OPUS-100 Public Corpora Demo",
        "",
        "This table was generated by `tame-mt demo opus100` using public",
        "OPUS-100 supervised train/test splits.",
        "",
        f"- Tier: {tier} ({TIERS[tier].description})",
        f"- Train cap per pair: {train_limit:,} aligned pairs",
        f"- Test cap per pair: {test_limit:,} aligned pairs",
        "- Direction: first language in the OPUS-100 pair name to second language",
        "- Mode: TAME-MT audit, so no system hypothesis was evaluated",
        "- Retrieval: exact by default; approximate runs require explicit opt-in",
        "",
        "| Pair | Direction | Train | Test | Retrieval | Backend | Audit s | TM-BLEU | "
        "TM-chrF | Mean SX | SourceNearDup@0.85 | PairLeakTopK@0.85 | ExactPair | "
        "Far % | Warnings |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {pair} | {direction} | {train_used:,} | {test_used:,} | {retrieval} | "
            "{backend} | {audit_seconds:.2f} | {tm_bleu:.2f} | {tm_chrf:.2f} | "
            "{mean_source_exposure:.3f} | {source_near_dup_at_085:.2%} | "
            "{pair_leak_topk_at_085:.2%} | {exact_pair_overlap:.2%} | {far_pct:.2%} | "
            "{warning_count} |".format(**cast(dict[str, Any], row))
        )
    lines.extend(
        [
            "",
            "Interpretation: lower TM-BLEU and lower PairLeakTopK suggest that this capped",
            "training subset does not make the test split easy to solve by source-side",
            "nearest-neighbor reuse. High far-bin coverage means the split contains many",
            "examples that are distant from the capped training subset under default",
            "character n-gram Jaccard exposure.",
            "",
            "These are demonstration numbers for capped subsets, not definitive claims",
            "about full OPUS-100 language-pair releases. If retrieval is approximate,",
            "treat exposure and TM metrics as candidate-set estimates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_profile_json(
    path: Path,
    *,
    args: argparse.Namespace,
    profiles: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tame_version": __version__,
        "command": command_from_args(args),
        "machine": machine_profile(),
        "settings": {
            "tier": args.tier,
            "tier_description": TIERS[args.tier].description,
            "pairs": args.pairs or list(DEFAULT_PAIRS),
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
            "retrieval": args.retrieval,
            "allow_approximate": args.allow_approximate,
            "index_mode": args.index_mode,
            "threads": args.threads,
            "validate_approx_sample": args.validate_approx_sample,
            "validate_approx_seed": args.validate_approx_seed,
            "download_retries": args.download_retries,
            "download_timeout": args.download_timeout,
        },
        "runs": profiles,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def machine_profile() -> dict[str, object]:
    profile: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "native": asdict(native_status()),
    }
    for key, command in {
        "rustc": ["rustc", "--version"],
        "cargo": ["cargo", "--version"],
    }.items():
        try:
            profile[key] = subprocess.check_output(command, text=True).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            profile[key] = f"unavailable: {exc}"
    return profile


def command_from_args(args: argparse.Namespace) -> list[str]:
    command = getattr(args, "command", None)
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        return command
    return [sys.executable, *sys.argv]


if __name__ == "__main__":
    raise SystemExit(main())
