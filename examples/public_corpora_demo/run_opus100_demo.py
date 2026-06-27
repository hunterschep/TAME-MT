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
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tame_mt import IndexConfig, RetrievalConfig, ScoreConfig, TameScorer
from tame_mt.io import write_lines
from tame_mt.report import write_json_report

DEFAULT_PAIRS = ("de-en", "en-hi", "en-tr", "ar-en")
OPUS100_BASE_URL = "https://object.pouta.csc.fi/OPUS-100/v1.0"


@dataclass(frozen=True)
class PairData:
    pair: str
    src_lang: str
    tgt_lang: str
    train_src: list[str]
    train_tgt: list[str]
    test_src: list[str]
    test_ref: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a TAME-MT audit demo on OPUS-100 public corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pair", action="append", dest="pairs", help="OPUS-100 pair, e.g. de-en")
    parser.add_argument("--train-limit", type=int, default=50_000)
    parser.add_argument("--test-limit", type=int, default=2_000)
    parser.add_argument("--retrieval", choices=["exact", "guarded", "approx"], default="exact")
    parser.add_argument("--allow-approximate", action="store_true")
    parser.add_argument("--index-mode", default="auto")
    parser.add_argument("--output-dir", default="demo_runs/opus100_public_corpora")
    parser.add_argument(
        "--summary-dir",
        default="examples/public_corpora_demo",
        help="directory for committed-style CSV and Markdown summaries",
    )
    args = parser.parse_args()

    pairs = tuple(args.pairs) if args.pairs else DEFAULT_PAIRS
    output_dir = Path(args.output_dir)
    downloads_dir = output_dir / "downloads"
    prepared_dir = output_dir / "prepared"
    reports_dir = output_dir / "reports"
    summary_dir = Path(args.summary_dir)
    for directory in (downloads_dir, prepared_dir, reports_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
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
        archive = download_pair(pair, downloads_dir)
        data = load_pair_data(archive, pair, args.train_limit, args.test_limit)
        write_prepared_files(data, prepared_dir)
        audit_start = time.perf_counter()
        result = scorer.evaluate_corpus(
            train_src=data.train_src,
            train_tgt=data.train_tgt,
            test_src=data.test_src,
            refs=[data.test_ref],
            hyp=None,
        )
        audit_seconds = time.perf_counter() - audit_start
        report_path = reports_dir / f"{pair}.audit.json"
        write_json_report(report_path, result.report)
        rows.append(summary_row(data, result.report.to_dict(), audit_seconds))

    write_summary_csv(summary_dir / "opus100_demo_summary.csv", rows)
    write_summary_markdown(
        summary_dir / "opus100_demo_summary.md", rows, args.train_limit, args.test_limit
    )
    print(f"Wrote {summary_dir / 'opus100_demo_summary.md'}")
    return 0


def download_pair(pair: str, downloads_dir: Path) -> Path:
    archive = downloads_dir / f"opus-100-corpus-{pair}-v1.0.tar.gz"
    if archive.exists() and is_complete_tar(archive):
        return archive
    if archive.exists():
        archive.unlink()
    url = f"{OPUS100_BASE_URL}/opus-100-corpus-{pair}-v1.0.tar.gz"
    print(f"Downloading {url}")
    partial = archive.with_suffix(archive.suffix + ".part")
    if partial.exists():
        partial.unlink()
    urllib.request.urlretrieve(url, partial)
    partial.replace(archive)
    return archive


def is_complete_tar(path: Path) -> bool:
    try:
        with tarfile.open(path) as tar:
            tar.getmembers()
    except (tarfile.TarError, EOFError, OSError):
        return False
    return True


def load_pair_data(archive: Path, pair: str, train_limit: int, test_limit: int) -> PairData:
    src_lang, tgt_lang = pair.split("-", maxsplit=1)
    prefix = f"opus-100-corpus/v1.0/supervised/{pair}/opus.{pair}"
    with tarfile.open(archive) as tar:
        train_src = read_member_lines(tar, f"{prefix}-train.{src_lang}", train_limit)
        train_tgt = read_member_lines(tar, f"{prefix}-train.{tgt_lang}", train_limit)
        test_src = read_member_lines(tar, f"{prefix}-test.{src_lang}", test_limit)
        test_ref = read_member_lines(tar, f"{prefix}-test.{tgt_lang}", test_limit)
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
    member = tar.getmember(member_name)
    handle = tar.extractfile(member)
    if handle is None:
        raise RuntimeError(f"could not read archive member {member_name}")
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
    bins = {item["name"]: item for item in report["bins"]}
    assert isinstance(quality, dict)
    assert isinstance(exposure, dict)
    assert isinstance(backend, dict)
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
        "signature": report["signature"],
    }


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    path: Path, rows: list[dict[str, object]], train_limit: int, test_limit: int
) -> None:
    lines = [
        "# OPUS-100 Public Corpora Demo",
        "",
        "This table was generated by `run_opus100_demo.py` using public OPUS-100",
        "supervised train/test splits.",
        "",
        f"- Train cap per pair: {train_limit:,} aligned pairs",
        f"- Test cap per pair: {test_limit:,} aligned pairs",
        "- Direction: first language in the OPUS-100 pair name to second language",
        "- Mode: TAME-MT audit, so no system hypothesis was evaluated",
        "- Retrieval: exact by default; approximate runs require explicit opt-in",
        "",
        "| Pair | Direction | Train | Test | Backend | Audit s | TM-BLEU | TM-chrF | "
        "Mean SX | SourceNearDup@0.85 | PairLeakTopK@0.85 | ExactPair | Far % |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {pair} | {direction} | {train_used:,} | {test_used:,} | {backend} | "
            "{audit_seconds:.2f} | {tm_bleu:.2f} | {tm_chrf:.2f} | "
            "{mean_source_exposure:.3f} | {source_near_dup_at_085:.2%} | "
            "{pair_leak_topk_at_085:.2%} | {exact_pair_overlap:.2%} | {far_pct:.2%} |".format(**row)
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


if __name__ == "__main__":
    raise SystemExit(main())
