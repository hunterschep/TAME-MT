from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import sacrebleu

from tame_mt.config import ScoreConfig
from tame_mt.exceptions import OutputError
from tame_mt.io import ensure_parent_dir, open_text
from tame_mt.json_utils import strict_json_dumps
from tame_mt.schema import SCHEMA_VERSION, SegmentExposure, SegmentTMResult, TameReport
from tame_mt.version import __version__

SEGMENT_METADATA_SUFFIX = ".meta.json"


def build_signature(config: ScoreConfig, backend_name: str | None = None) -> str:
    norm = _normalization_signature(config)
    orders = _orders_signature(config.similarity.ngram_orders)
    leaks = ",".join(f"{threshold:.2f}" for threshold in config.bins.leak_thresholds)
    metrics = ",".join(metric.lower() for metric in config.metrics)
    backend = backend_name or config.index.mode
    sacrebleu_version = dependency_versions()["sacrebleu"]
    return (
        f"tame-mt|v:{__version__}|norm:{norm}|sim:char_jaccard_{orders}_set|"
        f"retrieval:{config.retrieval.mode}|approx:{int(config.retrieval.mode == 'approx')}|"
        f"idx:{config.index.mode}|backend:{backend}|tm:src_nn_top1_zero_{config.tm.zero_policy}|"
        f"bins:far{config.bins.far_threshold:.2f}_near{config.bins.near_threshold:.2f}_leak{leaks}|"
        f"pair_k:{config.index.topk}|pair_exact:{int(config.pair.exact_thresholds)}|"
        f"fast:{config.index.candidate_gram_limit},"
        f"{config.index.posting_limit},{config.index.max_candidates},"
        f"{config.index.rerank_limit}|metrics:{metrics}|"
        f"sacrebleu:bleu_tok_{config.metric.bleu_tokenize}_lc_{int(config.metric.bleu_lowercase)}_"
        f"chrf_wo_{config.metric.chrf_word_order}|deps:sacrebleu_{sacrebleu_version}"
    )


def dependency_versions() -> dict[str, str]:
    return {"sacrebleu": str(sacrebleu.__version__)}


def config_to_dict(config: ScoreConfig) -> dict[str, Any]:
    return {
        "normalization": asdict(config.normalization),
        "similarity": {
            "type": config.similarity.similarity,
            "ngram_orders": list(config.similarity.ngram_orders),
        },
        "retrieval": asdict(config.retrieval),
        "index": asdict(config.index),
        "bins": {
            "far_threshold": config.bins.far_threshold,
            "near_threshold": config.bins.near_threshold,
            "leak_thresholds": list(config.bins.leak_thresholds),
            "min_bin_size_warning": config.bins.min_bin_size_warning,
        },
        "pair": {
            "pair_k": config.index.topk,
            "mode": "topk_rerank",
            "exact_thresholds": config.pair.exact_thresholds,
        },
        "tm": asdict(config.tm),
        "metrics": list(config.metrics),
        "sacrebleu": asdict(config.metric),
        "dependencies": dependency_versions(),
    }


def render_text_report(report: TameReport) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "TAME-MT report",
            "==============",
            "",
            "Data",
            "----",
            f"Train segments:  {report.num_train:,}",
            f"Test segments:   {report.num_test:,}",
            "",
        ]
    )
    lines.extend(_render_backend(report))
    lines.extend(_render_quality(report))
    lines.extend(_render_exposure(report))
    lines.extend(_render_bins(report))
    lines.extend(_render_gen_gap(report))
    if report.warnings:
        lines.extend(["Warnings", "--------"])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")
    lines.extend(["Signature", "---------", report.signature])
    return "\n".join(lines)


def _render_backend(report: TameReport) -> list[str]:
    backend = report.backend
    retrieval = report.retrieval
    exactness = "exact" if backend.get("exact") else "approximate"
    engine = "rust" if backend.get("native") else "cached artifact"
    return [
        "Backend",
        "-------",
        f"Name:            {backend.get('name', 'unknown')}",
        f"Engine:          {engine}",
        f"Retrieval:       {exactness}",
        f"Mode:            {retrieval.mode}",
        f"Source exposure: {retrieval.source_exposure_mode}",
        f"Target exposure: {retrieval.target_exposure_mode}",
        f"Pair exposure:   {retrieval.pair_exposure_mode}",
        f"TM retrieval:    {'exact' if retrieval.tm_retrieval_exact else 'approximate'}",
        "",
    ]


def write_json_report(path: str | Path, report: TameReport) -> None:
    output_path = Path(path)
    try:
        payload = strict_json_dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
    except ValueError as exc:
        raise OutputError(f"failed to serialize JSON report: {exc}") from exc
    ensure_parent_dir(output_path)
    with open_text(output_path, "w") as handle:
        handle.write(payload)


def segment_metadata_path(path: str | Path) -> Path:
    return Path(f"{path}{SEGMENT_METADATA_SUFFIX}")


def write_segment_metadata(
    path: str | Path,
    report: TameReport,
    *,
    fingerprints: dict[str, Any] | None = None,
    tm_text_included: bool = True,
    neighbor_text_included: bool = False,
) -> None:
    output_path = Path(path)
    try:
        payload = strict_json_dumps(
            _segment_metadata_payload(
                report,
                fingerprints=fingerprints,
                tm_text_included=tm_text_included,
                neighbor_text_included=neighbor_text_included,
            ),
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as exc:
        raise OutputError(f"failed to serialize segment metadata: {exc}") from exc
    ensure_parent_dir(output_path)
    with open_text(output_path, "w") as handle:
        handle.write(payload + "\n")


def write_segment_jsonl(
    path: str | Path,
    exposures: list[SegmentExposure],
    tm_results: list[SegmentTMResult],
    train_src: list[str],
    train_tgt: list[str] | None,
    test_src: list[str],
    refs: list[list[str]] | None,
    hyp: list[str] | None,
    include_neighbor_text: bool = False,
    include_source_text: bool = False,
    include_reference_text: bool = False,
    include_hyp_text: bool = False,
    include_tm_text: bool = True,
) -> None:
    tm_by_index = {result.index: result for result in tm_results}
    output_path = Path(path)
    ensure_parent_dir(output_path)
    with open_text(output_path, "w") as handle:
        for segment in exposures:
            tm_result = tm_by_index.get(segment.index)
            payload: dict[str, Any] = {
                "index": segment.index,
                "source_exposure": segment.source_exposure,
                "source_nn_index": segment.source_nn_index,
                "source_exact": segment.source_exact,
                "target_exposure": segment.target_exposure,
                "target_nn_index": segment.target_nn_index,
                "target_ref_index": segment.target_ref_index,
                "target_exact": segment.target_exact,
                "pair_exposure": segment.pair_exposure,
                "pair_nn_index": segment.pair_nn_index,
                "pair_ref_index": segment.pair_ref_index,
                "pair_exact": segment.pair_exact,
                "pair_exact_at_threshold": segment.pair_exact_at_threshold,
                "bin": segment.bin,
                "tm_source_index": tm_result.tm_source_index if tm_result else None,
                "tm_source_similarity": tm_result.tm_source_similarity if tm_result else None,
            }
            if include_tm_text:
                payload["tm_hyp"] = tm_result.tm_hyp if tm_result else ""
            if include_source_text:
                payload["source_text"] = test_src[segment.index]
            if include_reference_text and refs:
                ref_texts = [ref[segment.index] for ref in refs]
                if len(ref_texts) == 1:
                    payload["reference_text"] = ref_texts[0]
                else:
                    payload["reference_texts"] = ref_texts
            if include_hyp_text and hyp:
                payload["hyp_text"] = hyp[segment.index]
            if include_neighbor_text and segment.source_nn_index is not None:
                payload["neighbor_source_text"] = train_src[segment.source_nn_index]
                if train_tgt is not None:
                    payload["neighbor_target_text"] = train_tgt[segment.source_nn_index]
            try:
                line = strict_json_dumps(payload, ensure_ascii=False)
            except ValueError as exc:
                raise OutputError(
                    f"failed to serialize segment JSONL row {segment.index}: {exc}"
                ) from exc
            handle.write(line + "\n")


def _segment_metadata_payload(
    report: TameReport,
    *,
    fingerprints: dict[str, Any] | None,
    tm_text_included: bool,
    neighbor_text_included: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "segment_jsonl",
        "tame_version": report.tame_version,
        "signature": report.signature,
        "data": {
            "num_train": report.num_train,
            "num_test": report.num_test,
            "num_refs": report.num_refs,
        },
        "retrieval": asdict(report.retrieval),
        "config": report.config,
        "backend": report.backend,
        "privacy": {
            "tm_text_included": tm_text_included,
            "contains_training_target_text": tm_text_included,
            "contains_neighbor_training_text": neighbor_text_included,
        },
        "fingerprints": fingerprints or {},
    }


def _normalization_signature(config: ScoreConfig) -> str:
    parts = [config.normalization.unicode_form.lower()]
    parts.append("ws" if config.normalization.collapse_whitespace else "raw_ws")
    parts.append("lower" if config.normalization.lowercase else "case")
    if config.normalization.strip_diacritics:
        parts.append("stripdia")
    if config.normalization.normalize_punctuation:
        parts.append("normpunct")
    return "_".join(parts)


def _orders_signature(orders: tuple[int, ...]) -> str:
    sorted_orders = tuple(sorted(orders))
    if sorted_orders and sorted_orders == tuple(range(sorted_orders[0], sorted_orders[-1] + 1)):
        return f"{sorted_orders[0]}-{sorted_orders[-1]}"
    return ",".join(str(order) for order in sorted_orders)


def _render_quality(report: TameReport) -> list[str]:
    lines = ["Quality", "-------", "Metric       System      TM baseline      delta over TM"]
    for metric in report.system_scores:
        label = _metric_label(metric)
        lines.append(
            f"{label:<10} {_fmt_score(report.system_scores.get(metric)):>9}"
            f" {_fmt_score(report.tm_scores.get(metric)):>16}"
            f" {_fmt_signed(report.delta_scores.get(metric)):>17}"
        )
    lines.append("")
    return lines


def _render_exposure(report: TameReport) -> list[str]:
    lines = ["Exposure", "--------"]
    lines.extend(_render_exposure_side("Source exposure", report.exposure.source))
    if report.exposure.target is not None:
        lines.extend(_render_exposure_side("Target exposure", report.exposure.target))
    if report.exposure.pair is not None:
        lines.extend(
            _render_pair_exposure(report.exposure.pair, report.retrieval.pair_exposure_mode)
        )
    lines.append("")
    return lines


def _render_exposure_side(title: str, stats: dict[str, Any]) -> list[str]:
    lines = [f"{title}:"]
    lines.extend(
        [
            f"  mean:             {_fmt_fraction(stats.get('mean'))}",
            f"  median:           {_fmt_fraction(stats.get('median'))}",
            f"  p95:              {_fmt_fraction(stats.get('p95'))}",
            f"  exact overlap:    {_fmt_pct(stats.get('exact_overlap'))}",
        ]
    )
    thresholds = stats.get("at_threshold") or {}
    for threshold, value in thresholds.items():
        lines.append(f"  >= {threshold}:         {_fmt_pct(value)}")
    lines.append("")
    return lines


def _render_pair_exposure(stats: dict[str, Any], pair_mode: str) -> list[str]:
    lines = ["Pair exposure:"]
    lines.append(f"  exact overlap:    {_fmt_pct(stats.get('exact_overlap'))}")
    thresholds = stats.get("at_threshold") or {}
    label = "PairLeakTopK" if "topk" in pair_mode else "PairLeak"
    for threshold, value in thresholds.items():
        lines.append(f"  {label}@{threshold}:   {_fmt_pct(value)}")
    exact_thresholds = stats.get("exact_at_threshold") or {}
    for threshold, value in exact_thresholds.items():
        lines.append(f"  PairLeakExact@{threshold}: {_fmt_pct(value)}")
    lines.append("")
    return lines


def _render_bins(report: TameReport) -> list[str]:
    metric_names = list(report.system_scores)
    lines = [
        "Distance-stratified quality by source exposure",
        "----------------------------------------------",
    ]
    header = "Bin            N      %       MeanSX"
    for metric in metric_names:
        label = _metric_label(metric)
        header += f" {label:>9} {'TM-' + label:>10} {'dTM-' + label:>11}"
    lines.append(header)
    for item in report.bins:
        row = (
            f"{item.name:<12} {item.count:>5} {_fmt_pct_short(item.percentage):>7}"
            f" {_fmt_fraction(item.mean_source_exposure):>8}"
        )
        for metric in metric_names:
            row += (
                f" {_fmt_score(item.system_scores.get(metric)):>9}"
                f" {_fmt_score(item.tm_scores.get(metric)):>10}"
                f" {_fmt_signed(item.delta_scores.get(metric)):>11}"
            )
        lines.append(row)
    lines.append("")
    return lines


def _render_gen_gap(report: TameReport) -> list[str]:
    lines = ["Generalization gap", "------------------"]
    for metric, value in report.generalization_gap.items():
        lines.append(f"GenGap-{_metric_label(metric)}:  {_fmt_score(value)}")
    lines.append("")
    return lines


def _fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _metric_label(metric: str) -> str:
    return "chrF" if metric.lower() == "chrf" else metric.upper()


def _fmt_signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def _fmt_fraction(value: float | None | object) -> str:
    return "n/a" if not isinstance(value, (float, int)) else f"{value:.3f}"


def _fmt_pct(value: float | None | object) -> str:
    return "n/a" if not isinstance(value, (float, int)) else f"{value * 100:.2f}%"


def _fmt_pct_short(value: float | None | object) -> str:
    return "n/a" if not isinstance(value, (float, int)) else f"{value * 100:.1f}%"
