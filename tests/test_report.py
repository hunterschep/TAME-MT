import json
from pathlib import Path

import pytest
import sacrebleu

from tame_mt import BinConfig, IndexConfig, PairConfig, RetrievalConfig, ScoreConfig, TameScorer
from tame_mt.artifacts import read_segment_jsonl
from tame_mt.exceptions import OutputError
from tame_mt.report import render_text_report, write_json_report, write_segment_jsonl
from tame_mt.version import __version__


def test_report_to_json_contains_schema_version() -> None:
    pytest.importorskip("tame_mt._native")
    report = TameScorer().score_corpus(
        train_src=["hello world"],
        train_tgt=["hola mundo"],
        test_src=["hello world"],
        refs=[["hola mundo"]],
        hyp=["hola mundo"],
    )
    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["signature"].startswith(f"tame-mt|v:{__version__}|")
    assert f"|deps:sacrebleu_{sacrebleu.__version__}" in payload["signature"]
    assert payload["config"]["dependencies"]["sacrebleu"] == sacrebleu.__version__
    assert payload["backend"]["name"] == "native_exact"
    assert payload["retrieval"]["mode"] == "exact"
    assert payload["retrieval"]["source_exposure_mode"] == "exact"
    assert payload["retrieval"]["tm_retrieval_exact"] is True
    assert payload["performance"]["backend"] == "native_exact"
    assert payload["performance"]["index_reused"] is False
    assert isinstance(payload["performance"]["threads"], int)
    assert isinstance(payload["performance"]["timings_sec"], dict)
    assert "peak_rss_mb" in payload["performance"]["memory"]
    assert "|retrieval:exact|" in payload["signature"]


def test_approximate_report_is_labeled_and_warns() -> None:
    pytest.importorskip("tame_mt._native")
    report = TameScorer(
        ScoreConfig(retrieval=RetrievalConfig(mode="approx", allow_approximate=True))
    ).score_corpus(
        train_src=["hello world", "other source"],
        train_tgt=["hola mundo", "otro"],
        test_src=["hello"],
        refs=[["hola"]],
        hyp=["hola"],
    )
    payload = json.loads(report.to_json())

    assert payload["backend"]["name"] == "native_fast"
    assert payload["retrieval"]["mode"] == "approx"
    assert payload["retrieval"]["source_exposure_mode"] == "approx"
    assert payload["retrieval"]["tm_retrieval_exact"] is False
    assert any("Approximate retrieval is enabled" in warning for warning in payload["warnings"])
    assert "|retrieval:approx|" in payload["signature"]


def test_report_separates_pair_topk_and_exact_threshold_rates() -> None:
    test_source = "shared segment with many common words and token one"
    ref = "the quick brown fox jumps over the lazy dog"
    report = TameScorer(
        ScoreConfig(
            index=IndexConfig(topk=1),
            bins=BinConfig(leak_thresholds=(0.85,)),
            pair=PairConfig(exact_thresholds=True),
        )
    ).score_corpus(
        train_src=[
            test_source,
            "shared segment with many common words and token two",
            "unrelated source text",
        ],
        train_tgt=[
            "unrelated target text",
            "the quick brown fox jumps over the lazy cat",
            ref,
        ],
        test_src=[test_source],
        refs=[[ref]],
        hyp=[ref],
    )
    payload = json.loads(report.to_json())
    rendered = render_text_report(report)

    assert payload["retrieval"]["pair_exposure_mode"] == "topk_rerank+threshold_exact"
    assert payload["exposure"]["pair"]["at_threshold"]["0.85"] == 0.0
    assert payload["exposure"]["pair"]["exact_at_threshold"]["0.85"] == 1.0
    assert "PairLeakTopK@0.85" in rendered
    assert "PairLeakExact@0.85" in rendered
    assert "|pair_exact:1|" in payload["signature"]


def test_report_to_json_rejects_non_finite_numbers() -> None:
    report = TameScorer().score_corpus(
        train_src=["hello world"],
        train_tgt=["hola mundo"],
        test_src=["hello world"],
        refs=[["hola mundo"]],
        hyp=["hola mundo"],
    )
    report.system_scores["bleu"] = float("nan")

    with pytest.raises(ValueError, match="Out of range"):
        report.to_json()


def test_write_json_report_reports_non_finite_numbers_without_partial_file(tmp_path: Path) -> None:
    report = TameScorer().score_corpus(
        train_src=["hello world"],
        train_tgt=["hola mundo"],
        test_src=["hello world"],
        refs=[["hola mundo"]],
        hyp=["hola mundo"],
    )
    report.system_scores["bleu"] = float("nan")
    out = tmp_path / "report.json"

    with pytest.raises(OutputError, match="failed to serialize JSON report"):
        write_json_report(out, report)

    assert not out.exists()


def test_segment_jsonl_multi_ref_texts_are_explicit(tmp_path: Path) -> None:
    scorer = TameScorer()
    result = scorer.evaluate_corpus(
        train_src=["hello world"],
        train_tgt=["hola mundo"],
        test_src=["hello world"],
        refs=[["hola mundo"], ["saludos mundo"]],
        hyp=["hola mundo"],
    )
    out = tmp_path / "segments.jsonl"
    write_segment_jsonl(
        out,
        result.exposures,
        result.tm_results,
        train_src=["hello world"],
        train_tgt=["hola mundo"],
        test_src=["hello world"],
        refs=[["hola mundo"], ["saludos mundo"]],
        hyp=["hola mundo"],
        include_reference_text=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["reference_texts"] == ["hola mundo", "saludos mundo"]
    assert payload["target_ref_index"] == 0
    assert payload["pair_ref_index"] == 0


def test_segment_jsonl_supports_gzip(tmp_path: Path) -> None:
    scorer = TameScorer()
    result = scorer.evaluate_corpus(
        train_src=["hello world"],
        train_tgt=["hola mundo"],
        test_src=["hello world"],
        refs=[["hola mundo"]],
        hyp=["hola mundo"],
    )
    out = tmp_path / "segments.jsonl.gz"
    write_segment_jsonl(
        out,
        result.exposures,
        result.tm_results,
        train_src=["hello world"],
        train_tgt=["hola mundo"],
        test_src=["hello world"],
        refs=[["hola mundo"]],
        hyp=["hola mundo"],
    )

    exposures, tm_results = read_segment_jsonl(out)

    assert len(exposures) == 1
    assert tm_results[0].tm_hyp == "hola mundo"
