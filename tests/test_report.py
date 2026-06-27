import json
from pathlib import Path

import pytest
import sacrebleu

from tame_mt import TameScorer
from tame_mt.artifacts import read_segment_jsonl
from tame_mt.exceptions import OutputError
from tame_mt.report import write_json_report, write_segment_jsonl


def test_report_to_json_contains_schema_version() -> None:
    report = TameScorer().score_corpus(
        train_src=["hello world"],
        train_tgt=["hola mundo"],
        test_src=["hello world"],
        refs=[["hola mundo"]],
        hyp=["hola mundo"],
    )
    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "0.1"
    assert payload["signature"].startswith("tame-mt|v:0.1.0|")
    assert f"|deps:sacrebleu_{sacrebleu.__version__}" in payload["signature"]
    assert payload["config"]["dependencies"]["sacrebleu"] == sacrebleu.__version__
    assert payload["backend"]["name"] in {
        "native_exact",
        "python_exact",
    }


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
