import json
from pathlib import Path

from tame_mt import TameScorer
from tame_mt.report import write_segment_jsonl


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
    assert payload["backend"]["name"] in {
        "native_exact",
        "python_exact",
    }


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
