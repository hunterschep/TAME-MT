import json
from pathlib import Path

from tame_mt.cli import main


FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_score_json_and_segment_outputs(tmp_path: Path, capsys) -> None:
    json_out = tmp_path / "report.json"
    segment_out = tmp_path / "segments.jsonl"
    rc = main(
        [
            "score",
            "--train-src",
            str(FIXTURES / "train.src"),
            "--train-tgt",
            str(FIXTURES / "train.tgt"),
            "--test-src",
            str(FIXTURES / "test.src"),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--json-out",
            str(json_out),
            "--segment-out",
            str(segment_out),
            "--quiet",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out == ""
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["data"]["num_test"] == 4
    assert "bleu" in payload["quality"]["system"]
    segment_lines = segment_out.read_text(encoding="utf-8").splitlines()
    assert len(segment_lines) == 4
    assert json.loads(segment_lines[0])["source_exact"] is True


def test_cli_tm_baseline_writes_aligned_output(tmp_path: Path) -> None:
    out = tmp_path / "tm.out"
    rc = main(
        [
            "tm-baseline",
            "--train-src",
            str(FIXTURES / "train.src"),
            "--train-tgt",
            str(FIXTURES / "train.tgt"),
            "--test-src",
            str(FIXTURES / "test.src"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4
