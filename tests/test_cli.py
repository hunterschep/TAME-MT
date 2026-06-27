import json
from pathlib import Path

import pytest

from tame_mt.cli import main
from tame_mt.io import read_lines, write_lines
from tame_mt.report import segment_metadata_path

FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_score_json_and_segment_outputs(tmp_path: Path, capsys) -> None:
    json_out = tmp_path / "nested" / "report.json"
    segment_out = tmp_path / "nested" / "segments.jsonl"
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
            "--metrics",
            "bleu,chrf",
            "--quiet",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out == ""
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["data"]["num_test"] == 4
    assert "bleu" in payload["quality"]["system"]
    segment_lines = segment_out.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(segment_metadata_path(segment_out).read_text(encoding="utf-8"))
    assert len(segment_lines) == 4
    assert json.loads(segment_lines[0])["source_exact"] is True
    assert metadata["artifact"] == "segment_jsonl"
    assert metadata["signature"] == payload["signature"]
    assert metadata["data"]["num_test"] == 4


def test_cli_score_verbose_reports_stage_timings(tmp_path: Path, capsys) -> None:
    json_out = tmp_path / "report.json"
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
            "--quiet",
            "--verbose",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == ""
    assert "tame-mt: read evaluation inputs completed in " in captured.err
    assert "tame-mt: read training inputs completed in " in captured.err
    assert "tame-mt: evaluate corpus completed in " in captured.err
    assert "tame-mt: write outputs completed in " in captured.err


def test_cli_audit_verbose_reports_stage_timings(tmp_path: Path, capsys) -> None:
    json_out = tmp_path / "audit.json"
    rc = main(
        [
            "audit",
            "--train-src",
            str(FIXTURES / "train.src"),
            "--train-tgt",
            str(FIXTURES / "train.tgt"),
            "--test-src",
            str(FIXTURES / "test.src"),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--json-out",
            str(json_out),
            "--quiet",
            "--verbose",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == ""
    assert "tame-mt: read evaluation inputs completed in " in captured.err
    assert "tame-mt: read training inputs completed in " in captured.err
    assert "tame-mt: evaluate corpus completed in " in captured.err
    assert "tame-mt: write outputs completed in " in captured.err


def test_cli_score_supports_gzip_text_inputs_and_outputs(tmp_path: Path) -> None:
    train_src = tmp_path / "train.src.gz"
    train_tgt = tmp_path / "train.tgt.gz"
    test_src = tmp_path / "test.src.gz"
    ref = tmp_path / "test.ref.gz"
    hyp = tmp_path / "hyp.out.gz"
    json_out = tmp_path / "report.json.gz"
    segment_out = tmp_path / "segments.jsonl.gz"
    for source, target in [
        (FIXTURES / "train.src", train_src),
        (FIXTURES / "train.tgt", train_tgt),
        (FIXTURES / "test.src", test_src),
        (FIXTURES / "test.ref", ref),
        (FIXTURES / "hyp.out", hyp),
    ]:
        write_lines(target, read_lines(source))

    rc = main(
        [
            "score",
            "--train-src",
            str(train_src),
            "--train-tgt",
            str(train_tgt),
            "--test-src",
            str(test_src),
            "--ref",
            str(ref),
            "--hyp",
            str(hyp),
            "--json-out",
            str(json_out),
            "--segment-out",
            str(segment_out),
            "--quiet",
        ]
    )

    assert rc == 0
    payload = json.loads("\n".join(read_lines(json_out)))
    assert payload["data"]["num_test"] == 4
    assert len(read_lines(segment_out)) == 4
    assert segment_metadata_path(segment_out).exists()


def test_cli_doctor_reports_environment(capsys) -> None:
    rc = main(["doctor"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "TAME-MT:" in captured.out
    assert "Native backend:" in captured.out


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


def test_cli_tm_baseline_supports_gzip_outputs(tmp_path: Path) -> None:
    out = tmp_path / "tm.out.gz"
    metadata = tmp_path / "tm_metadata.jsonl.gz"
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
            "--metadata-out",
            str(metadata),
        ]
    )

    assert rc == 0
    assert len(read_lines(out)) == 4
    metadata_rows = [json.loads(line) for line in read_lines(metadata)]
    assert len(metadata_rows) == 4
    assert {"index", "tm_source_index", "tm_source_similarity"} <= set(metadata_rows[0])


def test_cli_tm_baseline_verbose_reports_stage_timings(tmp_path: Path, capsys) -> None:
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
            "--verbose",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == ""
    assert "tame-mt: read inputs completed in " in captured.err
    assert "tame-mt: evaluate tm baseline completed in " in captured.err
    assert "tame-mt: write outputs completed in " in captured.err


def test_cli_score_cached_matches_full_score(tmp_path: Path) -> None:
    full_json = tmp_path / "full.json"
    cached_json = tmp_path / "cached.json"
    segments = tmp_path / "segments.jsonl"
    full_rc = main(
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
            str(full_json),
            "--segment-out",
            str(segments),
            "--quiet",
        ]
    )
    cached_rc = main(
        [
            "score-cached",
            "--segment-in",
            str(segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--num-train",
            "4",
            "--json-out",
            str(cached_json),
            "--quiet",
        ]
    )
    assert full_rc == 0
    assert cached_rc == 0
    full = json.loads(full_json.read_text(encoding="utf-8"))
    cached = json.loads(cached_json.read_text(encoding="utf-8"))
    assert cached["quality"] == full["quality"]
    assert cached["exposure"] == full["exposure"]


def test_cli_score_cached_rejects_bin_threshold_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    segments = tmp_path / "segments.jsonl"
    json_out = tmp_path / "cached.json"
    full_rc = main(
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
            "--segment-out",
            str(segments),
            "--quiet",
        ]
    )
    cached_rc = main(
        [
            "score-cached",
            "--segment-in",
            str(segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--num-train",
            "4",
            "--near-threshold",
            "0.80",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert full_rc == 0
    assert cached_rc == 2
    assert "segment metadata bins.near_threshold does not match" in captured.err
    assert not json_out.exists()


def test_cli_score_cached_rejects_segment_metadata_config_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    segments = tmp_path / "segments.jsonl"
    json_out = tmp_path / "cached.json"
    full_rc = main(
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
            "--segment-out",
            str(segments),
            "--quiet",
        ]
    )
    cached_rc = main(
        [
            "score-cached",
            "--segment-in",
            str(segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--num-train",
            "4",
            "--lowercase",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert full_rc == 0
    assert cached_rc == 2
    assert "segment metadata normalization config does not match" in captured.err
    assert not json_out.exists()


def test_cli_score_cached_accepts_legacy_segments_without_metadata(tmp_path: Path) -> None:
    segments = tmp_path / "segments.jsonl"
    cached_json = tmp_path / "cached.json"
    full_rc = main(
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
            "--segment-out",
            str(segments),
            "--quiet",
        ]
    )
    segment_metadata_path(segments).unlink()
    cached_rc = main(
        [
            "score-cached",
            "--segment-in",
            str(segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--num-train",
            "4",
            "--json-out",
            str(cached_json),
            "--quiet",
        ]
    )

    assert full_rc == 0
    assert cached_rc == 0
    assert cached_json.exists()


def test_cli_score_cached_verbose_reports_stage_timings(tmp_path: Path, capsys) -> None:
    cached_json = tmp_path / "cached.json"
    segments = tmp_path / "segments.jsonl"
    full_rc = main(
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
            "--segment-out",
            str(segments),
            "--quiet",
        ]
    )
    cached_rc = main(
        [
            "score-cached",
            "--segment-in",
            str(segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--num-train",
            "4",
            "--json-out",
            str(cached_json),
            "--quiet",
            "--verbose",
        ]
    )
    captured = capsys.readouterr()

    assert full_rc == 0
    assert cached_rc == 0
    assert captured.out == ""
    assert "tame-mt: read cached inputs completed in " in captured.err
    assert "tame-mt: score cached hypothesis completed in " in captured.err
    assert "tame-mt: write outputs completed in " in captured.err


def test_cli_score_cached_batch_writes_per_system_reports(tmp_path: Path) -> None:
    segments = tmp_path / "segments.jsonl"
    output_dir = tmp_path / "reports"
    variant_hyp = tmp_path / "variant.out"
    write_lines(variant_hyp, ["hola mundo", "buenos dias", "hasta luego", "distinto"])
    full_rc = main(
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
            "--segment-out",
            str(segments),
            "--quiet",
        ]
    )
    batch_rc = main(
        [
            "score-cached-batch",
            "--segment-in",
            str(segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--system",
            f"baseline={FIXTURES / 'hyp.out'}",
            "--system",
            f"variant={variant_hyp}",
            "--num-train",
            "4",
            "--json-out-dir",
            str(output_dir),
            "--quiet",
        ]
    )

    assert full_rc == 0
    assert batch_rc == 0
    baseline = json.loads((output_dir / "baseline.json").read_text(encoding="utf-8"))
    variant = json.loads((output_dir / "variant.json").read_text(encoding="utf-8"))
    assert baseline["backend"]["resolved_mode"] == "cached_segments"
    assert baseline["quality"]["tm"] == variant["quality"]["tm"]
    assert baseline["quality"]["system"] != variant["quality"]["system"]


def test_cli_score_cached_batch_verbose_reports_stage_timings(
    tmp_path: Path,
    capsys,
) -> None:
    segments = tmp_path / "segments.jsonl"
    output_dir = tmp_path / "reports"
    variant_hyp = tmp_path / "variant.out"
    write_lines(variant_hyp, ["hola mundo", "buenos dias", "hasta luego", "distinto"])
    full_rc = main(
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
            "--segment-out",
            str(segments),
            "--quiet",
        ]
    )
    batch_rc = main(
        [
            "score-cached-batch",
            "--segment-in",
            str(segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--system",
            f"baseline={FIXTURES / 'hyp.out'}",
            "--system",
            f"variant={variant_hyp}",
            "--num-train",
            "4",
            "--json-out-dir",
            str(output_dir),
            "--quiet",
            "--verbose",
        ]
    )
    captured = capsys.readouterr()

    assert full_rc == 0
    assert batch_rc == 0
    assert captured.out == ""
    assert "tame-mt: read cached inputs completed in " in captured.err
    assert "tame-mt: score cached systems completed in " in captured.err
    assert "tame-mt: write outputs completed in " in captured.err


def test_cli_score_cached_batch_rejects_duplicate_system_names(tmp_path: Path, capsys) -> None:
    segments = tmp_path / "segments.jsonl"
    full_rc = main(
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
            "--segment-out",
            str(segments),
            "--quiet",
        ]
    )
    batch_rc = main(
        [
            "score-cached-batch",
            "--segment-in",
            str(segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--system",
            f"same={FIXTURES / 'hyp.out'}",
            "--system",
            f"same={FIXTURES / 'hyp.out'}",
            "--num-train",
            "4",
            "--json-out-dir",
            str(tmp_path / "reports"),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert full_rc == 0
    assert batch_rc == 2
    assert "duplicate system name" in captured.err


def test_cli_score_cached_reports_bad_segment_jsonl(tmp_path: Path, capsys) -> None:
    bad_segments = tmp_path / "bad.jsonl"
    bad_segments.write_text("{not-json}\n", encoding="utf-8")
    rc = main(
        [
            "score-cached",
            "--segment-in",
            str(bad_segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--num-train",
            "4",
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "invalid JSON" in captured.err


def test_cli_audit_source_only_works(tmp_path: Path) -> None:
    json_out = tmp_path / "audit.json"
    rc = main(
        [
            "audit",
            "--train-src",
            str(FIXTURES / "train.src"),
            "--test-src",
            str(FIXTURES / "test.src"),
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["quality"]["tm"] == {"bleu": None, "chrf": None}
    assert payload["exposure"]["target"] is None
    assert payload["exposure"]["pair"] is None


def test_cli_index_build_inspect_and_score_reuse(tmp_path: Path, capsys) -> None:
    pytest.importorskip("tame_mt._native")
    index_path = tmp_path / "train.tameidx"
    full_json = tmp_path / "full.json"
    indexed_json = tmp_path / "indexed.json"

    build_rc = main(
        [
            "index",
            "build",
            "--train-src",
            str(FIXTURES / "train.src"),
            "--train-tgt",
            str(FIXTURES / "train.tgt"),
            "--index-mode",
            "native_exact",
            "--out",
            str(index_path),
            "--quiet",
            "--verbose",
        ]
    )
    inspect_rc = main(["index", "inspect", str(index_path)])
    captured = capsys.readouterr()
    inspect_out = captured.out
    full_rc = main(
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
            "--index-mode",
            "native_exact",
            "--json-out",
            str(full_json),
            "--quiet",
        ]
    )
    indexed_rc = main(
        [
            "score",
            "--index",
            str(index_path),
            "--test-src",
            str(FIXTURES / "test.src"),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--index-mode",
            "native_exact",
            "--json-out",
            str(indexed_json),
            "--quiet",
        ]
    )

    assert build_rc == 0
    assert inspect_rc == 0
    assert "tame-mt: read training inputs completed in " in captured.err
    assert "tame-mt: build index bundle completed in " in captured.err
    assert json.loads(inspect_out)["format"] == "tameidx"
    assert full_rc == 0
    assert indexed_rc == 0
    full = json.loads(full_json.read_text(encoding="utf-8"))
    indexed = json.loads(indexed_json.read_text(encoding="utf-8"))
    assert indexed["quality"] == full["quality"]
    assert indexed["exposure"] == full["exposure"]
    assert indexed["backend"]["index_reused"] is True


def test_cli_reports_alignment_errors(tmp_path: Path, capsys) -> None:
    short_hyp = tmp_path / "short.out"
    short_hyp.write_text("only one line\n", encoding="utf-8")
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
            str(short_hyp),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "misaligned input files" in captured.err


def test_cli_reports_invalid_utf8_text_inputs(tmp_path: Path, capsys) -> None:
    bad_hyp = tmp_path / "bad.out"
    bad_hyp.write_bytes(b"\xff")
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
            str(bad_hyp),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "not valid UTF-8" in captured.err


def test_cli_reports_invalid_gzip_text_inputs(tmp_path: Path, capsys) -> None:
    bad_hyp = tmp_path / "bad.out.gz"
    bad_hyp.write_bytes(b"not gzip")
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
            str(bad_hyp),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "not a valid gzip file" in captured.err


def test_cli_score_cached_reports_invalid_utf8_segment_jsonl(
    tmp_path: Path,
    capsys,
) -> None:
    bad_segments = tmp_path / "bad.jsonl"
    bad_segments.write_bytes(b"\xff")
    rc = main(
        [
            "score-cached",
            "--segment-in",
            str(bad_segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--num-train",
            "4",
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "not valid UTF-8" in captured.err


def test_cli_score_cached_reports_invalid_gzip_segment_jsonl(
    tmp_path: Path,
    capsys,
) -> None:
    bad_segments = tmp_path / "bad.jsonl.gz"
    bad_segments.write_bytes(b"not gzip")
    rc = main(
        [
            "score-cached",
            "--segment-in",
            str(bad_segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--num-train",
            "4",
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "not a valid gzip file" in captured.err


def test_cli_score_cached_rejects_non_positive_num_train(tmp_path: Path, capsys) -> None:
    segments = tmp_path / "segments.jsonl"
    full_rc = main(
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
            "--segment-out",
            str(segments),
            "--quiet",
        ]
    )
    cached_rc = main(
        [
            "score-cached",
            "--segment-in",
            str(segments),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--num-train",
            "0",
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert full_rc == 0
    assert cached_rc == 2
    assert "num_train must be positive" in captured.err


def test_cli_reports_configuration_errors(capsys) -> None:
    rc = main(
        [
            "score",
            "--ngram-orders",
            "",
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
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "expected a comma-separated list of integers" in captured.err


def test_cli_rejects_non_finite_thresholds(tmp_path: Path, capsys) -> None:
    json_out = tmp_path / "report.json"
    rc = main(
        [
            "score",
            "--far-threshold",
            "nan",
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
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "finite number" in captured.err
    assert not json_out.exists()


@pytest.mark.parametrize(
    ("extra_args", "expected_error"),
    [
        (["--far-threshold", "1.01"], "far_threshold must be between 0 and 1"),
        (["--near-threshold", "1.01"], "near_threshold must be between 0 and 1"),
        (["--leak-thresholds", "0.7,1.01"], "leak_thresholds must be between 0 and 1"),
        (["--leak-thresholds", ","], "expected a comma-separated list of floats"),
        (["--ngram-orders", "3,,5"], "expected a comma-separated list of integers"),
    ],
)
def test_cli_rejects_malformed_numeric_configuration(
    extra_args: list[str], expected_error: str, capsys
) -> None:
    rc = main(
        [
            "score",
            *extra_args,
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
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert expected_error in captured.err


def test_cli_rejects_duplicate_metrics(capsys) -> None:
    rc = main(
        [
            "score",
            "--metrics",
            "bleu",
            "BLEU",
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
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "duplicate metrics" in captured.err
