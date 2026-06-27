import json
import subprocess
import sys
from pathlib import Path

import pytest

from tame_mt.approx_validation import ApproxValidation
from tame_mt.cli import main
from tame_mt.io import read_lines, write_lines
from tame_mt.report import segment_metadata_path

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_score_json_and_segment_outputs(tmp_path: Path, capsys) -> None:
    json_out = tmp_path / "nested" / "report.json"
    segment_out = tmp_path / "nested" / "segments.jsonl"
    profile_out = tmp_path / "nested" / "profile.json"
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
            "--profile-json",
            str(profile_out),
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
    assert payload["retrieval"]["mode"] == "exact"
    assert payload["performance"]["backend"] == "native_exact"
    assert payload["performance"]["index_reused"] is False
    assert "evaluate_corpus" in payload["performance"]["timings_sec"]
    profile = json.loads(profile_out.read_text(encoding="utf-8"))
    assert profile["command"] == "score"
    assert profile["performance"]["backend"] == "native_exact"
    assert "write_outputs" in profile["performance"]["timings_sec"]
    assert profile["reports"][0]["signature"] == payload["signature"]


def test_cli_diagnostic_and_cache_outputs_have_expected_privacy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_out = tmp_path / "report.json"
    diagnostic_out = tmp_path / "segments.diagnostic.jsonl"
    cache_out = tmp_path / "segments.tamecache"
    cached_json = tmp_path / "cached.json"
    unsafe_cached_json = tmp_path / "unsafe_cached.json"

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
            str(json_out),
            "--diagnostic-out",
            str(diagnostic_out),
            "--cache-out",
            str(cache_out),
            "--quiet",
        ]
    )
    assert full_rc == 0

    diagnostic_row = json.loads(diagnostic_out.read_text(encoding="utf-8").splitlines()[0])
    cache_row = json.loads(cache_out.read_text(encoding="utf-8").splitlines()[0])
    diagnostic_metadata = json.loads(
        segment_metadata_path(diagnostic_out).read_text(encoding="utf-8")
    )
    cache_metadata = json.loads(segment_metadata_path(cache_out).read_text(encoding="utf-8"))
    assert "tm_hyp" not in diagnostic_row
    assert "tm_hyp" in cache_row
    assert diagnostic_metadata["privacy"]["tm_text_included"] is False
    assert cache_metadata["privacy"]["tm_text_included"] is True

    cached_rc = main(
        [
            "score-cached",
            "--cache-in",
            str(cache_out),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--json-out",
            str(cached_json),
            "--quiet",
        ]
    )
    unsafe_cached_rc = main(
        [
            "score-cached",
            "--cache-in",
            str(diagnostic_out),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--json-out",
            str(unsafe_cached_json),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    full_payload = json.loads(json_out.read_text(encoding="utf-8"))
    cached_payload = json.loads(cached_json.read_text(encoding="utf-8"))
    assert cached_rc == 0
    assert full_payload["quality"] == cached_payload["quality"]
    assert full_payload["exposure"] == cached_payload["exposure"]
    assert unsafe_cached_rc == 2
    assert "does not contain TM hypotheses" in captured.err
    assert not unsafe_cached_json.exists()


def test_cli_cache_in_segment_in_conflict(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "score-cached",
            "--cache-in",
            str(tmp_path / "a.tamecache"),
            "--segment-in",
            str(tmp_path / "b.jsonl"),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "--cache-in and --segment-in refer to different files" in captured.err


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


def test_cli_demo_opus100_help_is_first_class() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tame_mt", "demo", "opus100", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "usage: tame-mt demo opus100" in result.stdout
    assert "--quick" in result.stdout
    assert "--standard" in result.stdout
    assert "--paper" in result.stdout


def test_example_opus100_script_wraps_packaged_demo() -> None:
    result = subprocess.run(
        [sys.executable, "examples/public_corpora_demo/run_opus100_demo.py", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "usage: run_opus100_demo.py" in result.stdout
    assert "--quick" in result.stdout


def test_cli_demo_opus100_dispatches_to_packaged_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str | None, list[str]]] = []

    def fake_run_demo(args, *, command: list[str]) -> int:
        calls.append((args.tier, command))
        return 0

    monkeypatch.setattr("tame_mt.cli.opus100_demo.run_demo", fake_run_demo)

    rc = main(["demo", "opus100", "--quick", "--pair", "de-en"])

    assert rc == 0
    assert calls == [
        (
            "quick",
            ["tame-mt", "demo", "opus100", "--quick", "--pair", "de-en"],
        )
    ]


def test_cli_threads_are_deterministic_across_subprocess_runs(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    semantic_payloads = []
    for label, thread_args in {
        "default": [],
        "one": ["--threads", "1"],
        "four": ["--threads", "4"],
    }.items():
        json_out = tmp_path / f"{label}.json"
        command = [
            sys.executable,
            "-m",
            "tame_mt",
            "score",
            *thread_args,
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
        subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        if thread_args:
            assert payload["performance"]["threads"] == int(thread_args[1])
        semantic_payloads.append(_without_performance(payload))

    assert semantic_payloads[1] == semantic_payloads[0]
    assert semantic_payloads[2] == semantic_payloads[0]


def test_cli_doctor_threads_option_reports_requested_thread_count(tmp_path: Path) -> None:
    _ = tmp_path
    pytest.importorskip("tame_mt._native")
    result = subprocess.run(
        [sys.executable, "-m", "tame_mt", "doctor", "--threads", "1"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Rayon threads: 1" in result.stdout


def test_cli_rejects_approx_backend_without_approx_mode(tmp_path: Path, capsys) -> None:
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
            "--index-mode",
            "native_fast",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "cannot use approximate backend" in captured.err


def test_cli_approx_mode_labels_report(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
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
            "--retrieval",
            "approx",
            "--allow-approximate",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    payload = json.loads(json_out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["backend"]["name"] == "native_fast"
    assert payload["retrieval"]["mode"] == "approx"
    assert payload["retrieval"]["pair_exposure_mode"] == "approx_topk"
    assert any("Approximate retrieval is enabled" in warning for warning in payload["warnings"])


def test_cli_exact_pair_thresholds_write_pairleak_exact(tmp_path: Path) -> None:
    train_src = tmp_path / "train.src"
    train_tgt = tmp_path / "train.tgt"
    test_src = tmp_path / "test.src"
    ref = tmp_path / "test.ref"
    hyp = tmp_path / "hyp.out"
    json_out = tmp_path / "report.json"
    write_lines(
        train_src,
        [
            "shared segment with many common words and token one",
            "shared segment with many common words and token two",
            "unrelated source text",
        ],
    )
    write_lines(
        train_tgt,
        [
            "unrelated target text",
            "the quick brown fox jumps over the lazy cat",
            "the quick brown fox jumps over the lazy dog",
        ],
    )
    write_lines(test_src, ["shared segment with many common words and token one"])
    write_lines(ref, ["the quick brown fox jumps over the lazy dog"])
    write_lines(hyp, ["the quick brown fox jumps over the lazy dog"])

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
            "--pair-k",
            "1",
            "--leak-thresholds",
            "0.85",
            "--exact-pair-thresholds",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    payload = json.loads(json_out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["retrieval"]["pair_exposure_mode"] == "topk_rerank+threshold_exact"
    assert payload["exposure"]["pair"]["at_threshold"]["0.85"] == 0.0
    assert payload["exposure"]["pair"]["exact_at_threshold"]["0.85"] == 1.0
    assert "|pair_exact:1|" in payload["signature"]


def test_cli_approx_validation_writes_report_payload(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    json_out = tmp_path / "report.json"
    profile_out = tmp_path / "profile.json"
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
            "--retrieval",
            "approx",
            "--allow-approximate",
            "--validate-approx-sample",
            "4",
            "--json-out",
            str(json_out),
            "--profile-json",
            str(profile_out),
            "--quiet",
        ]
    )
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    profile = json.loads(profile_out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["approx_validation"]["passed"] is True
    assert payload["approx_validation"]["sample_size"] == 4
    assert payload["approx_validation"]["source_top1_agreement"] == 1.0
    assert payload["approx_validation"]["tm_bleu_abs_delta_on_sample"] == 0.0
    assert "validate_approximate_retrieval" in profile["performance"]["timings_sec"]


def test_cli_approx_validation_requires_approx_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
            "--validate-approx-sample",
            "1",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "--validate-approx-sample requires --retrieval approx" in captured.err
    assert not json_out.exists()


def test_cli_approx_validation_supports_source_only_audit(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    json_out = tmp_path / "audit.json"
    rc = main(
        [
            "audit",
            "--train-src",
            str(FIXTURES / "train.src"),
            "--test-src",
            str(FIXTURES / "test.src"),
            "--retrieval",
            "approx",
            "--allow-approximate",
            "--validate-approx-sample",
            "2",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    payload = json.loads(json_out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["approx_validation"]["passed"] is True
    assert payload["approx_validation"]["target_top1_agreement"] is None
    assert payload["approx_validation"]["tm_bleu_abs_delta_on_sample"] is None


def test_cli_approx_validation_failure_is_fatal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tame_mt._native")
    json_out = tmp_path / "report.json"

    def fail_validation(**_: object) -> ApproxValidation:
        return ApproxValidation(
            payload={"passed": False, "failures": ["source_top1_agreement 0.0 < 0.95"]},
            failures=["source_top1_agreement 0.0 < 0.95"],
        )

    monkeypatch.setattr("tame_mt.cli.validate_approximate_run", fail_validation)
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
            "--retrieval",
            "approx",
            "--allow-approximate",
            "--validate-approx-sample",
            "1",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "approximate validation failed" in captured.err
    assert not json_out.exists()


def test_cli_approx_validation_failure_can_be_written_as_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tame_mt._native")
    json_out = tmp_path / "report.json"

    def fail_validation(**_: object) -> ApproxValidation:
        return ApproxValidation(
            payload={"passed": False, "failures": ["source_top1_agreement 0.0 < 0.95"]},
            failures=["source_top1_agreement 0.0 < 0.95"],
        )

    monkeypatch.setattr("tame_mt.cli.validate_approximate_run", fail_validation)
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
            "--retrieval",
            "approx",
            "--allow-approximate",
            "--validate-approx-sample",
            "1",
            "--allow-approx-validation-failure",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    payload = json.loads(json_out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["approx_validation"]["passed"] is False
    assert any("approximate validation failed" in warning for warning in payload["warnings"])


def test_cli_cached_scoring_rejects_approx_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        [
            "score-cached",
            "--segment-in",
            str(tmp_path / "missing.jsonl"),
            "--ref",
            str(FIXTURES / "test.ref"),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--validate-approx-sample",
            "1",
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "cannot be used with cached scoring" in captured.err


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
    assert cached["backend"]["artifact_backend"] == full["backend"]


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


def test_cli_score_cached_rejects_reference_hash_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    segments = tmp_path / "segments.jsonl"
    json_out = tmp_path / "cached.json"
    stale_ref = tmp_path / "stale.ref"
    stale_ref.write_text("uno\nDOS CAMBIADO\ntres\ncuatro\n", encoding="utf-8")
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
            str(stale_ref),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert full_rc == 0
    assert cached_rc == 2
    assert "reference hash mismatch" in captured.err
    assert not json_out.exists()


def test_cli_score_cached_allows_reference_hash_mismatch_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    segments = tmp_path / "segments.jsonl"
    json_out = tmp_path / "cached.json"
    stale_ref = tmp_path / "stale.ref"
    stale_ref.write_text("uno\nDOS CAMBIADO\ntres\ncuatro\n", encoding="utf-8")
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
            str(stale_ref),
            "--hyp",
            str(FIXTURES / "hyp.out"),
            "--allow-reference-hash-mismatch",
            "--json-out",
            str(json_out),
            "--quiet",
        ]
    )
    captured = capsys.readouterr()

    assert full_rc == 0
    assert cached_rc == 0
    assert "may reuse stale target/pair exposure" in captured.err
    assert json_out.exists()


def test_cli_no_tm_text_segment_artifact_is_not_cacheable(
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
            "--no-tm-text-in-segments",
            "--quiet",
        ]
    )
    row = json.loads(segments.read_text(encoding="utf-8").splitlines()[0])
    metadata = json.loads(segment_metadata_path(segments).read_text(encoding="utf-8"))
    cached_rc = main(
        [
            "score-cached",
            "--segment-in",
            str(segments),
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

    assert full_rc == 0
    assert "tm_hyp" not in row
    assert metadata["privacy"]["tm_text_included"] is False
    assert cached_rc == 2
    assert "does not contain TM hypotheses" in captured.err
    assert not json_out.exists()


def test_cli_score_cached_rejects_legacy_segments_without_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    captured = capsys.readouterr()

    assert full_rc == 0
    assert cached_rc == 2
    assert "segment metadata sidecar is required" in captured.err
    assert not cached_json.exists()


def test_cli_score_cached_accepts_unsafe_legacy_segments_with_override(tmp_path: Path) -> None:
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
            "--allow-unsafe-no-metadata",
            "--json-out",
            str(cached_json),
            "--quiet",
        ]
    )

    assert full_rc == 0
    assert cached_rc == 0
    cached = json.loads(cached_json.read_text(encoding="utf-8"))
    assert "artifact_backend" not in cached["backend"]


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
    assert baseline["backend"]["artifact_backend"] == variant["backend"]["artifact_backend"]
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


def test_cli_index_verify_checks_bundle(tmp_path: Path, capsys) -> None:
    pytest.importorskip("tame_mt._native")
    index_path = tmp_path / "train.tameidx"
    build_rc = main(
        [
            "index",
            "build",
            "--train-src",
            str(FIXTURES / "train.src"),
            "--train-tgt",
            str(FIXTURES / "train.tgt"),
            "--out",
            str(index_path),
            "--quiet",
        ]
    )
    verify_rc = main(
        [
            "index",
            "verify",
            str(index_path),
            "--train-src",
            str(FIXTURES / "train.src"),
            "--train-tgt",
            str(FIXTURES / "train.tgt"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert build_rc == 0
    assert verify_rc == 0
    assert payload["format"] == "tameidx"
    assert payload["num_train"] == 4
    assert payload["train_src_matches"] is True
    assert payload["train_tgt_matches"] is True
    assert "source.index.bin" in payload["checked_native_indexes"]


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


def _without_performance(payload: dict[str, object]) -> dict[str, object]:
    semantic_payload = dict(payload)
    semantic_payload.pop("performance", None)
    return semantic_payload
