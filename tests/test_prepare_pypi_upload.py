from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prepare_pypi_upload.py"


def test_prepare_pypi_upload_keeps_new_distributions(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    wheel = _write_dist(dist, "tame_mt-0.2.1-py3-none-any.whl", b"new")
    existing_json = _write_existing_json(tmp_path, {})
    output = tmp_path / "github_output.txt"

    result = _run_prepare(dist, existing_json, output)

    assert result.returncode == 0
    assert wheel.exists()
    assert _output_values(output)["should_publish"] == "true"


def test_prepare_pypi_upload_removes_matching_existing_distributions(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    wheel = _write_dist(dist, "tame_mt-0.2.0-py3-none-any.whl", b"published")
    sdist = _write_dist(dist, "tame_mt-0.2.0.tar.gz", b"published-sdist")
    existing_json = _write_existing_json(
        tmp_path,
        {
            wheel.name: _sha256(wheel),
            sdist.name: _sha256(sdist),
        },
    )
    output = tmp_path / "github_output.txt"

    result = _run_prepare(dist, existing_json, output)

    assert result.returncode == 0
    assert not wheel.exists()
    assert not sdist.exists()
    assert _output_values(output)["should_publish"] == "false"


def test_prepare_pypi_upload_keeps_missing_files_after_removing_matches(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    existing_wheel = _write_dist(dist, "tame_mt-0.2.0-py3-none-any.whl", b"published")
    missing_wheel = _write_dist(dist, "tame_mt-0.2.0-cp312-cp312-win_amd64.whl", b"missing")
    existing_json = _write_existing_json(
        tmp_path,
        {
            existing_wheel.name: _sha256(existing_wheel),
        },
    )
    output = tmp_path / "github_output.txt"

    result = _run_prepare(dist, existing_json, output)

    assert result.returncode == 0
    assert not existing_wheel.exists()
    assert missing_wheel.exists()
    assert _output_values(output)["should_publish"] == "true"


def test_prepare_pypi_upload_rejects_existing_filename_with_different_hash(
    tmp_path: Path,
) -> None:
    dist = tmp_path / "dist"
    wheel = _write_dist(dist, "tame_mt-0.2.0-py3-none-any.whl", b"local")
    existing_json = _write_existing_json(
        tmp_path,
        {
            wheel.name: hashlib.sha256(b"different").hexdigest(),
        },
    )
    output = tmp_path / "github_output.txt"

    result = _run_prepare(dist, existing_json, output)

    assert result.returncode == 1
    assert wheel.exists()
    assert "local file hash differs" in result.stderr
    assert not output.exists()


def _run_prepare(
    dist: Path,
    existing_json: Path,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GITHUB_OUTPUT"] = str(output)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project",
            "tame-mt",
            "--version",
            "0.2.0",
            "--dist",
            str(dist),
            "--existing-json",
            str(existing_json),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_dist(dist: Path, filename: str, content: bytes) -> Path:
    dist.mkdir(parents=True, exist_ok=True)
    path = dist / filename
    path.write_bytes(content)
    return path


def _write_existing_json(tmp_path: Path, hashes: dict[str, str]) -> Path:
    urls = [
        {
            "filename": filename,
            "digests": {"sha256": sha256},
        }
        for filename, sha256 in hashes.items()
    ]
    path = tmp_path / "existing.json"
    path.write_text(json.dumps({"urls": urls}), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_values(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())
