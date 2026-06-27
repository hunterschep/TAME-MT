import re
import subprocess
import sys
from pathlib import Path

import pytest

from tame_mt import __version__
from tame_mt import native as native_mod

ROOT = Path(__file__).resolve().parents[1]


def test_static_package_versions_match() -> None:
    assert _metadata_version(ROOT / "pyproject.toml") == __version__
    assert _metadata_version(ROOT / "Cargo.toml") == __version__


def test_release_version_check_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_versions.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout


def test_native_version_matches_python_version_when_available() -> None:
    native_module = pytest.importorskip("tame_mt._native")

    assert native_module.native_version() == __version__


def test_native_status_rejects_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeNativeModule:
        @staticmethod
        def native_version() -> str:
            return "0.0.0"

    monkeypatch.setattr(native_mod, "import_module", lambda name: FakeNativeModule())

    status = native_mod.native_status()

    assert status.available is False
    assert status.version == "0.0.0"
    assert status.error is not None
    assert "does not match" in status.error


def test_native_loader_rejects_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeNativeModule:
        @staticmethod
        def native_version() -> str:
            return "0.0.0"

    monkeypatch.setattr(native_mod, "import_module", lambda name: FakeNativeModule())

    with pytest.raises(RuntimeError, match="does not match"):
        native_mod.build_native_index(
            normalized_lines=[],
            ngram_orders=(3,),
            mode="fast",
            candidate_gram_limit=8,
            posting_limit=500,
            max_candidates=3000,
            rerank_limit=1000,
        )


def _metadata_version(path: Path) -> str:
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"{path.name} does not define a version")
    return match.group(1)
