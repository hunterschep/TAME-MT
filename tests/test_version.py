import re
from pathlib import Path

import pytest

from tame_mt import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_static_package_versions_match() -> None:
    assert _metadata_version(ROOT / "pyproject.toml") == __version__
    assert _metadata_version(ROOT / "Cargo.toml") == __version__


def test_native_version_matches_python_version_when_available() -> None:
    native_module = pytest.importorskip("tame_mt._native")

    assert native_module.native_version() == __version__


def _metadata_version(path: Path) -> str:
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"{path.name} does not define a version")
    return match.group(1)
