#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]+)?$")


@dataclass(frozen=True, slots=True)
class VersionSource:
    name: str
    path: Path
    pattern: re.Pattern[str]


VERSION_SOURCES = (
    VersionSource(
        "pyproject.toml",
        ROOT / "pyproject.toml",
        re.compile(r'(?m)^version\s*=\s*"([^"]+)"'),
    ),
    VersionSource(
        "Cargo.toml",
        ROOT / "Cargo.toml",
        re.compile(r'(?m)^version\s*=\s*"([^"]+)"'),
    ),
    VersionSource(
        "src/tame_mt/version.py",
        ROOT / "src/tame_mt/version.py",
        re.compile(r'(?m)^__version__\s*=\s*"([^"]+)"'),
    ),
    VersionSource(
        "CITATION.cff",
        ROOT / "CITATION.cff",
        re.compile(r'(?m)^version:\s*"([^"]+)"'),
    ),
)

DOC_VERSION_PATTERNS = (
    (ROOT / "README.md", re.compile(r"(?m)\bversion = \{([^}]+)\}")),
    (ROOT / "README.md", re.compile(r"(?m)tame-mt\|v:([0-9][^|]+)\|")),
    (ROOT / "README.md", re.compile(r'(?m)"tame_version":\s*"([^"]+)"')),
    (ROOT / "docs/json_schema.md", re.compile(r"(?m)tame-mt\|v:([0-9][^|]+)\|")),
    (ROOT / "docs/json_schema.md", re.compile(r'(?m)"tame_version":\s*"([^"]+)"')),
)


def main() -> int:
    errors = _version_errors()
    if errors:
        for error in errors:
            print(f"check_versions: {error}", file=sys.stderr)
        return 1
    version = _collect_versions()[VERSION_SOURCES[0].name]
    print(f"All release version references match {version}")
    return 0


def _version_errors() -> list[str]:
    versions = _collect_versions()
    errors: list[str] = []
    for name, version in versions.items():
        if not VERSION_PATTERN.fullmatch(version):
            errors.append(f"{name} has invalid version {version!r}")

    unique_versions = sorted(set(versions.values()))
    if len(unique_versions) != 1:
        rendered = ", ".join(f"{name}={version}" for name, version in sorted(versions.items()))
        errors.append(f"version mismatch: {rendered}")
    if not errors:
        expected = unique_versions[0]
        errors.extend(_doc_version_errors(expected))
    return errors


def _collect_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for source in VERSION_SOURCES:
        text = source.path.read_text(encoding="utf-8")
        match = source.pattern.search(text)
        if match is None:
            raise SystemExit(f"check_versions: {source.name} does not define a version")
        versions[source.name] = match.group(1)
    return versions


def _doc_version_errors(expected: str) -> list[str]:
    errors: list[str] = []
    for path, pattern in DOC_VERSION_PATTERNS:
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            observed = match.group(1)
            if observed != expected:
                errors.append(
                    f"{path.relative_to(ROOT)} contains version {observed!r}, expected {expected!r}"
                )
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
