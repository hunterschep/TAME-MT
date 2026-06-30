#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dist_dir = Path(args.dist)
    distributions = _distribution_files(dist_dir)
    if not distributions:
        print(f"no distributions found in {dist_dir}", file=sys.stderr)
        return 2

    existing = _load_existing_hashes(
        project=args.project,
        version=args.version,
        existing_json=Path(args.existing_json) if args.existing_json else None,
    )
    removed: list[str] = []
    remaining: list[str] = []

    for path in distributions:
        expected_sha256 = existing.get(path.name)
        if expected_sha256 is None:
            remaining.append(path.name)
            continue
        local_sha256 = _sha256(path)
        if local_sha256 != expected_sha256:
            print(
                (
                    f"PyPI already has {path.name}, but the local file hash differs: "
                    f"local sha256={local_sha256}, PyPI sha256={expected_sha256}. "
                    "Do not overwrite or reuse a published filename; release a new version."
                ),
                file=sys.stderr,
            )
            return 1
        path.unlink()
        removed.append(path.name)

    should_publish = bool(remaining)
    _write_output("should_publish", str(should_publish).lower())
    _write_summary(args.project, args.version, removed, remaining)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove already-published PyPI distributions before a retry-safe upload.",
    )
    parser.add_argument("--project", required=True, help="PyPI project name")
    parser.add_argument("--version", required=True, help="release version without leading v")
    parser.add_argument("--dist", default="dist", help="distribution directory")
    parser.add_argument(
        "--existing-json",
        help="test hook: read PyPI release JSON from this file instead of pypi.org",
    )
    return parser.parse_args(argv)


def _distribution_files(dist_dir: Path) -> list[Path]:
    if not dist_dir.exists():
        return []
    return sorted(
        path
        for path in dist_dir.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )


def _load_existing_hashes(
    *,
    project: str,
    version: str,
    existing_json: Path | None,
) -> dict[str, str]:
    if existing_json is not None:
        payload = json.loads(existing_json.read_text(encoding="utf-8"))
    else:
        url = f"https://pypi.org/pypi/{quote(project, safe='')}/{quote(version, safe='')}/json"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {}
            raise
    urls = payload.get("urls", [])
    if not isinstance(urls, list):
        raise ValueError("PyPI JSON field 'urls' must be a list")
    existing: dict[str, str] = {}
    for item in urls:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        digests = item.get("digests")
        if not isinstance(filename, str) or not isinstance(digests, dict):
            continue
        sha256 = digests.get("sha256")
        if isinstance(sha256, str):
            existing[filename] = sha256
    return existing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    else:
        print(f"{name}={value}")


def _write_summary(
    project: str,
    version: str,
    removed: list[str],
    remaining: list[str],
) -> None:
    lines = [
        f"## PyPI upload preflight for `{project}=={version}`",
        "",
        f"- already published and removed locally: {len(removed)}",
        f"- remaining to publish: {len(remaining)}",
    ]
    if removed:
        lines.extend(["", "Already present on PyPI with matching SHA-256:"])
        lines.extend(f"- `{filename}`" for filename in removed)
    if remaining:
        lines.extend(["", "Will upload:"])
        lines.extend(f"- `{filename}`" for filename in remaining)
    else:
        lines.extend(["", "All distributions already exist on PyPI; upload step can be skipped."])

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
