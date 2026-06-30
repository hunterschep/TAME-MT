# TAME-MT Release Checklist

Use this checklist for every public release. Release artifacts should be built
and published by GitHub Actions with PyPI trusted publishing, not by manually
uploading a local `dist/` directory.

## 1. Confirm Scope

- Confirm the version in `pyproject.toml`, `Cargo.toml`,
  `src/tame_mt/version.py`, `CITATION.cff`, and public examples is the intended
  release version.
- Confirm `CHANGELOG.md` has a dated entry for the release.
- Confirm no raw benchmark corpora, local caches, build products, or native
  extension artifacts are staged.

## 2. Local Gate

Run the full local release gate from a clean worktree:

```bash
scripts/acceptance.sh
python scripts/check_versions.py
python -m twine check dist/*
```

The acceptance script must pass before tagging. It covers Python and Rust static
checks, full tests, dependency audits, exact/approx validation, synthetic
performance thresholds, wheel build checks, wheel smoke, cached scoring, index
verification, and the OPUS-100 standard demo smoke.

## 3. Commit And Tag

```bash
git status --short
git add -A
git diff --cached --stat
git commit -m "Prepare TAME-MT X.Y.Z"
git push origin main
git tag -a vX.Y.Z -m "TAME-MT vX.Y.Z"
git push origin vX.Y.Z
```

Do not move a published tag. If a pushed tag fails validation before PyPI
publication, delete and recreate it only after confirming no package has been
published for that version.

## 4. Validate GitHub Release Artifacts

The tag-triggered `Release` workflow must pass. It builds the sdist, builds
cross-platform wheels for CPython 3.10 through 3.13, runs installed-wheel smoke
tests, runs `twine check`, installs a Linux CPython 3.12 wheel from the release
artifacts, runs small synthetic benchmark smoke checks from that installed
artifact, and uploads the SBOM.

```bash
gh run list --workflow Release --limit 5
gh run watch --exit-status
```

Do not publish if the release workflow fails or if any artifact was not produced
by the tag workflow.

## 5. Publish

Publish only from the validated tag through the protected `pypi` environment:

```bash
gh workflow run Release --ref vX.Y.Z -f publish=true
gh run list --workflow Release --limit 5
gh run watch --exit-status
```

The publish job must use trusted publishing and provenance attestation. Do not
use a long-lived PyPI API token for normal releases.

If a publish run is retried after some or all files already reached PyPI, the
workflow preflight must either skip matching already-published files or fail on
same-name/different-hash files. PyPI filenames are immutable; never delete and
reuse a filename to force a different artifact under the same version.

## 6. Post-Release Smoke

After PyPI shows the new version:

```bash
python -m venv /tmp/tame-mt-release-smoke
/tmp/tame-mt-release-smoke/bin/python -m pip install --upgrade pip
/tmp/tame-mt-release-smoke/bin/python -m pip install --no-cache-dir tame-mt==X.Y.Z
/tmp/tame-mt-release-smoke/bin/tame-mt doctor
/tmp/tame-mt-release-smoke/bin/python scripts/wheel_smoke.py
rm -rf /tmp/tame-mt-release-smoke
```

The post-release `doctor` output must show the native backend as available on
the tested platform. If the native backend is unavailable from PyPI for a common
platform, treat the release as broken and document the platform-specific issue.
