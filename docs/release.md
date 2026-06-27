# Release Process

TAME-MT releases should be reproducible, tested from built wheels, and published
only through PyPI trusted publishing.

## Local Dry Run

Run the full acceptance script from a clean checkout:

```bash
scripts/acceptance.sh
```

That script installs the dev extra, checks formatting, linting, typing, Rust
format/clippy/tests, Python tests, small and 100k-scale performance guards,
fast-retrieval recall characterization, source/wheel build checks, clean-venv
wheel smoke tests, CLI smoke tests, index reuse, cached scoring, and the public
corpus demo.

Before committing a release candidate, remove generated artifacts:

```bash
rm -rf build dist src/tame_mt.egg-info
find src/tame_mt -maxdepth 1 \( -name '_native*.so' -o -name '_native*.pyd' -o -name '_native*.dylib' \) -delete
```

## Version And Changelog

1. Update the package version in `pyproject.toml`, `Cargo.toml`, and
   `src/tame_mt/version.py`. Also update `CITATION.cff` and README/schema
   examples that embed the release version.
2. Update `CHANGELOG.md`.
3. Run `python scripts/check_versions.py`.
4. Confirm `tame-mt doctor` reports matching Python and native versions from a
   freshly built wheel.

## GitHub Release Workflow

The `Release` workflow supports two modes:

- `workflow_dispatch` from a branch or tag builds and validates distributions as
  a dry run.
- pushing a `v*` tag builds the source distribution and wheels, runs wheel smoke
  tests, runs `twine check`, and generates an SBOM artifact. It does not
  publish automatically.
- `workflow_dispatch` from a `v*` tag with `publish=true` publishes the already
  validated artifacts to PyPI through trusted publishing after the protected
  `pypi` environment allows the job to proceed.

Publishing requires the `pypi` GitHub environment to be configured for trusted
publishing on PyPI. Do not use long-lived PyPI API tokens for normal releases.
The workflow pins third-party actions to immutable commit SHAs; update those
pins deliberately during release-maintenance work rather than floating them on
every run.

## Supply-Chain Checks

CI runs:

- Python dependency audit with `pip-audit`;
- Rust dependency audit with `cargo audit`;
- SacreBLEU compatibility tests for the minimum supported 2.x range and the
  current supported 2.x range;
- cross-platform wheel builds and installed-wheel smoke tests.
- a dedicated larger staged performance smoke test in CI, plus the heavier
  local 100k benchmark in `scripts/acceptance.sh`.

Release artifacts should include:

- `.tar.gz` source distribution;
- platform wheels for CPython 3.10 through 3.13;
- build provenance attestations;
- SPDX JSON SBOM artifact from the release workflow.

## Post-Release Smoke

After PyPI publishing finishes:

```bash
python -m venv /tmp/tame-mt-release-smoke
/tmp/tame-mt-release-smoke/bin/python -m pip install tame-mt
/tmp/tame-mt-release-smoke/bin/python -m tame_mt doctor
/tmp/tame-mt-release-smoke/bin/python scripts/wheel_smoke.py
```

Delete the smoke environment after verification.
