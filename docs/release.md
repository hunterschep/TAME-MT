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
   `src/tame_mt/version.py`.
2. Update `CHANGELOG.md`.
3. Confirm `tame-mt doctor` reports matching Python and native versions from a
   freshly built wheel.

## GitHub Release Workflow

The `Release` workflow supports two modes:

- `workflow_dispatch` from a branch or tag builds and validates distributions as
  a dry run.
- pushing a `v*` tag builds the source distribution and wheels, runs wheel smoke
  tests, runs `twine check`, generates an SBOM artifact, attests build
  provenance, and publishes to PyPI through trusted publishing.

Publishing requires the `pypi` GitHub environment to be configured for trusted
publishing on PyPI. Do not use long-lived PyPI API tokens for normal releases.

## Supply-Chain Checks

CI runs:

- Python dependency audit with `pip-audit`;
- Rust dependency audit with `cargo audit`;
- SacreBLEU compatibility tests for the minimum supported 2.x range and the
  current supported 2.x range;
- cross-platform wheel builds and installed-wheel smoke tests.

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
