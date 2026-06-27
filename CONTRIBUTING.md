# Contributing

TAME-MT is intended to be a small, deterministic, low-dependency evaluation
tool. Contributions should preserve that shape.

## Development Setup

```bash
pip install -e '.[dev]'
pytest
ruff check .
mypy src/tame_mt
python -m build
python -m twine check dist/*
```

## Design Constraints

- Do not add heavyweight runtime dependencies without a strong reason.
- Do not download models or external resources at runtime.
- Do not change metric definitions without updating the signature and
  changelog.
- Do not print training-neighbor text by default.
- Keep source-side nearest-neighbor retrieval as the TM baseline behavior.
- Add tests for user-visible CLI, JSON, or metric-definition changes.

## Release Checklist

Use [`docs/release.md`](docs/release.md) as the canonical checklist.

1. Update `CHANGELOG.md`.
2. Run `scripts/acceptance.sh`.
3. Confirm the staged synthetic benchmark output stays under the release
   thresholds for fresh, indexed, cached, prepared cached, and batch cached
   scoring.
4. Confirm the fast-retrieval recall validation, built-wheel smoke test, and
   `twine check` pass.
5. Confirm CI dependency audits and SacreBLEU compatibility jobs pass.
6. Remove generated `dist/`, build, and editable native-extension artifacts
   before committing.
