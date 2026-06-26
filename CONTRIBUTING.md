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

1. Update `CHANGELOG.md`.
2. Confirm `pytest`, `ruff check .`, and `mypy src/tame_mt` pass.
3. Run `python -m build`.
4. Run `python -m twine check dist/*`.
5. Smoke-test the toy example.
