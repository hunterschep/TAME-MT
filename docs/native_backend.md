# Native Backend

TAME-MT ships an internal Rust extension, `tame_mt._native`, built with PyO3 and
maturin. Public Python users should import `tame_mt`, not the private extension.

## Check Installation

```bash
tame-mt doctor
```

If the native backend is installed, `doctor` reports:

```text
Native backend: available
Default backend: auto -> native_exact/native_fast
```

If it is unavailable, `auto` falls back to the pure-Python exact/fast backends.
That fallback preserves usability, but large audits will be slower.

## Build From Source

Install Rust and then run:

```bash
pip install -e '.[dev]'
cargo test
pytest
```

Build release artifacts:

```bash
python -m build
python -m twine check dist/*
```

## Semantics

Python owns normalization and report generation. The native layer receives
already-normalized strings, builds character n-gram postings, and returns
deterministically sorted nearest-neighbor results. Ties are resolved by lower
training index.

`native_exact` preserves exact nearest-neighbor retrieval over candidates that
share query n-grams. `native_fast` is approximate because it bounds rare-gram
candidate generation before exact Jaccard reranking. The report signature records
the resolved backend.

Corpus-level batch queries release the Python GIL and use Rayon for parallel
query execution inside Rust. Python still owns file IO, SacreBLEU scoring, JSON
serialization, and report formatting.
