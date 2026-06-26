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
already-normalized strings, interns character n-grams into compact integer IDs,
builds integer postings, and returns deterministically sorted nearest-neighbor
results. Ties are resolved by lower training index.

`native_exact` preserves exact nearest-neighbor retrieval over candidates that
share query n-grams. `native_fast` is approximate because it bounds rare-gram
candidate generation before exact Jaccard reranking. The report signature records
the resolved backend.

Corpus-level batch queries release the Python GIL and use Rayon for parallel
query execution inside Rust. Python still owns file IO, SacreBLEU scoring, JSON
serialization, and report formatting.

Pair exposure uses bulk candidate scoring: each source/reference query is
featurized once, then scored against the candidate ID set. This avoids the
per-candidate n-gram regeneration that makes pure Python nearest-neighbor loops
slow at corpus scale.

## Persistent Indexes

The native index can serialize its compact gram-ID tables, postings, document
gram sets, and exact-match map with `bincode`. The public workflow is:

```bash
tame-mt index build \
  --train-src train.src \
  --train-tgt train.tgt \
  --out train.tameidx

tame-mt score \
  --index train.tameidx \
  --test-src test.src \
  --ref test.ref \
  --hyp system.out
```

Loading a `.tameidx` bundle deserializes the native source/target indexes
instead of reconstructing character n-grams and postings from the training
files. The bundle also stores raw training lines so TM outputs, exact pair
overlap, and optional neighbor-text segment reports remain identical to a fresh
run.

Bundles are uncompressed zip containers by default. This favors load speed over
minimum disk size. Because they contain raw training text and normalized
exact-match keys, they should be handled as training data.
