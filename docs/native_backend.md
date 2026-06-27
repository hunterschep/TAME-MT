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

If `doctor` reports a native backend version mismatch, rebuild or reinstall the
package. TAME-MT refuses to use a compiled extension whose version differs from
the Python package version, because stale native code could change persistence
or nearest-neighbor behavior without changing the report signature.

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
already-normalized strings, derives character n-grams as UTF-8 byte slices,
interns each distinct n-gram into compact integer IDs, builds integer postings,
and returns deterministically sorted nearest-neighbor results. Exact-match maps
store normalized strings, so exact source/target overlap is literal after
normalization. Ties are resolved by lower training index.

`native_exact` preserves exact nearest-neighbor retrieval over candidates that
share query n-grams. `native_fast` is approximate because it bounds rare-gram
candidate generation before exact Jaccard reranking. The report signature records
the resolved backend.

Corpus-level batch queries and batched pair reranking release the Python GIL and
use Rayon for parallel execution inside Rust. Python still owns file IO,
SacreBLEU scoring, JSON serialization, and report formatting.

During fresh native audits, Python keeps normalized training strings only long
enough to build exact source/target pair keys. It then releases those Python
copies; the Rust indexes retain the compact grams, postings, and exact maps
needed for retrieval. Loaded `.tameidx` bundles follow the same low-memory shape
and do not materialize Python normalized training-line copies.

Pair exposure uses native pair reranking when both source and target indexes are
native. Python builds deterministic candidate ID lists from source and target
top-k results, then Rust scores each source/reference query against the shared
candidate set and returns the best paired neighbor. The corpus path batches
those pair reranks through one native call, which reduces Python/Rust boundary
overhead at large test sizes. Pure-Python indexes use the same scoring
semantics through a fallback path.

## Persistent Indexes

The native index can serialize its compact n-gram-to-ID tables, postings,
document gram sets, and exact-match map with `bincode`. The public workflow is:

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

Bundles record both the outer `.tameidx` format version and the private native
index schema version. If either version is unsupported, TAME-MT rejects the
bundle before deserializing native bytes. Rebuild the index with the current
`tame-mt index build` command after native-index upgrades.

Bundle loading also validates strict manifest field types, unexpected or
duplicate ZIP member names, total uncompressed size, member-specific hard caps,
declared native-member byte sizes, and compression ratios before native
deserialization. This keeps corrupt, hand-edited, or zip-bomb-style `.tameidx`
files from silently loading under the wrong settings or forcing unbounded reads.

Bundles are low-compression zip containers by default. Level-1 deflate keeps
load time low while avoiding very large cache artifacts on public-corpus-scale
training sets. Because bundles contain raw training text and normalized
exact-match and pair keys, they should be handled as training data.

Bundle writes are atomic at the file level: `tame-mt index build` writes a
temporary file in the destination directory and replaces the requested output
only after the ZIP archive closes successfully. If the write fails, an existing
bundle at that path is left untouched.
