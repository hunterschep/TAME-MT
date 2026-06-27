# Python API

TAME-MT exposes both a simple file API and a lower-level evaluation API that
returns reusable artifacts.

## File Scoring

```python
from tame_mt import TameScorer

report = TameScorer().score_files(
    train_src="train.src",
    train_tgt="train.tgt",
    test_src="test.src",
    refs=["test.ref"],
    hyp="system.out",
)
```

`report` is a `TameReport` dataclass with:

```python
report.system_scores
report.tm_scores
report.delta_scores
report.exposure
report.bins
report.generalization_gap
report.warnings
report.retrieval
report.backend
report.performance
report.signature
```

Use `report.to_dict()` for structured data or `report.to_json()` for serialized
JSON.

## In-Memory Scoring

```python
from tame_mt import ScoreConfig, TameScorer

scorer = TameScorer(ScoreConfig())

report = scorer.score_corpus(
    train_src=train_src_lines,
    train_tgt=train_tgt_lines,
    test_src=test_src_lines,
    refs=[ref_lines],
    hyp=hyp_lines,
)
```

## Full Evaluation Result

When you need segment diagnostics or TM outputs:

```python
result = scorer.evaluate_corpus(
    train_src=train_src_lines,
    train_tgt=train_tgt_lines,
    test_src=test_src_lines,
    refs=[ref_lines],
    hyp=hyp_lines,
)

report = result.report
segments = result.exposures
tm_hypotheses = result.tm_hyp
tm_metadata = result.tm_results
```

## Persistent Index Bundles

For repeated runs over the same training corpus, build and save a native index
bundle:

```python
from tame_mt import (
    ScoreConfig,
    TameScorer,
    load_index_bundle,
    save_index_bundle,
    verify_index_bundle,
)

config = ScoreConfig()
save_index_bundle(
    "train.tameidx",
    train_src=train_src_lines,
    train_tgt=train_tgt_lines,
    config=config,
)

bundle = load_index_bundle("train.tameidx", config)
result = TameScorer(config).evaluate_index_bundle(
    bundle=bundle,
    test_src=test_src_lines,
    refs=[ref_lines],
    hyp=hyp_lines,
)
```

`result.report.backend["index_reused"]` is `True` for this path. Bundle loading
validates normalization, similarity, backend mode, and fast-mode settings before
scoring so stale indexes do not silently produce mismatched signatures.
Exact bundles can still be reused with different query-time settings such as
`IndexConfig.topk` and `IndexConfig.batch_size`; those settings affect
evaluation, not the persisted index bytes.
It also enforces a default uncompressed load-memory budget before reading raw
training text, native index bytes, or exact-pair fingerprints into memory. Pass
`max_load_bytes=...` only for trusted large bundles on machines with enough
RAM, or `max_load_bytes=None` to disable that guard deliberately.

Use `verify_index_bundle()` to validate a bundle before reuse or transfer:

```python
verification = verify_index_bundle(
    "train.tameidx",
    train_src=train_src_lines,
    train_tgt=train_tgt_lines,
)
print(verification.to_dict()["checked_hashes"])
```

Index bundles store raw training text plus exact-match and pair fingerprints.
Treat them as training data.

## Retrieval Internals

Public scoring code should normally use `TameScorer`, but lower-level retrieval
types are available from `tame_mt.index` for tests, diagnostics, and benchmark
work:

```python
from tame_mt.index import (
    NgramInvertedIndex,
    PythonExactSimilarityIndex,
    SimilarityIndex,
)
```

`NgramInvertedIndex` is the production native-backed wrapper. It resolves
`auto` to `native_exact` when the installed extension is available and refuses
to run if the extension is missing. `PythonExactSimilarityIndex` is a small
reference implementation used for parity tests and debugging; it is not the CLI
fallback for large or production audits.

## Cached Scoring

For a fixed train/test/reference setup, compute exposure once with
`evaluate_corpus()` or `audit_files()`, write a cache artifact, and reuse that
artifact for later system outputs.

```python
from tame_mt import load_cached_artifact
from tame_mt.io import read_lines

refs = [read_lines("test.ref")]
artifact = load_cached_artifact(
    "segments.tamecache",
    refs=refs,
    config=scorer.config,
)

report = scorer.score_from_cached_artifact(
    artifact,
    refs=refs,
    hyp=read_lines("system.out"),
)
```

This path does not inspect the training corpus or rebuild nearest-neighbor
indexes. The loader reads the `.meta.json` sidecar, validates the scorer
configuration, infers `num_train`, checks reference hashes, checks TM-hypothesis
hashes, and rejects privacy-safe diagnostics that omit TM text.

For many systems on the same cached diagnostics, score them together:

```python
reports = scorer.score_many_from_cached_artifact(
    artifact,
    refs=[read_lines("test.ref")],
    systems={
        "system_a": read_lines("system_a.out"),
        "system_b": read_lines("system_b.out"),
    },
)

system_a_report = reports["system_a"]
```

Batch artifact scoring validates segment diagnostics once, reuses the same
reference cache for every system, and computes TM baseline scores once for the
batch.

For services, notebooks, and leaderboards that receive hypotheses over time,
prepare a cached scorer once and reuse it:

```python
cached = scorer.prepare_from_cached_artifact(
    artifact,
    refs=[read_lines("test.ref")],
)

system_a_report = cached.score(read_lines("system_a.out"), system_name="system_a")

later_reports = cached.score_many(
    {
        "system_b": read_lines("system_b.out"),
        "system_c": read_lines("system_c.out"),
    }
)
```

The prepared scorer owns a validated snapshot of the segment diagnostics,
references, exposure bins, SacreBLEU reference caches, and TM baseline scores.
Mutating the original artifact/reference lists after preparation does not
change later `score()` or `score_many()` calls.

Artifact indices are validated and canonicalized before scoring. They must be
unique and contiguous from `0` to `N-1`; valid rows may be supplied out of order.
Cached segment rows also carry their exposure-bin labels.
`score_from_cached_artifact()` and prepared cached scorers verify that each
stored bin still matches the current `ScoreConfig.bins`; if you change
`far_threshold` or `near_threshold`, rerun the audit or score cached artifacts
with the original bin settings.

The CLI writes a `.meta.json` sidecar next to new diagnostic/cache JSONL outputs
and validates it on cached CLI runs. `load_cached_artifact()` uses the same
rules.
By default, missing metadata is a hard error. For trusted legacy artifacts only:

```python
legacy_artifact = load_cached_artifact(
    "legacy_segments.jsonl",
    refs=refs,
    config=scorer.config,
    num_train=125000,
    allow_unsafe_no_metadata=True,
)
```

Metadata validation checks normalization, similarity, retrieval settings, TM
zero policy, bin thresholds, reference content hashes, and TM hypothesis hashes.
The artifact object stores the original backend metadata so generated reports
preserve provenance:

```python
report = scorer.score_from_cached_artifact(
    artifact,
    refs=[read_lines("test.ref")],
    hyp=read_lines("system.out"),
)

print(report.backend["artifact_backend"]["name"])
```

The lower-level `read_segment_jsonl()`, `read_segment_metadata()`, and
`validate_segment_metadata()` helpers remain public for advanced integrations,
but new applications should prefer `load_cached_artifact()`.

Cache artifacts contain TM baseline hypotheses. Diagnostic artifacts written by
`--diagnostic-out` omit TM hypotheses by default; they are useful for private
exposure diagnostics but cannot be used for cached scoring because TM-BLEU would
be incorrect.

## Custom Configuration

```python
from tame_mt import BinConfig, IndexConfig, MetricConfig, ScoreConfig, SimilarityConfig

config = ScoreConfig(
    metrics=("bleu", "chrf"),
    similarity=SimilarityConfig(ngram_orders=(2, 3, 4, 5)),
    index=IndexConfig(topk=100, batch_size=4096),
    bins=BinConfig(far_threshold=0.25, near_threshold=0.75),
    metric=MetricConfig(bleu_tokenize="13a", chrf_word_order=2),
)
```

Metric-changing configuration, retrieval mode, approximation flag, resolved
backend, and metric-affecting dependency versions are recorded in the report
signature. Default `ScoreConfig()` uses exact retrieval. Approximate retrieval
must be requested explicitly:

```python
from tame_mt import RetrievalConfig, ScoreConfig

config = ScoreConfig(
    retrieval=RetrievalConfig(mode="approx", allow_approximate=True)
)
```

Configuration dataclasses validate both structure and scalar values at
construction time. Integer settings reject booleans and floats, boolean flags
must be real booleans, exposure thresholds must be finite numbers in \([0, 1]\),
Unicode normalization must be one of the supported forms, nested config objects
must be the expected dataclass types, and metrics must be an ordered sequence
with no duplicates. Invalid values raise `ConfigurationError` before any corpus
work starts.

## Exceptions

TAME-MT raises subclasses of `tame_mt.exceptions.TameMTError` for user-facing
failures:

```python
from tame_mt.exceptions import (
    AlignmentError,
    BackendError,
    ConfigurationError,
    InputDataError,
    OutputError,
)
```

`AlignmentError` means aligned files have different lengths. `ConfigurationError`
means a config value is invalid. `InputDataError` means the corpus is
structurally invalid, such as an empty training source file. `BackendError`
means an explicitly requested native backend was unavailable or failed to build.
`OutputError` means TAME-MT could not serialize or write an output artifact.
