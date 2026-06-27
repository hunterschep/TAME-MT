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
It also enforces a default uncompressed load-memory budget before reading raw
training text, native index bytes, or exact-pair keys into memory. Pass
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

Index bundles store raw training text and normalized exact-match and pair keys.
Treat them as training data.

## Cached Scoring

For a fixed train/test/reference setup, compute exposure once with
`evaluate_corpus()` or `audit_files()`, write segment JSONL, and reuse those
artifacts for later system outputs.

```python
from tame_mt import read_segment_jsonl
from tame_mt.io import read_lines

exposures, tm_results = read_segment_jsonl("segments.jsonl")

report = scorer.score_from_artifacts(
    exposures=exposures,
    tm_results=tm_results,
    refs=[read_lines("test.ref")],
    hyp=read_lines("system.out"),
    num_train=125000,
)
```

This path does not inspect the training corpus or rebuild nearest-neighbor
indexes.

For many systems on the same cached diagnostics, score them together:

```python
reports = scorer.score_many_from_artifacts(
    exposures=exposures,
    tm_results=tm_results,
    refs=[read_lines("test.ref")],
    systems={
        "system_a": read_lines("system_a.out"),
        "system_b": read_lines("system_b.out"),
    },
    num_train=125000,
)

system_a_report = reports["system_a"]
```

Batch artifact scoring validates segment diagnostics once, reuses the same
reference cache for every system, and computes TM baseline scores once for the
batch.

For services, notebooks, and leaderboards that receive hypotheses over time,
prepare a cached scorer once and reuse it:

```python
cached = scorer.prepare_from_artifacts(
    exposures=exposures,
    tm_results=tm_results,
    refs=[read_lines("test.ref")],
    num_train=125000,
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
Cached segment rows also carry their exposure-bin labels. `score_from_artifacts`
and prepared cached scorers verify that each stored bin still matches the
current `ScoreConfig.bins`; if you change `far_threshold` or `near_threshold`,
rerun the audit or score cached artifacts with the original bin settings.

The CLI writes a `.meta.json` sidecar next to new segment JSONL outputs and
validates it on cached CLI runs. Python callers can use the same helpers:

```python
from tame_mt import read_segment_metadata, segment_metadata_path, validate_segment_metadata

metadata = read_segment_metadata("segments.jsonl")
if metadata is None:
    raise RuntimeError("cached scoring requires segment metadata")

validate_segment_metadata(
    metadata,
    config=config,
    num_train=125000,
    num_test=len(exposures),
    num_refs=len(refs),
    refs=refs,
    tm_results=tm_results,
    require_tm_text=True,
)

print(segment_metadata_path("segments.jsonl"))
```

Metadata validation checks normalization, similarity, retrieval settings, TM
zero policy, bin thresholds, reference content hashes, and TM hypothesis hashes.
When metadata is available, pass its backend into cached scoring so generated
reports preserve provenance:

```python
artifact_backend = metadata["backend"]

report = scorer.score_from_artifacts(
    exposures=exposures,
    tm_results=tm_results,
    refs=refs,
    hyp=read_lines("system.out"),
    num_train=125000,
    artifact_backend=artifact_backend,
)

print(report.backend["artifact_backend"]["name"])
```

Segment JSONL contains TM baseline hypotheses by default. If a segment artifact
was written with `--no-tm-text-in-segments`, it is useful for private exposure
diagnostics but cannot be used for cached scoring because TM-BLEU would be
incorrect.

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
