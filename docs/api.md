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
report.backend
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
from tame_mt import ScoreConfig, TameScorer, load_index_bundle, save_index_bundle

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

Index bundles store raw training text and normalized exact-match and pair keys.
Treat them as training data.

## Cached Scoring

For a fixed train/test/reference setup, compute exposure once with
`evaluate_corpus()` or `audit_files()`, write segment JSONL, and reuse those
artifacts for later system outputs.

```python
from tame_mt.artifacts import read_segment_jsonl
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

## Custom Configuration

```python
from tame_mt import BinConfig, IndexConfig, ScoreConfig, SimilarityConfig

config = ScoreConfig(
    metrics=("bleu", "chrf"),
    similarity=SimilarityConfig(ngram_orders=(2, 3, 4, 5)),
    index=IndexConfig(topk=100),
    bins=BinConfig(far_threshold=0.25, near_threshold=0.75),
)
```

Metric-changing configuration and the resolved retrieval backend are recorded in
the report signature.

## Exceptions

TAME-MT raises subclasses of `tame_mt.exceptions.TameMTError` for user-facing
failures:

```python
from tame_mt.exceptions import AlignmentError, BackendError, ConfigurationError, InputDataError
```

`AlignmentError` means aligned files have different lengths. `ConfigurationError`
means a config value is invalid. `InputDataError` means the corpus is
structurally invalid, such as an empty training source file. `BackendError`
means an explicitly requested native backend was unavailable or failed to build.
