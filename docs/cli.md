# CLI Reference

## Full Score

```bash
tame-mt score \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --hyp system.out
```

Outputs a human-readable report to stdout. Optional outputs:

```bash
--json-out report.json
--segment-out segments.jsonl
--tm-out tm.out
```

## Audit

```bash
tame-mt audit \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref
```

Audit mode computes exposure, leakage, bin counts, and TM baseline scores when
references and training targets are available. It does not compute system scores
or delta-over-TM scores.

## Translation-Memory Baseline

```bash
tame-mt tm-baseline \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --out tm.out
```

Optional metadata:

```bash
--metadata-out tm_metadata.jsonl
```

## Cached Scoring

For large training corpora, build a reusable native index once:

```bash
tame-mt index build \
  --train-src train.src \
  --train-tgt train.tgt \
  --out train.tameidx
```

Then score or audit without passing the training files again:

```bash
tame-mt score \
  --index train.tameidx \
  --test-src test.src \
  --ref test.ref \
  --hyp system.out \
  --json-out system.tame.json
```

Inspect metadata without loading the native indexes:

```bash
tame-mt index inspect train.tameidx
```

Index bundles store raw training text and normalized exact-match and pair keys.
Protect them with the same access controls as the original training corpus.

For large corpora or repeated system comparisons, cache segment diagnostics once:

```bash
tame-mt audit \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --segment-out segments.jsonl \
  --json-out audit.json
```

Then score each system without rebuilding the training index:

```bash
tame-mt score-cached \
  --segment-in segments.jsonl \
  --ref test.ref \
  --hyp system.out \
  --num-train 125000 \
  --json-out system.tame.json
```

`score-cached` reuses source/target/pair exposure and TM hypotheses from the
segment JSONL file. It recomputes only system metrics, TM metrics, delta over
TM, bin scores, warnings, and the final report.

Segment rows are validated before scoring. Indices must be unique and
contiguous from `0` to `N-1`; valid rows may appear in any order and are sorted
by index before metrics are computed.

Any corpus, hypothesis, reference, JSON, or JSONL path ending in `.gz` is read
or written as gzip-compressed UTF-8 text.

## Shared Options

```bash
--metrics bleu chrf
--ngram-orders 3,4,5
--far-threshold 0.30
--near-threshold 0.70
--leak-thresholds 0.70,0.85,0.95
--pair-k 50
--index-mode auto
--auto-exact-cutoff 5000
--candidate-gram-limit 8
--posting-limit 500
--max-candidates 3000
--rerank-limit 1000
--min-bin-size-warning 30
--tm-zero-policy empty
--lowercase
--strip-diacritics
--normalize-punctuation
--bleu-tokenize 13a
--bleu-lowercase
--chrf-word-order 2
```

## Doctor

```bash
tame-mt doctor
```

`doctor` prints the TAME-MT version, Python/platform details, SacreBLEU
version, and whether the native Rust backend is importable. Use it first when a
large run seems unexpectedly slow.

## Segment Text Options

By default, segment JSONL contains indices, scores, bins, and TM hypotheses,
but not raw source/reference/hypothesis/neighbor text.

```bash
--include-source-text
--include-reference-text
--include-hyp-text
--include-neighbor-text
```

`--include-neighbor-text` may write raw training text and should be used only
when that is appropriate for the corpus.

## Exit Behavior

TAME-MT returns exit code `0` on success and `2` for user-facing input,
alignment, configuration, or file errors. It preserves empty lines inside input
files as aligned segments, but it rejects empty training-source or test-source
files.

## Metric Selection

Metrics can be passed as separate arguments:

```bash
--metrics bleu chrf
```

or comma-separated:

```bash
--metrics bleu,chrf
```

## Retrieval Performance

`--index-mode auto` is the default. It uses `native_exact` or `native_fast`
when the Rust extension is installed. If the native extension is unavailable,
it falls back to `python_exact` or `python_fast`.

Use exact mode explicitly when the corpus is small or when exact nearest-neighbor
exposure is required:

```bash
--index-mode native_exact
```

Use fast mode explicitly for large public corpora:

```bash
--index-mode native_fast
```

Fast mode selects rare query n-grams, reads bounded postings, keeps an
approximate shortlist, and reranks that shortlist with exact Jaccard similarity.

The older `inverted_exact` and `inverted_fast` names remain accepted as
pure-Python aliases for compatibility.
