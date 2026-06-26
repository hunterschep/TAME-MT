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

## Shared Options

```bash
--metrics bleu chrf
--ngram-orders 3,4,5
--far-threshold 0.30
--near-threshold 0.70
--leak-thresholds 0.70,0.85,0.95
--pair-k 50
--lowercase
--strip-diacritics
--normalize-punctuation
--bleu-tokenize 13a
--bleu-lowercase
--chrf-word-order 2
```

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
