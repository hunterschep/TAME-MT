# OPUS-100 Demo Performance Notes

This demo originally exposed an important issue: train-test nearest-neighbor
search is more expensive than BLEU because it has to inspect training data.
TAME-MT now uses native exact retrieval by default for paper-facing demo
summaries and reserves approximate retrieval for explicitly labeled exploratory
runs.

## Retrieval Modes

- `native_exact`: Rust exact character n-gram Jaccard retrieval. This is the
  default for release demo summaries and paper-facing numbers.
- `native_fast`: Rust rare-gram candidate generation plus exact Jaccard
  reranking of a bounded shortlist. Use only with `--retrieval approx`,
  `--allow-approximate`, and validation for paper-critical work.
- `auto`: resolves to the safest configured backend for the selected retrieval
  mode. With the native extension installed, exact demo runs resolve to
  `native_exact`.

## Cached Scoring Workflow

For repeated runs over the same training corpus, build a reusable native index:

```bash
tame-mt index build \
  --train-src train.src \
  --train-tgt train.tgt \
  --out train.tameidx

tame-mt score \
  --index train.tameidx \
  --test-src test.src \
  --ref test.ref \
  --json-out audit.json
```

For repeated system comparisons on the same train/test/reference setup, compute
train-aware diagnostics once:

```bash
tame-mt score \
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
  --num-train 50000 \
  --json-out system.tame.json
```

## Local Timing Snapshot

On the local development machine, using OPUS-100 `de-en` capped at 50,000 train
pairs and 2,000 test pairs:

| Step | Backend | Time |
| --- | --- | ---: |
| `tame-mt demo opus100 --standard --pair de-en --retrieval exact` after download cache | `native_exact` | ~3.2s |
| Four-pair OPUS-100 standard demo after download cache | `native_exact` | ~9.9s |
| `tame-mt demo opus100 --quick --retrieval exact` after download cache | `native_exact` | ~0.6s |
| Synthetic 100k train / 2k test, fresh audit | `native_exact` | ~4.1s |
| Synthetic 100k train / 2k test, audit from `.tameidx` | `native_exact`, reused index | ~2.1s |
| Synthetic 100k train / 2k test, prepared cached score | cached diagnostics | ~0.2s |

The index-build step is the reusable training-corpus cost. The audit step is
the train/test/reference diagnostic cost. The cached-score step is the
repeated-system cost and is much closer to ordinary BLEU/chrF evaluation.

`.tameidx` bundles use low-compression ZIP storage by default. Bundle size
depends on corpus size, script, target availability, and exact-pair overlap
storage, so report the actual file size alongside timing results.

These timings are a smoke benchmark, not a formal performance claim. They are
included to make the intended large-corpus workflow concrete.
