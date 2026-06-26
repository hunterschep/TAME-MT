# OPUS-100 Demo Performance Notes

This demo originally exposed an important issue: exact train-test nearest
neighbor search is much slower than BLEU because it has to inspect training
data. TAME-MT now uses a production-oriented two-stage path.

## Retrieval Modes

- `native_exact`: Rust exact character n-gram Jaccard retrieval. Best for small
  corpora or targeted verification when exact exposure is required.
- `native_fast`: Rust rare-gram candidate generation plus exact Jaccard
  reranking of a bounded shortlist. Best for large audits.
- `python_exact` / `python_fast`: pure-Python fallbacks for debugging and
  source installs without the native extension.
- `auto`: native exact/fast when the extension is installed, otherwise Python
  exact/fast fallback. This is the default.

## Cached Scoring Workflow

For repeated system comparisons, compute train-aware diagnostics once:

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
  --num-train 50000 \
  --json-out system.tame.json
```

## Local Timing Snapshot

On the local development machine, using OPUS-100 `de-en` capped at 50,000 train
pairs and 2,000 test pairs:

| Step | Backend | Time |
| --- | --- | ---: |
| `run_opus100_demo.py --pair de-en --train-limit 50000 --test-limit 2000` after download cache | `native_fast` | ~5.7s |
| `tame-mt audit --segment-out` on prepared files | `native_fast` | ~6.4s |
| `tame-mt score-cached` for one hypothesis | cached diagnostics | ~1.8s |
| Four-pair OPUS-100 standard demo after download cache | `native_fast` | ~21.4s |
| OPUS-100 `de-en`, 100k train / 2k test, after download cache | `native_fast` | ~11.4s |
| Synthetic 100k train / 2k test | `native_fast` | ~5.1s |

The first step is the train-aware cost. The second step is the repeated-system
cost and is much closer to ordinary BLEU/chrF evaluation.

These timings are a smoke benchmark, not a formal performance claim. They are
included to make the intended large-corpus workflow concrete.
