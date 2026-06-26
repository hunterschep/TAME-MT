# Performance Guide

TAME-MT is more expensive than BLEU because it compares each test segment to a
training corpus. BLEU scores only hypothesis/reference files. TAME-MT also asks
which training segment is closest to each test source and reference.

## Backends

| Backend | Exact nearest neighbor | Native | Intended use |
| --- | --- | --- | --- |
| `native_exact` | Yes | Yes | Exact audits on small and medium corpora. |
| `native_fast` | No | Yes | Large public corpora and repeated evaluation workflows. |
| `python_exact` | Yes | No | Debugging and parity checks. |
| `python_fast` | No | No | Fallback when native wheels are unavailable. |

`auto` chooses `native_exact` up to `--auto-exact-cutoff` and `native_fast`
above it when the native extension is installed. Without the extension, `auto`
falls back to the corresponding Python backend.

Fast backends are approximate for nearest-neighbor retrieval. They select rare
query n-grams, read bounded postings, keep a bounded candidate set, and rerank
that shortlist with exact Jaccard similarity. Exact source, target, and pair
overlap rates remain exact.

## Recommended Workflow

Run the train-aware audit once:

```bash
tame-mt audit \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --segment-out segments.jsonl \
  --json-out audit.json
```

Then score every system output from cached diagnostics:

```bash
tame-mt score-cached \
  --segment-in segments.jsonl \
  --ref test.ref \
  --hyp system.out \
  --num-train 50000 \
  --json-out system.tame.json
```

This makes each additional system close to ordinary BLEU/chrF cost because the
training-corpus retrieval has already been done.

## Local Smoke Timing

On the local development machine, OPUS-100 `de-en` capped at 50,000 train pairs
and 2,000 test pairs completed as follows:

| Step | Backend | Time |
| --- | --- | ---: |
| Public demo after download cache | `native_fast` | ~5.7s |
| Direct CLI audit on prepared files | `native_fast` | ~6.4s |
| `score-cached` for one hypothesis | cached diagnostics | ~1.8s |
| Four-pair OPUS-100 standard demo after download cache | `native_fast` | ~21.4s |
| OPUS-100 `de-en`, 100k train / 2k test, after download cache | `native_fast` | ~11.4s |
| Synthetic 100k train / 2k test | `native_fast` | ~5.1s |

These numbers are smoke timings, not universal performance claims. Report the
machine, corpus size, backend, and full TAME-MT signature for benchmark tables.
