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

There are two reuse levels.

Use an index bundle when the training corpus is fixed but test sets or
configuration may change:

```bash
tame-mt index build \
  --train-src train.src \
  --train-tgt train.tgt \
  --out train.tameidx

tame-mt audit \
  --index train.tameidx \
  --test-src test.src \
  --ref test.ref \
  --json-out audit.json
```

This skips native source/target index construction on later runs. The bundle is
a low-compression zip container tuned for much smaller cache files while keeping
load time low. It stores raw training text and normalized exact-match and pair
keys, so treat it as training data.

Use cached segment diagnostics when the train/test/reference setup is fixed and
only system outputs change. Run the train-aware audit once:

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
| OPUS-100 `de-en`, 100k train / 2k test, fresh audit | `native_fast` | ~10.0s |
| OPUS-100 `de-en`, 100k train / 2k test, one-time index build | `native_fast` | ~9.6s |
| OPUS-100 `de-en`, 100k train / 2k test, load + audit from `.tameidx` | `native_fast`, reused index | ~2.3s |
| Synthetic 100k train / 2k test, fresh audit | `native_fast` | ~2.5s |
| Synthetic 100k train / 2k test, one-time compressed index build | `native_fast` | ~3.5s |
| Synthetic 100k train / 2k test, load + audit from `.tameidx` | `native_fast`, reused index | ~0.9s |
| Synthetic 100k train / 2k test, cached hypothesis scoring | cached diagnostics | ~0.4s |
| Synthetic 100k train / 2k test, prepare cached scorer | cached diagnostics | ~0.3s |
| Synthetic 100k train / 2k test, prepared cached hypothesis scoring | cached diagnostics | ~0.2s |
| Synthetic 100k train / 2k test, prepared batch cached scoring, 5 systems | cached diagnostics | ~0.2s/system |
| Synthetic 100k train / 10k test, fresh audit | `native_fast` | ~4.3s |
| Synthetic 100k train / 10k test, one-time compressed index build | `native_fast` | ~3.5s |
| Synthetic 100k train / 10k test, load + audit from `.tameidx` | `native_fast`, reused index | ~2.8s |
| Synthetic 100k train / 10k test, cached hypothesis scoring | cached diagnostics | ~2.2s |
| Synthetic 100k train / 10k test, prepared cached hypothesis scoring | cached diagnostics | ~0.8s |

These numbers are smoke timings, not universal performance claims. Report the
machine, corpus size, backend, and full TAME-MT signature for benchmark tables.
The synthetic 100k source+target `.tameidx` bundle in the table above is about
67 MB with low-compression ZIP storage. The same payload was about 323 MB when
stored uncompressed, so acceptance checks now include a bundle-size ceiling as
well as runtime ceilings.

The cached path still runs SacreBLEU/chrF over system and TM outputs, but it no
longer touches the training corpus. TAME-MT aggregates SacreBLEU segment
statistics once per metric and output, then reuses those statistics for the
whole corpus and all exposure bins without copying the full segment-stat list
again. Ordered cached segment artifacts also take a fast validation path before
scoring, while malformed or reordered artifacts still go through the full
duplicate/missing-index validator. If a future SacreBLEU release changes those
internal segment-stat APIs, TAME-MT falls back to SacreBLEU's public corpus
scoring APIs so scoring remains correct, with reduced bin-scoring performance
until the optimized adapter is updated.

Fresh and indexed audits also avoid avoidable Python work around retrieval:
native query candidate maps use the same lightweight FNV hashing strategy as
the compact n-gram table, per-segment result objects use slotted dataclasses to
reduce memory pressure, and exposure summaries collect source/target/pair
scores in one pass before sorting each side once. Threshold counts then use
binary search over the sorted scores instead of rescanning every segment for
each threshold. Fresh native audits also release Python-side normalized
training-line copies after exact pair keys are prepared; the Rust indexes keep
the state needed for exact checks and nearest-neighbor queries.

When pair exposure is not requested, for example `tm-baseline` or source-only
audits without references, TAME-MT queries only the nearest source neighbor
instead of the configured pair-candidate `top-k`.

For many systems on the same cached diagnostics, use `score-cached-batch`
instead of one `score-cached` process per system. Batch mode reads and validates
the segment JSONL once, keeps SacreBLEU reference caches alive across all
systems for each metric, and computes TM baseline scores once for the batch. On
the local synthetic 100,000 train / 2,000 test benchmark, a prepared cached
scorer takes about 0.26 seconds to build and then scores five cached systems in
about 0.83 seconds total, or roughly 0.17 seconds per system. In Python
applications that receive systems over time, use
`TameScorer.prepare_from_artifacts()` and keep the returned scorer alive so
later calls skip artifact validation, reference preprocessing, and TM baseline
scoring.
