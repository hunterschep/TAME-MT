# Performance Guide

TAME-MT is more expensive than BLEU because it compares each test segment to a
training corpus. BLEU scores only hypothesis/reference files. TAME-MT also asks
which training segment is closest to each test source and reference.

Every report includes a `performance` object with backend, thread count,
index-reuse status, available stage timings, and peak RSS. For full command
timings, including output writing, pass `--profile-json profile.json` to the CLI
command. Benchmark tables should include the report signature and the profile
artifact when possible.

## Retrieval Engine

| Mode | Exact nearest neighbor | Intended use |
| --- | --- | --- |
| `auto` | Yes | Default production mode; requires Rust and resolves to `native_exact`. |
| `native_exact` | Yes | Explicit exact audits when nearest-neighbor exposure must be exact. |
| `native_fast` | No | Explicit approximate exploratory runs and recall-characterized workflows. |

The Rust extension is the only retrieval engine. A missing native backend is an
installation error, not a production fallback. Approximate `native_fast` must be
requested with `--retrieval approx --allow-approximate`.

`native_fast` is approximate for nearest-neighbor retrieval. It selects rare
query n-grams, reads bounded postings, keeps a bounded candidate set, and reranks
that shortlist with exact Jaccard similarity. Exact source and exact pair
overlap flags remain exact, but approximate SourceExposure/TargetExposure,
TM-BLEU, and PairLeakTopK are candidate-set estimates.

For speed-critical exploratory runs, add `--validate-approx-sample N`. TAME-MT
reruns a deterministic test-segment sample with exact `native_exact` retrieval,
records an `approx_validation` block in the report, and fails the command when
agreement drops below the built-in release thresholds. This is a corpus-specific
guardrail for fast mode; exact mode remains the required choice for canonical
paper-facing numbers.

Release acceptance includes `benchmarks/validate_fast_recall.py`, which compares
fast retrieval with exact retrieval on deterministic domain-template,
multilingual, lexical-family, duplicate-heavy, and noisy-perturbation corpora.
The guard requires exact-match recall of 1.0 and enforces top-1 agreement and
score-gap ceilings. This is not a proof that every corpus is recall-safe; it is
a regression test that the approximate path stays characterized instead of
drifting silently. CI also runs a larger non-matrix staged benchmark so obvious
performance regressions fail before release.

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

On the local development machine, exact-default and approximate smoke runs
completed as follows. These are not universal performance claims; use them to
understand the cost profile and always report the full TAME-MT signature with
benchmark numbers.

| Step | Backend | Time |
| --- | --- | ---: |
| Synthetic 100k train / 2k test, fresh audit | `native_exact` | ~35.0s |
| Synthetic 100k train / 2k test, one-time compressed index build | `native_exact` | ~4.0s |
| Synthetic 100k train / 2k test, load + audit from `.tameidx` | `native_exact`, reused index | ~43.2s |
| Synthetic 100k train / 2k test, cached hypothesis scoring | cached diagnostics | ~0.5s |
| Synthetic 100k train / 2k test, prepared cached hypothesis scoring | cached diagnostics | ~0.2s |
| Public demo after download cache | `native_fast` | ~5.7s |
| Direct CLI audit on prepared files | `native_fast` | ~6.4s |
| `score-cached` for one hypothesis | cached diagnostics | ~1.8s |
| Four-pair OPUS-100 standard demo after download cache | `native_fast` | ~21.4s |
| OPUS-100 `de-en`, 100k train / 2k test, fresh audit | `native_fast` | ~10.0s |
| OPUS-100 `de-en`, 100k train / 2k test, one-time index build | `native_fast` | ~9.6s |
| OPUS-100 `de-en`, 100k train / 2k test, load + audit from `.tameidx` | `native_fast`, reused index | ~2.3s |
| Synthetic 100k train / 2k test, fresh audit | `native_fast` | ~2.8s |
| Synthetic 100k train / 2k test, one-time compressed index build | `native_fast` | ~4.0s |
| Synthetic 100k train / 2k test, load + audit from `.tameidx` | `native_fast`, reused index | ~2.4s |
| Synthetic 100k train / 2k test, cached hypothesis scoring | cached diagnostics | ~0.4s |
| Synthetic 100k train / 2k test, prepare cached scorer | cached diagnostics | ~0.3s |
| Synthetic 100k train / 2k test, prepared cached hypothesis scoring | cached diagnostics | ~0.2s |
| Synthetic 100k train / 2k test, prepared batch cached scoring, 5 systems | cached diagnostics | ~0.2s/system |
| Synthetic 100k train / 10k test, fresh audit | `native_fast` | ~4.3s |
| Synthetic 100k train / 10k test, one-time compressed index build | `native_fast` | ~3.5s |
| Synthetic 100k train / 10k test, load + audit from `.tameidx` | `native_fast`, reused index | ~2.8s |
| Synthetic 100k train / 10k test, cached hypothesis scoring | cached diagnostics | ~2.2s |
| Synthetic 100k train / 10k test, prepared cached hypothesis scoring | cached diagnostics | ~0.8s |

When publishing benchmark tables, report the machine, corpus size, backend, and
full TAME-MT signature.
The synthetic 100k source+target `.tameidx` bundle in the table above is about
78 MB with low-compression ZIP storage. The same payload was about 323 MB when
stored uncompressed, so acceptance checks now include a bundle-size ceiling as
well as runtime ceilings.

The cached path still runs SacreBLEU/chrF over system and TM outputs, but it no
longer touches the training corpus. TAME-MT aggregates SacreBLEU segment
statistics once per metric and output, then reuses those statistics for the
whole corpus and all exposure bins without copying the full segment-stat list
again. Ordered cached segment artifacts also take a fast validation path before
scoring, while malformed or reordered artifacts still go through the full
duplicate/missing-index validator. TAME-MT pins the supported SacreBLEU major
range to `sacrebleu>=2.4,<3` and CI tests both the minimum supported 2.x line
and the current 2.x line. If a forced install removes the internal segment-stat
hooks anyway, TAME-MT falls back to SacreBLEU's public corpus scoring APIs so
scoring remains correct, with reduced bin-scoring performance until the
optimized adapter is updated.

Fresh and indexed audits also avoid avoidable Python work around retrieval:
per-segment result objects use slotted dataclasses to reduce memory pressure,
evaluation retrieval runs in configurable chunks, and exposure summaries collect
source/target/pair scores in one pass before sorting each side once. Threshold
counts then use binary search over the sorted scores instead of rescanning every
segment for each threshold. Fresh native audits also release Python-side
normalized training-line copies after exact pair keys are prepared; the Rust
indexes keep the state needed for exact checks and nearest-neighbor queries.
Native maps use Rust's randomized default hashing rather than a fixed public
hash, reducing collision-DoS exposure when indexing untrusted text.

For large indexed runs, `score --index` and `audit --index` enforce a default
uncompressed bundle load budget before reading raw training text, native index
bytes, or exact-pair keys into memory. Raise `--max-index-load-bytes` only for a
trusted `.tameidx` bundle on a machine with enough RAM. Lower `--batch-size`
when test/reference retrieval batches need a smaller peak memory footprint.

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
