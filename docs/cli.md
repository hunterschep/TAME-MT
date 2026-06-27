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
--diagnostic-out segments.diagnostic.jsonl
--cache-out segments.tamecache
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

Verify a bundle before reuse or transfer:

```bash
tame-mt index verify train.tameidx \
  --train-src train.src \
  --train-tgt train.tgt
```

`index verify` checks the manifest format, archive member set, declared sizes,
load-memory budget, member SHA-256 hashes, optional supplied training-file
hashes, and native-index invariants. Use `--json` for a machine-readable
summary.

Index bundles store raw training text plus exact-match and pair fingerprints.
Protect them with the same access controls as the original training corpus.
Index bundle writes are atomic: the output path is replaced only after the new
ZIP bundle is fully written and closed. Bundle loading rejects unexpected ZIP
members, duplicate members, unsafe member sizes, excessive compression ratios,
load-memory budget violations, and unsupported schema versions before native
deserialization. Native bytes are also invariant-checked before queries can run.
If `score --index` or `audit --index` reports an unsupported bundle or native
schema version after an upgrade, rebuild the `.tameidx` file with the current
`tame-mt index build` command.
Exact index bundles may be reused with different query-time settings such as
`--pair-k` and `--batch-size`. They are not reusable across normalization,
ngram/similarity, backend-mode, or fast-mode cap changes.

For trusted large bundles on high-memory machines, raise the load budget:

```bash
--max-index-load-bytes 8589934592
```

For large corpora or repeated system comparisons, cache train-aware diagnostics once:

```bash
tame-mt audit \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --cache-out segments.tamecache \
  --json-out audit.json
```

Then score each system without rebuilding the training index:

```bash
tame-mt score-cached \
  --cache-in segments.tamecache \
  --ref test.ref \
  --hyp system.out \
  --json-out system.tame.json
```

For many systems, score them in one process so references, segment artifacts,
and TM baseline scores are reused:

```bash
tame-mt score-cached-batch \
  --cache-in segments.tamecache \
  --ref test.ref \
  --system system_a=system_a.out \
  --system system_b=system_b.out \
  --json-out-dir tame_reports
```

Cached segment rows include the source-exposure bin assigned during the original
audit. `score-cached` and `score-cached-batch` verify those labels against the
current `--far-threshold` and `--near-threshold`; if the thresholds differ,
TAME-MT fails instead of producing a mixed-configuration report.
New diagnostic/cache outputs also include an automatic `.meta.json` sidecar.
Cached commands require and validate that sidecar, including
normalization, similarity, retrieval settings, TM zero policy, train/test/ref
counts, bin thresholds, reference content hashes, and TM hypothesis hashes.
Older segment JSONL files without a sidecar fail by default; pass
`--allow-unsafe-no-metadata` only for legacy artifacts you trust.
When a sidecar is present, cached JSON reports copy the original producing
backend into `backend.artifact_backend`.

`score-cached` reuses source/target/pair exposure and TM hypotheses from the
cache artifact. It recomputes only system metrics, TM metrics, delta over TM,
bin scores, warnings, and the final report.
`score-cached-batch` does the same work for multiple systems while reading and
validating the segment JSONL once and computing the TM baseline once.

`--cache-out` contains TM hypotheses and may contain raw training-target text.
`--diagnostic-out` omits TM hypotheses by default for privacy-safer diagnostics.
Pass `--include-tm-text` only when a diagnostic artifact may safely store TM
hypotheses. `--segment-out` remains a deprecated alias for `--cache-out`.

Segment rows are validated before scoring. Indices must be unique and
contiguous from `0` to `N-1`; valid rows may appear in any order and are sorted
by index before metrics are computed.

Any corpus, hypothesis, reference, JSON, JSONL, TM output, or TM metadata path
ending in `.gz` is read or written as gzip-compressed UTF-8 text.

## Shared Options

```bash
--metrics bleu chrf
--ngram-orders 3,4,5
--far-threshold 0.30
--near-threshold 0.70
--leak-thresholds 0.70,0.85,0.95
--pair-k 50
--batch-size 8192
--index-mode auto
--auto-exact-cutoff 5000
--candidate-gram-limit 8
--posting-limit 500
--max-candidates 3000
--rerank-limit 1000
--validate-approx-sample 0
--validate-approx-seed 13
--validate-approx-exact-mode native_exact
--allow-approx-validation-failure
--min-bin-size-warning 30
--tm-zero-policy empty
--lowercase
--strip-diacritics
--normalize-punctuation
--bleu-tokenize 13a
--bleu-lowercase
--chrf-word-order 2
--verbose
--profile-json profile.json
```

Numeric options must be finite decimal values. `nan`, `inf`, and `-inf` are
rejected because TAME-MT JSON outputs are strict JSON and never emit
non-standard floating-point tokens. Exposure thresholds are fractions in the
closed interval \([0, 1]\), and comma-separated numeric lists must not contain
empty items.

Long-running commands support `--verbose`, which writes stage timings to
stderr without changing report output. This covers `score`, `audit`,
`score-cached`, `score-cached-batch`, `index build`, and `tm-baseline`.
Lower `--batch-size` if source/reference retrieval batches use too much memory
on a very large test set.

Use `--profile-json profile.json` on long-running commands to write structured
performance metadata. Report JSON includes a `performance` object with backend,
thread count, peak RSS, index-reuse status, and report-available timings. The
profile artifact records final command timings, including output writing, and a
small summary of any reports produced.

For approximate runs, add `--validate-approx-sample N` to score a deterministic
sample with exact retrieval and fail if the approximate run disagrees too much:

```bash
tame-mt audit \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --retrieval approx \
  --allow-approximate \
  --validate-approx-sample 1000 \
  --json-out audit.json
```

Validation writes `approx_validation` into the JSON report and adds
`validate_approximate_retrieval` to profile timings. It is intentionally
available only when creating fresh diagnostics with `score`, `audit`, or
`tm-baseline`; cached scoring reuses existing segment diagnostics and rejects
`--validate-approx-sample`.

## Doctor

```bash
tame-mt doctor
```

`doctor` prints the TAME-MT version, Python/platform details, SacreBLEU
version, Rayon thread count when native is available, and whether the native
Rust backend is importable. Use it first when a large run seems unexpectedly
slow.

To force a native thread count for a run:

```bash
tame-mt score --threads 4 ...
```

`--threads auto` uses Rayon defaults. Numeric thread counts must be positive
and must be set before native retrieval starts; the CLI configures this before
loading or building native indexes.

## Segment Text Options

By default, diagnostic JSONL contains indices, scores, and bins, but not raw
source/reference/hypothesis/neighbor text or TM hypotheses. Cache artifacts
include TM hypotheses because cached scoring requires them.

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

`--retrieval exact --index-mode auto` is the default. It requires the Rust
extension and resolves to `native_exact`. If the native extension is
unavailable, default scoring exits with an installation error.

Use exact mode explicitly when the corpus is small or when exact nearest-neighbor
exposure is required:

```bash
--index-mode native_exact
```

Use approximate fast mode only for exploratory or recall-validated workflows:

```bash
--retrieval approx --allow-approximate --index-mode native_fast
```

Fast mode selects rare query n-grams, reads bounded postings, keeps an
approximate shortlist, and reranks that shortlist with exact Jaccard similarity.
Reports label this as approximate retrieval and show `PairLeakTopK` for pair
thresholds.

For exact no-false-negative pair threshold rates, use exact retrieval with:

```bash
--exact-pair-thresholds
```

This adds `PairLeakExact@t` values under `exposure.pair.exact_at_threshold`.
It can be substantially slower than top-k pair reranking on large corpora.

For large approximate runs, use `--validate-approx-sample N` as a per-corpus
guardrail. The validation sample compares source nearest-neighbor identity,
source-bin agreement, source-score error, target nearest-neighbor identity,
pair-threshold decisions, and sample TM-BLEU against exact `native_exact`
retrieval. By default, validation failures exit with code `2`. Pass
`--allow-approx-validation-failure` only when you deliberately want the report
written with the failure recorded as a warning.

`auto`, `native_exact`, and `native_fast` are the only accepted index modes.
There are no Python retrieval backends; if the Rust extension is unavailable,
TAME-MT reports an installation error.

Native retrieval uses Rayon. Use `--threads N` on `score`, `audit`,
`tm-baseline`, `index build`, and cached scoring commands when you need a fixed
thread count for reproducible benchmarking or resource limits. Reports include
the actual native thread count under `performance.threads`.
