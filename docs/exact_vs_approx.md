# Exact vs Approximate Retrieval

TAME-MT is an exposure diagnostic. False negatives are worse than slower runs:
if a contaminated or narrow-domain test set is labeled far from training, the
metric has failed its main job.

## Default

The default is exact:

```bash
tame-mt score ...
```

This uses `--retrieval exact --index-mode auto`, and `auto` resolves to
`native_exact` when the Rust backend is installed.

## Approximate Mode

Approximate mode must be explicit:

```bash
tame-mt score \
  --retrieval approx \
  --allow-approximate \
  --index-mode native_fast \
  ...
```

`native_fast` selects a bounded set of rare query grams, reads bounded postings,
keeps a bounded candidate set, and reranks that shortlist with exact Jaccard
similarity. It can miss a true nearest neighbor outside the candidate set.

Approximate reports include:

- `retrieval.mode = "approx"`
- `retrieval.approximate = true`
- `retrieval.tm_retrieval_exact = false`
- `source_exposure_mode = "approx"`
- `target_exposure_mode = "approx"`
- `pair_exposure_mode = "approx_topk"`

Human-readable reports also include an approximation warning.

## Per-Run Validation

For large exploratory runs, validate the approximate candidate search against
exact retrieval on a deterministic sample:

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

The validation sample compares approximate output with exact `native_exact`
retrieval for source top-1 identity, source-bin agreement, source-score error,
target top-1 identity, pair-threshold decisions, and sample TM-BLEU. Passing
validation means the approximate settings behaved well on that sampled corpus;
it does not convert the run into canonical exact TAME-MT. Failed validation
exits nonzero unless `--allow-approx-validation-failure` is passed, in which
case the JSON report records the failure under `approx_validation` and in
`warnings`.

## PairLeakTopK

Exact pair overlap is exact. Pair threshold rates are top-k candidate limited in
the current implementation, so they are labeled `PairLeakTopK@t`.

Do not write `PairLeak@0.85` unless the run used an exact or no-false-negative
pair threshold search for that threshold.

## Recommended Use

Use exact mode for paper-facing numbers. Use approximate mode for exploratory
triage, large sweeps, or engineering benchmarks, and report the approximate
retrieval fields with the results.
