# JSON Output

`tame-mt score --json-out report.json` writes a stable JSON object suitable for
paper tables, dashboards, and downstream scripts.

Machine-readable JSON Schema files live in `schemas/`:

- `schemas/tame_report.v1.schema.json`
- `schemas/segment_diagnostic.v1.schema.json`
- `schemas/tame_cache.v1.schema.json`
- `schemas/tame_index_manifest.v1.schema.json`

The current report `schema_version` is `1.0`; the `v1` schema filenames are the
public contract file names used by downstream validators while the package is
still pre-1.0.

## Top-Level Fields

```json
{
  "schema_version": "1.0",
  "tame_version": "0.2.1",
  "signature": "tame-mt|v:0.2.1|...",
  "data": {},
  "config": {},
  "retrieval": {},
  "backend": {},
  "quality": {},
  "exposure": {},
  "bins": [],
  "generalization_gap": {},
  "approx_validation": null,
  "performance": {},
  "warnings": []
}
```

## Retrieval

```json
{
  "mode": "exact",
  "source_exposure_mode": "exact",
  "target_exposure_mode": "exact",
  "pair_exposure_mode": "topk_rerank",
  "tm_retrieval_exact": true,
  "false_negative_safe_thresholds": [0.3, 0.7, 0.85, 0.95],
  "approximate": false
}
```

Approximate runs set `approximate` to `true`, set source/target exposure modes
to `approx`, and report pair thresholds as top-k candidate-set rates.

## Approximate Validation

`approx_validation` is `null` unless the CLI run used
`--validate-approx-sample N`. When enabled, TAME-MT samples test segments with a
deterministic seed, reruns those segments with exact `native_exact` retrieval
against the same training corpus, and compares the approximate run to the exact
sample.

```json
{
  "sample_size": 1000,
  "requested_sample_size": 1000,
  "seed": 13,
  "exact_mode": "native_exact",
  "sample_indices": [0, 17, 42],
  "source_top1_agreement": 0.997,
  "source_bin_agreement": 0.999,
  "source_mean_abs_error": 0.0021,
  "source_p95_abs_error": 0.014,
  "target_top1_agreement": 0.996,
  "pair_threshold_agreement": {
    "0.70": 1.0,
    "0.85": 0.999,
    "0.95": 1.0
  },
  "tm_bleu_abs_delta_on_sample": 0.2,
  "passed": true,
  "failures": [],
  "thresholds": {
    "min_source_top1_agreement": 0.95,
    "min_source_bin_agreement": 0.99,
    "max_source_mean_abs_error": 0.05,
    "max_source_p95_abs_error": 0.15,
    "min_target_top1_agreement": 0.95,
    "min_pair_threshold_agreement": 0.99,
    "max_tm_bleu_abs_delta": 1.0
  }
}
```

Source-only audits set target, pair, and TM-BLEU validation fields to `null`.
By default, failed approximate validation stops the command before writing
outputs. If `--allow-approx-validation-failure` is used, the report is written
with `passed: false`, non-empty `failures`, and a warning.

## Backend

```json
{
  "name": "native_exact",
  "native": true,
  "exact": true,
  "requested_mode": "auto",
  "resolved_mode": "native_exact",
  "index_reused": false
}
```

For `score --index` or `audit --index`, `index_reused` is `true` and `name`
remains the resolved retrieval backend such as `native_exact`. For
`score-cached`, `name` is `cached_segments` and `index_reused` is also `true`.
If segment metadata was available, cached reports also include
`backend.artifact_backend`, a copy of the backend object from the report that
created the segment JSONL.

## Performance

Every report includes structured performance metadata:

```json
{
  "backend": "native_exact",
  "threads": 8,
  "index_reused": false,
  "timings_sec": {
    "read_evaluation_inputs": 0.012,
    "read_training_inputs": 0.034,
    "evaluate_corpus": 1.245,
    "total": 1.301
  },
  "memory": {
    "peak_rss_mb": 512.4
  }
}
```

Reports created through the Python API include backend, thread, index-reuse,
and peak-RSS metadata. CLI-created reports also include observed stage timings
available before the report file is written. Use `--profile-json profile.json`
for a command-level profile artifact that includes final command timings such
as `write_outputs`.

## Config Dependencies

The `config` object includes metric-defining package versions:

```json
{
  "dependencies": {
    "sacrebleu": "2.6.0"
  }
}
```

The report signature also includes the SacreBLEU version, because BLEU/chrF
implementation changes can affect reported scores.

## Data

```json
{
  "num_train": 125000,
  "num_test": 1000,
  "num_refs": 1
}
```

## Quality

```json
{
  "system": {"bleu": 31.4, "chrf": 54.2},
  "tm": {"bleu": 23.8, "chrf": 47.1},
  "delta_tm": {"bleu": 7.6, "chrf": 7.1}
}
```

In audit mode, system and delta scores are `null`.

## Exposure

Exposure summaries use fractions, not percentages:

```json
{
  "source": {
    "mean": 0.71,
    "median": 0.76,
    "p05": 0.08,
    "p25": 0.39,
    "p75": 0.84,
    "p95": 0.94,
    "max": 1.0,
    "exact_overlap": 0.042,
    "at_threshold": {
      "0.70": 0.253,
      "0.85": 0.184,
      "0.95": 0.067
    }
  }
}
```

Target and pair exposure are `null` when the corresponding target/reference
inputs are unavailable.

Pair exposure may additionally contain `exact_at_threshold` when exact
pair-threshold computation was requested:

```json
{
  "pair": {
    "at_threshold": {
      "0.85": 0.10
    },
    "exact_at_threshold": {
      "0.85": 0.14
    }
  }
}
```

## Segment Diagnostics And Cache JSONL

`--diagnostic-out segments.diagnostic.jsonl` and `--cache-out segments.tamecache`
write one object per test segment. By default, diagnostic artifacts do not
include raw source, reference, hypothesis, training-neighbor text, or TM
hypotheses. Cache artifacts include TM hypotheses because `score-cached` needs
them to recompute TM-BLEU and delta over TM.

```json
{
  "index": 0,
  "source_exposure": 0.8123,
  "source_nn_index": 42,
  "source_exact": false,
  "target_exposure": 0.7761,
  "target_nn_index": 42,
  "target_ref_index": 0,
  "target_exact": false,
  "pair_exposure": 0.7761,
  "pair_nn_index": 42,
  "pair_ref_index": 0,
  "pair_exact": false,
  "bin": "near",
  "tm_source_index": 42,
  "tm_source_similarity": 0.8123,
  "tm_hyp": "..."
}
```

For multi-reference inputs, `target_ref_index` is the zero-based reference file
that produced `target_exposure`, and `pair_ref_index` is the zero-based
reference file that produced `pair_exposure`. They are `null` when target or
pair exposure is unavailable. Older cached segment JSONL files without these
fields remain valid; missing values are read as `null`.

Raw text fields are opt-in:

```bash
--include-source-text
--include-reference-text
--include-hyp-text
--include-neighbor-text
```

Use `.json.gz` or `.jsonl.gz` output paths to write gzip-compressed UTF-8 JSON
reports or segment diagnostics.

When `--diagnostic-out` or `--cache-out` is used, TAME-MT also writes a
`.meta.json` sidecar next to the JSONL artifact. The sidecar records the
artifact type, TAME-MT version, report signature, train/test/reference counts,
config, and backend used to create the diagnostics. It also records privacy
flags and content fingerprints for the metric-affecting inputs:

- `fingerprints.config_sha256`;
- `fingerprints.train_src_sha256` and normalized training-source hash;
- `fingerprints.train_tgt_sha256` and normalized training-target hash, when
  training targets are available;
- `fingerprints.test_src_sha256` and normalized test-source hash;
- `fingerprints.refs_sha256` and normalized reference hashes;
- `fingerprints.tm_hyp_sha256`.

`score-cached` and `score-cached-batch` validate this sidecar, including
reference and TM-hypothesis hashes. Segment JSONL files created by older
TAME-MT versions without a sidecar fail by default; pass
`--allow-unsafe-no-metadata` only for trusted legacy artifacts whose provenance
you have verified outside TAME-MT.

The metadata `privacy.tm_text_included` flag records whether TM hypotheses are
present. `--diagnostic-out` sets it to `false` unless `--include-tm-text` is
used. `--cache-out` sets it to `true`. Artifacts without TM hypotheses are
useful for exposure diagnostics but intentionally fail cached scoring, because
cached TM-BLEU would otherwise be wrong.

Consumers should treat `index` as the authoritative original segment position.
`score-cached` requires indices to be unique and contiguous from `0` to `N-1`
and sorts valid rows by index before scoring. For current artifacts,
`--num-train` is inferred from metadata; if you pass it manually, it must match
the metadata value. Cached segment rows store exposure-bin labels, so cached
scoring rejects rows whose stored `bin` does not match the current
`--far-threshold` and `--near-threshold` settings. Regenerate the segment JSONL,
or pass the same bin thresholds used to create it.

All numeric JSON values are finite JSON numbers. Cached segment diagnostics and
index-bundle manifests are parsed as strict JSON, so non-standard `NaN`,
`Infinity`, and `-Infinity` constants and duplicate object keys are rejected
even outside fields TAME-MT uses directly. Report writers use strict JSON
serialization.
