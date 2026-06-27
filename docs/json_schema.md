# JSON Output

`tame-mt score --json-out report.json` writes a stable JSON object suitable for
paper tables, dashboards, and downstream scripts.

## Top-Level Fields

```json
{
  "schema_version": "0.1",
  "tame_version": "0.1.0",
  "signature": "tame-mt|v:0.1.0|...",
  "data": {},
  "config": {},
  "backend": {},
  "quality": {},
  "exposure": {},
  "bins": [],
  "generalization_gap": {},
  "warnings": []
}
```

## Backend

```json
{
  "name": "native_fast",
  "native": true,
  "exact": false,
  "requested_mode": "auto",
  "resolved_mode": "native_fast",
  "index_reused": false
}
```

For `score --index` or `audit --index`, `index_reused` is `true` and `name`
remains the resolved retrieval backend such as `native_fast`. For
`score-cached`, `name` is `cached_segments` and `index_reused` is also `true`.

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

## Segment JSONL

`--segment-out segments.jsonl` writes one object per test segment. By default it
does not include raw source, reference, hypothesis, or training-neighbor text.

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

Consumers should treat `index` as the authoritative original segment position.
`score-cached` requires indices to be unique and contiguous from `0` to `N-1`
and sorts valid rows by index before scoring. `--num-train` must be the positive
training segment count used to create the segment diagnostics. Cached segment
rows store exposure-bin labels, so cached scoring rejects rows whose stored
`bin` does not match the current `--far-threshold` and `--near-threshold`
settings. Regenerate the segment JSONL, or pass the same bin thresholds used to
create it.

All numeric JSON values are finite JSON numbers. Cached segment diagnostics and
index-bundle manifests are parsed as strict JSON, so non-standard `NaN`,
`Infinity`, and `-Infinity` constants and duplicate object keys are rejected
even outside fields TAME-MT uses directly. Report writers use strict JSON
serialization.
