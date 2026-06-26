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
  "target_exact": false,
  "pair_exposure": 0.7761,
  "pair_nn_index": 42,
  "pair_exact": false,
  "bin": "near",
  "tm_source_index": 42,
  "tm_source_similarity": 0.8123,
  "tm_hyp": "..."
}
```

Raw text fields are opt-in:

```bash
--include-source-text
--include-reference-text
--include-hyp-text
--include-neighbor-text
```
