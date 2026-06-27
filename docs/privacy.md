# Privacy And Artifact Handling

TAME-MT runs locally. It does not download models, call remote services, send
telemetry, or upload corpus text.

The privacy risk is in artifacts you choose to write.

## Artifact Sensitivity

Treat these as sensitive:

- `.tameidx` index bundles;
- `.tamecache` files, because they include TM hypotheses;
- diagnostic/cache JSONL files written with raw text flags;
- segment metadata sidecars, because they reveal corpus fingerprints and
  experiment provenance.

`.tameidx` bundles store raw training source text and, when available, raw
training target text. They also store fixed-size exact-match and pair
fingerprints so the bundle can reproduce exposure diagnostics without rebuilding
from raw files.

Cache artifacts include TM baseline hypotheses. A TM hypothesis is usually
copied from the nearest training target segment, so it may contain raw
training-target text even when `--include-neighbor-text` is not used.

Use this option for privacy-safer diagnostics:

```bash
--diagnostic-out segments.diagnostic.jsonl
```

Diagnostic artifacts omit TM hypotheses by default and intentionally cannot be
used by `score-cached`. Cached scoring needs TM hypotheses to recompute TM-BLEU
and delta-over-TM, so use `--cache-out` only when that artifact can be protected
like training-target text.

## Raw Text Flags

Raw source, reference, hypothesis, and neighbor text are opt-in:

```bash
--include-source-text
--include-reference-text
--include-hyp-text
--include-neighbor-text
```

`--include-neighbor-text` may write raw training source and target text. Use it
only for trusted local diagnostics.

## Untrusted Inputs

Do not load `.tameidx` bundles from unknown sources unless you are willing to
treat them as untrusted archives. TAME-MT validates archive shape, declared
sizes, compression ratios, schema versions, and native-index invariants before
using a bundle, but the file can still contain corpus text.

For cached segment JSONL, TAME-MT validates the metadata sidecar and content
hashes by default. Missing metadata fails unless
`--allow-unsafe-no-metadata` is explicitly passed.
