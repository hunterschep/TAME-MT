# Privacy And Artifact Handling

TAME-MT runs locally. It does not download models, call remote services, send
telemetry, or upload corpus text.

The privacy risk is in artifacts you choose to write.

## Artifact Sensitivity

Treat these as sensitive:

- `.tameidx` index bundles;
- segment JSONL files that include TM hypotheses;
- segment JSONL files written with raw text flags;
- segment metadata sidecars, because they reveal corpus fingerprints and
  experiment provenance.

`.tameidx` bundles store raw training source text and, when available, raw
training target text. They also store normalized exact-match and pair keys so
the bundle can reproduce exposure diagnostics without rebuilding from raw
files.

Segment JSONL includes TM baseline hypotheses by default. A TM hypothesis is
usually copied from the nearest training target segment, so it may contain raw
training-target text even when `--include-neighbor-text` is not used.

Use this option for privacy-safer diagnostics:

```bash
--no-tm-text-in-segments
```

Those artifacts intentionally cannot be used by `score-cached`, because cached
scoring needs the TM hypotheses to recompute TM-BLEU and delta-over-TM.

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
