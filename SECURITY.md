# Security And Privacy

TAME-MT runs locally and does not send data to remote services.

## Reporting Security Issues

Please report vulnerabilities privately through GitHub security advisories for
the repository when available. If that is not available, contact the maintainer
directly before opening a public issue.

## Data Handling

Training corpora can contain sensitive, licensed, community-owned, or otherwise
restricted text. TAME-MT's default behavior avoids printing nearest-neighbor
training examples and avoids including raw segment text in JSONL diagnostics.

The following flags can write raw text and should be used only when appropriate:

```bash
--include-source-text
--include-reference-text
--include-hyp-text
--include-neighbor-text
```

`--include-neighbor-text` can write raw training text.

Segment JSONL contains TM baseline hypotheses by default. A TM hypothesis is a
training-target sentence copied from the nearest training-source neighbor, so it
may reveal training-target text even when `--include-neighbor-text` is not used.
Use `--no-tm-text-in-segments` for privacy-safer exposure diagnostics. Those
artifacts intentionally cannot be used by `score-cached`, because cached
TM-BLEU requires the original TM hypotheses.

## Untrusted Artifacts

Treat `.tameidx`, segment JSONL, and segment metadata files from unknown sources
as untrusted inputs. Prefer rebuilding index bundles from trusted training text.

TAME-MT validates `.tameidx` archives before native deserialization:

- unexpected or duplicate ZIP members are rejected;
- manifest-declared native member sizes must match the ZIP entries;
- native index, exact-key, training-text, manifest, and total uncompressed sizes
  have hard caps;
- suspicious compression ratios are rejected to limit zip-bomb-style inputs;
- unsupported bundle and native-index schema versions fail closed.

Segment JSONL and sidecar metadata use strict JSON parsing. Non-standard
`NaN`/`Infinity` tokens, duplicate object keys, non-finite numbers, duplicate
segment indices, missing segment indices, config drift, reference hash
mismatches, TM-hypothesis hash mismatches, and missing current sidecar metadata
are rejected before cached scoring. `--allow-unsafe-no-metadata` exists only for
trusted legacy artifacts whose provenance has been verified outside TAME-MT.

These checks are designed to keep accidental or malicious artifacts from
silently corrupting reports or exhausting memory. They are not a reason to
publish private training text in index bundles or raw-text segment reports.
