# Public Corpora Demo

The public-corpora demo computes TAME-MT on small OPUS-100 slices that resemble
datasets people often download for MT experiments. It is a demonstration and
smoke benchmark, not a canonical leaderboard.

## Quick Run

```bash
tame-mt demo opus100 \
  --quick \
  --retrieval approx \
  --allow-approximate \
  --validate-approx-sample 100 \
  --threads 4 \
  --require-native \
  --profile-json demo_runs/opus100_quick/profile.json \
  --summary-dir demo_runs/opus100_quick/summary \
  --output-dir demo_runs/opus100_quick/run
```

Use exact retrieval for paper-facing examples when the corpus size is feasible:

```bash
tame-mt demo opus100 \
  --standard \
  --retrieval exact \
  --threads auto \
  --require-native \
  --profile-json demo_runs/opus100_standard/profile.json \
  --summary-dir demo_runs/opus100_standard/summary \
  --output-dir demo_runs/opus100_standard/run
```

## What To Report

A reproducible public-corpus summary should include:

- language pair;
- train/test limits;
- command;
- retrieval mode and backend;
- exactness/approximation fields;
- report signature;
- warnings;
- performance metadata;
- hardware, OS, Python, Rust, and TAME-MT versions;
- whether corpora were freshly downloaded or read from cache.

Do not commit raw downloaded corpora. Commit only small summary artifacts.

Committed example summaries live under
`examples/public_corpora_demo/results/`. Recreate them with the CLI rather than
editing numbers by hand. The script at
`examples/public_corpora_demo/run_opus100_demo.py` is a compatibility wrapper
around the packaged CLI implementation.

The demo writes Markdown, CSV, and JSON summaries. Use `--profile-json` for a
separate machine/profile artifact with per-pair timing, report paths,
signatures, backend metadata, retrieval metadata, performance metadata, and
approximate-validation payloads when requested.

## Tiers

- `--quick`: 10k train / 500 test / one pair (`de-en`).
- `--standard`: 50k train / 2k test / four pairs (`de-en`, `en-hi`, `en-tr`,
  `ar-en`).
- `--paper`: larger configurable tier. Override `--pair`, `--train-limit`, and
  `--test-limit` for the exact paper audit.

The default tier is `--standard`. Manual `--pair`, `--train-limit`, and
`--test-limit` values override the tier defaults.
