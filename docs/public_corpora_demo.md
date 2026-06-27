# Public Corpora Demo

The public-corpora demo computes TAME-MT on small OPUS-100 slices that resemble
datasets people often download for MT experiments. It is a demonstration and
smoke benchmark, not a canonical leaderboard.

## Quick Run

```bash
python examples/public_corpora_demo/run_opus100_demo.py \
  --pair de-en \
  --train-limit 10000 \
  --test-limit 500 \
  --retrieval approx \
  --allow-approximate \
  --validate-approx-sample 100 \
  --summary-dir demo_runs/opus100_quick/summary \
  --output-dir demo_runs/opus100_quick/run
```

Use exact retrieval for paper-facing examples when the corpus size is feasible:

```bash
python examples/public_corpora_demo/run_opus100_demo.py \
  --pair de-en \
  --train-limit 50000 \
  --test-limit 2000 \
  --retrieval exact \
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

Existing example summaries live under
`examples/public_corpora_demo/`. Recreate them with the script rather than
editing numbers by hand.
