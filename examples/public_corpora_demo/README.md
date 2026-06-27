# OPUS-100 Public Corpora Demo

This example runs TAME-MT on capped OPUS-100 train/test slices that resemble
public corpora people often use for MT experiments. It is a reproducible demo
and smoke benchmark, not a leaderboard.

## Run

Quick laptop smoke:

```bash
tame-mt demo opus100 \
  --quick \
  --retrieval exact \
  --require-native
```

Standard release demo:

```bash
tame-mt demo opus100 \
  --standard \
  --retrieval exact \
  --threads auto \
  --require-native \
  --profile-json demo_runs/opus100_standard/profile.json \
  --summary-dir examples/public_corpora_demo/results \
  --output-dir demo_runs/opus100_standard/run
```

Approximate exploratory run with per-run validation:

```bash
tame-mt demo opus100 \
  --quick \
  --retrieval approx \
  --allow-approximate \
  --validate-approx-sample 100 \
  --require-native
```

## Tiers

- `--quick`: 10k train / 500 test / one pair (`de-en`).
- `--standard`: 50k train / 2k test / four pairs (`de-en`, `en-hi`, `en-tr`,
  `ar-en`).
- `--paper`: larger configurable tier. Override `--pair`, `--train-limit`, and
  `--test-limit` for the actual paper audit.

Manual `--pair`, `--train-limit`, and `--test-limit` values override tier
defaults.

## Outputs

The CLI writes:

- `{tier}.summary.md`
- `{tier}.summary.json`
- `{tier}.summary.csv`
- one JSON audit report per pair under the run directory
- an optional profile JSON when `--profile-json` is passed

Committed summaries live in `examples/public_corpora_demo/results/`. Do not
commit downloaded OPUS archives or prepared raw corpora.

The summaries include command, hardware, versions, retrieval mode, backend,
signature, exactness fields, warning counts, and timing fields so the numbers
are reproducible and not presented as universal performance claims.

`run_opus100_demo.py` is kept as a compatibility wrapper around
`tame-mt demo opus100`.
