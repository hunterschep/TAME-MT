# Benchmarks

Benchmarks are smoke checks for regressions, not formal leaderboard claims.

Run the synthetic benchmark:

```bash
python benchmarks/bench_synthetic.py --small --assert-thresholds
```

Run the production-style staged benchmark:

```bash
python benchmarks/bench_synthetic.py --small --staged --assert-thresholds
```

Staged output reports index build time, `.tameidx` load time, indexed audit
time, load-plus-audit total time, end-to-end cached scoring time, prepared
cached scorer setup time, prepared cached hypothesis time, prepared batch
per-system time, and bundle size.

Run the fast-retrieval recall guard:

```bash
python benchmarks/validate_fast_recall.py --require-native
```

It compares fast retrieval against exact retrieval on deterministic
domain-template, multilingual, lexical-family, duplicate-heavy, and noisy
perturbation cases. This is a regression guard for approximation drift, not a
proof that every future corpus has perfect recall.

Run a larger local check:

```bash
python benchmarks/bench_synthetic.py \
  --train-size 100000 \
  --test-size 2000 \
  --staged \
  --max-seconds 12 \
  --max-index-build-seconds 8 \
  --max-indexed-seconds 4 \
  --max-cached-seconds 3 \
  --max-prepared-cached-seconds 1 \
  --max-cached-batch-per-system-seconds 1 \
  --max-index-bytes 120000000 \
  --assert-thresholds
```

CI runs a 50k train / 1k test staged benchmark outside the Python-version
matrix. The 100k local check above remains the release-candidate gate in
`scripts/acceptance.sh`.

For public-corpus timing, use:

```bash
python examples/public_corpora_demo/run_opus100_demo.py \
  --pair de-en \
  --train-limit 50000 \
  --test-limit 2000
```

For production-like local timing, measure all three stages separately:

```bash
tame-mt index build --train-src train.src --train-tgt train.tgt --out train.tameidx
tame-mt audit --index train.tameidx --test-src test.src --ref test.ref --segment-out segments.jsonl
tame-mt score-cached-batch \
  --segment-in segments.jsonl \
  --ref test.ref \
  --system system_a=system_a.out \
  --system system_b=system_b.out \
  --num-train 100000 \
  --json-out-dir tame-reports
```

Always report corpus size, backend, runtime, machine, and the TAME-MT signature.
