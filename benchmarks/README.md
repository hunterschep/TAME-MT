# Benchmarks

Benchmarks are smoke checks for regressions, not formal leaderboard claims.

Run the synthetic benchmark:

```bash
python benchmarks/bench_synthetic.py --small --assert-thresholds
```

Run a larger local check:

```bash
python benchmarks/bench_synthetic.py \
  --train-size 50000 \
  --test-size 2000 \
  --index-mode auto
```

For public-corpus timing, use:

```bash
python examples/public_corpora_demo/run_opus100_demo.py \
  --pair de-en \
  --train-limit 50000 \
  --test-limit 2000
```

Always report corpus size, backend, runtime, machine, and the TAME-MT signature.
