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

For production-like local timing, measure all three stages separately:

```bash
tame-mt index build --train-src train.src --train-tgt train.tgt --out train.tameidx
tame-mt audit --index train.tameidx --test-src test.src --ref test.ref --segment-out segments.jsonl
tame-mt score-cached --segment-in segments.jsonl --ref test.ref --hyp system.out --num-train 100000
```

Always report corpus size, backend, runtime, machine, and the TAME-MT signature.
