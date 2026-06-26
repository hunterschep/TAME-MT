#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e '.[dev]'

ruff format --check .
ruff check .
mypy src/tame_mt
cargo fmt --check
cargo clippy -- -D warnings
cargo test
pytest

python benchmarks/bench_synthetic.py --small --assert-thresholds

rm -rf build dist src/tame_mt.egg-info
python -m build
python -m twine check dist/*

tame-mt doctor
tame-mt score \
  --train-src tests/fixtures/train.src \
  --train-tgt tests/fixtures/train.tgt \
  --test-src tests/fixtures/test.src \
  --ref tests/fixtures/test.ref \
  --hyp tests/fixtures/hyp.out \
  --index-mode auto \
  --json-out /tmp/tame_report.json \
  --segment-out /tmp/tame_segments.jsonl \
  --tm-out /tmp/tame_tm.out \
  --quiet

python examples/public_corpora_demo/run_opus100_demo.py \
  --pair de-en \
  --train-limit 50000 \
  --test-limit 2000 \
  --output-dir /tmp/tame_opus100_acceptance \
  --summary-dir /tmp/tame_opus100_acceptance/summary
