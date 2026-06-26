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
python benchmarks/bench_synthetic.py \
  --train-size 100000 \
  --test-size 2000 \
  --max-seconds 30 \
  --assert-thresholds

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

tame-mt index build \
  --train-src tests/fixtures/train.src \
  --train-tgt tests/fixtures/train.tgt \
  --out /tmp/tame_fixture.tameidx \
  --quiet

tame-mt score \
  --index /tmp/tame_fixture.tameidx \
  --test-src tests/fixtures/test.src \
  --ref tests/fixtures/test.ref \
  --hyp tests/fixtures/hyp.out \
  --json-out /tmp/tame_index_report.json \
  --quiet

python - <<'PY'
import json
from pathlib import Path

fresh = json.loads(Path("/tmp/tame_report.json").read_text(encoding="utf-8"))
indexed = json.loads(Path("/tmp/tame_index_report.json").read_text(encoding="utf-8"))
assert fresh["quality"] == indexed["quality"]
assert fresh["exposure"] == indexed["exposure"]
assert indexed["backend"]["index_reused"] is True
PY

python examples/public_corpora_demo/run_opus100_demo.py \
  --pair de-en \
  --train-limit 50000 \
  --test-limit 2000 \
  --output-dir /tmp/tame_opus100_acceptance \
  --summary-dir /tmp/tame_opus100_acceptance/summary
