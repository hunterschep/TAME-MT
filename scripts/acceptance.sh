#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e '.[dev]'

ruff format --check .
ruff check .
mypy src/tame_mt
python scripts/check_versions.py
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
pytest

python -m pip_audit . --progress-spinner off
if ! command -v cargo-audit >/dev/null 2>&1; then
  cargo install cargo-audit --locked --version 0.22.2
fi
cargo audit --deny warnings

python benchmarks/bench_synthetic.py --small --assert-thresholds
python benchmarks/bench_synthetic.py --small --staged --assert-thresholds
python benchmarks/validate_fast_recall.py --require-native
python benchmarks/validate_threshold_exact.py --require-native
python benchmarks/bench_synthetic.py \
  --train-size 100000 \
  --test-size 2000 \
  --staged \
  --max-seconds 15 \
  --max-index-build-seconds 8 \
  --max-indexed-seconds 10 \
  --max-indexed-audit-seconds 5 \
  --max-cached-seconds 3 \
  --max-prepared-cached-seconds 1 \
  --max-cached-batch-per-system-seconds 1 \
  --max-index-bytes 120000000 \
  --assert-thresholds

python benchmarks/bench_synthetic.py \
  --train-size 100000 \
  --test-size 2000 \
  --retrieval approx \
  --allow-approximate \
  --index-mode native_fast \
  --staged \
  --max-seconds 12 \
  --max-index-build-seconds 8 \
  --max-indexed-seconds 4 \
  --max-cached-seconds 3 \
  --max-prepared-cached-seconds 1 \
  --max-cached-batch-per-system-seconds 1 \
  --max-index-bytes 120000000 \
  --assert-thresholds

rm -rf build dist src/tame_mt.egg-info
python -m build
python -m twine check dist/*

wheel_smoke_venv=$(mktemp -d)
trap 'rm -rf "$wheel_smoke_venv"' EXIT
python -m venv "$wheel_smoke_venv"
"$wheel_smoke_venv/bin/python" -m pip install dist/*.whl
"$wheel_smoke_venv/bin/python" -m pip check
"$wheel_smoke_venv/bin/python" scripts/wheel_smoke.py
rm -rf "$wheel_smoke_venv"
trap - EXIT

tame-mt doctor
tame-mt score \
  --train-src tests/fixtures/train.src \
  --train-tgt tests/fixtures/train.tgt \
  --test-src tests/fixtures/test.src \
  --ref tests/fixtures/test.ref \
  --hyp tests/fixtures/hyp.out \
  --index-mode auto \
  --json-out /tmp/tame_report.json \
  --diagnostic-out /tmp/tame_segments.diagnostic.jsonl \
  --cache-out /tmp/tame_segments.tamecache \
  --tm-out /tmp/tame_tm.out \
  --profile-json /tmp/tame_profile.json \
  --quiet

tame-mt index build \
  --train-src tests/fixtures/train.src \
  --train-tgt tests/fixtures/train.tgt \
  --out /tmp/tame_fixture.tameidx \
  --quiet

tame-mt index verify \
  /tmp/tame_fixture.tameidx \
  --train-src tests/fixtures/train.src \
  --train-tgt tests/fixtures/train.tgt \
  --json

tame-mt score \
  --index /tmp/tame_fixture.tameidx \
  --test-src tests/fixtures/test.src \
  --ref tests/fixtures/test.ref \
  --hyp tests/fixtures/hyp.out \
  --json-out /tmp/tame_index_report.json \
  --quiet

tame-mt score-cached \
  --cache-in /tmp/tame_segments.tamecache \
  --ref tests/fixtures/test.ref \
  --hyp tests/fixtures/hyp.out \
  --json-out /tmp/tame_cached_report.json \
  --quiet

python - <<'PY'
import json
from pathlib import Path

fresh = json.loads(Path("/tmp/tame_report.json").read_text(encoding="utf-8"))
indexed = json.loads(Path("/tmp/tame_index_report.json").read_text(encoding="utf-8"))
cached = json.loads(Path("/tmp/tame_cached_report.json").read_text(encoding="utf-8"))
assert fresh["quality"] == indexed["quality"]
assert fresh["exposure"] == indexed["exposure"]
assert indexed["backend"]["index_reused"] is True
assert fresh["quality"] == cached["quality"]
assert fresh["exposure"] == cached["exposure"]
assert cached["backend"]["resolved_mode"] == "cached_segments"
PY

tame-mt demo opus100 \
  --standard \
  --pair de-en \
  --retrieval exact \
  --require-native \
  --output-dir /tmp/tame_opus100_acceptance \
  --summary-dir /tmp/tame_opus100_acceptance/summary
