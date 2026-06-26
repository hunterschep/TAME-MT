# Changelog

## 0.1.0

- Initial TAME-MT package and CLI.
- Added source, target, and approximate pair exposure diagnostics.
- Added nearest-neighbor translation-memory baseline scoring with BLEU and chrF.
- Added text, JSON, segment JSONL, TM output modes, and cached scoring from
  prior segment diagnostics.
- Added a native Rust/PyO3 nearest-neighbor backend with exact and bounded fast
  retrieval modes, compact integer n-gram indexes, parallel batch query, bulk
  candidate scoring, and pure-Python fallback modes.
- Added `tame-mt doctor` for install/backend diagnostics.
- Added OPUS-100 public-corpus demo outputs and local performance notes for the
  50k-train/2k-test benchmark scale.
- Added synthetic benchmark smoke checks and an acceptance script.
- Added toy corpus, documentation, unit tests, native tests, and CI checks.
- Added typed exceptions, typed package marker, maturin package metadata, and
  release-oriented docs for public distribution.
- Removed unnecessary runtime dependencies beyond SacreBLEU.
