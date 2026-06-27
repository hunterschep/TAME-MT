# Changelog

## 0.1.0

- Initial TAME-MT package and CLI.
- Added source, target, and approximate pair exposure diagnostics.
- Added nearest-neighbor translation-memory baseline scoring with BLEU and chrF.
- Added text, JSON, segment JSONL, TM output modes, and cached scoring from
  prior segment diagnostics.
- Added real `score --verbose` and `audit --verbose` stage timing output to
  stderr for long-running jobs.
- Extended verbose stage timing to cached scoring, cached batch scoring, index
  building, and TM-baseline generation.
- Added a native Rust/PyO3 nearest-neighbor backend with exact and bounded fast
  retrieval modes, compact integer n-gram indexes, parallel batch query, bulk
  candidate scoring, and pure-Python fallback modes.
- Optimized the native backend by interning character n-grams from UTF-8 slices
  instead of allocating per-occurrence gram strings, reducing native index build
  time while preserving exact n-gram identity and literal normalized-string
  exact-match maps.
- Parallelized batched native pair reranking and removed redundant exact-match
  checks from exposure assembly for faster large-test audits.
- Reused normalized source/reference batches across native search, exact-pair
  checks, and pair reranking to reduce repeated Python preprocessing in
  large-test audits.
- Added fast-path normalization for ASCII text and common strip-plus-whitespace
  normalization.
- Added persistent `.tameidx` native index bundles plus `tame-mt index build`,
  `tame-mt index inspect`, and `score`/`audit --index` reuse workflows for
  large training corpora.
- Switched persisted `.tameidx` bundles to low-compression ZIP storage, cutting
  the synthetic 100k source+target bundle from about 323 MB to about 67 MB in
  local smoke timing while preserving fast indexed reuse.
- Added explicit `.tameidx` bundle and native index schema versioning so
  incompatible persisted indexes fail with clear rebuild guidance.
- Made native backend availability require the compiled extension version to
  match the Python package version, preventing stale editable builds from being
  selected silently.
- Hardened `.tameidx` loading with strict manifest types, duplicate-member
  rejection, and manifest-vs-ZIP member size checks before native index
  deserialization.
- Optimized native index reuse by avoiding duplicate Python exact maps and by
  persisting normalized exact-pair keys for faster repeated audits.
- Reduced native query-loop overhead by using the package's lightweight FNV
  hasher for exact-match and per-query candidate-count maps.
- Reduced large-audit Python overhead by using slotted dataclasses for
  configs, reports, per-segment diagnostics, and internal result containers.
- Optimized exposure-summary generation by collecting side statistics in one
  pass, sorting each score side once, and using binary search for threshold
  counts.
- Reduced fresh native-audit memory pressure by releasing Python-side
  normalized training-line copies after exact pair keys have been prepared.
- Hardened cached segment artifact scoring with strict index validation,
  canonical ordering, and safer JSON type parsing.
- Rejected cached segment artifacts whose stored exposure-bin labels do not
  match the current bin thresholds, preventing mixed-configuration cached
  reports.
- Added automatic segment JSONL metadata sidecars and cached-CLI validation for
  artifact-defining config and count drift while preserving legacy sidecar-free
  segment JSONL compatibility.
- Exposed segment metadata path/read/validation helpers in the public Python
  namespace so services can enforce the same cached-artifact checks as the CLI.
- Added `target_ref_index` and `pair_ref_index` to segment diagnostics so
  multi-reference audits show which reference produced the target and pair
  exposure maxima; older cached JSONL files without those fields remain valid.
- Hardened cached segment JSONL parsing so malformed bin labels and non-string
  TM hypotheses fail closed instead of being coerced.
- Rejected non-finite numeric config and cached-artifact values and made JSON
  report/segment writers fail closed instead of emitting non-standard `NaN` or
  infinity values.
- Rejected out-of-range exposure thresholds and malformed comma-separated
  numeric CLI lists so large batch runs fail fast on invalid configuration.
- Tightened Python API configuration validation so malformed numeric, boolean,
  metric-list, normalization, and nested config options raise
  `ConfigurationError` consistently.
- Rejected unordered and duplicate metric selections in `ScoreConfig` so report
  key order and signatures remain deterministic.
- Rejected duplicate CLI metric selections instead of silently deduplicating
  them before configuration validation.
- Centralized strict JSON parsing/serialization for package artifacts so
  cached segment JSONL and `.tameidx` manifests reject non-standard `NaN` and
  infinity constants and duplicate object keys even in ignored fields.
- Wrapped output serialization failures in user-facing `OutputError` exceptions
  and serialize full JSON reports before opening the destination path to avoid
  partial report files.
- Improved malformed artifact and index-bundle errors so corrupt numeric fields,
  manifests, and UTF-8 members fail with user-facing TAME-MT exceptions.
- Optimized cached and repeated scoring by aggregating SacreBLEU segment
  statistics once per metric/system instead of rescoring every exposure bin
  separately.
- Reduced cached-scoring allocation by fast-pathing already ordered segment
  artifacts, building exposure-bin groups in one pass, and reusing whole-corpus
  SacreBLEU statistics without copying.
- Added `score-cached-batch` and a batch artifact-scoring API so many systems
  can share one segment-artifact read, one reference cache, and one TM baseline
  scoring pass.
- Added `CachedSegmentScorer` and `TameScorer.prepare_from_artifacts()` so
  services and notebooks can validate cached diagnostics once, keep SacreBLEU
  reference caches and TM baseline scores alive, and score later hypotheses with
  prepared cached-score latency.
- Made prepared cached scorers snapshot validated segment diagnostics so later
  caller-side mutation of artifact objects cannot corrupt cached state.
- Exposed cached-scoring artifact types, `read_segment_jsonl`, and
  `MetricConfig` from the top-level Python package API.
- Reduced source-only audit and TM-baseline retrieval work by querying only the
  nearest source neighbor unless pair exposure is being computed.
- Moved native pair-exposure reranking into a batched Rust path to reduce
  Python/Rust boundary overhead in large indexed audits.
- Hardened text-file decoding so invalid UTF-8 in corpus or cached segment
  artifacts produces user-facing TAME-MT errors instead of raw tracebacks.
- Hardened malformed gzip handling so compressed corpus and cached segment
  inputs produce user-facing TAME-MT input errors.
- Tightened API and cached-scoring validation for empty references and
  non-positive training counts.
- Removed an unused experimental weighted-BLEU helper from the package surface.
- Added transparent `.gz` support for corpus inputs and text/JSONL/JSON outputs.
- Applied `.gz` output support consistently to TM baseline metadata JSONL.
- Extended the synthetic benchmark with staged index-build, indexed-audit, and
  cached-scoring timings.
- Extended staged benchmark guards with prepared cached-score and batch
  per-system latency checks.
- Added the staged synthetic benchmark to CI so indexed, cached, prepared
  cached, and batch cached performance regressions are caught before merge.
- Made staged benchmark indexed timings include persisted `.tameidx` load time.
- Tightened release acceptance performance checks to cover the 100k staged
  benchmark path with explicit build, indexed-audit, and cached-score
  thresholds.
- Added a persisted index-size threshold to the staged benchmark acceptance
  guard.
- Replaced inline wheel smoke tests with a cross-platform script covering gzip
  IO, index reuse, cached scoring, and TM metadata outputs.
- Added clean-venv built-wheel smoke testing to the acceptance script.
- Added built-wheel smoke testing to package CI and ensured wheel CI reruns when
  smoke fixtures or the smoke script change.
- Extended wheel smoke coverage to exercise the public prepared cached-scoring
  API from the installed package.
- Added `tame-mt doctor` for install/backend diagnostics.
- Added OPUS-100 public-corpus demo outputs and local performance notes for the
  50k-train/2k-test benchmark scale.
- Added synthetic benchmark smoke checks and an acceptance script.
- Added toy corpus, documentation, unit tests, native tests, and CI checks.
- Added typed exceptions, typed package marker, maturin package metadata, and
  release-oriented docs for public distribution.
- Polished package classifiers and contributor release guidance for the
  hardened acceptance workflow.
- Removed unnecessary runtime dependencies beyond SacreBLEU.
