# Changelog

## Unreleased

- Made the PyPI publish job retry-safe by checking existing PyPI filenames and
  SHA-256 hashes before upload. Matching already-published artifacts are
  removed from the local upload set, all-present releases skip upload cleanly,
  and same-name/different-hash artifacts fail with explicit guidance to release
  a new version.

## 0.2.0 - 2026-06-27

- Added a structured `CachedArtifact` public API and `load_cached_artifact()`
  loader that applies the same metadata, reference-hash, TM-hypothesis, train
  count, and privacy checks as cached CLI scoring.
- Added first-class `--diagnostic-out`, `--cache-out`, and `--cache-in` CLI
  artifact names so privacy-safer diagnostics are clearly separated from
  cacheable artifacts that intentionally store TM hypotheses.
- Added `TameScorer.prepare_from_cached_artifact()`,
  `score_from_cached_artifact()`, and `score_many_from_cached_artifact()` so
  services and notebooks no longer need to manually pass metadata dictionaries.
- Refactored `score-cached` and `score-cached-batch` to use the same
  artifact-loader validation path as the Python API, reducing validation drift.
- Added typed `ArtifactValidationError`, `ApproximationError`, and
  `SecurityError` exception classes for more precise downstream error handling.
- Added exact source-threshold APIs on `native_exact` plus
  `benchmarks/validate_threshold_exact.py` so no-false-negative threshold flags
  and source-bin decisions are regression-tested and never computed from
  `native_fast`.
- Optimized native exact retrieval to reuse vector-backed per-worker query
  workspaces instead of allocating per-query candidate maps, with deterministic
  exact/fast native query entry points and parity tests.
- Split the Python retrieval architecture from a flat `index.py` module into
  `tame_mt.index` protocols, mode helpers, native wrapper, factory helper, and
  a Python exact reference index for parity/debugging.
- Added CLI `--threads` control for native Rayon retrieval and subprocess tests
  proving deterministic report semantics with one thread, four threads, and the
  default thread pool.
- Hardened the OPUS-100 public-corpora demo with `--quick`, `--standard`, and
  `--paper` tiers, first-class `tame-mt demo opus100` CLI access, retrying
  downloads with timeouts, summary JSON/CSV/Markdown under
  `examples/public_corpora_demo/results/`, and a dedicated example README.
- Added Hypothesis property tests for Jaccard invariants, exact-match exposure,
  batch-vs-single query parity, native-index round trips, persisted-index
  round trips, exact threshold no-false-negative flags, and approximate-report
  labeling.
- Relaxed exact index-bundle compatibility for query-time settings such as
  `topk` and `batch_size` while still rejecting normalization, similarity,
  backend-mode, and fast-cap mismatches.
- Tightened the local acceptance gate with locked Rust checks, Python and Rust
  dependency audits, and clean-wheel `pip check` before smoke testing.
- Bumped package, native crate, citation, schema examples, and README examples
  to `0.2.0`.

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
- Hardened `.tameidx` loading further with unexpected-member rejection,
  member-specific hard caps, total uncompressed-size limits, compression-ratio
  checks, and streaming UTF-8 training-text reads before native index
  deserialization.
- Made `.tameidx` writes atomic by writing a temporary bundle in the destination
  directory and replacing the output path only after the archive closes
  successfully.
- Optimized native index reuse by avoiding duplicate Python exact maps and by
  persisting compact exact-pair fingerprints for faster repeated audits.
- Reduced large-audit memory pressure by storing native exact-match keys and
  Python exact-pair membership as fixed-size fingerprints instead of full
  normalized strings.
- Reduced 1M-train/2k-test exact audit peak RSS below the 4 GiB target on the
  release-candidate machine by avoiding retained Python normalized training
  copies in fresh scoring and persisted-index builds.
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
- Preserved the original segment-artifact backend in cached JSON reports via
  `backend.artifact_backend` when metadata sidecars are available.
- Added SacreBLEU runtime versions to report signatures and JSON config
  metadata so metric-affecting dependency changes are visible and reproducible.
- Bounded the supported SacreBLEU dependency range to `sacrebleu>=2.4,<3` and
  added CI compatibility tests for supported 2.x ranges.
- Exposed segment metadata path/read/validation helpers in the public Python
  namespace so services can enforce the same cached-artifact checks as the CLI.
- Extended the built-wheel smoke test to require the native backend up front
  and validate segment metadata sidecars through the public API.
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
- Added `benchmarks/validate_fast_recall.py` and CI/acceptance coverage for
  deterministic fast-vs-exact retrieval recall, agreement, and score-gap
  characterization.
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
- Added deterministic fuzz-style parser tests for segment JSONL scalar
  encodings, native/Python retrieval parity tests, and a SacreBLEU segment-stat
  acceleration sentinel.
- Added release CI for dependency audits, tag-driven trusted PyPI publishing,
  build provenance attestation, and SPDX SBOM artifact generation.
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
