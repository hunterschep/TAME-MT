# TAME-MT

[![CI](https://github.com/hunterschep/TAME-MT/actions/workflows/ci.yml/badge.svg)](https://github.com/hunterschep/TAME-MT/actions/workflows/ci.yml)
[![Wheels](https://github.com/hunterschep/TAME-MT/actions/workflows/wheels.yml/badge.svg)](https://github.com/hunterschep/TAME-MT/actions/workflows/wheels.yml)

**Training-Aware Machine Translation Evaluation.**

TAME-MT is a command-line tool and Python package for evaluating machine
translation systems while accounting for the data they were trained on.

Standard MT metrics such as BLEU and chrF answer:

> How close are the system translations to the references?

TAME-MT keeps those scores, but adds a second question:

> How close is each test example to the training corpus?

That second question matters because many MT benchmarks, especially
low-resource benchmarks, are built from narrow and overlapping sources:
scripture, government documents, educational text, health leaflets, NGO
materials, oral narratives, dictionaries, crawled web pages, or small public
corpora reused across projects. A high BLEU score on a train-similar test set
can be a valid in-domain result, but it is weaker evidence of broad translation
ability.

TAME-MT makes this visible.

It reports ordinary system quality, a nearest-neighbor translation-memory
baseline built from the training corpus, train-test exposure scores,
leakage-style overlap diagnostics, and quality broken out by distance from the
training data.

## Contents

- [When To Use It](#when-to-use-it)
- [What It Reports](#what-it-reports)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Large Corpora](#large-corpora)
- [Many Systems](#many-systems)
- [How To Read The Results](#how-to-read-the-results)
- [How It Works](#how-it-works)
- [The Math](#the-math)
- [Command Line Reference](#command-line-reference)
- [Python API](#python-api)
- [Outputs And Reproducibility](#outputs-and-reproducibility)
- [Package Architecture](#package-architecture)
- [Performance](#performance)
- [Privacy And Security](#privacy-and-security)
- [Limitations](#limitations)
- [Development And Contributing](#development-and-contributing)
- [Release Process](#release-process)
- [Citation](#citation)
- [License](#license)

## When To Use It

Use TAME-MT when you have:

- a machine translation training corpus;
- a test source file;
- one or more reference translation files;
- one or more system output files;
- a need to understand whether high scores come from broad generalization or
  from test examples that look very close to training examples.

TAME-MT is especially useful for:

- low-resource MT evaluation;
- benchmark audits before publishing scores;
- comparing several MT systems on the same train/test split;
- finding train-test exact overlaps and near duplicates;
- reporting whether BLEU/chrF claims are mostly supported by far-from-training
  examples or by near-training examples.

TAME-MT is not a replacement for BLEU, chrF, COMET, human evaluation, or error
analysis. It is a training-aware layer around corpus evaluation.

## What It Reports

| Output | Meaning |
| --- | --- |
| `BLEU`, `chrF` | Standard corpus quality scores for the system output. |
| `TM-BLEU`, `TM-chrF` | BLEU/chrF of a simple translation-memory baseline built from nearest training examples. |
| `delta over TM` | System score minus translation-memory baseline score. |
| `SourceExposure` | For each test source, similarity to the closest training source. |
| `TargetExposure` | For each reference, similarity to the closest training target. |
| `PairExposure` | Whether a test source/reference pair is close to the same training source/target pair. |
| `PairLeakTopK@0.85` | Fraction of test pairs whose top-k pair-reranked exposure is at least `0.85`. |
| `PairLeakExact@0.85` | Optional exact no-false-negative pair-threshold rate when `--exact-pair-thresholds` is used. |
| `source_exact` | Test source exactly appears in normalized training sources. |
| `pair_exact` | Test source and reference exactly appear as a normalized training pair. |
| `Far-BLEU`, `Far-chrF` | Quality on test examples far from the training source corpus. |
| `GenGap-BLEU`, `GenGap-chrF` | Near-bin score minus far-bin score. |

The shortest interpretation is:

- high system score and high TM baseline means the benchmark is partly solvable
  by training-set nearest-neighbor reuse;
- high delta over TM means the system is doing more than that baseline;
- strong far-bin scores are better evidence of generalization than strong
  near-bin scores alone;
- high PairLeakTopK or exact-pair overlap should be reported next to raw MT scores.

## Installation

```bash
pip install tame-mt
```

Check that the package and native backend are available:

```bash
tame-mt doctor
```

Published wheels include the Rust backend. If `doctor` reports `Native backend:
unavailable`, the install is incomplete and the default `auto` mode will refuse
to score until the native extension is installed.

For local development from a checkout:

```bash
pip install -e '.[dev]'
```

If an editable install was created before the Rust extension was built, rebuild
it with:

```bash
python -m pip install --force-reinstall --no-deps -e .
```

## Quick Start

TAME-MT expects aligned UTF-8 text files, one segment per line:

```text
train.src     source side of the training corpus
train.tgt     target side of the training corpus
test.src      source side of the test set
test.ref      reference translation for the test set
system.out    system hypothesis translations
```

Run a full train-aware score:

```bash
tame-mt score \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --hyp system.out
```

Run the bundled toy example:

```bash
tame-mt score \
  --train-src examples/toy/train.src \
  --train-tgt examples/toy/train.tgt \
  --test-src examples/toy/test.src \
  --ref examples/toy/test.ref \
  --hyp examples/toy/hyp.out
```

The toy report includes the main pieces:

```text
Quality
-------
Metric       System      TM baseline      delta over TM
BLEU           85.02            59.00            +26.01
chrF           85.45            61.73            +23.72

Exposure
--------
Source exposure:
  mean:             0.484
  exact overlap:    25.00%
  >= 0.85:         25.00%

Pair exposure:
  exact overlap:    25.00%
  PairLeakTopK@0.85:   25.00%

Generalization gap
------------------
GenGap-BLEU:  53.09
GenGap-chrF:  42.53
```

Write machine-readable artifacts:

```bash
tame-mt score \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --hyp system.out \
  --json-out report.json \
  --diagnostic-out segments.diagnostic.jsonl \
  --cache-out segments.tamecache \
  --tm-out tm.out
```

The JSON report is for dashboards and reproducible experiment records. The
diagnostic JSONL file is for per-example analysis. The `.tamecache` artifact
contains the TM hypotheses needed by `score-cached`.

## Large Corpora

Nearest-neighbor retrieval over the training corpus is the expensive part.
For large corpora, build a reusable training index once:

```bash
tame-mt index build \
  --train-src train.src \
  --train-tgt train.tgt \
  --out train.tameidx
```

Then score or audit without passing the training files again:

```bash
tame-mt score \
  --index train.tameidx \
  --test-src test.src \
  --ref test.ref \
  --hyp system_a.out \
  --json-out system_a.tame.json
```

Inspect an index bundle without loading the full native indexes:

```bash
tame-mt index inspect train.tameidx
```

Important: `.tameidx` bundles contain enough training text to reproduce
translation-memory outputs and neighbor diagnostics. Treat them like the
original training corpus.

## Many Systems

When the training corpus, test source, and references are fixed, cache the
train-aware diagnostics once:

```bash
tame-mt audit \
  --index train.tameidx \
  --test-src test.src \
  --ref test.ref \
  --cache-out segments.tamecache \
  --json-out audit.json
```

Then score one hypothesis without another training-corpus pass:

```bash
tame-mt score-cached \
  --cache-in segments.tamecache \
  --ref test.ref \
  --hyp system_a.out \
  --json-out system_a.tame.json
```

Or score many systems in one batch:

```bash
tame-mt score-cached-batch \
  --cache-in segments.tamecache \
  --ref test.ref \
  --system baseline=baseline.out \
  --system model_a=model_a.out \
  --system model_b=model_b.out \
  --json-out-dir tame_reports
```

This is the closest TAME-MT path to ordinary BLEU/chrF runtime. The expensive
exposure pass has already happened.

## How To Read The Results

| Pattern | Interpretation |
| --- | --- |
| High BLEU, high TM-BLEU, high PairLeakTopK, low Far-BLEU | Raw corpus score may overstate broad generalization. Report this as high-exposure or narrow-domain performance. |
| Moderate BLEU, low TM-BLEU, high delta over TM, good Far-BLEU | Stronger evidence that the system is doing more than nearest-neighbor reuse. |
| Low BLEU, low TM-BLEU, low Far-BLEU | The test set is not easy for the TM baseline, and the system is also weak. |
| High BLEU, low TM-BLEU, high Far-BLEU, low PairLeakTopK | Stronger evidence of generalization under this test distribution. |
| High exposure and no far-bin data | Valid in-domain result, but weak evidence for broad out-of-domain generalization. |

Use these diagnostics as context, not as automatic pass/fail rules. A
high-exposure benchmark can still be useful if the claim is in-domain
performance. It is a problem when high-exposure scores are presented as broad
generalization evidence.

## How It Works

TAME-MT builds a simple translation-memory baseline from the training corpus:

1. Normalize all strings for exposure scoring.
2. Represent each string as a set of character n-grams.
3. For each test source, find the most similar training source.
4. Reuse that training source's paired target translation as the TM hypothesis.
5. Score the system output and the TM hypotheses with BLEU and chrF.
6. Compute exposure summaries and source-distance bins.

The default similarity is tokenizer-free character n-gram Jaccard similarity.
That keeps the method deterministic and usable for languages without reliable
tokenizers, segmenters, pretrained encoders, or external downloads.

By default TAME-MT normalizes only:

```python
text = unicodedata.normalize("NFKC", text)
text = text.strip()
text = re.sub(r"\s+", " ", text)
```

It does not lowercase, strip diacritics, strip punctuation, or normalize digits
unless you ask for those options.

## The Math

This section uses simple GitHub-supported Markdown math and avoids custom
operator macros.

Assume a training corpus with `m` aligned pairs:

```text
(u_1, v_1), (u_2, v_2), ..., (u_m, v_m)
```

and a test set with `n` examples:

```text
(x_1, r_1), (x_2, r_2), ..., (x_n, r_n)
```

Here:

- `u_j` is a normalized training source;
- `v_j` is the paired normalized training target;
- `x_i` is a normalized test source;
- `r_i` is a normalized reference translation;
- `h_i` is a system hypothesis;
- `h_i^TM` is the translation-memory hypothesis.

### Character N-Gram Sets

For a normalized string `s`, let `G_k(s)` be the set of character n-grams of
length `k`.

$$
G(s) = G_3(s) \cup G_4(s) \cup G_5(s)
$$

### String Similarity

Similarity is Jaccard similarity over the character n-gram sets:

$$
\mathrm{sim}(a,b) =
\frac{|G(a) \cap G(b)|}{|G(a) \cup G(b)|}
$$

If both strings are empty, similarity is defined as `1.0`. If only one string
is empty, similarity is `0.0`.

### Source Exposure

For each test source, source exposure is the similarity to the nearest training
source:

$$
E_i^{src} =
\max_{1 \le j \le m} \mathrm{sim}(x_i,u_j)
$$

The nearest training source index is:

$$
n_i^{src} =
\min \{j : \mathrm{sim}(x_i,u_j) = E_i^{src}\}
$$

Ties are broken by choosing the lowest training index.

### Target Exposure

For each reference translation, target exposure is the similarity to the
nearest training target:

$$
E_i^{tgt} =
\max_{1 \le j \le m} \mathrm{sim}(r_i,v_j)
$$

### Pair Exposure

Pair exposure asks a stricter question: is there one training pair whose source
and target sides are both close to the test source/reference pair?

For test pair `i` and training pair `j`:

$$
P_{ij} =
\min(\mathrm{sim}(x_i,u_j), \mathrm{sim}(r_i,v_j))
$$

The pair exposure for test example `i` is:

$$
E_i^{pair} =
\max_{1 \le j \le m} P_{ij}
$$

For threshold `t`, pair leak is:

$$
\mathrm{PairLeakTopK}_{t} =
\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[E_i^{pair} \ge t]
$$

For example, `PairLeakTopK@0.85 = 0.20` means 20% of test examples have a
source/reference pair that is very close to one training pair within the
top-k candidate set. Exact pair overlap is exact; top-k pair leak is labeled
separately because it is candidate-set limited.

When `--exact-pair-thresholds` is enabled, TAME-MT also reports
`PairLeakExact@t`. That value checks whether any same-index training pair has
both source similarity and target similarity at least `t`. It has no false
negatives for the configured threshold, but it can be much slower than top-k
pair reranking on large corpora.

### Translation-Memory Baseline

The TM hypothesis for test source `x_i` is selected using only the source side:

$$
h_i^{TM} = v_{n_i^{src}}
$$

If no useful candidate shares character n-grams with the test source, the
default TM hypothesis is an empty string.

TAME-MT never uses the reference translation to choose the TM hypothesis.
References are used only for scoring and diagnostics.

### Delta Over TM

The system and TM baseline are scored with the same corpus metric:

$$
B_{sys} = \mathrm{BLEU}(h,r)
$$

$$
B_{TM} = \mathrm{BLEU}(h^{TM},r)
$$

The BLEU improvement over the training-memory baseline is:

$$
\Delta B = B_{sys} - B_{TM}
$$

The same definition is used for chrF.

### Distance Bins

TAME-MT bins test examples by source exposure:

| Bin | Definition |
| --- | --- |
| `source_exact` | Normalized test source appears exactly in normalized `train.src`. |
| `near` | Not exact, and `SourceExposure >= 0.70`. |
| `medium` | `0.30 <= SourceExposure < 0.70`. |
| `far` | `SourceExposure < 0.30`. |

Generalization gap is near-bin quality minus far-bin quality:

$$
Gap_{BLEU} = B_{near} - B_{far}
$$

A large positive gap means the system performs much better on train-similar
examples than on train-distant examples. A small gap is not automatically good:
a weak system can score poorly in every bin.

## Command Line Reference

Main commands:

| Command | Use |
| --- | --- |
| `tame-mt doctor` | Show package, Python, dependency, and native backend status. |
| `tame-mt score` | Run full train-aware scoring for one hypothesis. |
| `tame-mt audit` | Compute exposure diagnostics without a system hypothesis. |
| `tame-mt tm-baseline` | Write nearest-neighbor TM hypotheses. |
| `tame-mt index build` | Build a reusable `.tameidx` training index. |
| `tame-mt index inspect` | Inspect a `.tameidx` bundle manifest. |
| `tame-mt index verify` | Verify bundle hashes, archive shape, and native-index invariants. |
| `tame-mt score-cached` | Score one hypothesis from cached segment diagnostics. |
| `tame-mt score-cached-batch` | Score many hypotheses from one cached segment file. |

Common options:

```bash
--metrics bleu chrf
--ngram-orders 3,4,5
--far-threshold 0.30
--near-threshold 0.70
--leak-thresholds 0.70,0.85,0.95
--pair-k 50
--index-mode auto
--lowercase
--strip-diacritics
--normalize-punctuation
--bleu-tokenize 13a
--bleu-lowercase
--chrf-word-order 2
--verbose
```

Raw text is excluded from segment reports by default. These flags write raw
text and should be used only when the data can safely be stored:

```bash
--include-source-text
--include-reference-text
--include-hyp-text
--include-neighbor-text
```

`--include-neighbor-text` may write raw training text.

Plain UTF-8 files and `.gz` files are supported for corpus inputs and for
text, JSONL, and JSON outputs.

For exhaustive CLI details, see [docs/cli.md](docs/cli.md).

## Python API

Score files:

```python
from tame_mt import ScoreConfig, TameScorer

scorer = TameScorer(ScoreConfig())

report = scorer.score_files(
    train_src="train.src",
    train_tgt="train.tgt",
    test_src="test.src",
    refs=["test.ref"],
    hyp="system.out",
)

print(report.system_scores["bleu"])
print(report.tm_scores["bleu"])
print(report.delta_scores["bleu"])
print(report.signature)
```

Score in-memory corpora:

```python
from tame_mt import TameScorer

scorer = TameScorer()

report = scorer.score_corpus(
    train_src=train_src_lines,
    train_tgt=train_tgt_lines,
    test_src=test_src_lines,
    refs=[ref_lines],
    hyp=hyp_lines,
)
```

Get segment diagnostics and TM outputs:

```python
result = scorer.evaluate_corpus(
    train_src=train_src_lines,
    train_tgt=train_tgt_lines,
    test_src=test_src_lines,
    refs=[ref_lines],
    hyp=hyp_lines,
)

report = result.report
segments = result.exposures
tm_hypotheses = result.tm_hyp
```

Reuse a saved training index:

```python
from tame_mt import load_index_bundle, save_index_bundle

save_index_bundle("train.tameidx", train_src_lines, train_tgt_lines, scorer.config)

bundle = load_index_bundle("train.tameidx", scorer.config)
result = scorer.evaluate_index_bundle(
    bundle=bundle,
    test_src=test_src_lines,
    refs=[ref_lines],
    hyp=hyp_lines,
)
```

Prepare cached diagnostics once and score many systems:

```python
from tame_mt import load_cached_artifact

artifact = load_cached_artifact(
    "segments.tamecache",
    refs=[ref_lines],
    config=scorer.config,
)
cached = scorer.prepare_from_cached_artifact(
    artifact,
    refs=[ref_lines],
)

reports = cached.score_many(
    {
        "baseline": baseline_lines,
        "model_a": model_a_lines,
        "model_b": model_b_lines,
    }
)
```

For deeper API documentation, see [docs/api.md](docs/api.md).

## Outputs And Reproducibility

`--json-out` writes a stable report structure:

```json
{
  "schema_version": "1.0",
  "tame_version": "0.2.1",
  "signature": "tame-mt|v:0.2.1|...",
  "data": {
    "num_train": 125000,
    "num_test": 1000,
    "num_refs": 1
  },
  "retrieval": {
    "mode": "exact",
    "source_exposure_mode": "exact",
    "target_exposure_mode": "exact",
    "pair_exposure_mode": "topk_rerank",
    "tm_retrieval_exact": true
  },
  "backend": {
    "name": "native_exact",
    "native": true,
    "exact": true,
    "requested_mode": "auto",
    "resolved_mode": "native_exact",
    "index_reused": false
  },
  "quality": {
    "system": {"bleu": 31.4, "chrf": 54.2},
    "tm": {"bleu": 23.8, "chrf": 47.1},
    "delta_tm": {"bleu": 7.6, "chrf": 7.1}
  },
  "exposure": {
    "source": {"mean": 0.71, "exact_overlap": 0.042},
    "target": {"mean": 0.684, "exact_overlap": 0.038},
    "pair": {"exact_overlap": 0.031}
  },
  "bins": [],
  "generalization_gap": {"bleu": 20.7, "chrf": 19.3},
  "warnings": []
}
```

Every report includes a deterministic signature that records:

- TAME-MT version;
- normalization settings;
- similarity function;
- retrieval mode, approximation flag, backend, and index mode;
- TM baseline policy;
- bin and leak thresholds;
- pair reranking top-k and whether exact pair thresholds were computed;
- BLEU/chrF settings;
- metric-affecting dependency versions.

Example:

```text
tame-mt|v:0.2.1|norm:nfkc_ws_case|sim:char_jaccard_3-5_set|retrieval:exact|approx:0|idx:auto|backend:native_exact|tm:src_nn_top1_zero_empty|bins:far0.30_near0.70_leak0.70,0.85,0.95|pair_k:50|pair_exact:0|fast:8,500,3000,1000|metrics:bleu,chrf|sacrebleu:bleu_tok_13a_lc_0_chrf_wo_2|deps:sacrebleu_2.6.0
```

For the JSON schema, see [docs/json_schema.md](docs/json_schema.md).

## Package Architecture

TAME-MT is split into small, testable modules:

```text
src/tame_mt/
  cli.py          command-line interface
  api.py          public Python scoring API
  config.py       typed configuration and validation
  schema.py       report and segment dataclasses
  normalize.py    deterministic text normalization
  ngrams.py       character n-gram extraction
  similarity.py   Jaccard similarity logic
  index/          retrieval interfaces, native wrapper, and exact reference index
  native.py       native backend selection/status
  exact.py        exact overlap and exact-match helpers
  exposure.py     source, target, and pair exposure assembly
  approx_validation.py per-run fast-mode validation against exact retrieval
  tm.py           source-nearest-neighbor TM baseline
  bins.py         source-distance bins and gap metrics
  scoring.py      BLEU/chrF corpus and bin scoring
  metrics/        SacreBLEU integration
  artifacts.py    cached segment JSONL validation
  persistence.py  .tameidx bundle save/load/inspection
  performance.py  timing, thread, and memory metadata
  report.py       text, JSON, and JSONL report rendering
  io.py           UTF-8 and gzip input/output helpers
```

The retrieval package is intentionally split:

```text
src/tame_mt/index/
  __init__.py     stable import surface for retrieval types
  base.py         retrieval protocols and result/backend dataclasses
  modes.py        backend mode and threshold validation helpers
  native.py       Rust-backed production index wrapper
  python_exact.py exact Python reference index for parity/debugging
  factory.py      construction helper for internal callers
```

The Rust/PyO3 native extension is the production retrieval engine. Python owns
packaging, the public API, CLI argument parsing, file IO, normalization,
SacreBLEU integration, aggregation, and report rendering. The internal
`tame_mt.index` package separates retrieval protocols, mode resolution, the
native wrapper, and a small Python exact reference index used for parity tests
and debugging. Rust owns high-throughput nearest-neighbor search, exact
reranking, pair reranking, and serialized native indexes. If the Rust extension
is unavailable, normal scoring fails with an installation error instead of
falling back silently.

Rust crate layout:

```text
src/lib.rs              PyO3 module registration
src/index/mod.rs        native index construction and Python-exposed methods
src/index/query.rs      top-k query orchestration and exact/fast dispatch
src/index/exact.rs      exact nearest-neighbor ranking with reusable workspaces
src/index/fast.rs       approximate rare-gram candidate collection and rerank
src/index/pair.rs       pair-candidate reranking
src/index/validation.rs serialized-index invariant checks
src/index/workspace.rs  reusable per-query native work buffers
src/index/tests.rs      Rust native-index tests
src/ngrams.rs           UTF-8-safe character n-gram slicing
src/similarity.rs       integer Jaccard helpers
src/validation.rs       shared validation helpers
src/types.rs            compact native type aliases
```

Supporting directories:

```text
tests/        Python unit and CLI tests
benchmarks/   synthetic performance and recall checks
examples/     toy data and public-corpus demo scripts/results
docs/         CLI, API, method, privacy, reproducibility, release, and demo docs
schemas/      JSON Schema contracts for reports and artifacts
.github/      CI, wheel, release, and Dependabot workflows
```

## Performance

TAME-MT does more work than BLEU because it compares test examples to the
training corpus. The package is designed so that cost is paid once whenever
possible.

Retrieval modes:

| Mode | Behavior | Use |
| --- | --- | --- |
| `auto` | Default. Requires Rust and resolves to exact native retrieval. | Production scoring and audits. |
| `native_exact` | Rust exact character n-gram Jaccard retrieval over shared-gram candidates. | Small and medium corpora when exact nearest-neighbor exposure is required. |
| `native_fast` | Rust rare-gram candidate generation plus exact reranking of a bounded shortlist. | Explicit approximate exploratory runs with `--retrieval approx --allow-approximate`. |

There are no Python retrieval backends. The small Python `ngrams.py` and
`similarity.py` modules define the public metric math for tests and examples;
they are not indexing engines.

Fast mode can be validated per corpus by sampling exact retrieval:

```bash
tame-mt audit \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --retrieval approx \
  --allow-approximate \
  --validate-approx-sample 1000 \
  --json-out audit.json
```

The report then includes `approx_validation`, which compares approximate and
exact retrieval for nearest-neighbor agreement, source-bin decisions,
source-score error, pair-threshold decisions, and sample TM-BLEU. Validation is
a guardrail for fast exploratory runs; exact mode remains the canonical mode
for published numbers.

Recommended production workflow:

1. Build `train.tameidx` once for a fixed training corpus.
2. Run `audit --index train.tameidx --cache-out segments.tamecache` once for a
   fixed test/reference setup.
3. Run `score-cached-batch` for all systems.

Local benchmark notes for public corpora and synthetic corpora are in
[docs/performance.md](docs/performance.md) and
[examples/public_corpora_demo/](examples/public_corpora_demo/).

Every JSON report includes a `performance` block with backend, thread count,
index-reuse status, peak RSS, and available stage timings. Use
`--profile-json profile.json` on CLI runs to write a separate command profile
that includes final output-writing time.

Native retrieval uses Rayon. Pass `--threads N` to fix the worker count for a
CLI run; `--threads auto` keeps Rayon defaults. Report JSON records the actual
thread count in `performance.threads`.

## Privacy And Security

TAME-MT runs locally. It does not download models, call remote services, or send
text anywhere.

Data can still leak through artifacts you choose to write:

- `--include-neighbor-text` can write raw training text;
- `.tameidx` bundles contain training text needed for reproducible TM outputs;
- `.tamecache` files contain TM baseline hypotheses and may therefore contain
  raw training-target text;
- diagnostic/cache JSONL files may contain raw test/reference/hypothesis text
  if raw text flags are enabled.

Use `--diagnostic-out` for privacy-safer diagnostics that omit TM hypotheses by
default. Use `--include-tm-text` only when that diagnostic artifact may safely
store TM hypotheses. Cached scoring uses `--cache-out`, which intentionally
includes TM hypotheses because they are required to recompute TM-BLEU and delta
over TM.

Treat `.tameidx`, `.tamecache`, diagnostic JSONL, and metadata files from
unknown sources as untrusted. The loader rejects malformed or suspicious index
bundles before native deserialization, including unexpected ZIP members,
duplicate names, unsafe declared sizes, excessive compression ratios,
load-memory budget violations, unsupported schema versions, and invalid
native-index invariants.

For more detail, see [SECURITY.md](SECURITY.md).

## Limitations

TAME-MT does not prove that a neural model memorized a sentence. It does not
measure semantic adequacy. It does not decide whether a benchmark is valid. It
does not replace human evaluation.

The default similarity is surface-based. It can miss semantic paraphrases and
can overemphasize orthographic similarity.

Fast retrieval mode is approximate for nearest-neighbor retrieval, although
shortlisted candidates are reranked with the exact Jaccard formula and exact
normalized source, target, and pair overlaps remain exact.

Pair exposure in v0.1 reranks a candidate set for speed instead of scoring
every possible training pair. Exact pair overlap is still exact.

## Development And Contributing

Install development dependencies:

```bash
pip install -e '.[dev]'
```

Run the core checks:

```bash
pytest
ruff format --check .
ruff check .
mypy src/tame_mt
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
python benchmarks/validate_fast_recall.py --require-native
python -m build
python -m twine check dist/*
```

Run the full release-candidate acceptance suite:

```bash
scripts/acceptance.sh
```

Contribution rules:

- keep runtime dependencies small;
- do not add model downloads or remote calls;
- do not change metric definitions without updating signatures, tests, and the
  changelog;
- do not print raw training-neighbor text by default;
- add tests for user-visible CLI, JSON, artifact, or metric changes.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Release Process

Releases are built by GitHub Actions and published through PyPI Trusted
Publishing.

Before a release:

1. Update versions in `pyproject.toml`, `Cargo.toml`, and
   `src/tame_mt/version.py`, plus `CITATION.cff` and versioned README/schema
   examples.
2. Update [CHANGELOG.md](CHANGELOG.md).
3. Run `python scripts/check_versions.py`.
4. Run `scripts/acceptance.sh`.
5. Push a `v*` tag.
6. Let the tag-triggered release workflow build wheels, validate distributions,
   and generate the SBOM artifact.
7. Manually dispatch the release workflow from the tag with `publish=true`.
8. Approve the protected `pypi` environment deployment.

See [docs/release.md](docs/release.md).

## Citation

```bibtex
@software{tame_mt_2026,
  title = {TAME-MT: Training-Aware Machine Translation Evaluation},
  year = {2026},
  version = {0.2.1},
  url = {https://github.com/hunterschep/TAME-MT}
}
```

## License

TAME-MT is released under the MIT License. See [LICENSE](LICENSE).
