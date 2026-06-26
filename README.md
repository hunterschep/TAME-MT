# TAME-MT

Training-Aware Machine Translation Evaluation for Machine Translation.

TAME-MT is a command-line tool and Python package that explains MT scores in
light of the data the model was trained on. It does not replace BLEU or chrF.
It answers a different question:

> How much of this system's measured quality is earned on test examples that
> are close to the training corpus?

That question matters in low-resource MT because train and test data often come
from narrow domains: Bible text, government forms, education materials, health
leaflets, NGO documents, oral narratives, or a small number of web sources.
High BLEU on a highly exposed test set can be a valid in-domain result, but it
is weaker evidence of broad general-purpose translation ability.

TAME-MT makes that visible by reporting:

| Number | Meaning |
| --- | --- |
| `BLEU`, `chrF` | Standard system quality scores. |
| `TM-BLEU`, `TM-chrF` | Score of a training-set nearest-neighbor translation-memory baseline. |
| `delta over TM` | System score minus translation-memory baseline score. |
| `SourceExposure` | How similar each test source is to the closest training source. |
| `PairLeak@0.85` | How many test source/reference pairs are close to the same training pair. |
| `Far-BLEU` | BLEU on test examples far from the training source corpus. |
| `GenGap-BLEU` | Near-bin BLEU minus far-bin BLEU. |

## Installation

```bash
pip install tame-mt
```

For local development:

```bash
pip install -e '.[dev]'
```

## Quick Start

TAME-MT expects aligned UTF-8 text files:

```text
train.src     source side of the training corpus
train.tgt     target side of the training corpus
test.src      source side of the test set
test.ref      reference translation for the test set
system.out    system hypothesis translations
```

Run the full report:

```bash
tame-mt score \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --hyp system.out
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
  --segment-out segments.jsonl \
  --tm-out tm.out
```

For large corpora, build the training index once and reuse it:

```bash
tame-mt index build \
  --train-src train.src \
  --train-tgt train.tgt \
  --out train.tameidx

tame-mt score \
  --index train.tameidx \
  --test-src test.src \
  --ref test.ref \
  --hyp system_a.out \
  --json-out system_a.tame.json
```

The `.tameidx` bundle stores the native source/target indexes plus the raw
training text needed for TM outputs and optional segment reports. Treat it like
the original training corpus for privacy and access control.

If the train/test/reference setup is fixed and only hypotheses change, cache
the train-aware diagnostics once and reuse them:

```bash
tame-mt audit \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --segment-out segments.jsonl \
  --json-out audit.json

tame-mt score-cached \
  --segment-in segments.jsonl \
  --ref test.ref \
  --hyp system_a.out \
  --num-train 125000 \
  --json-out system_a.tame.json
```

`score-cached` does not rebuild the training index. That makes repeated scoring
of new hypotheses close to ordinary BLEU/chrF scoring cost after the first
audit.

Try the bundled toy example:

```bash
tame-mt score \
  --train-src examples/toy/train.src \
  --train-tgt examples/toy/train.tgt \
  --test-src examples/toy/test.src \
  --ref examples/toy/test.ref \
  --hyp examples/toy/hyp.out
```

## The Core Idea

TAME-MT builds a simple translation-memory baseline from the training corpus:

1. For each test source sentence, find the most similar training source
   sentence.
2. Reuse that training sentence's paired target translation.
3. Score those reused translations with BLEU and chrF.

If this simple baseline already scores well, the test set is highly
training-solvable under surface nearest-neighbor reuse. That does not mean the
system is bad. It means raw corpus scores should be interpreted as
high-exposure or narrow-domain performance unless far-bin results support a
broader claim.

## How Similarity Works

TAME-MT uses tokenizer-free character n-gram Jaccard similarity by default. This
keeps the method deterministic and usable for low-resource languages without
pretrained models, language-specific tokenizers, or downloads.

First, text is normalized for exposure calculations:

```python
text = unicodedata.normalize("NFKC", text)
text = text.strip()
text = re.sub(r"\s+", " ", text)
```

By default TAME-MT does not lowercase, strip diacritics, strip punctuation, or
normalize digits.

For a normalized string \(s\), define \(G(s)\) as the set of character n-grams
for orders 3, 4, and 5:

```math
G(s) = \{ \text{all character n-grams in } s \mid n \in \{3,4,5\} \}
```

The similarity between two strings \(a\) and \(b\) is Jaccard similarity:

```math
\operatorname{sim}(a,b) =
\frac{|G(a) \cap G(b)|}{|G(a) \cup G(b)|}
```

If both strings are empty, similarity is defined as 1.0. If only one is empty,
similarity is 0.0.

## Exposure Metrics

For each test source \(x_i\), TAME-MT finds the closest training source:

```math
\operatorname{SourceExposure}_i =
\max_j \operatorname{sim}(x_i, \operatorname{train\_src}_j)
```

The nearest-neighbor index is:

```math
\operatorname{SourceNNIndex}_i =
\arg\max_j \operatorname{sim}(x_i, \operatorname{train\_src}_j)
```

Ties are broken by choosing the lowest training index.

For a reference translation \(r_i\), target exposure is:

```math
\operatorname{TargetExposure}_i =
\max_j \operatorname{sim}(r_i, \operatorname{train\_tgt}_j)
```

Pair exposure asks a stricter question: is there one training pair whose source
and target sides are both close to the test source/reference pair?

For a test pair \((x_i, r_i)\) and training pair
\((\operatorname{train\_src}_j, \operatorname{train\_tgt}_j)\):

```math
\operatorname{pair\_sim}(i,j) =
\min(
  \operatorname{sim}(x_i, \operatorname{train\_src}_j),
  \operatorname{sim}(r_i, \operatorname{train\_tgt}_j)
)
```

Then:

```math
\operatorname{PairExposure}_i =
\max_j \operatorname{pair\_sim}(i,j)
```

`PairLeak@0.85` is the fraction of test examples where
\(\operatorname{PairExposure}_i \ge 0.85\).

In v0.1, pair exposure reranks the union of source and target top-k candidates
instead of scoring every training pair. Exact pair overlap is still computed
exactly.

## Translation-Memory Baseline

The translation-memory hypothesis for test source \(x_i\) is selected using
only the source side:

```math
j^\*(i) = \arg\max_j \operatorname{sim}(x_i, \operatorname{train\_src}_j)
```

```math
\operatorname{tm\_hyp}_i =
\operatorname{train\_tgt}_{j^\*(i)}
```

If no candidate shares character n-grams with the test source, the default
policy is:

```math
\operatorname{tm\_hyp}_i = ""
```

The baseline scores are ordinary corpus metrics:

```math
\operatorname{TM\text{-}BLEU} =
\operatorname{BLEU}(\operatorname{tm\_hyp}, \operatorname{ref})
```

```math
\Delta\operatorname{TM\text{-}BLEU} =
\operatorname{SystemBLEU} - \operatorname{TM\text{-}BLEU}
```

The same definitions apply to chrF.

Important: TAME-MT never uses the reference translation to choose the TM
hypothesis. References are only used for scoring and diagnostics.

## Distance-Stratified Quality

TAME-MT bins test examples by source exposure:

| Bin | Definition |
| --- | --- |
| `source_exact` | normalized test source appears exactly in normalized `train.src` |
| `near` | not exact and `SourceExposure >= 0.70` |
| `medium` | `0.30 <= SourceExposure < 0.70` |
| `far` | `SourceExposure < 0.30` |

Each bin reports count, percentage, mean source exposure, system BLEU/chrF,
TM-BLEU/TM-chrF, and delta over TM.

Generalization gap is:

```math
\operatorname{GenGap\text{-}BLEU} =
\operatorname{BLEU}_{near} - \operatorname{BLEU}_{far}
```

A large positive gap means the system performs much better on train-similar
examples than on train-distant examples. A small gap is not automatically good:
a weak system can perform poorly in every bin.

## Worked Example

Suppose a training corpus contains:

```text
train.src: god created the heaven and the earth
train.tgt: dios creó el cielo y la tierra
```

And the test set contains the exact same source/reference pair:

```text
test.src:  god created the heaven and the earth
test.ref:  dios creó el cielo y la tierra
```

TAME-MT will report:

```text
SourceExposure = 1.0
source_exact = true
PairExposure = 1.0
pair_exact = true
tm_hyp = "dios creó el cielo y la tierra"
```

That segment contributes to `source_exact`, not `near`. If many test examples
look like this, raw BLEU may be measuring training-set reuse as much as system
generalization.

Now suppose another test source is similar but not exact:

```text
test.src: and the earth was without form and void
```

If its closest training source is:

```text
train.src: and the earth was without form
```

the example may fall in the `near` bin. The TM baseline will reuse the paired
training translation. If that baseline gets high BLEU, the test item is partly
solvable by nearest-neighbor reuse.

## Reading Common Patterns

| Pattern | Plain-language interpretation |
| --- | --- |
| High BLEU, high TM-BLEU, high PairLeak, low Far-BLEU | Raw score may overstate broad generalization. Treat as high-exposure or narrow-domain performance. |
| Moderate BLEU, low TM-BLEU, high delta over TM, good Far-BLEU | Stronger evidence that the system is doing more than nearest-neighbor reuse. |
| Low BLEU, low TM-BLEU, low Far-BLEU | The test set is not TM-solvable, and the system also performs poorly. |
| High BLEU, low TM-BLEU, high Far-BLEU, low PairLeak | Stronger evidence of real generalization under this test distribution. |
| High exposure and no far-bin data | Valid in-domain result, weak evidence for broad out-of-domain generalization. |

## CLI Reference

Full scoring:

```bash
tame-mt score \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref \
  --hyp system.out
```

Audit a benchmark before evaluating systems:

```bash
tame-mt audit \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --ref test.ref
```

Write translation-memory hypotheses:

```bash
tame-mt tm-baseline \
  --train-src train.src \
  --train-tgt train.tgt \
  --test-src test.src \
  --out tm.out \
  --metadata-out tm_metadata.jsonl
```

Score a system from cached segment diagnostics:

```bash
tame-mt score-cached \
  --segment-in segments.jsonl \
  --ref test.ref \
  --hyp system.out \
  --num-train 125000
```

Build a reusable training index and use it without passing train files again:

```bash
tame-mt index build \
  --train-src train.src \
  --train-tgt train.tgt \
  --out train.tameidx

tame-mt score \
  --index train.tameidx \
  --test-src test.src \
  --ref test.ref \
  --hyp system.out
```

Common options:

```bash
--metrics bleu chrf
--ngram-orders 3,4,5
--far-threshold 0.30
--near-threshold 0.70
--leak-thresholds 0.70,0.85,0.95
--pair-k 50
--index-mode auto
--auto-exact-cutoff 5000
--candidate-gram-limit 8
--posting-limit 500
--max-candidates 3000
--rerank-limit 1000
--min-bin-size-warning 30
--tm-zero-policy empty
--lowercase
--strip-diacritics
--normalize-punctuation
--bleu-tokenize 13a
--bleu-lowercase
--chrf-word-order 2
```

Segment reports do not include raw text by default. Use these only when it is
safe to write the data:

```bash
--include-source-text
--include-reference-text
--include-hyp-text
--include-neighbor-text
```

`--include-neighbor-text` may write raw training text.

## Python API

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

In-memory corpora:

```python
report = scorer.score_corpus(
    train_src=train_src_lines,
    train_tgt=train_tgt_lines,
    test_src=test_src_lines,
    refs=[ref_lines],
    hyp=hyp_lines,
)
```

Need segment diagnostics too:

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
tm_output = result.tm_hyp
```

Reuse a saved training index from Python:

```python
from tame_mt import load_index_bundle, save_index_bundle

save_index_bundle("train.tameidx", train_src_lines, train_tgt_lines, scorer.config)
bundle = load_index_bundle("train.tameidx", scorer.config)
result = scorer.evaluate_index_bundle(bundle, test_src_lines, [ref_lines], hyp_lines)
```

## JSON Output

`--json-out` writes a stable top-level structure:

```json
{
  "schema_version": "0.1",
  "tame_version": "0.1.0",
  "signature": "tame-mt|v:0.1.0|...",
  "data": {
    "num_train": 125000,
    "num_test": 1000,
    "num_refs": 1
  },
  "backend": {
    "name": "native_fast",
    "native": true,
    "exact": false,
    "requested_mode": "auto",
    "resolved_mode": "native_fast",
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

JSON uses fractions such as `0.184`; the human report renders percentages such
as `18.40%`.

## Reproducibility

Every report includes a deterministic signature, for example:

```text
tame-mt|v:0.1.0|norm:nfkc_ws_case|sim:char_jaccard_3-5_set|idx:auto|backend:native_fast|tm:src_nn_top1_zero_empty|bins:far0.30_near0.70_leak0.70,0.85,0.95|pair_k:50|fast:8,500,3000,1000|metrics:bleu,chrf|sacrebleu:bleu_tok_13a_lc_0_chrf_wo_2
```

The signature records the TAME-MT version, normalization, similarity function,
requested index mode, resolved backend, TM zero policy, bin thresholds, pair
reranking top-k, selected metrics, and SacreBLEU settings.

## Performance Modes

Nearest-neighbor search over a training corpus is the expensive part of
TAME-MT. BLEU only compares hypotheses to references; TAME-MT also needs to
ask, for every test source, "what training source is most similar?"

TAME-MT has native Rust retrieval backends plus pure-Python fallbacks:

| Mode | Behavior | Use |
| --- | --- | --- |
| `native_exact` | Rust exact character n-gram Jaccard retrieval over shared-gram candidates. | Small and medium corpora when exact nearest-neighbor exposure is required. |
| `native_fast` | Rust rare-gram candidate generation plus exact Jaccard reranking of a bounded shortlist. | Large corpora and interactive audits. |
| `python_exact` | Pure-Python exact fallback. | Debugging and source installs without the native extension. |
| `python_fast` | Pure-Python bounded fast fallback. | Large-corpus fallback when native wheels are unavailable. |
| `auto` | Uses native exact/fast when the extension is installed; otherwise uses Python exact/fast. | Default. |

Fast mode is approximate for nearest-neighbor retrieval. Exact normalized source,
target, and pair overlap checks remain exact, and shortlisted candidates are
reranked with the exact Jaccard formula. For paper-critical numbers, report the
signature and consider rerunning exact mode on smaller filtered subsets or
high-risk bins.

Check the installed backend with:

```bash
tame-mt doctor
```

On the local development machine, the OPUS-100 `de-en` public-corpus audit at
50,000 training pairs and 2,000 test pairs completed in about 6 seconds with
`native_fast`; `score-cached` on the saved segment diagnostics took under 2
seconds for an additional hypothesis. On the 100,000 train / 2,000 test
OPUS-100 `de-en` slice, a fresh `native_fast` audit took about 10.0 seconds, the
one-time `.tameidx` build took about 9.6 seconds, and a later audit from the
saved index took about 2.3 seconds with identical exposure outputs.

On a synthetic 100,000 train / 2,000 test benchmark on the same machine, a fresh
`native_fast` audit takes about 4.2 seconds, an indexed audit takes about 1.0
second, and cached scoring for another hypothesis takes about 0.6 seconds. The
cached stage is the closest analogue to ordinary BLEU/chrF scoring because it no
longer touches the training corpus.

For production evaluation, use a staged workflow:

1. Run `tame-mt index build --out train.tameidx` once for a fixed training
   corpus.
2. Run `tame-mt audit --index train.tameidx --segment-out segments.jsonl` once
   for a fixed train/test/reference setup.
3. Run `tame-mt score-cached` for every system output.

The index stage avoids rebuilding source/target postings. The audit stage is
train-aware and does nearest-neighbor retrieval. The cached-score stage reuses
exposure and TM hypotheses, so adding another system output does not require
another pass over the training corpus.

## Privacy

TAME-MT runs locally. It does not download models, call remote services, or send
text anywhere.

Training data can still be sensitive. By default, TAME-MT does not print
nearest-neighbor training text. Segment reports contain indices and scores
unless raw text fields are explicitly requested.

Index bundles created by `tame-mt index build` store raw training source/target
lines and normalized exact-match and pair keys so later runs can produce
identical TM outputs and optional neighbor-text diagnostics. Do not publish
`.tameidx` files unless the underlying training corpus can also be published.

## Limitations

TAME-MT does not prove that a neural model memorized a sentence. It does not
measure semantic adequacy. It does not make a high-exposure benchmark invalid.
It does not replace human evaluation.

The default similarity is surface-based. It can miss semantic paraphrases and
can overemphasize orthographic similarity. Pair exposure in v0.1 uses top-k
candidate reranking for speed; exact pair overlap is exact.

Fast retrieval mode is approximate. It is designed to make large-corpus audits
usable while retaining exact scoring within the selected candidate set.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
mypy src/tame_mt
cargo fmt --check
cargo clippy -- -D warnings
cargo test
python -m build
python -m twine check dist/*
```

## Citation

```bibtex
@software{tame_mt_2026,
  title = {TAME-MT: Training-Aware Machine Translation Evaluation for Machine Translation},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/hunterschep/TAME-MT}
}
```

## License

MIT.
