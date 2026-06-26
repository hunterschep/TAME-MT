# TAME-MT

Training-Aware Machine Translation Evaluation for Machine Translation.

TAME-MT is a lightweight companion report for BLEU and chrF. It asks how much
of a system's measured MT quality is earned on test examples that are close to
the training corpus.

## Why TAME-MT?

BLEU and chrF score system outputs against references, but they do not inspect
the training data. In low-resource MT, train and test corpora often come from a
small number of domains or sources. A high raw score can therefore reflect
valid narrow-domain performance, train-test near-duplication, or broad
generalization. TAME-MT makes that distinction visible.

The core diagnostic is simple: for each test source sentence, find the most
similar training source sentence and reuse its paired training translation.
The BLEU/chrF of that nearest-neighbor translation memory is reported as the
TM baseline.

## Installation

```bash
pip install tame-mt
```

For local development:

```bash
pip install -e '.[dev]'
```

## Quick Start

```bash
tame-mt score \
  --train-src examples/toy/train.src \
  --train-tgt examples/toy/train.tgt \
  --test-src examples/toy/test.src \
  --ref examples/toy/test.ref \
  --hyp examples/toy/hyp.out
```

Write machine-readable outputs:

```bash
tame-mt score \
  --train-src examples/toy/train.src \
  --train-tgt examples/toy/train.tgt \
  --test-src examples/toy/test.src \
  --ref examples/toy/test.ref \
  --hyp examples/toy/hyp.out \
  --json-out report.json \
  --segment-out segments.jsonl \
  --tm-out tm.out
```

## Main Numbers

`BLEU` and `chrF` are standard corpus scores for the system output.

`TM-BLEU` and `TM-chrF` score a nearest-neighbor translation-memory baseline
built only from the training corpus. Retrieval uses the source side only.

`delta over TM` is the system score minus the TM baseline score. A small delta
means the system is not gaining much over direct training-set reuse under this
test distribution.

`SourceExposure` is the maximum character n-gram Jaccard similarity between a
test source segment and the training source corpus.

`PairLeak@0.85` is the fraction of test source/reference pairs whose source
and target sides are both close to one training pair under the default
top-k reranking approximation.

`Far-BLEU` is BLEU on examples with low source exposure. `GenGap-BLEU` is
near-bin BLEU minus far-bin BLEU.

## Input Format

The canonical scoring mode uses five aligned UTF-8 plain-text files:

```text
train.src
train.tgt
test.src
test.ref
system.out
```

Each line is one segment. TAME-MT validates:

```text
len(train.src) == len(train.tgt)
len(test.src) == len(test.ref)
len(test.src) == len(system.out)
```

Multiple references can be passed by repeating `--ref`.

## CLI

Full scoring:

```bash
tame-mt score --train-src train.src --train-tgt train.tgt --test-src test.src --ref test.ref --hyp system.out
```

Audit a benchmark without system outputs:

```bash
tame-mt audit --train-src train.src --train-tgt train.tgt --test-src test.src --ref test.ref
```

Write nearest-neighbor TM hypotheses:

```bash
tame-mt tm-baseline --train-src train.src --train-tgt train.tgt --test-src test.src --out tm.out
```

Common options:

```bash
--metrics bleu chrf
--ngram-orders 3,4,5
--far-threshold 0.30
--near-threshold 0.70
--leak-thresholds 0.70,0.85,0.95
--pair-k 50
--lowercase
--strip-diacritics
--normalize-punctuation
```

Segment reports do not include raw text by default. Use
`--include-neighbor-text`, `--include-source-text`, `--include-reference-text`,
or `--include-hyp-text` only when that data is safe to write.

## Python API

```python
from tame_mt import TameScorer, ScoreConfig

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
```

In-memory corpora are supported:

```python
report = scorer.score_corpus(
    train_src=train_src_lines,
    train_tgt=train_tgt_lines,
    test_src=test_src_lines,
    refs=[ref_lines],
    hyp=hyp_lines,
)
```

## Interpretation Examples

High BLEU, high TM-BLEU, high PairLeak, and low Far-BLEU means the raw score may
overstate broad generalization. It can still be a valid in-domain result.

Moderate BLEU, low TM-BLEU, high delta over TM, good Far-BLEU, and low PairLeak
is stronger evidence that the system is doing more than nearest-neighbor reuse.

High exposure with no far-bin data supports a narrow-domain claim, not a broad
out-of-domain generalization claim.

## Limitations

TAME-MT does not replace BLEU, chrF, or human evaluation. It does not prove that
a neural model memorized a sentence. It does not measure semantic adequacy. The
default similarity is surface-based character 3-5-gram Jaccard, and v0.1 pair
exposure uses top-k candidate reranking rather than exhaustive all-pairs search.

The report is intended to contextualize raw MT scores with training-data
proximity, especially for low-resource and narrow-domain evaluation.

## Citation

```bibtex
@software{tame_mt_2026,
  title = {TAME-MT: Training-Aware Machine Translation Evaluation for Machine Translation},
  year = {2026},
  version = {0.1.0}
}
```

## License

MIT.
