# Scientific Validation Plan

This plan describes experiments needed for paper-level validation of TAME-MT.
It is not a claim that those experiments have all been completed.

## Synthetic Contamination

Construct training/test splits with controlled exact and near-duplicate
contamination rates: 0%, 1%, 5%, 10%, and 25%.

Expected evidence:

- raw BLEU/chrF can rise as contamination increases;
- TM-BLEU rises when nearest-neighbor reuse explains more of the benchmark;
- exact-pair overlap and PairLeakTopK rise with injected overlap;
- far-bin scores are less sensitive to injected near duplicates.

## Domain Narrowness

Compare narrow train/test domains, such as scripture-like or formulaic text,
with mixed-domain train/test setups.

Expected evidence:

- high raw BLEU on narrow data should be interpreted alongside high exposure;
- delta over TM and far-bin scores should separate broad generalization from
  nearest-neighbor reuse.

## Public Test-Set Audits

Run OPUS-100-style and FLORES-style audits where licensing permits. Record
commands, corpus sizes, signatures, timings, warnings, and exactness modes.

Approximate runs must include `approx_validation` or be labeled exploratory.

## Approximation Validation

For small corpora, compare exact and approximate retrieval over the full test
set. For larger corpora, sample with `--validate-approx-sample` and report:

- source top-1 agreement;
- source-bin agreement;
- source-score error;
- target top-1 agreement;
- pair-threshold agreement;
- TM-BLEU delta on the validation sample.

## Human Sanity Check

Sample examples from far, medium, near, source-exact, and high-pair-exposure
bins. Ask annotators to judge adequacy and whether the nearest training example
looks like a plausible explanation for the score.

The purpose is not to prove memorization. The purpose is to verify that the
exposure bins are interpretable and useful for cautious MT evaluation.
