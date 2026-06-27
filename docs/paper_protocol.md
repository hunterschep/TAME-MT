# Paper Reporting Protocol

Use TAME-MT as a companion to standard MT quality metrics, not as a replacement.

## Recommended Methods Text

```text
We report SacreBLEU and chrF together with TAME-MT train-aware diagnostics.
TAME-MT computes source-side train-test exposure using character 3-5-gram
Jaccard similarity over normalized text. TM-BLEU is the BLEU score of a
nearest-neighbor translation-memory baseline that retrieves the closest
training source sentence for each test source and outputs its paired target
sentence. delta over TM is system BLEU minus TM-BLEU. We also report
PairLeakTopK@0.85 and BLEU stratified by source exposure.
```

Report the TAME-MT signature alongside results so normalization, similarity,
requested index mode, resolved backend, bin thresholds, TM zero policy, pair
reranking top-k, and SacreBLEU settings are recoverable.

The recommended paper-facing mode is exact retrieval. When using
`--retrieval approx --allow-approximate`, describe the retrieval mode as
approximate nearest-neighbor retrieval with exact Jaccard reranking of a bounded
rare-gram candidate shortlist, report `tm_retrieval_exact=false`, and label pair
thresholds as `PairLeakTopK`, not exact `PairLeak`.

## Recommended Table Columns

```text
System
BLEU
chrF
TM-BLEU
delta TM-BLEU
MeanSourceExposure
PairLeakTopK@0.85
TM retrieval exact?
Retrieval signature
Far-BLEU
GenGap-BLEU
```

## Claim Discipline

If PairLeakTopK or TM-BLEU is high, describe results as high-exposure or
in-domain. If far-bin coverage is small, do not claim broad out-of-domain
generalization from this test set. If a model performs well in the far bin and
has high delta over TM, that is stronger evidence of generalization.

Avoid claiming that TAME-MT replaces BLEU, proves memorization, detects all
contamination, or fully solves low-resource MT evaluation.
