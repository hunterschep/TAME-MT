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
PairLeak@0.85 and BLEU stratified by source exposure.
```

## Recommended Table Columns

```text
System
BLEU
chrF
TM-BLEU
delta TM-BLEU
MeanSourceExposure
PairLeak@0.85
Far-BLEU
GenGap-BLEU
```

## Claim Discipline

If PairLeak or TM-BLEU is high, describe results as high-exposure or in-domain.
If far-bin coverage is small, do not claim broad out-of-domain generalization
from this test set. If a model performs well in the far bin and has high delta
over TM, that is stronger evidence of generalization.

Avoid claiming that TAME-MT replaces BLEU, proves memorization, detects all
contamination, or fully solves low-resource MT evaluation.
