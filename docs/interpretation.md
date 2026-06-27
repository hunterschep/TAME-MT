# Interpretation Guide

## What TAME-MT Tells You

TAME-MT tells you how close the test corpus is to the training corpus, how well
a simple nearest-neighbor translation memory can score, how much the evaluated
system improves over that memory, and how quality changes as examples move
farther from training data.

## What TAME-MT Does Not Tell You

TAME-MT does not prove that a model memorized a sentence. It does not prove a
system is bad. It does not decide that a test set is invalid. It does not
measure semantic adequacy or guarantee real-world utility.

## Common Patterns

High BLEU, high TM-BLEU, high PairLeakTopK, and low Far-BLEU indicate that raw
corpus metrics may partly reflect train-test near-duplication. Describe the
result as in-domain or high-exposure unless additional far-bin evidence supports
a broader claim.

Moderate BLEU, low TM-BLEU, high delta over TM, good Far-BLEU, and low PairLeakTopK
suggest the system may generalize better than raw BLEU alone implies.

Low BLEU, low TM-BLEU, and low Far-BLEU indicate that the test set is not easily
solved by nearest-neighbor reuse and the system also performs poorly.

High BLEU, low TM-BLEU, high Far-BLEU, and low PairLeakTopK are stronger evidence
of generalization under the tested distribution.

High exposure, high in-domain quality, and no far-bin coverage support a valid
narrow-domain result, but weak evidence for broad MT generalization.
