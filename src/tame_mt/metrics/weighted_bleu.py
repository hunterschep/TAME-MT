from __future__ import annotations

import math
from collections import Counter


def weighted_bleu(
    hyps: list[str],
    refs: list[str],
    weights: list[float],
    max_order: int = 4,
    smooth: float = 1e-9,
) -> float:
    """Compute an experimental single-reference weighted BLEU score.

    This helper is intentionally not exposed in the v0.1 CLI. TAME-MT's main
    report uses standard SacreBLEU corpus scores, TM baselines, exposure
    statistics, and distance-stratified scores.
    """
    if len(hyps) != len(refs) or len(hyps) != len(weights):
        raise ValueError("hyps, refs, and weights must have the same length")
    if not hyps:
        return 0.0

    weighted_matches = [0.0 for _ in range(max_order)]
    weighted_totals = [0.0 for _ in range(max_order)]
    weighted_hyp_len = 0.0
    weighted_ref_len = 0.0

    for hyp, ref, weight in zip(hyps, refs, weights):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()
        weighted_hyp_len += weight * len(hyp_tokens)
        weighted_ref_len += weight * len(ref_tokens)
        for order in range(1, max_order + 1):
            hyp_counts = _ngram_counts(hyp_tokens, order)
            ref_counts = _ngram_counts(ref_tokens, order)
            clipped = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
            weighted_matches[order - 1] += weight * clipped
            weighted_totals[order - 1] += weight * sum(hyp_counts.values())

    if weighted_hyp_len <= 0:
        return 0.0
    precisions = [
        (weighted_matches[idx] + smooth) / (weighted_totals[idx] + smooth)
        for idx in range(max_order)
    ]
    bp = min(1.0, math.exp(1.0 - (weighted_ref_len / weighted_hyp_len)))
    return 100.0 * bp * math.exp(sum(math.log(value) for value in precisions) / max_order)


def _ngram_counts(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))
