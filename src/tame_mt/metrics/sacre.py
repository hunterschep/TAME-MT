from __future__ import annotations

import sacrebleu

from tame_mt.config import MetricConfig


def score_bleu(hyps: list[str], refs: list[list[str]], config: MetricConfig) -> float:
    score = sacrebleu.corpus_bleu(
        hyps,
        refs,
        tokenize=config.bleu_tokenize,
        lowercase=config.bleu_lowercase,
    )
    return float(score.score)


def score_chrf(hyps: list[str], refs: list[list[str]], config: MetricConfig) -> float:
    score = sacrebleu.corpus_chrf(
        hyps,
        refs,
        word_order=config.chrf_word_order,
    )
    return float(score.score)
