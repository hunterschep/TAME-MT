from __future__ import annotations

from tame_mt.config import ScoreConfig
from tame_mt.schema import BinReport, ExposureSummary


def generate_warnings(
    exposure: ExposureSummary,
    system_scores: dict[str, float | None],
    tm_scores: dict[str, float | None],
    bin_reports: list[BinReport],
    generalization_gap: dict[str, float | None],
    config: ScoreConfig,
    num_test: int,
) -> list[str]:
    warnings: list[str] = []
    pair = exposure.pair or {}
    exact_pair = pair.get("exact_overlap")
    if isinstance(exact_pair, float) and exact_pair > 0:
        warnings.append(
            f"{_pct(exact_pair)} of test source/reference pairs exactly appear in training. "
            "Report exact-pair overlap separately from general MT quality."
        )

    pair_thresholds = pair.get("at_threshold") if isinstance(pair, dict) else None
    if isinstance(pair_thresholds, dict):
        pair_leak = pair_thresholds.get("0.85")
        if isinstance(pair_leak, float) and pair_leak >= 0.05:
            warnings.append(
                f"{_pct(pair_leak)} of test pairs have PairExposure >= 0.85. "
                "Raw corpus metrics may partly reflect train-test near-duplication."
            )

    system_bleu = system_scores.get("bleu")
    tm_bleu = tm_scores.get("bleu")
    if (
        system_bleu is not None
        and tm_bleu is not None
        and system_bleu > 0
        and tm_bleu >= 0.5 * system_bleu
    ):
        warnings.append(
            f"TM-BLEU is {(tm_bleu / system_bleu) * 100:.1f}% of system BLEU. "
            "A nearest-neighbor translation memory explains a substantial fraction of raw BLEU."
        )

    far = next((item for item in bin_reports if item.name == "far"), None)
    if far is not None:
        if far.count < config.bins.min_bin_size_warning:
            warnings.append(
                f"Far bin contains only {far.count} segments. Far-bin scores may be unstable."
            )
        if num_test and far.count / num_test < 0.10:
            warnings.append(
                "Less than 10% of the test set is source-far under the default threshold. "
                "This test set provides limited evidence of "
                "out-of-training-distribution generalization."
            )

    has_system_scores = any(value is not None for value in system_scores.values())
    if has_system_scores and any(value is None for value in generalization_gap.values()):
        warnings.append("GenGap cannot be computed because the near or far bin is empty.")

    return warnings


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"
