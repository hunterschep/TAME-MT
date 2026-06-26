from pathlib import Path

from tame_mt import TameScorer

FIXTURES = Path(__file__).parent / "fixtures"


def test_score_files_produces_report() -> None:
    report = TameScorer().score_files(
        train_src=str(FIXTURES / "train.src"),
        train_tgt=str(FIXTURES / "train.tgt"),
        test_src=str(FIXTURES / "test.src"),
        refs=[str(FIXTURES / "test.ref")],
        hyp=str(FIXTURES / "hyp.out"),
    )
    assert report.num_train == 4
    assert report.num_test == 4
    assert report.exposure.source["max"] == 1.0
    assert report.tm_scores["bleu"] is not None
    assert report.signature.startswith("tame-mt|v:0.1.0|")


def test_audit_without_system_scores_does_not_warn_about_gengap() -> None:
    report = TameScorer().audit_files(
        train_src=str(FIXTURES / "train.src"),
        train_tgt=str(FIXTURES / "train.tgt"),
        test_src=str(FIXTURES / "test.src"),
        refs=[str(FIXTURES / "test.ref")],
    )
    assert not any("GenGap cannot be computed" in warning for warning in report.warnings)
