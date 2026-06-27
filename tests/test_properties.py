from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tame_mt import IndexConfig, RetrievalConfig, ScoreConfig, TameScorer
from tame_mt.index import NgramInvertedIndex
from tame_mt.persistence import load_index_bundle, save_index_bundle
from tame_mt.similarity import jaccard, text_similarity

TEXT_ALPHABET = tuple(" abcdefghijklmnopqrstuvwxyz0123456789") + (
    "á",
    "é",
    "ñ",
    "ü",
    "測",
    "試",
    "句",
    "न",
    "म",
    "स",
    "ل",
    "س",
)
SMALL_TEXT = st.text(alphabet=TEXT_ALPHABET, min_size=0, max_size=32)
NONEMPTY_TEXT = st.text(alphabet=TEXT_ALPHABET, min_size=1, max_size=32)
SMALL_CORPUS = st.lists(SMALL_TEXT, min_size=1, max_size=8)
SMALL_QUERIES = st.lists(SMALL_TEXT, min_size=0, max_size=8)
THRESHOLDS = st.lists(
    st.sampled_from((0.0, 0.30, 0.70, 0.85, 0.95, 1.0)),
    min_size=1,
    max_size=6,
    unique=True,
)
PROPERTY_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.function_scoped_fixture),
)


@PROPERTY_SETTINGS
@given(
    st.sets(SMALL_TEXT, max_size=12),
    st.sets(SMALL_TEXT, max_size=12),
)
def test_jaccard_is_symmetric_and_bounded(
    left: set[str],
    right: set[str],
) -> None:
    left_score = jaccard(left, right)
    right_score = jaccard(right, left)

    assert left_score == right_score
    assert 0.0 <= left_score <= 1.0


def test_jaccard_empty_string_semantics_are_stable() -> None:
    assert text_similarity("", "") == 1.0
    assert text_similarity("", "nonempty") == 0.0
    assert text_similarity("nonempty", "") == 0.0


@PROPERTY_SETTINGS
@given(NONEMPTY_TEXT)
def test_exact_match_gives_exposure_one(text: str) -> None:
    index = NgramInvertedIndex.build(
        [f"{text} unrelated tail", text],
        index_config=IndexConfig(mode="native_exact"),
    )

    result = index.query_best(text)

    assert result.index == 1
    assert result.score == 1.0
    assert result.exact is True


@PROPERTY_SETTINGS
@given(SMALL_CORPUS, SMALL_QUERIES)
def test_batch_topk_equals_repeated_single_queries(
    train: list[str],
    queries: list[str],
) -> None:
    index = NgramInvertedIndex.build(train, index_config=IndexConfig(mode="native_exact"))

    assert index.batch_query_topk(queries, 3) == [index.query_topk(query, 3) for query in queries]


@PROPERTY_SETTINGS
@given(SMALL_CORPUS, SMALL_QUERIES)
def test_native_index_bytes_roundtrip_preserves_queries(
    train: list[str],
    queries: list[str],
) -> None:
    native_module = __import__("tame_mt._native", fromlist=["NativeNgramIndex"])
    index_config = IndexConfig(mode="native_exact")
    index = NgramInvertedIndex.build(train, index_config=index_config)
    restored_native = native_module.NativeNgramIndex.from_bytes(index.native_bytes())
    restored = NgramInvertedIndex.from_native(
        native_index=restored_native,
        norm_config=index.norm_config,
        sim_config=index.sim_config,
        index_config=index_config,
        resolved_mode="native_exact",
        lines=train,
    )

    assert restored.batch_query_topk(queries, 3) == index.batch_query_topk(queries, 3)


@PROPERTY_SETTINGS
@given(SMALL_CORPUS, SMALL_QUERIES, THRESHOLDS)
def test_exact_threshold_flags_have_no_false_negatives_against_exact_top1(
    train: list[str],
    queries: list[str],
    thresholds: list[float],
) -> None:
    index = NgramInvertedIndex.build(train, index_config=IndexConfig(mode="native_exact"))

    flags = index.batch_threshold_flags(queries, thresholds)
    exact_best = [index.query_best(query) for query in queries]

    for row, best in zip(flags, exact_best, strict=True):
        for threshold in thresholds:
            assert row[threshold] is (best.score >= threshold)


@PROPERTY_SETTINGS
@given(train_src=SMALL_CORPUS, queries=SMALL_QUERIES)
def test_saved_index_bundle_preserves_source_retrieval(
    train_src: list[str],
    queries: list[str],
    tmp_path: Path,
) -> None:
    train_tgt = [f"target {idx} {line}" for idx, line in enumerate(train_src)]
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    bundle_path = tmp_path / "train.tameidx"
    saved = save_index_bundle(bundle_path, train_src, train_tgt, config)
    loaded = load_index_bundle(bundle_path, config)

    assert loaded.source_index.batch_query_topk(queries, 3) == saved.source_index.batch_query_topk(
        queries, 3
    )


def test_approximate_report_cannot_label_itself_exact() -> None:
    config = ScoreConfig(
        index=IndexConfig(mode="native_fast"),
        retrieval=RetrievalConfig(mode="approx", allow_approximate=True),
    )
    report = (
        TameScorer(config)
        .evaluate_corpus(
            train_src=["shared source alpha", "distant source beta"],
            train_tgt=["shared target alpha", "distant target beta"],
            test_src=["shared source alfa"],
            refs=[["shared target alfa"]],
            hyp=None,
        )
        .report
    )
    payload = report.to_dict()

    assert payload["retrieval"]["mode"] == "approx"
    assert payload["retrieval"]["approximate"] is True
    assert payload["retrieval"]["source_exposure_mode"] == "approx"
    assert payload["retrieval"]["tm_retrieval_exact"] is False
    assert payload["backend"]["exact"] is False
    assert "|approx:1|" in payload["signature"]
