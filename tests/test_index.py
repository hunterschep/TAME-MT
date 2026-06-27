import pytest

from tame_mt.config import IndexConfig, NormalizationConfig, SimilarityConfig
from tame_mt.exceptions import BackendError
from tame_mt.index import NgramInvertedIndex
from tame_mt.similarity import text_similarity


def test_index_retrieves_best_neighbor() -> None:
    index = NgramInvertedIndex.build(["abcdef", "uvwxyz", "abcxyz"])
    result = index.query_best("abcdeg")
    assert result.index == 0
    assert result.score > 0


def test_index_tie_breaks_to_lowest_index() -> None:
    index = NgramInvertedIndex.build(["abcdef", "abcdef"])
    result = index.query_best("abcdef")
    assert result.index == 0
    assert result.exact is True


def test_index_no_candidate_returns_none() -> None:
    index = NgramInvertedIndex.build(["abcdef"])
    result = index.query_best("uvwxyz")
    assert result.index is None
    assert result.score == 0.0


def test_index_auto_resolves_to_exact_for_small_corpora() -> None:
    index = NgramInvertedIndex.build(["abcdef"], index_config=IndexConfig(mode="auto"))
    assert index.resolved_mode == "native_exact"
    assert index.backend_info.exact is True


def test_index_auto_remains_exact_for_larger_corpora() -> None:
    lines = [f"sentence {idx:05d}" for idx in range(6)]
    index = NgramInvertedIndex.build(
        lines,
        index_config=IndexConfig(mode="auto", auto_exact_cutoff=5),
    )
    assert index.resolved_mode == "native_exact"
    assert index.backend_info.exact is True


def test_index_auto_requires_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tame_mt.index.is_native_available", lambda: False)

    with pytest.raises(BackendError, match="native Rust backend is required"):
        NgramInvertedIndex.build(["abcdef"], index_config=IndexConfig(mode="auto"))


def test_fast_index_preserves_exact_match_shortcut() -> None:
    index = NgramInvertedIndex.build(
        ["abcdef", "uvwxyz"],
        index_config=IndexConfig(
            mode="native_fast",
            posting_limit=1,
            max_candidates=1,
            rerank_limit=1,
        ),
    )
    result = index.query_best("uvwxyz")
    assert result.index == 1
    assert result.score == 1.0
    assert result.exact is True


def test_fast_index_matches_exact_when_limits_cover_candidate_space() -> None:
    lines = [
        f"domain {idx % 5} source sentence {idx:03d} topic {idx % 7} shared tail"
        for idx in range(40)
    ]
    queries = [
        lines[0],
        "domain 2 source sentence 017 topic 3 shared tile",
        "domain 4 heldout source sentence topic shared tail",
    ]
    exact = NgramInvertedIndex.build(lines, index_config=IndexConfig(mode="native_exact"))
    fast = NgramInvertedIndex.build(
        lines,
        index_config=IndexConfig(
            mode="native_fast",
            candidate_gram_limit=1_000,
            posting_limit=1_000,
            max_candidates=1_000,
            rerank_limit=1_000,
        ),
    )

    for query in queries:
        assert fast.query_topk(query, 5) == exact.query_topk(query, 5)


def test_native_exact_matches_metric_definition() -> None:
    lines = ["abcdef", "uvwxyz", "abcxyz", "नमस्ते दुनिया"]
    query = "abcdeg"
    native_index = NgramInvertedIndex.build(lines, index_config=IndexConfig(mode="native_exact"))
    expected_scores = {
        index: text_similarity(query, candidate) for index, candidate in enumerate(lines)
    }
    expected_top3 = [
        (index, score)
        for index, score in sorted(expected_scores.items(), key=lambda item: (-item[1], item[0]))
        if score > 0
    ][:3]

    assert native_index.backend_info.native is True
    assert [(item.index, item.score) for item in native_index.query_topk(query, 3)] == expected_top3
    assert native_index.batch_query_topk([query, "नमस्ते दुनिया"], 2)[0] == native_index.query_topk(
        query, 2
    )
    assert native_index.score_candidates(query, [0, 1, 2, 3]) == expected_scores


def test_native_exact_orders_seeded_unicode_corpus_by_metric_definition() -> None:
    lines = [
        "alpha beta gamma",
        "alpha beta delta",
        "mañana será otro día",
        "mañana sera otro dia",
        "測試 句子 甲",
        "測試 句子 乙",
        "नमस्ते दुनिया",
        "नमस्ते संसार",
    ]
    queries = [
        "alpha beta gamma",
        "alpha beta epsilon",
        "mañana será un día distinto",
        "測試 句子 丙",
        "नमस्ते दुनिया",
    ]
    native_index = NgramInvertedIndex.build(lines, index_config=IndexConfig(mode="native_exact"))

    for query in queries:
        expected = [
            (index, score)
            for index, score in sorted(
                (
                    (index, text_similarity(query, candidate))
                    for index, candidate in enumerate(lines)
                ),
                key=lambda item: (-item[1], item[0]),
            )
            if score > 0
        ][:4]
        if expected[0][1] == 1.0:
            expected = [expected[0]]
        assert [(item.index, item.score) for item in native_index.query_topk(query, 4)] == expected


def test_native_index_can_release_python_normalized_lines() -> None:
    index = NgramInvertedIndex.build(
        ["abcdef", "uvwxyz", "abcxyz"],
        index_config=IndexConfig(mode="native_exact"),
    )
    expected = index.query_topk("abcdeg", 3)

    assert index.normalized_lines
    assert index.release_python_normalized_lines() is True
    assert index.normalized_lines == []
    assert index.release_python_normalized_lines() is False
    assert index.query_topk("abcdeg", 3) == expected
    assert index.contains_exact_normalized("abcdef") is True
    assert index.score_candidate("abcdef", 0) == 1.0


def test_score_candidates_matches_single_candidate_scoring() -> None:
    lines = ["abcdef", "uvwxyz", "abcxyz"]
    index = NgramInvertedIndex.build(lines, index_config=IndexConfig(mode="native_exact"))
    indices = [0, 1, 2]
    bulk_scores = index.score_candidates("abcdeg", indices)
    single_scores = {idx: index.score_candidate("abcdeg", idx) for idx in indices}
    assert bulk_scores == single_scores


def test_native_index_bytes_roundtrip_when_available() -> None:
    native_module = pytest.importorskip("tame_mt._native")
    lines = ["abcdef", "uvwxyz", "abcxyz", "नमस्ते दुनिया"]
    norm_config = NormalizationConfig()
    sim_config = SimilarityConfig()
    index_config = IndexConfig(mode="native_exact")
    index = NgramInvertedIndex.build(
        lines,
        norm_config=norm_config,
        sim_config=sim_config,
        index_config=index_config,
    )

    native_bytes = index.native_bytes()
    restored_native = native_module.NativeNgramIndex.from_bytes(native_bytes)
    restored = NgramInvertedIndex.from_native(
        lines=lines,
        native_index=restored_native,
        norm_config=norm_config,
        sim_config=sim_config,
        index_config=index_config,
        resolved_mode="native_exact",
    )

    assert restored.query_topk("abcdeg", 3) == index.query_topk("abcdeg", 3)
    assert restored.batch_query_topk(["abcdeg", "नमस्ते दुनिया"], 2) == index.batch_query_topk(
        ["abcdeg", "नमस्ते दुनिया"], 2
    )


def test_native_contains_exact_without_python_normalized_lines_when_available() -> None:
    native_module = pytest.importorskip("tame_mt._native")
    norm_config = NormalizationConfig()
    sim_config = SimilarityConfig()
    index_config = IndexConfig(mode="native_exact")
    native_index = native_module.NativeNgramIndex(
        ["hello world"],
        list(sim_config.ngram_orders),
        "exact",
        index_config.candidate_gram_limit,
        index_config.posting_limit,
        index_config.max_candidates,
        index_config.rerank_limit,
    )
    index = NgramInvertedIndex.from_native(
        native_index=native_index,
        norm_config=norm_config,
        sim_config=sim_config,
        index_config=index_config,
        resolved_mode="native_exact",
        lines=["hello world"],
        normalized_lines=[],
    )

    assert index.normalized_lines == []
    assert index.contains_exact_normalized("hello world") is True
    assert index.contains_exact_normalized("goodbye") is False
    assert index.score_candidate("hello world", 0) == 1.0


def test_native_best_pair_candidate_matches_separate_candidate_scores() -> None:
    pytest.importorskip("tame_mt._native")
    index_config = IndexConfig(mode="native_exact")
    source_index = NgramInvertedIndex.build(
        ["abcdef", "abcxyz", "uvwxyz"],
        index_config=index_config,
    )
    target_index = NgramInvertedIndex.build(
        ["klmnop", "klmxyz", "qrstuv"],
        index_config=index_config,
    )
    candidates = [2, 1, 0]
    source_text = "abcdeg"
    refs = ["klmxyy"]

    source_scores = source_index.score_candidates(source_text, candidates)
    target_scores = target_index.score_candidates(refs[0], candidates)
    expected_index, expected_score = max(
        ((index, min(source_scores[index], target_scores[index])) for index in sorted(candidates)),
        key=lambda item: (item[1], -item[0]),
    )

    result = source_index.best_pair_candidate(target_index, source_text, refs, candidates)

    assert result is not None
    assert result.index == expected_index
    assert result.score == expected_score


def test_native_batch_best_pair_candidates_matches_single_calls() -> None:
    pytest.importorskip("tame_mt._native")
    index_config = IndexConfig(mode="native_exact")
    source_index = NgramInvertedIndex.build(
        ["abcdef", "abcxyz", "uvwxyz"],
        index_config=index_config,
    )
    target_index = NgramInvertedIndex.build(
        ["klmnop", "klmxyz", "qrstuv"],
        index_config=index_config,
    )
    source_texts = ["abcdeg", "uvwxyy"]
    refs_by_segment = [["klmxyy"], ["qrstuv"]]
    candidate_lists = [[2, 1, 0], [0, 2]]

    batch_results = source_index.batch_best_pair_candidates(
        target_index,
        source_texts,
        refs_by_segment,
        candidate_lists,
    )
    single_results = [
        source_index.best_pair_candidate(target_index, source, refs, candidates)
        for source, refs, candidates in zip(
            source_texts,
            refs_by_segment,
            candidate_lists,
            strict=True,
        )
    ]

    assert batch_results == single_results
