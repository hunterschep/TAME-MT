import pytest

from tame_mt.config import IndexConfig, NormalizationConfig, SimilarityConfig
from tame_mt.index import NgramInvertedIndex


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
    assert index.resolved_mode in {"python_exact", "native_exact"}
    assert index.backend_info.exact is True


def test_index_auto_resolves_to_fast_for_larger_corpora() -> None:
    lines = [f"sentence {idx:05d}" for idx in range(6)]
    index = NgramInvertedIndex.build(
        lines,
        index_config=IndexConfig(mode="auto", auto_exact_cutoff=5),
    )
    assert index.resolved_mode in {"python_fast", "native_fast"}
    assert index.backend_info.exact is False


def test_fast_index_preserves_exact_match_shortcut() -> None:
    index = NgramInvertedIndex.build(
        ["abcdef", "uvwxyz"],
        index_config=IndexConfig(
            mode="inverted_fast",
            posting_limit=1,
            max_candidates=1,
            rerank_limit=1,
        ),
    )
    result = index.query_best("uvwxyz")
    assert result.index == 1
    assert result.score == 1.0
    assert result.exact is True


def test_native_exact_matches_python_exact_when_available() -> None:
    pytest.importorskip("tame_mt._native")
    lines = ["abcdef", "uvwxyz", "abcxyz", "नमस्ते दुनिया"]
    query = "abcdeg"
    python_index = NgramInvertedIndex.build(lines, index_config=IndexConfig(mode="python_exact"))
    native_index = NgramInvertedIndex.build(lines, index_config=IndexConfig(mode="native_exact"))

    assert native_index.backend_info.native is True
    assert native_index.query_topk(query, 3) == python_index.query_topk(query, 3)
    assert native_index.batch_query_topk([query, "नमस्ते दुनिया"], 2) == python_index.batch_query_topk(
        [query, "नमस्ते दुनिया"], 2
    )
    assert native_index.score_candidates(query, [0, 1, 2, 3]) == python_index.score_candidates(
        query, [0, 1, 2, 3]
    )


def test_score_candidates_matches_single_candidate_scoring() -> None:
    lines = ["abcdef", "uvwxyz", "abcxyz"]
    index = NgramInvertedIndex.build(lines, index_config=IndexConfig(mode="python_exact"))
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


def test_native_contains_exact_without_python_exact_map_when_available() -> None:
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
