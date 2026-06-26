import pytest

from tame_mt.config import IndexConfig
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
