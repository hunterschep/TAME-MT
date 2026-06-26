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
