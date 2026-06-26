from tame_mt.similarity import jaccard


def test_jaccard_edge_cases() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 1.0
    assert jaccard(set(), {"a"}) == 0.0
