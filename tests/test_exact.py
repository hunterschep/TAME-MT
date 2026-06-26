from tame_mt.exact import exact_pair_key


def test_exact_pair_key_is_unambiguous() -> None:
    assert exact_pair_key("ab", "c") != exact_pair_key("a", "bc")
