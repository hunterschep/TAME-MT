from tame_mt.exact import (
    EXACT_PAIR_KEY_BYTES,
    build_exact_pair_keys,
    contains_exact_pair_key,
    exact_pair_key,
)


def test_exact_pair_key_is_unambiguous() -> None:
    assert exact_pair_key("ab", "c") != exact_pair_key("a", "bc")
    assert len(exact_pair_key("ab", "c")) == EXACT_PAIR_KEY_BYTES


def test_exact_pair_keys_are_packed_and_searchable() -> None:
    keys = build_exact_pair_keys(["b", "a"], ["y", "x"])
    assert isinstance(keys, bytes)
    assert len(keys) == 2 * EXACT_PAIR_KEY_BYTES
    assert contains_exact_pair_key(keys, exact_pair_key("a", "x"))
    assert contains_exact_pair_key(keys, exact_pair_key("b", "y"))
    assert not contains_exact_pair_key(keys, exact_pair_key("a", "y"))
