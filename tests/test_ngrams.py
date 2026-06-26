from tame_mt.ngrams import char_ngrams


def test_char_ngrams_default_rules() -> None:
    assert char_ngrams("abcd", orders=(3,)) == frozenset({"abc", "bcd"})
    assert char_ngrams("ab", orders=(3, 4, 5)) == frozenset({"ab"})
    assert char_ngrams("", orders=(3, 4, 5)) == frozenset()
