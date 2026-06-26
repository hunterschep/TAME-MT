from tame_mt.config import NormalizationConfig
from tame_mt.normalize import normalize_text


def test_normalize_whitespace_and_nfkc() -> None:
    assert normalize_text("  hello   world  ") == "hello world"
    assert normalize_text("ＡＢＣ") == "ABC"


def test_case_preservation_and_lowercase_option() -> None:
    assert normalize_text("Hello") == "Hello"
    assert normalize_text("Hello", NormalizationConfig(lowercase=True)) == "hello"
