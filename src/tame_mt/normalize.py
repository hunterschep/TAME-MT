from __future__ import annotations

import re
import unicodedata

from tame_mt.config import NormalizationConfig

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u00a0": " ",
    }
)


def normalize_text(text: str, config: NormalizationConfig | None = None) -> str:
    config = config or NormalizationConfig()
    normalized = text if text.isascii() else unicodedata.normalize(config.unicode_form, text)
    if config.normalize_punctuation:
        normalized = normalized.translate(_PUNCT_TRANSLATION)
    if config.strip_diacritics:
        normalized = _strip_diacritics(normalized)
    if config.lowercase:
        normalized = normalized.casefold()
    if config.strip and config.collapse_whitespace:
        normalized = " ".join(normalized.split())
    else:
        if config.strip:
            normalized = normalized.strip()
        if config.collapse_whitespace:
            normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)
