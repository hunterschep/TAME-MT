from __future__ import annotations


def char_ngrams(text: str, orders: tuple[int, ...] = (3, 4, 5)) -> frozenset[str]:
    if not text:
        return frozenset()
    min_order = min(orders)
    if len(text) < min_order:
        return frozenset({text})

    grams: set[str] = set()
    for order in orders:
        if order <= len(text):
            grams.update(text[i : i + order] for i in range(len(text) - order + 1))
    return frozenset(grams)
