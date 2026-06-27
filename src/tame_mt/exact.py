from __future__ import annotations

from collections.abc import Iterable
from hashlib import blake2b

EXACT_PAIR_KEY_BYTES = 16
ExactPairKeys = bytes


def exact_pair_key(normalized_source: str, normalized_target: str) -> bytes:
    """Return a deterministic fixed-size fingerprint for a normalized source/target pair."""

    digest = blake2b(digest_size=EXACT_PAIR_KEY_BYTES)
    source_bytes = normalized_source.encode("utf-8")
    target_bytes = normalized_target.encode("utf-8")
    digest.update(len(source_bytes).to_bytes(8, byteorder="little", signed=False))
    digest.update(source_bytes)
    digest.update(target_bytes)
    return digest.digest()


def build_exact_pair_keys(
    normalized_sources: Iterable[str],
    normalized_targets: Iterable[str],
) -> ExactPairKeys:
    keys = [
        exact_pair_key(source, target)
        for source, target in zip(normalized_sources, normalized_targets, strict=True)
    ]
    keys.sort()
    return b"".join(keys)


def contains_exact_pair_key(keys: ExactPairKeys, key: bytes) -> bool:
    if len(key) != EXACT_PAIR_KEY_BYTES or len(keys) % EXACT_PAIR_KEY_BYTES != 0:
        return False
    lo = 0
    hi = len(keys) // EXACT_PAIR_KEY_BYTES
    while lo < hi:
        mid = (lo + hi) // 2
        start = mid * EXACT_PAIR_KEY_BYTES
        current = keys[start : start + EXACT_PAIR_KEY_BYTES]
        if current < key:
            lo = mid + 1
        elif current > key:
            hi = mid
        else:
            return True
    return False
