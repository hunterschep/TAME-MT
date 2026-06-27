from __future__ import annotations

import json
from typing import Any, NoReturn, TextIO


def strict_json_loads(payload: str) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_standard_constant,
    )


def strict_json_dumps(
    value: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def strict_json_dump(
    value: Any,
    handle: TextIO,
    *,
    ensure_ascii: bool = False,
    indent: int | None = None,
    sort_keys: bool = False,
) -> None:
    json.dump(
        value,
        handle,
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def _reject_non_standard_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
        payload[key] = value
    return payload
