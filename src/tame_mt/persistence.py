from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from tame_mt.config import ScoreConfig
from tame_mt.exceptions import BackendError, ConfigurationError, InputDataError
from tame_mt.index import NgramInvertedIndex
from tame_mt.io import ensure_parent_dir, validate_equal_lengths
from tame_mt.native import native_index_from_bytes
from tame_mt.report import config_to_dict
from tame_mt.version import __version__

INDEX_FORMAT = "tameidx"
FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"
TRAIN_SRC_NAME = "train.src"
TRAIN_TGT_NAME = "train.tgt"
SOURCE_INDEX_NAME = "source.index.bin"
TARGET_INDEX_NAME = "target.index.bin"


@dataclass(frozen=True)
class IndexBundle:
    """A loaded persistent training index and the raw training corpus it indexes."""

    train_src: list[str]
    train_tgt: list[str] | None
    source_index: NgramInvertedIndex
    target_index: NgramInvertedIndex | None
    manifest: dict[str, Any]


def save_index_bundle(
    path: str | Path,
    train_src: list[str],
    train_tgt: list[str] | None,
    config: ScoreConfig,
) -> IndexBundle:
    """Build and save a reusable native training index bundle."""

    if not train_src:
        raise InputDataError("train.src must contain at least one segment")
    if train_tgt is not None:
        validate_equal_lengths("train.src", train_src, "train.tgt", train_tgt)

    source_index = NgramInvertedIndex.build(
        train_src,
        norm_config=config.normalization,
        sim_config=config.similarity,
        index_config=config.index,
    )
    _require_native_index(source_index, "source")

    target_index = None
    if train_tgt is not None:
        target_index = NgramInvertedIndex.build(
            train_tgt,
            norm_config=config.normalization,
            sim_config=config.similarity,
            index_config=config.index,
        )
        _require_native_index(target_index, "target")

    source_index_bytes = source_index.native_bytes()
    target_index_bytes = target_index.native_bytes() if target_index is not None else None
    manifest = _build_manifest(
        train_src,
        train_tgt,
        source_index,
        target_index,
        source_index_bytes,
        target_index_bytes,
        config,
    )
    output_path = Path(path)
    ensure_parent_dir(output_path)
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        archive.writestr(TRAIN_SRC_NAME, _encode_lines(train_src))
        if train_tgt is not None:
            archive.writestr(TRAIN_TGT_NAME, _encode_lines(train_tgt))
        archive.writestr(SOURCE_INDEX_NAME, source_index_bytes)
        if target_index_bytes is not None:
            archive.writestr(TARGET_INDEX_NAME, target_index_bytes)

    return IndexBundle(
        train_src=train_src,
        train_tgt=train_tgt,
        source_index=source_index,
        target_index=target_index,
        manifest=manifest,
    )


def load_index_bundle(path: str | Path, config: ScoreConfig) -> IndexBundle:
    """Load and validate a native training index bundle for the supplied config."""

    try:
        with zipfile.ZipFile(Path(path), mode="r") as archive:
            manifest = _read_manifest(archive)
            _validate_manifest(manifest, config)
            train_src = _read_lines_member(archive, TRAIN_SRC_NAME)
            train_tgt = (
                _read_lines_member(archive, TRAIN_TGT_NAME)
                if bool(manifest.get("has_target"))
                else None
            )
            if int(manifest.get("num_train", -1)) != len(train_src):
                raise ConfigurationError(
                    "index bundle manifest does not match stored train.src line count"
                )
            if train_tgt is not None:
                validate_equal_lengths("train.src", train_src, "train.tgt", train_tgt)

            source_backend = _backend_manifest(manifest, "source_backend")
            source_index = NgramInvertedIndex.from_native(
                lines=train_src,
                native_index=_load_native_bytes(
                    _read_bytes_member(archive, SOURCE_INDEX_NAME), "source"
                ),
                norm_config=config.normalization,
                sim_config=config.similarity,
                index_config=config.index,
                resolved_mode=str(source_backend["resolved_mode"]),
            )

            target_index = None
            if train_tgt is not None:
                target_backend = _backend_manifest(manifest, "target_backend")
                target_index = NgramInvertedIndex.from_native(
                    lines=train_tgt,
                    native_index=_load_native_bytes(
                        _read_bytes_member(archive, TARGET_INDEX_NAME), "target"
                    ),
                    norm_config=config.normalization,
                    sim_config=config.similarity,
                    index_config=config.index,
                    resolved_mode=str(target_backend["resolved_mode"]),
                )
    except zipfile.BadZipFile as exc:
        raise ConfigurationError("index bundle is not a valid zip file") from exc

    return IndexBundle(
        train_src=train_src,
        train_tgt=train_tgt,
        source_index=source_index,
        target_index=target_index,
        manifest=manifest,
    )


def inspect_index_bundle(path: str | Path) -> dict[str, Any]:
    """Read bundle metadata without deserializing native indexes."""

    try:
        with zipfile.ZipFile(Path(path), mode="r") as archive:
            return _read_manifest(archive)
    except zipfile.BadZipFile as exc:
        raise ConfigurationError("index bundle is not a valid zip file") from exc


def _build_manifest(
    train_src: list[str],
    train_tgt: list[str] | None,
    source_index: NgramInvertedIndex,
    target_index: NgramInvertedIndex | None,
    source_index_bytes: bytes,
    target_index_bytes: bytes | None,
    config: ScoreConfig,
) -> dict[str, Any]:
    return {
        "format": INDEX_FORMAT,
        "format_version": FORMAT_VERSION,
        "tame_version": __version__,
        "num_train": len(train_src),
        "has_target": train_tgt is not None,
        "config": config_to_dict(config),
        "normalization": _jsonable(asdict(config.normalization)),
        "similarity": {
            "type": config.similarity.similarity,
            "ngram_orders": list(config.similarity.ngram_orders),
        },
        "index": _jsonable(asdict(config.index)),
        "source_backend": _backend_to_dict(source_index),
        "target_backend": _backend_to_dict(target_index) if target_index is not None else None,
        "storage": {
            "container": "zip",
            "compression": "stored",
            "source_index_bytes": len(source_index_bytes),
            "target_index_bytes": len(target_index_bytes) if target_index_bytes is not None else 0,
        },
        "privacy": {
            "stores_raw_training_text": True,
            "stores_normalized_exact_match_keys": True,
        },
    }


def _backend_to_dict(index: NgramInvertedIndex | None) -> dict[str, Any]:
    if index is None:
        return {}
    return cast(dict[str, Any], _jsonable(asdict(index.backend_info)))


def _read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        payload = archive.read(MANIFEST_NAME).decode("utf-8")
    except KeyError as exc:
        raise ConfigurationError("index bundle is missing manifest.json") from exc
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"index bundle manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ConfigurationError("index bundle manifest must be a JSON object")
    return cast(dict[str, Any], manifest)


def _validate_manifest(manifest: dict[str, Any], config: ScoreConfig) -> None:
    if manifest.get("format") != INDEX_FORMAT:
        raise ConfigurationError("not a TAME-MT index bundle")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ConfigurationError(
            f"unsupported index bundle format version: {manifest.get('format_version')}"
        )

    expected_normalization = _jsonable(asdict(config.normalization))
    expected_similarity = {
        "type": config.similarity.similarity,
        "ngram_orders": list(config.similarity.ngram_orders),
    }
    if manifest.get("normalization") != expected_normalization:
        raise ConfigurationError(
            "index bundle normalization settings do not match the requested config"
        )
    if manifest.get("similarity") != expected_similarity:
        raise ConfigurationError(
            "index bundle similarity settings do not match the requested config"
        )

    source_backend = _backend_manifest(manifest, "source_backend")
    _validate_requested_backend(source_backend, config)
    if bool(manifest.get("has_target")):
        target_backend = _backend_manifest(manifest, "target_backend")
        _validate_requested_backend(target_backend, config)

    saved_index = manifest.get("index")
    if not isinstance(saved_index, dict):
        raise ConfigurationError("index bundle manifest is missing index settings")
    _validate_build_settings(saved_index, config, source_backend)


def _validate_requested_backend(backend: dict[str, Any], config: ScoreConfig) -> None:
    resolved_mode = str(backend.get("resolved_mode"))
    if not resolved_mode.startswith("native_"):
        raise ConfigurationError("index bundle was not built with a native backend")
    requested = config.index.mode
    if requested == "auto":
        return
    if requested != resolved_mode:
        raise ConfigurationError(
            f"index bundle backend is {resolved_mode}, but requested --index-mode {requested}"
        )


def _validate_build_settings(
    saved_index: dict[str, Any],
    config: ScoreConfig,
    source_backend: dict[str, Any],
) -> None:
    saved_mode = str(source_backend.get("resolved_mode"))
    if saved_mode != "native_fast":
        return

    for key in ("candidate_gram_limit", "posting_limit", "max_candidates", "rerank_limit"):
        if saved_index.get(key) != getattr(config.index, key):
            raise ConfigurationError(
                f"index bundle fast setting {key}={saved_index.get(key)} does not match "
                f"requested value {getattr(config.index, key)}"
            )


def _backend_manifest(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    backend = manifest.get(key)
    if not isinstance(backend, dict):
        raise ConfigurationError(f"index bundle manifest is missing {key}")
    return cast(dict[str, Any], backend)


def _read_lines_member(archive: zipfile.ZipFile, name: str) -> list[str]:
    try:
        return _decode_lines(archive.read(name))
    except KeyError as exc:
        raise ConfigurationError(f"index bundle is missing {name}") from exc


def _read_bytes_member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except KeyError as exc:
        raise ConfigurationError(f"index bundle is missing {name}") from exc


def _load_native_bytes(payload: bytes, label: str) -> Any:
    try:
        return native_index_from_bytes(payload)
    except Exception as exc:
        raise BackendError(f"failed to load {label} native index: {exc}") from exc


def _require_native_index(index: NgramInvertedIndex, label: str) -> None:
    if not index.backend_info.native:
        raise BackendError(
            f"{label} index persistence requires the native backend; "
            f"resolved backend was {index.backend_info.resolved_mode}"
        )


def _encode_lines(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


def _decode_lines(payload: bytes) -> list[str]:
    return payload.decode("utf-8").splitlines()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
