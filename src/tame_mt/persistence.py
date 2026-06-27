from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from tame_mt.config import ScoreConfig
from tame_mt.exact import build_exact_pair_keys
from tame_mt.exceptions import BackendError, ConfigurationError, InputDataError
from tame_mt.index import NgramInvertedIndex
from tame_mt.io import ensure_parent_dir, validate_equal_lengths
from tame_mt.json_utils import strict_json_dumps, strict_json_loads
from tame_mt.native import native_index_from_bytes
from tame_mt.normalize import normalize_text
from tame_mt.report import config_to_dict
from tame_mt.version import __version__

INDEX_FORMAT = "tameidx"
FORMAT_VERSION = 3
NATIVE_INDEX_SCHEMA_VERSION = 3

MANIFEST_NAME = "manifest.json"
TRAIN_SRC_NAME = "train.src"
TRAIN_TGT_NAME = "train.tgt"
SOURCE_INDEX_NAME = "source.index.bin"
TARGET_INDEX_NAME = "target.index.bin"
EXACT_PAIR_KEYS_NAME = "exact_pairs.keys"
ZIP_COMPRESSION = zipfile.ZIP_DEFLATED
ZIP_COMPRESSION_NAME = "deflated"
ZIP_COMPRESSLEVEL = 1
MAX_MANIFEST_BYTES = 1_000_000
MAX_TRAIN_TEXT_MEMBER_BYTES = 4 * 1024 * 1024 * 1024
MAX_NATIVE_INDEX_BYTES = 8 * 1024 * 1024 * 1024
MAX_EXACT_PAIR_KEYS_BYTES = 8 * 1024 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 24 * 1024 * 1024 * 1024
MAX_BUNDLE_LOAD_BYTES = 4 * 1024 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 500.0


@dataclass(frozen=True, slots=True)
class IndexBundle:
    """A loaded persistent training index and the raw training corpus it indexes."""

    train_src: list[str]
    train_tgt: list[str] | None
    source_index: NgramInvertedIndex
    target_index: NgramInvertedIndex | None
    exact_pair_keys: set[str] | None
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IndexVerification:
    """Summary returned by index-bundle verification."""

    manifest: dict[str, Any]
    checked_members: list[str]
    checked_hashes: list[str]
    checked_native_indexes: list[str]
    train_src_matches: bool | None = None
    train_tgt_matches: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.manifest.get("format"),
            "format_version": self.manifest.get("format_version"),
            "tame_version": self.manifest.get("tame_version"),
            "num_train": self.manifest.get("num_train"),
            "has_target": self.manifest.get("has_target"),
            "source_backend": self.manifest.get("source_backend"),
            "target_backend": self.manifest.get("target_backend"),
            "checked_members": self.checked_members,
            "checked_hashes": self.checked_hashes,
            "checked_native_indexes": self.checked_native_indexes,
            "train_src_matches": self.train_src_matches,
            "train_tgt_matches": self.train_tgt_matches,
        }


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
    exact_pair_keys = (
        build_exact_pair_keys(source_index.normalized_lines, target_index.normalized_lines)
        if target_index is not None
        else None
    )
    exact_pair_keys_bytes = (
        _encode_exact_pair_keys(exact_pair_keys) if exact_pair_keys is not None else None
    )
    manifest = _build_manifest(
        train_src,
        train_tgt,
        source_index,
        target_index,
        source_index_bytes,
        target_index_bytes,
        exact_pair_keys_bytes,
        config,
    )
    output_path = Path(path)
    ensure_parent_dir(output_path)
    source_index.release_python_normalized_lines()
    if target_index is not None:
        target_index.release_python_normalized_lines()
    tmp_path = _temporary_bundle_path(output_path)
    try:
        with zipfile.ZipFile(
            tmp_path,
            mode="w",
            compression=ZIP_COMPRESSION,
            compresslevel=ZIP_COMPRESSLEVEL,
        ) as archive:
            archive.writestr(
                MANIFEST_NAME,
                strict_json_dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
            archive.writestr(TRAIN_SRC_NAME, _encode_lines(train_src))
            if train_tgt is not None:
                archive.writestr(TRAIN_TGT_NAME, _encode_lines(train_tgt))
            archive.writestr(SOURCE_INDEX_NAME, source_index_bytes)
            if target_index_bytes is not None:
                archive.writestr(TARGET_INDEX_NAME, target_index_bytes)
            if exact_pair_keys_bytes is not None:
                archive.writestr(EXACT_PAIR_KEYS_NAME, exact_pair_keys_bytes)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return IndexBundle(
        train_src=train_src,
        train_tgt=train_tgt,
        source_index=source_index,
        target_index=target_index,
        exact_pair_keys=exact_pair_keys,
        manifest=manifest,
    )


def load_index_bundle(
    path: str | Path,
    config: ScoreConfig,
    *,
    max_load_bytes: int | None = MAX_BUNDLE_LOAD_BYTES,
) -> IndexBundle:
    """Load and validate a native training index bundle for the supplied config."""

    _validate_max_load_bytes(max_load_bytes)
    try:
        with zipfile.ZipFile(Path(path), mode="r") as archive:
            manifest = _read_manifest(archive)
            _validate_manifest(manifest, config)
            _validate_archive_members(archive, manifest, max_load_bytes=max_load_bytes)
            _validate_archive_hashes(archive, manifest, config=config)
            has_target = _manifest_bool(manifest, "has_target")
            train_src = _read_lines_member(archive, TRAIN_SRC_NAME)
            train_tgt = _read_lines_member(archive, TRAIN_TGT_NAME) if has_target else None
            if _manifest_int(manifest, "num_train") != len(train_src):
                raise ConfigurationError(
                    "index bundle manifest does not match stored train.src line count"
                )
            if train_tgt is not None:
                validate_equal_lengths("train.src", train_src, "train.tgt", train_tgt)

            source_backend = _backend_manifest(manifest, "source_backend")
            source_index = NgramInvertedIndex.from_native(
                native_index=_load_native_bytes(
                    _read_bytes_member(archive, SOURCE_INDEX_NAME), "source"
                ),
                norm_config=config.normalization,
                sim_config=config.similarity,
                index_config=config.index,
                resolved_mode=str(source_backend["resolved_mode"]),
                lines=train_src,
                normalized_lines=[],
            )

            target_index = None
            exact_pair_keys = None
            if has_target:
                target_backend = _backend_manifest(manifest, "target_backend")
                target_index = NgramInvertedIndex.from_native(
                    native_index=_load_native_bytes(
                        _read_bytes_member(archive, TARGET_INDEX_NAME), "target"
                    ),
                    norm_config=config.normalization,
                    sim_config=config.similarity,
                    index_config=config.index,
                    resolved_mode=str(target_backend["resolved_mode"]),
                    lines=train_tgt,
                    normalized_lines=[],
                )
                exact_pair_keys = _read_exact_pair_keys_member(archive)
    except zipfile.BadZipFile as exc:
        raise ConfigurationError("index bundle is not a valid zip file") from exc

    return IndexBundle(
        train_src=train_src,
        train_tgt=train_tgt,
        source_index=source_index,
        target_index=target_index,
        exact_pair_keys=exact_pair_keys,
        manifest=manifest,
    )


def inspect_index_bundle(path: str | Path) -> dict[str, Any]:
    """Read bundle metadata without deserializing native indexes."""

    try:
        with zipfile.ZipFile(Path(path), mode="r") as archive:
            return _read_manifest(archive)
    except zipfile.BadZipFile as exc:
        raise ConfigurationError("index bundle is not a valid zip file") from exc


def verify_index_bundle(
    path: str | Path,
    *,
    config: ScoreConfig | None = None,
    train_src: list[str] | None = None,
    train_tgt: list[str] | None = None,
    max_load_bytes: int | None = MAX_BUNDLE_LOAD_BYTES,
) -> IndexVerification:
    """Verify a `.tameidx` bundle without constructing a scoring bundle."""

    _validate_max_load_bytes(max_load_bytes)
    try:
        with zipfile.ZipFile(Path(path), mode="r") as archive:
            manifest = _read_manifest(archive)
            if config is None:
                _validate_manifest_format(manifest)
            else:
                _validate_manifest(manifest, config)
            checked_members = _validate_archive_members(
                archive,
                manifest,
                max_load_bytes=max_load_bytes,
            )
            checked_hashes = _validate_archive_hashes(archive, manifest, config=config)
            checked_native_indexes = _verify_native_index_members(archive, manifest)
            train_src_matches, train_tgt_matches = _verify_supplied_training_hashes(
                manifest,
                train_src=train_src,
                train_tgt=train_tgt,
                config=config,
            )
    except zipfile.BadZipFile as exc:
        raise ConfigurationError("index bundle is not a valid zip file") from exc

    return IndexVerification(
        manifest=manifest,
        checked_members=checked_members,
        checked_hashes=checked_hashes,
        checked_native_indexes=checked_native_indexes,
        train_src_matches=train_src_matches,
        train_tgt_matches=train_tgt_matches,
    )


def _build_manifest(
    train_src: list[str],
    train_tgt: list[str] | None,
    source_index: NgramInvertedIndex,
    target_index: NgramInvertedIndex | None,
    source_index_bytes: bytes,
    target_index_bytes: bytes | None,
    exact_pair_keys_bytes: bytes | None,
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
            "compression": ZIP_COMPRESSION_NAME,
            "compresslevel": ZIP_COMPRESSLEVEL,
            "native_index_schema_version": NATIVE_INDEX_SCHEMA_VERSION,
            "source_index_bytes": len(source_index_bytes),
            "target_index_bytes": len(target_index_bytes) if target_index_bytes is not None else 0,
            "exact_pair_keys_bytes": (
                len(exact_pair_keys_bytes) if exact_pair_keys_bytes is not None else 0
            ),
        },
        "hashes": _build_manifest_hashes(
            train_src=train_src,
            train_tgt=train_tgt,
            source_normalized=source_index.normalized_lines,
            target_normalized=target_index.normalized_lines if target_index is not None else None,
            source_index_bytes=source_index_bytes,
            target_index_bytes=target_index_bytes,
            exact_pair_keys_bytes=exact_pair_keys_bytes,
        ),
        "privacy": {
            "stores_raw_training_text": True,
            "stores_normalized_exact_match_keys": True,
            "stores_normalized_pair_keys": exact_pair_keys_bytes is not None,
        },
    }


def _backend_to_dict(index: NgramInvertedIndex | None) -> dict[str, Any]:
    if index is None:
        return {}
    return cast(dict[str, Any], _jsonable(asdict(index.backend_info)))


def _build_manifest_hashes(
    *,
    train_src: list[str],
    train_tgt: list[str] | None,
    source_normalized: list[str],
    target_normalized: list[str] | None,
    source_index_bytes: bytes,
    target_index_bytes: bytes | None,
    exact_pair_keys_bytes: bytes | None,
) -> dict[str, str]:
    hashes = {
        "train_src_sha256": _hash_lines(train_src),
        "train_src_normalized_sha256": _hash_lines(source_normalized),
        "source_index_sha256": _sha256_bytes(source_index_bytes),
    }
    if train_tgt is not None:
        hashes["train_tgt_sha256"] = _hash_lines(train_tgt)
    if target_normalized is not None:
        hashes["train_tgt_normalized_sha256"] = _hash_lines(target_normalized)
    if target_index_bytes is not None:
        hashes["target_index_sha256"] = _sha256_bytes(target_index_bytes)
    if exact_pair_keys_bytes is not None:
        hashes["exact_pair_keys_sha256"] = _sha256_bytes(exact_pair_keys_bytes)
    return hashes


def _read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        payload = _read_member_bytes(
            archive,
            MANIFEST_NAME,
            max_size=MAX_MANIFEST_BYTES,
        ).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError("index bundle manifest is not valid UTF-8") from exc
    try:
        manifest = strict_json_loads(payload)
    except ValueError as exc:
        raise ConfigurationError(f"index bundle manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ConfigurationError("index bundle manifest must be a JSON object")
    return cast(dict[str, Any], manifest)


def _validate_manifest(manifest: dict[str, Any], config: ScoreConfig) -> None:
    _validate_manifest_format(manifest)
    _validate_manifest_compatibility(manifest, config)


def _validate_manifest_format(manifest: dict[str, Any]) -> None:
    if manifest.get("format") != INDEX_FORMAT:
        raise ConfigurationError("not a TAME-MT index bundle")
    format_version = _manifest_int(manifest, "format_version")
    if format_version != FORMAT_VERSION:
        raise ConfigurationError(
            "unsupported index bundle format version: "
            f"{format_version}; rebuild the .tameidx file with the current "
            "TAME-MT version"
        )
    storage = _storage_manifest(manifest)
    native_schema_version = _storage_int(storage, "native_index_schema_version")
    if native_schema_version != NATIVE_INDEX_SCHEMA_VERSION:
        raise ConfigurationError(
            "unsupported native index schema version: "
            f"{native_schema_version}; rebuild the .tameidx file with the "
            "current TAME-MT version"
        )
    _manifest_int(manifest, "num_train")
    _manifest_bool(manifest, "has_target")
    _backend_manifest(manifest, "source_backend")
    if _manifest_bool(manifest, "has_target"):
        _backend_manifest(manifest, "target_backend")
    _hashes_manifest(manifest)


def _validate_manifest_compatibility(manifest: dict[str, Any], config: ScoreConfig) -> None:
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
    if _manifest_bool(manifest, "has_target"):
        target_backend = _backend_manifest(manifest, "target_backend")
        _validate_requested_backend(target_backend, config)

    saved_index = manifest.get("index")
    if not isinstance(saved_index, dict):
        raise ConfigurationError("index bundle manifest is missing index settings")
    _validate_build_settings(saved_index, config, source_backend)


def _validate_archive_members(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
    *,
    max_load_bytes: int | None,
) -> list[str]:
    storage = _storage_manifest(manifest)
    has_target = _manifest_bool(manifest, "has_target")

    _validate_archive_shape(archive, has_target=has_target)
    checked_members = [MANIFEST_NAME]
    _validate_text_member(archive, TRAIN_SRC_NAME)
    checked_members.append(TRAIN_SRC_NAME)
    _validate_member_size(
        archive,
        SOURCE_INDEX_NAME,
        _positive_storage_int(storage, "source_index_bytes"),
        max_size=MAX_NATIVE_INDEX_BYTES,
    )
    checked_members.append(SOURCE_INDEX_NAME)

    target_index_bytes = _storage_int(storage, "target_index_bytes")
    exact_pair_keys_bytes = _storage_int(storage, "exact_pair_keys_bytes")
    if has_target:
        _validate_text_member(archive, TRAIN_TGT_NAME)
        checked_members.append(TRAIN_TGT_NAME)
        _validate_member_size(
            archive,
            TARGET_INDEX_NAME,
            _positive_storage_int(storage, "target_index_bytes"),
            max_size=MAX_NATIVE_INDEX_BYTES,
        )
        checked_members.append(TARGET_INDEX_NAME)
        _validate_member_size(
            archive,
            EXACT_PAIR_KEYS_NAME,
            _positive_storage_int(storage, "exact_pair_keys_bytes"),
            max_size=MAX_EXACT_PAIR_KEYS_BYTES,
        )
        checked_members.append(EXACT_PAIR_KEYS_NAME)
    elif target_index_bytes != 0 or exact_pair_keys_bytes != 0:
        raise ConfigurationError(
            "index bundle manifest has target storage bytes but has_target is false"
        )
    _validate_load_budget(archive, has_target=has_target, max_load_bytes=max_load_bytes)
    return checked_members


def _validate_archive_hashes(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
    *,
    config: ScoreConfig | None = None,
) -> list[str]:
    hashes = _hashes_manifest(manifest)
    checked: list[str] = []
    train_src = _read_lines_member(archive, TRAIN_SRC_NAME)
    _validate_hash(hashes, "train_src_sha256", _hash_lines(train_src))
    checked.append("train_src_sha256")
    if config is not None:
        _validate_hash(
            hashes,
            "train_src_normalized_sha256",
            _hash_lines(_normalized_lines(train_src, config)),
        )
        checked.append("train_src_normalized_sha256")
    _validate_hash(
        hashes, "source_index_sha256", _sha256_bytes(_read_bytes_member(archive, SOURCE_INDEX_NAME))
    )
    checked.append("source_index_sha256")
    if _manifest_bool(manifest, "has_target"):
        train_tgt = _read_lines_member(archive, TRAIN_TGT_NAME)
        _validate_hash(
            hashes,
            "train_tgt_sha256",
            _hash_lines(train_tgt),
        )
        checked.append("train_tgt_sha256")
        if config is not None:
            _validate_hash(
                hashes,
                "train_tgt_normalized_sha256",
                _hash_lines(_normalized_lines(train_tgt, config)),
            )
            checked.append("train_tgt_normalized_sha256")
        _validate_hash(
            hashes,
            "target_index_sha256",
            _sha256_bytes(_read_bytes_member(archive, TARGET_INDEX_NAME)),
        )
        checked.append("target_index_sha256")
        _validate_hash(
            hashes,
            "exact_pair_keys_sha256",
            _sha256_bytes(
                _read_member_bytes(
                    archive,
                    EXACT_PAIR_KEYS_NAME,
                    max_size=MAX_EXACT_PAIR_KEYS_BYTES,
                )
            ),
        )
        checked.append("exact_pair_keys_sha256")
    return checked


def _verify_native_index_members(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
) -> list[str]:
    _load_native_bytes(_read_bytes_member(archive, SOURCE_INDEX_NAME), "source")
    checked = [SOURCE_INDEX_NAME]
    if _manifest_bool(manifest, "has_target"):
        _load_native_bytes(_read_bytes_member(archive, TARGET_INDEX_NAME), "target")
        checked.append(TARGET_INDEX_NAME)
    return checked


def _verify_supplied_training_hashes(
    manifest: dict[str, Any],
    *,
    train_src: list[str] | None,
    train_tgt: list[str] | None,
    config: ScoreConfig | None,
) -> tuple[bool | None, bool | None]:
    hashes = _hashes_manifest(manifest)
    train_src_matches = None
    train_tgt_matches = None
    if train_src is not None:
        _validate_hash(hashes, "train_src_sha256", _hash_lines(train_src))
        train_src_matches = True
        if config is not None:
            _validate_hash(
                hashes,
                "train_src_normalized_sha256",
                _hash_lines(_normalized_lines(train_src, config)),
            )
    if train_tgt is not None:
        if not _manifest_bool(manifest, "has_target"):
            raise ConfigurationError("--train-tgt was supplied but index bundle has no target side")
        _validate_hash(hashes, "train_tgt_sha256", _hash_lines(train_tgt))
        train_tgt_matches = True
        if config is not None:
            _validate_hash(
                hashes,
                "train_tgt_normalized_sha256",
                _hash_lines(_normalized_lines(train_tgt, config)),
            )
    return train_src_matches, train_tgt_matches


def _validate_max_load_bytes(max_load_bytes: int | None) -> None:
    if max_load_bytes is None:
        return
    if isinstance(max_load_bytes, bool) or not isinstance(max_load_bytes, int):
        raise ConfigurationError("max_load_bytes must be an integer or None")
    if max_load_bytes <= 0:
        raise ConfigurationError("max_load_bytes must be positive")


def _validate_load_budget(
    archive: zipfile.ZipFile,
    *,
    has_target: bool,
    max_load_bytes: int | None,
) -> None:
    if max_load_bytes is None:
        return
    member_names = [TRAIN_SRC_NAME, SOURCE_INDEX_NAME]
    if has_target:
        member_names.extend([TRAIN_TGT_NAME, TARGET_INDEX_NAME, EXACT_PAIR_KEYS_NAME])
    load_bytes = sum(_archive_member_info(archive, name).file_size for name in member_names)
    if load_bytes > max_load_bytes:
        raise ConfigurationError(
            "index bundle load footprint exceeds maximum: "
            f"{load_bytes} > {max_load_bytes}. "
            "Raise max_load_bytes only for trusted bundles on machines with enough memory."
        )


def _manifest_int(manifest: dict[str, Any], key: str) -> int:
    value = manifest.get(key)
    if value is None:
        raise ConfigurationError(f"index bundle manifest field {key} must be an integer")
    if type(value) is not int:
        raise ConfigurationError(f"index bundle manifest field {key} must be an integer")
    return value


def _manifest_bool(manifest: dict[str, Any], key: str) -> bool:
    value = manifest.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"index bundle manifest field {key} must be a boolean")
    return value


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


def _storage_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    storage = manifest.get("storage")
    if not isinstance(storage, dict):
        raise ConfigurationError("index bundle manifest is missing storage")
    return cast(dict[str, Any], storage)


def _hashes_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        raise ConfigurationError("index bundle manifest is missing hashes")
    required = ["train_src_sha256", "source_index_sha256"]
    if _manifest_bool(manifest, "has_target"):
        required.extend(["train_tgt_sha256", "target_index_sha256", "exact_pair_keys_sha256"])
    for key in required:
        value = hashes.get(key)
        if not isinstance(value, str) or not _looks_like_sha256(value):
            raise ConfigurationError(f"index bundle hash {key} must be a SHA-256 hex digest")
    return cast(dict[str, Any], hashes)


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_hash(hashes: dict[str, Any], key: str, actual: str) -> None:
    expected = hashes.get(key)
    if expected != actual:
        raise ConfigurationError(f"index bundle hash mismatch for {key}")


def _storage_int(storage: dict[str, Any], key: str) -> int:
    value = storage.get(key)
    if value is None:
        raise ConfigurationError(f"index bundle storage field {key} must be an integer")
    if type(value) is not int:
        raise ConfigurationError(f"index bundle storage field {key} must be an integer")
    if value < 0:
        raise ConfigurationError(f"index bundle storage field {key} must be non-negative")
    return value


def _positive_storage_int(storage: dict[str, Any], key: str) -> int:
    value = _storage_int(storage, key)
    if value <= 0:
        raise ConfigurationError(f"index bundle storage field {key} must be positive")
    return value


def _archive_member_info(archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
    matches = [item for item in archive.infolist() if item.filename == name]
    if not matches:
        raise ConfigurationError(f"index bundle is missing {name}")
    if len(matches) > 1:
        raise ConfigurationError(f"index bundle has duplicate member {name}")
    return matches[0]


def _validate_archive_shape(archive: zipfile.ZipFile, *, has_target: bool) -> None:
    expected = {
        MANIFEST_NAME,
        TRAIN_SRC_NAME,
        SOURCE_INDEX_NAME,
    }
    if has_target:
        expected.update({TRAIN_TGT_NAME, TARGET_INDEX_NAME, EXACT_PAIR_KEYS_NAME})
    names = [item.filename for item in archive.infolist()]
    unexpected = sorted(set(names) - expected)
    if unexpected:
        raise ConfigurationError(f"index bundle has unexpected member {unexpected[0]}")
    total_size = sum(item.file_size for item in archive.infolist())
    if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ConfigurationError(
            "index bundle uncompressed size exceeds maximum: "
            f"{total_size} > {MAX_TOTAL_UNCOMPRESSED_BYTES}"
        )


def _has_archive_member(archive: zipfile.ZipFile, name: str) -> bool:
    return any(item.filename == name for item in archive.infolist())


def _validate_text_member(archive: zipfile.ZipFile, name: str) -> None:
    _validate_member_safety(
        _archive_member_info(archive, name),
        max_size=MAX_TRAIN_TEXT_MEMBER_BYTES,
    )


def _validate_member_size(
    archive: zipfile.ZipFile,
    name: str,
    expected_size: int,
    *,
    max_size: int,
) -> None:
    if expected_size > max_size:
        raise ConfigurationError(
            f"index bundle member {name} declared size exceeds maximum: "
            f"{expected_size} > {max_size}"
        )
    member = _archive_member_info(archive, name)
    _validate_member_safety(member, max_size=max_size)
    actual_size = member.file_size
    if actual_size != expected_size:
        raise ConfigurationError(
            f"index bundle member {name} size {actual_size} does not match "
            f"manifest value {expected_size}"
        )


def _validate_member_safety(member: zipfile.ZipInfo, *, max_size: int) -> None:
    if member.file_size > max_size:
        raise ConfigurationError(
            f"index bundle member {member.filename} is too large: {member.file_size} > {max_size}"
        )
    if member.file_size and member.compress_size == 0:
        raise ConfigurationError(f"index bundle member {member.filename} has invalid compression")
    if member.compress_size:
        ratio = member.file_size / member.compress_size
        if ratio > MAX_ZIP_COMPRESSION_RATIO:
            raise ConfigurationError(
                f"index bundle member {member.filename} compression ratio is too high: "
                f"{ratio:.1f} > {MAX_ZIP_COMPRESSION_RATIO:.1f}"
            )


def _read_member_bytes(
    archive: zipfile.ZipFile,
    name: str,
    *,
    max_size: int | None = None,
) -> bytes:
    member = _archive_member_info(archive, name)
    if max_size is not None and member.file_size > max_size:
        raise ConfigurationError(
            f"index bundle member {name} is too large: {member.file_size} > {max_size}"
        )
    _validate_member_safety(member, max_size=max_size or MAX_TOTAL_UNCOMPRESSED_BYTES)
    return archive.read(member)


def _read_lines_member(archive: zipfile.ZipFile, name: str) -> list[str]:
    member = _archive_member_info(archive, name)
    _validate_member_safety(member, max_size=MAX_TRAIN_TEXT_MEMBER_BYTES)
    try:
        with (
            archive.open(member, "r") as raw_handle,
            io.TextIOWrapper(
                raw_handle,
                encoding="utf-8",
            ) as text_handle,
        ):
            return [line.rstrip("\n\r") for line in text_handle]
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"index bundle member {name} is not valid UTF-8") from exc


def _read_bytes_member(archive: zipfile.ZipFile, name: str) -> bytes:
    return _read_member_bytes(archive, name, max_size=MAX_NATIVE_INDEX_BYTES)


def _read_exact_pair_keys_member(archive: zipfile.ZipFile) -> set[str] | None:
    if not _has_archive_member(archive, EXACT_PAIR_KEYS_NAME):
        return None
    try:
        return _decode_exact_pair_keys(
            _read_member_bytes(
                archive,
                EXACT_PAIR_KEYS_NAME,
                max_size=MAX_EXACT_PAIR_KEYS_BYTES,
            )
        )
    except UnicodeDecodeError as exc:
        raise ConfigurationError("index bundle exact-pair key member is not valid UTF-8") from exc


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


def _encode_exact_pair_keys(keys: set[str]) -> bytes:
    payload = bytearray()
    for key in sorted(keys):
        encoded = key.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, byteorder="little", signed=False))
        payload.extend(encoded)
    return bytes(payload)


def _decode_exact_pair_keys(payload: bytes) -> set[str]:
    keys: set[str] = set()
    offset = 0
    payload_len = len(payload)
    while offset < payload_len:
        if offset + 8 > payload_len:
            raise ConfigurationError("index bundle exact-pair key member is truncated")
        key_len = int.from_bytes(payload[offset : offset + 8], byteorder="little", signed=False)
        offset += 8
        end = offset + key_len
        if end > payload_len:
            raise ConfigurationError("index bundle exact-pair key member is truncated")
        keys.add(payload[offset:end].decode("utf-8"))
        offset = end
    return keys


def _jsonable(value: Any) -> Any:
    return strict_json_loads(strict_json_dumps(value, ensure_ascii=False))


def _hash_lines(lines: list[str]) -> str:
    digest = sha256()
    for line in lines:
        encoded = line.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _normalized_lines(lines: list[str], config: ScoreConfig) -> list[str]:
    return [normalize_text(line, config.normalization) for line in lines]


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _temporary_bundle_path(output_path: Path) -> Path:
    parent = output_path.parent if output_path.parent != Path("") else Path(".")
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=parent,
        delete=False,
    ) as handle:
        return Path(handle.name)
