import json
import zipfile
from pathlib import Path

import pytest

import tame_mt.persistence as persistence
from tame_mt.api import TameScorer
from tame_mt.config import IndexConfig, NormalizationConfig, RetrievalConfig, ScoreConfig
from tame_mt.exceptions import ConfigurationError
from tame_mt.persistence import (
    FORMAT_VERSION,
    MAX_BUNDLE_LOAD_BYTES,
    NATIVE_INDEX_SCHEMA_VERSION,
    ZIP_COMPRESSION,
    ZIP_COMPRESSION_NAME,
    ZIP_COMPRESSLEVEL,
    inspect_index_bundle,
    load_index_bundle,
    save_index_bundle,
    verify_index_bundle,
)


def test_index_bundle_roundtrip_when_native_available(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    train_src = ["abcdef", "uvwxyz", "abcxyz"]
    train_tgt = ["alpha", "omega", "mixed"]
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    path = tmp_path / "train.tameidx"

    saved = save_index_bundle(path, train_src, train_tgt, config)
    loaded = load_index_bundle(path, config)
    manifest = inspect_index_bundle(path)

    assert saved.source_index.query_topk("abcdeg", 2) == loaded.source_index.query_topk("abcdeg", 2)
    assert saved.source_index.normalized_lines == []
    assert saved.target_index is not None
    assert saved.target_index.normalized_lines == []
    assert loaded.train_src == train_src
    assert loaded.train_tgt == train_tgt
    assert loaded.source_index.normalized_lines == []
    assert loaded.target_index is not None
    assert loaded.target_index.normalized_lines == []
    assert loaded.exact_pair_keys is not None
    assert manifest["format"] == "tameidx"
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["storage"]["compression"] == ZIP_COMPRESSION_NAME
    assert manifest["storage"]["compresslevel"] == ZIP_COMPRESSLEVEL
    assert manifest["storage"]["native_index_schema_version"] == NATIVE_INDEX_SCHEMA_VERSION
    assert manifest["hashes"]["train_src_sha256"]
    assert manifest["hashes"]["train_src_normalized_sha256"]
    assert manifest["hashes"]["source_index_sha256"]
    assert manifest["hashes"]["train_tgt_sha256"]
    assert manifest["hashes"]["train_tgt_normalized_sha256"]
    assert manifest["hashes"]["target_index_sha256"]
    assert manifest["hashes"]["exact_pair_keys_sha256"]
    assert manifest["privacy"]["stores_raw_training_text"] is True
    assert manifest["privacy"]["stores_exact_match_fingerprints"] is True
    assert manifest["privacy"]["stores_pair_fingerprints"] is True
    with zipfile.ZipFile(path, "r") as archive:
        assert {item.compress_type for item in archive.infolist()} == {ZIP_COMPRESSION}

    result = TameScorer(config).evaluate_index_bundle(
        loaded,
        test_src=["abcdef"],
        refs=[["alpha"]],
        hyp=["alpha"],
    )
    assert result.exposures[0].source_exact is True
    assert result.exposures[0].target_exact is True
    assert result.exposures[0].pair_exact is True


def test_verify_index_bundle_checks_hashes_and_native_invariants(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    train_src = ["abcdef", "uvwxyz", "abcxyz"]
    train_tgt = ["alpha", "omega", "mixed"]
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    path = tmp_path / "train.tameidx"
    save_index_bundle(path, train_src, train_tgt, config)

    verification = verify_index_bundle(
        path,
        config=config,
        train_src=train_src,
        train_tgt=train_tgt,
    )
    payload = verification.to_dict()

    assert payload["format"] == "tameidx"
    assert payload["num_train"] == 3
    assert verification.train_src_matches is True
    assert verification.train_tgt_matches is True
    assert "source_index_sha256" in verification.checked_hashes
    assert "train_src_normalized_sha256" in verification.checked_hashes
    assert "train_tgt_normalized_sha256" in verification.checked_hashes
    assert "target.index.bin" in verification.checked_native_indexes


def test_verify_index_bundle_rejects_tampered_member_hash(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    bad_path = tmp_path / "bad.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)
    _copy_bundle_with_member_override(bad_path, path, "train.src", b"abcdeg\n")

    with pytest.raises(ConfigurationError, match="hash mismatch for train_src_sha256"):
        verify_index_bundle(bad_path)


def test_exact_pair_key_member_is_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    first = tmp_path / "first.tameidx"
    second = tmp_path / "second.tameidx"
    train_src = ["same source", "same source", "other source"]
    train_tgt = ["target b", "target a", "target c"]

    save_index_bundle(first, train_src, train_tgt, config)
    save_index_bundle(second, train_src, train_tgt, config)

    with (
        zipfile.ZipFile(first, "r") as first_archive,
        zipfile.ZipFile(second, "r") as second_archive,
    ):
        assert first_archive.read("exact_pairs.keys") == second_archive.read("exact_pairs.keys")


def test_exact_pair_key_member_must_be_sorted() -> None:
    unsorted_payload = (b"\x02" * 16) + (b"\x01" * 16)

    with pytest.raises(ConfigurationError, match="not sorted"):
        persistence._decode_exact_pair_keys(unsorted_payload)


def test_index_bundle_rejects_incompatible_backend_mode(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    save_index_bundle(
        path,
        ["abcdef", "uvwxyz"],
        ["alpha", "omega"],
        ScoreConfig(index=IndexConfig(mode="native_exact")),
    )

    with pytest.raises(ConfigurationError, match="backend is native_exact"):
        load_index_bundle(
            path,
            ScoreConfig(
                index=IndexConfig(mode="native_fast"),
                retrieval=RetrievalConfig(mode="approx", allow_approximate=True),
            ),
        )


def test_scorer_allows_query_only_index_setting_changes(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef", "abcxyz"], ["alpha", "beta"], config)
    bundle = load_index_bundle(path, config)

    scorer = TameScorer(ScoreConfig(index=IndexConfig(mode="native_exact", topk=1, batch_size=1)))
    result = scorer.evaluate_index_bundle(bundle, ["abcdef"], [["alpha"]], ["alpha"])

    assert result.report.backend["index_reused"] is True
    assert result.report.num_train == 2


def test_scorer_rejects_bundle_loaded_with_different_normalization(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)
    bundle = load_index_bundle(path, config)

    scorer = TameScorer(
        ScoreConfig(
            index=IndexConfig(mode="native_exact"),
            normalization=NormalizationConfig(lowercase=True),
        )
    )
    with pytest.raises(ConfigurationError, match="source index normalization"):
        scorer.evaluate_index_bundle(bundle, ["ABCDEF"], [["alpha"]], ["alpha"])


def test_scorer_rejects_bundle_with_incompatible_backend_mode(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    exact_config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], exact_config)
    bundle = load_index_bundle(path, exact_config)

    scorer = TameScorer(
        ScoreConfig(
            index=IndexConfig(mode="native_fast"),
            retrieval=RetrievalConfig(mode="approx", allow_approximate=True),
        )
    )
    with pytest.raises(ConfigurationError, match="source index backend is native_exact"):
        scorer.evaluate_index_bundle(bundle, ["abcdef"], [["alpha"]], ["alpha"])


def test_inspect_index_bundle_reports_invalid_utf8_manifest(tmp_path: Path) -> None:
    path = tmp_path / "bad.tameidx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", b"\xff")

    with pytest.raises(ConfigurationError, match="manifest is not valid UTF-8"):
        inspect_index_bundle(path)


def test_inspect_index_bundle_rejects_duplicate_manifest_member(tmp_path: Path) -> None:
    path = tmp_path / "bad.tameidx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}\n")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("manifest.json", "{}\n")

    with pytest.raises(ConfigurationError, match="duplicate member manifest.json"):
        inspect_index_bundle(path)


def test_inspect_index_bundle_rejects_non_standard_json_constant(tmp_path: Path) -> None:
    path = tmp_path / "bad.tameidx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", '{"format": NaN}\n')

    with pytest.raises(ConfigurationError, match="non-standard JSON constant"):
        inspect_index_bundle(path)


def test_inspect_index_bundle_rejects_duplicate_json_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.tameidx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", '{"format": "tameidx", "format": "other"}\n')

    with pytest.raises(ConfigurationError, match="duplicate JSON object key"):
        inspect_index_bundle(path)


def test_load_index_bundle_reports_invalid_num_train_manifest(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    bad_path = tmp_path / "bad.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)

    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(bad_path, "w") as target:
        manifest = json.loads(source.read("manifest.json").decode("utf-8"))
        manifest["num_train"] = "not-an-int"
        for item in source.infolist():
            if item.filename == "manifest.json":
                target.writestr(item, json.dumps(manifest))
            else:
                target.writestr(item, source.read(item.filename))

    with pytest.raises(ConfigurationError, match="num_train must be an integer"):
        load_index_bundle(bad_path, config)


def test_load_index_bundle_rejects_float_num_train_manifest(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    bad_path = tmp_path / "bad.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)

    _copy_bundle_with_manifest_override(path, bad_path, {"num_train": 1.5})

    with pytest.raises(ConfigurationError, match="num_train must be an integer"):
        load_index_bundle(bad_path, config)


def test_load_index_bundle_rejects_non_boolean_has_target(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    bad_path = tmp_path / "bad.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)

    _copy_bundle_with_manifest_override(path, bad_path, {"has_target": "false"})

    with pytest.raises(ConfigurationError, match="has_target must be a boolean"):
        load_index_bundle(bad_path, config)


def test_load_index_bundle_rejects_manifest_member_size_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    bad_path = tmp_path / "bad.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)
    manifest = inspect_index_bundle(path)
    source_bytes = manifest["storage"]["source_index_bytes"]
    assert isinstance(source_bytes, int)

    _copy_bundle_with_manifest_override(
        path,
        bad_path,
        {"storage": {"source_index_bytes": source_bytes + 1}},
    )

    with pytest.raises(ConfigurationError, match="source.index.bin size"):
        load_index_bundle(bad_path, config)


def test_load_index_bundle_rejects_unexpected_archive_member(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    bad_path = tmp_path / "bad.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)

    _copy_bundle_with_extra_member(path, bad_path, "surprise.bin", b"not part of the format")

    with pytest.raises(ConfigurationError, match="unexpected member surprise.bin"):
        load_index_bundle(bad_path, config)


def test_load_index_bundle_rejects_declared_member_size_above_cap(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    bad_path = tmp_path / "bad.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)

    _copy_bundle_with_manifest_override(
        path,
        bad_path,
        {
            "storage": {
                "source_index_bytes": persistence.MAX_NATIVE_INDEX_BYTES + 1,
            }
        },
    )

    with pytest.raises(ConfigurationError, match="declared size exceeds maximum"):
        load_index_bundle(bad_path, config)


def test_load_index_bundle_rejects_excessive_zip_compression_ratio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)

    monkeypatch.setattr(persistence, "MAX_ZIP_COMPRESSION_RATIO", 1.0)

    with pytest.raises(ConfigurationError, match="compression ratio is too high"):
        load_index_bundle(path, config)


def test_load_index_bundle_rejects_load_budget_excess(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)

    with pytest.raises(ConfigurationError, match="load footprint exceeds maximum"):
        load_index_bundle(path, config, max_load_bytes=1)

    loaded = load_index_bundle(path, config, max_load_bytes=MAX_BUNDLE_LOAD_BYTES)

    assert loaded.train_src == ["abcdef"]


def test_load_index_bundle_rejects_invalid_load_budget(tmp_path: Path) -> None:
    path = tmp_path / "empty.tameidx"

    with pytest.raises(ConfigurationError, match="max_load_bytes"):
        load_index_bundle(path, ScoreConfig(), max_load_bytes=0)


def test_save_index_bundle_is_atomic_when_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    path.write_bytes(b"old bundle bytes")
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))

    def fail_writestr(
        self: zipfile.ZipFile,
        zinfo_or_arcname: str | zipfile.ZipInfo,
        data: str | bytes,
        compress_type: int | None = None,
        compresslevel: int | None = None,
    ) -> None:
        _ = self, zinfo_or_arcname, data, compress_type, compresslevel
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail_writestr)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        save_index_bundle(path, ["abcdef"], ["alpha"], config)

    assert path.read_bytes() == b"old bundle bytes"
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_load_index_bundle_rejects_unsupported_format_version(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    bad_path = tmp_path / "bad.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)

    _copy_bundle_with_manifest_override(
        path,
        bad_path,
        {"format_version": FORMAT_VERSION - 1},
    )

    with pytest.raises(ConfigurationError, match="unsupported index bundle format version"):
        load_index_bundle(bad_path, config)


def test_load_index_bundle_rejects_unsupported_native_schema(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    bad_path = tmp_path / "bad.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)

    _copy_bundle_with_manifest_override(
        path,
        bad_path,
        {"storage": {"native_index_schema_version": NATIVE_INDEX_SCHEMA_VERSION - 1}},
    )

    with pytest.raises(ConfigurationError, match="unsupported native index schema version"):
        load_index_bundle(bad_path, config)


def _copy_bundle_with_manifest_override(
    source_path: Path,
    target_path: Path,
    overrides: dict[str, object],
) -> None:
    with zipfile.ZipFile(source_path, "r") as source, zipfile.ZipFile(target_path, "w") as target:
        manifest = json.loads(source.read("manifest.json").decode("utf-8"))
        _merge_manifest_overrides(manifest, overrides)
        for item in source.infolist():
            if item.filename == "manifest.json":
                target.writestr(item, json.dumps(manifest))
            else:
                target.writestr(item, source.read(item.filename))


def _copy_bundle_with_extra_member(
    source_path: Path,
    target_path: Path,
    member_name: str,
    payload: bytes,
) -> None:
    with zipfile.ZipFile(source_path, "r") as source, zipfile.ZipFile(target_path, "w") as target:
        for item in source.infolist():
            target.writestr(item, source.read(item.filename))
        target.writestr(member_name, payload)


def _copy_bundle_with_member_override(
    target_path: Path,
    source_path: Path,
    member_name: str,
    payload: bytes,
) -> None:
    with zipfile.ZipFile(source_path, "r") as source, zipfile.ZipFile(target_path, "w") as target:
        for item in source.infolist():
            if item.filename == member_name:
                target.writestr(item, payload)
            else:
                target.writestr(item, source.read(item.filename))


def _merge_manifest_overrides(
    manifest: dict[str, object],
    overrides: dict[str, object],
) -> None:
    for key, value in overrides.items():
        existing = manifest.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_manifest_overrides(existing, value)
        else:
            manifest[key] = value
