import json
import zipfile
from pathlib import Path

import pytest

from tame_mt.api import TameScorer
from tame_mt.config import IndexConfig, ScoreConfig
from tame_mt.exceptions import ConfigurationError
from tame_mt.persistence import inspect_index_bundle, load_index_bundle, save_index_bundle


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
    assert loaded.train_src == train_src
    assert loaded.train_tgt == train_tgt
    assert loaded.source_index.normalized_lines == []
    assert loaded.target_index is not None
    assert loaded.target_index.normalized_lines == []
    assert loaded.exact_pair_keys is not None
    assert manifest["format"] == "tameidx"
    assert manifest["privacy"]["stores_raw_training_text"] is True
    assert manifest["privacy"]["stores_normalized_pair_keys"] is True

    result = TameScorer(config).evaluate_index_bundle(
        loaded,
        test_src=["abcdef"],
        refs=[["alpha"]],
        hyp=["alpha"],
    )
    assert result.exposures[0].source_exact is True
    assert result.exposures[0].target_exact is True
    assert result.exposures[0].pair_exact is True


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
        load_index_bundle(path, ScoreConfig(index=IndexConfig(mode="native_fast")))


def test_scorer_rejects_bundle_loaded_with_different_config(tmp_path: Path) -> None:
    pytest.importorskip("tame_mt._native")
    path = tmp_path / "train.tameidx"
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    save_index_bundle(path, ["abcdef"], ["alpha"], config)
    bundle = load_index_bundle(path, config)

    scorer = TameScorer(ScoreConfig(index=IndexConfig(mode="native_exact", topk=10)))
    with pytest.raises(ConfigurationError, match="retrieval settings"):
        scorer.evaluate_index_bundle(bundle, ["abcdef"], [["alpha"]], ["alpha"])


def test_inspect_index_bundle_reports_invalid_utf8_manifest(tmp_path: Path) -> None:
    path = tmp_path / "bad.tameidx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", b"\xff")

    with pytest.raises(ConfigurationError, match="manifest is not valid UTF-8"):
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
