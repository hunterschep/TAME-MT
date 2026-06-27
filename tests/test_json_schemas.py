import json
from pathlib import Path

from jsonschema import Draft202012Validator

from tame_mt import TameScorer
from tame_mt.artifacts import build_segment_fingerprints, read_segment_metadata
from tame_mt.config import IndexConfig, ScoreConfig
from tame_mt.persistence import inspect_index_bundle, save_index_bundle
from tame_mt.report import segment_metadata_path, write_segment_jsonl, write_segment_metadata

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"


def test_json_schemas_are_valid() -> None:
    for schema_path in sorted(SCHEMAS.glob("*.schema.json")):
        Draft202012Validator.check_schema(_load_json(schema_path))


def test_report_json_matches_schema() -> None:
    report = TameScorer().score_corpus(
        train_src=["hello world", "other source"],
        train_tgt=["hola mundo", "otro"],
        test_src=["hello world"],
        refs=[["hola mundo"]],
        hyp=["hola mundo"],
    )

    _validate("tame_report.v1.schema.json", report.to_dict())


def test_segment_jsonl_row_and_metadata_match_schemas(tmp_path: Path) -> None:
    train_src = ["hello world", "other source"]
    train_tgt = ["hola mundo", "otro"]
    test_src = ["hello world"]
    refs = [["hola mundo"]]
    hyp = ["hola mundo"]
    result = TameScorer().evaluate_corpus(train_src, train_tgt, test_src, refs, hyp)
    segment_path = tmp_path / "segments.jsonl"

    write_segment_jsonl(
        segment_path,
        result.exposures,
        result.tm_results,
        train_src=train_src,
        train_tgt=train_tgt,
        test_src=test_src,
        refs=refs,
        hyp=hyp,
    )
    write_segment_metadata(
        segment_metadata_path(segment_path),
        result.report,
        fingerprints=build_segment_fingerprints(
            ScoreConfig(),
            train_src=train_src,
            train_tgt=train_tgt,
            test_src=test_src,
            refs=refs,
            tm_results=result.tm_results,
        ),
    )

    row = json.loads(segment_path.read_text(encoding="utf-8").splitlines()[0])
    metadata = read_segment_metadata(segment_path)

    assert metadata is not None
    _validate("segment_diagnostic.v1.schema.json", row)
    _validate("tame_cache.v1.schema.json", metadata)


def test_index_manifest_matches_schema(tmp_path: Path) -> None:
    config = ScoreConfig(index=IndexConfig(mode="native_exact"))
    path = tmp_path / "train.tameidx"
    save_index_bundle(
        path,
        train_src=["hello world", "other source"],
        train_tgt=["hola mundo", "otro"],
        config=config,
    )

    _validate("tame_index_manifest.v1.schema.json", inspect_index_bundle(path))


def _validate(schema_name: str, payload: object) -> None:
    Draft202012Validator(_load_json(SCHEMAS / schema_name)).validate(payload)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
