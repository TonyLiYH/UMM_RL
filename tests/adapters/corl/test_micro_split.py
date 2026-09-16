from __future__ import annotations

import json
from pathlib import Path

import pytest

from comppareto.adapters.corl.micro_split import (
    MicroSplitValidationError,
    load_records,
    validate_records,
)

RECORD_TEMPLATE = {
    "idx": 0,
    "prompt": "a man making donuts",
    "qa_problem": "Question: what?",
    "qa_solution": "<answer>oil</answer>",
    "qa_type": "OE",
    "image_path": "/dockerdata/t710-corl/assets/micro-split-images/000.png",
}


def _record(idx: int, prompt: str, qa_type: str = "OE") -> dict:
    record = dict(RECORD_TEMPLATE)
    record["idx"] = idx
    record["prompt"] = prompt
    record["qa_type"] = qa_type
    return record


def test_load_records_roundtrip(tmp_path: Path):
    path = tmp_path / "micro-split.jsonl"
    records = [_record(0, "prompt a"), _record(1, "prompt b")]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    loaded = load_records(path)
    assert loaded == records


def test_load_records_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "micro-split.jsonl"
    path.write_text(json.dumps(_record(0, "p")) + "\n\n", encoding="utf-8")
    assert len(load_records(path)) == 1


def test_load_records_bad_json_raises(tmp_path: Path):
    path = tmp_path / "micro-split.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(MicroSplitValidationError):
        load_records(path)


def test_validate_records_ok():
    records = [_record(i, f"prompt {i}", "MC" if i % 2 else "OE") for i in range(5)]
    summary = validate_records(records)
    assert summary["num_records"] == 5
    assert summary["num_unique_prompts"] == 5
    assert summary["num_with_image"] == 5
    assert summary["qa_type_counts"] == {"OE": 3, "MC": 2}


def test_validate_records_rejects_over_envelope():
    records = [_record(i, f"prompt {i}") for i in range(33)]
    with pytest.raises(MicroSplitValidationError):
        validate_records(records, max_records=32)


def test_validate_records_rejects_duplicate_prompts():
    records = [_record(0, "same"), _record(1, "same")]
    with pytest.raises(MicroSplitValidationError):
        validate_records(records)


def test_validate_records_rejects_missing_field():
    record = dict(RECORD_TEMPLATE)
    del record["qa_type"]
    with pytest.raises(MicroSplitValidationError):
        validate_records([record])
