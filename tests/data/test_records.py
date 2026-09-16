from __future__ import annotations

from comppareto.data.records import validate_record


def _valid_record(**overrides):
    record = {
        "record_id": "d1-coco-caption-abc123",
        "source": "coco_captions_2017",
        "role": "D1_paired",
        "split": "pilot_train",
        "group_key": "42",
        "task_directions": ["i2t", "t2i"],
        "license_tag": "cc-by-4.0",
        "source_native_id": "42:7",
        "image": {"source_dataset": "coco_train2017", "source_relative_path": "train2017/x.jpg"},
        "text": {"caption": "a cat"},
    }
    record.update(overrides)
    return record


def test_valid_record_has_no_errors() -> None:
    assert validate_record(_valid_record()) == []


def test_missing_field_is_reported() -> None:
    record = _valid_record()
    del record["license_tag"]
    errors = validate_record(record)
    assert any("license_tag" in e for e in errors)


def test_invalid_role_is_reported() -> None:
    errors = validate_record(_valid_record(role="not_a_role"))
    assert any("invalid role" in e for e in errors)


def test_invalid_split_is_reported() -> None:
    errors = validate_record(_valid_record(split="not_a_split"))
    assert any("invalid split" in e for e in errors)


def test_invalid_task_direction_is_reported() -> None:
    errors = validate_record(_valid_record(task_directions=["not_a_direction"]))
    assert any("invalid task_direction" in e for e in errors)


def test_empty_task_directions_is_reported() -> None:
    errors = validate_record(_valid_record(task_directions=[]))
    assert any("task_directions must be a non-empty list" in e for e in errors)


def test_missing_image_path_is_reported() -> None:
    errors = validate_record(_valid_record(image={"source_dataset": "x", "source_relative_path": ""}))
    assert any("source_relative_path" in e for e in errors)


def test_evaluation_only_requires_evaluation_role() -> None:
    errors = validate_record(_valid_record(split="evaluation_only", role="D1_paired"))
    assert any("evaluation_only split must carry role D4_evaluation" in e for e in errors)


def test_evaluation_only_with_evaluation_role_is_valid() -> None:
    errors = validate_record(_valid_record(split="evaluation_only", role="D4_evaluation"))
    assert errors == []
