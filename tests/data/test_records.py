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
        "image": {
            "source_dataset": "coco_train2017",
            "source_relative_path": "train2017/x.jpg",
            "metadata_admitted": True,
            "media_materialized": False,
            "media_verified_available": None,
            "media_sha256": None,
            "media_bytes": None,
        },
        "text": {"caption": "a cat"},
        "training_constraints": {"restricted_as_training_target": False, "note": None},
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


def test_image_missing_media_status_flags_is_reported() -> None:
    errors = validate_record(
        _valid_record(image={"source_dataset": "x", "source_relative_path": "y.jpg"})
    )
    assert any("metadata_admitted/media_materialized" in e for e in errors)


def test_image_non_bool_media_status_flags_is_reported() -> None:
    errors = validate_record(
        _valid_record(
            image={
                "source_dataset": "x",
                "source_relative_path": "y.jpg",
                "metadata_admitted": "yes",
                "media_materialized": False,
            }
        )
    )
    assert any("must be booleans" in e for e in errors)


def test_media_materialized_true_requires_hash_and_bytes() -> None:
    errors = validate_record(
        _valid_record(
            image={
                "source_dataset": "x",
                "source_relative_path": "y.jpg",
                "metadata_admitted": True,
                "media_materialized": True,
                "media_sha256": None,
                "media_bytes": None,
            }
        )
    )
    assert any("media_materialized=True requires" in e for e in errors)


def test_media_materialized_true_with_hash_and_bytes_is_valid() -> None:
    errors = validate_record(
        _valid_record(
            image={
                "source_dataset": "x",
                "source_relative_path": "y.jpg",
                "metadata_admitted": True,
                "media_materialized": True,
                "media_verified_available": True,
                "media_sha256": "abc123",
                "media_bytes": 42,
            }
        )
    )
    assert errors == []


def test_media_materialized_false_must_not_carry_hash() -> None:
    errors = validate_record(
        _valid_record(
            image={
                "source_dataset": "x",
                "source_relative_path": "y.jpg",
                "metadata_admitted": True,
                "media_materialized": False,
                "media_sha256": "abc123",
                "media_bytes": None,
            }
        )
    )
    assert any("must not carry a media_sha256/media_bytes" in e for e in errors)


def test_missing_training_constraints_is_reported() -> None:
    record = _valid_record()
    del record["training_constraints"]
    errors = validate_record(record)
    assert any("training_constraints" in e for e in errors)


def test_restricted_training_constraints_without_note_is_reported() -> None:
    errors = validate_record(
        _valid_record(training_constraints={"restricted_as_training_target": True, "note": None})
    )
    assert any("explanatory note" in e for e in errors)


def test_restricted_training_constraints_with_note_is_valid() -> None:
    errors = validate_record(
        _valid_record(
            training_constraints={"restricted_as_training_target": True, "note": "see audit"}
        )
    )
    assert errors == []


def test_evaluation_only_requires_evaluation_role() -> None:
    errors = validate_record(_valid_record(split="evaluation_only", role="D1_paired"))
    assert any("evaluation_only split must carry role D4_evaluation" in e for e in errors)


def test_evaluation_only_with_evaluation_role_is_valid() -> None:
    errors = validate_record(_valid_record(split="evaluation_only", role="D4_evaluation"))
    assert errors == []
