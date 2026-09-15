from __future__ import annotations

from comppareto.data.dedup import (
    cross_split_duplicate_groups,
    duplicate_record_ids,
    evaluation_records_in_training,
)


def _record(record_id: str, group_key: str, split: str) -> dict:
    return {"record_id": record_id, "group_key": group_key, "split": split}


def test_cross_split_duplicate_groups_detects_shared_group_in_two_splits() -> None:
    records = [
        _record("r1", "img-1", "pilot_train"),
        _record("r2", "img-1", "pilot_validation"),
        _record("r3", "img-2", "pilot_train"),
    ]
    assert cross_split_duplicate_groups(records) == ["img-1"]


def test_cross_split_duplicate_groups_empty_when_disjoint() -> None:
    records = [
        _record("r1", "img-1", "pilot_train"),
        _record("r2", "img-2", "pilot_validation"),
    ]
    assert cross_split_duplicate_groups(records) == []


def test_evaluation_records_in_training_counts_overlap() -> None:
    records = [
        _record("r1", "img-1", "evaluation_only"),
        _record("r2", "img-1", "pilot_train"),
        _record("r3", "img-2", "evaluation_only"),
        _record("r4", "img-3", "pilot_validation"),
    ]
    assert evaluation_records_in_training(records) == 1


def test_evaluation_records_in_training_zero_when_disjoint() -> None:
    records = [
        _record("r1", "img-1", "evaluation_only"),
        _record("r2", "img-2", "pilot_train"),
    ]
    assert evaluation_records_in_training(records) == 0


def test_duplicate_record_ids_detects_repeats() -> None:
    records = [
        _record("dup", "img-1", "pilot_train"),
        _record("dup", "img-2", "pilot_validation"),
        _record("unique", "img-3", "pilot_train"),
    ]
    assert duplicate_record_ids(records) == ["dup"]


def test_duplicate_record_ids_empty_when_unique() -> None:
    records = [
        _record("a", "img-1", "pilot_train"),
        _record("b", "img-2", "pilot_train"),
    ]
    assert duplicate_record_ids(records) == []
