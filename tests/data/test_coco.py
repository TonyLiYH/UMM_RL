from __future__ import annotations

import json
import zipfile

from comppareto.data import coco


def _make_captions_zip(zip_path) -> None:
    train_payload = {
        "images": [
            {"id": 1, "file_name": "000001.jpg"},
            {"id": 2, "file_name": "000002.jpg"},
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "caption": "a dog running"},
            {"id": 102, "image_id": 1, "caption": "a brown dog"},
            {"id": 201, "image_id": 2, "caption": "a red car"},
        ],
    }
    val_payload = {
        "images": [{"id": 9001, "file_name": "900001.jpg"}],
        "annotations": [{"id": 5001, "image_id": 9001, "caption": "a mountain"}],
    }
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(coco.TRAIN_ANNOTATION_MEMBER, json.dumps(train_payload))
        archive.writestr(coco.VAL_ANNOTATION_MEMBER, json.dumps(val_payload))


def test_iter_train_records_one_caption_per_image(tmp_path) -> None:
    zip_path = tmp_path / "annotations.zip"
    _make_captions_zip(zip_path)
    records = list(coco.iter_train_records(zip_path, captions_per_image=1))
    assert len(records) == 2
    image_ids = {r["group_key"] for r in records}
    assert image_ids == {"1", "2"}
    for record in records:
        assert record["role"] == "D1_paired"
        assert record["source"] == coco.SOURCE_NAME
        assert record["license_tag"] == coco.LICENSE_TAG
        assert record["task_directions"] == ["i2t", "t2i"]
        assert record["split"] != "evaluation_only"


def test_iter_train_records_picks_lowest_annotation_id_first(tmp_path) -> None:
    zip_path = tmp_path / "annotations.zip"
    _make_captions_zip(zip_path)
    records = {r["group_key"]: r for r in coco.iter_train_records(zip_path, captions_per_image=1)}
    assert records["1"]["text"]["annotation_id"] == 101


def test_iter_train_records_respects_captions_per_image_cap(tmp_path) -> None:
    zip_path = tmp_path / "annotations.zip"
    _make_captions_zip(zip_path)
    records = list(coco.iter_train_records(zip_path, captions_per_image=2))
    per_image = {}
    for record in records:
        per_image.setdefault(record["group_key"], 0)
        per_image[record["group_key"]] += 1
    assert per_image["1"] == 2
    assert per_image["2"] == 1


def test_iter_evaluation_records_are_disjoint_from_train_ids(tmp_path) -> None:
    zip_path = tmp_path / "annotations.zip"
    _make_captions_zip(zip_path)
    train_ids = {r["group_key"] for r in coco.iter_train_records(zip_path)}
    eval_records = list(coco.iter_evaluation_records(zip_path))
    eval_ids = {r["group_key"] for r in eval_records}
    assert eval_ids.isdisjoint(train_ids)
    for record in eval_records:
        assert record["split"] == "evaluation_only"
        assert record["role"] == "D4_evaluation"


def test_iter_train_records_are_deterministic_across_calls(tmp_path) -> None:
    zip_path = tmp_path / "annotations.zip"
    _make_captions_zip(zip_path)
    first = list(coco.iter_train_records(zip_path))
    second = list(coco.iter_train_records(zip_path))
    assert first == second
