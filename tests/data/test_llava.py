from __future__ import annotations

import json

from comppareto.data import llava


def _write_fixture(path) -> None:
    entries = [
        {
            "id": "000000033471",
            "image": "000000033471.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is in this photo?"},
                {"from": "gpt", "value": "A cat sitting on a windowsill."},
                {"from": "human", "value": "What color is it?"},
                {"from": "gpt", "value": "It looks orange and white."},
            ],
        },
        {
            "id": "000000033472",
            "image": "000000033472.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe the scene."},
                {"from": "gpt", "value": "A busy street with several cars."},
            ],
        },
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_iter_records_basic_shape(tmp_path) -> None:
    path = tmp_path / "llava_instruct_150k.json"
    _write_fixture(path)
    records = list(llava.iter_records(path))
    assert len(records) == 2
    for record in records:
        assert record["role"] == "D2_understanding"
        assert record["source"] == llava.SOURCE_NAME
        assert record["license_tag"] == llava.LICENSE_TAG
        assert record["task_directions"] == ["vqa"]
        assert record["image"]["source_dataset"] == "coco_train2017"


def test_iter_records_group_key_matches_normalized_coco_image_id(tmp_path) -> None:
    path = tmp_path / "llava_instruct_150k.json"
    _write_fixture(path)
    records = {r["source_native_id"]: r for r in llava.iter_records(path)}
    assert records["000000033471"]["group_key"] == "33471"


def test_iter_records_preview_strips_image_token_and_truncates(tmp_path) -> None:
    path = tmp_path / "llava_instruct_150k.json"
    _write_fixture(path)
    records = list(llava.iter_records(path))
    first = records[0]["text"]
    assert "<image>" not in first["first_human_turn"]
    assert first["first_human_turn"].startswith("What is in this photo?")
    assert first["first_assistant_turn"] == "A cat sitting on a windowsill."
    assert first["num_turns"] == 4


def test_iter_records_shares_split_with_coco_group_key(tmp_path) -> None:
    from comppareto.data.split import assign_split

    path = tmp_path / "llava_instruct_150k.json"
    _write_fixture(path)
    records = {r["group_key"]: r for r in llava.iter_records(path)}
    assert records["33471"]["split"] == assign_split("33471")
