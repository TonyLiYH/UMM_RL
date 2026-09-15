from __future__ import annotations

import json
import zipfile

import yaml

from comppareto.data import build, coco


def _make_repo(tmp_path):
    coco_zip = tmp_path / "annotations_trainval2017.zip"
    train_payload = {
        "images": [{"id": i, "file_name": f"{i:012d}.jpg"} for i in range(1, 401)],
        "annotations": [
            {"id": i * 10, "image_id": i, "caption": f"caption for image {i}"}
            for i in range(1, 401)
        ],
    }
    val_payload = {
        "images": [{"id": 90000 + i, "file_name": f"{90000 + i:012d}.jpg"} for i in range(1, 21)],
        "annotations": [
            {"id": (90000 + i) * 10, "image_id": 90000 + i, "caption": f"eval caption {i}"}
            for i in range(1, 21)
        ],
    }
    with zipfile.ZipFile(coco_zip, "w") as archive:
        archive.writestr(coco.TRAIN_ANNOTATION_MEMBER, json.dumps(train_payload))
        archive.writestr(coco.VAL_ANNOTATION_MEMBER, json.dumps(val_payload))

    llava_json = tmp_path / "llava_instruct_150k.json"
    llava_entries = [
        {
            "id": f"{i:012d}",
            "image": f"{i:012d}.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is happening?"},
                {"from": "gpt", "value": f"Description {i}"},
            ],
        }
        for i in range(1, 51)
    ]
    llava_json.write_text(json.dumps(llava_entries), encoding="utf-8")

    diffusiondb_cache = tmp_path / "metadata.jsonl"
    with diffusiondb_cache.open("w", encoding="utf-8") as handle:
        for i in range(200):
            row = {
                "image_name": f"ddb-{i:06d}.png",
                "prompt": f"prompt {i}",
                "part_id": 1,
                "seed": i,
                "cfg": 7.0,
                "sampler": 8,
                "width": 512,
                "height": 512,
                "image_nsfw": 0.0,
                "prompt_nsfw": 0.0,
            }
            handle.write(json.dumps(row))
            handle.write("\n")

    return build.BuildInputs(coco_zip=coco_zip, llava_json=llava_json, diffusiondb_cache=diffusiondb_cache)


def test_end_to_end_build_produces_disjoint_valid_splits(tmp_path) -> None:
    inputs = _make_repo(tmp_path)
    all_records = build.build_all_records(inputs)
    by_split = build.split_records(all_records)
    output_dir = tmp_path / "configs"
    counts = build.write_manifests(by_split, output_dir)
    metrics = build.compute_metrics(all_records, counts)

    assert metrics["splits"]["cross_split_duplicate_groups"] == 0
    assert metrics["splits"]["evaluation_records_in_training"] == 0
    assert metrics["paired_core"]["bidirectional_mapping_verified"] is True
    assert metrics["resources"]["gpu_hours"] == 0
    assert metrics["duplicate_record_ids"] == 0

    for split_name in build.JSONL_SPLITS:
        path = output_dir / f"{split_name}.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == counts[split_name]
        for line in lines:
            json.loads(line)  # every line must be valid JSON

    eval_path = output_dir / "evaluation_only.yaml"
    payload = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    assert payload["record_count"] == counts["evaluation_only"]
    assert payload["record_count"] == 20


def test_build_is_deterministic_across_runs(tmp_path) -> None:
    inputs = _make_repo(tmp_path)
    first = build.build_all_records(inputs)
    second = build.build_all_records(inputs)
    assert first == second


def test_llava_and_coco_share_image_lands_in_same_split(tmp_path) -> None:
    inputs = _make_repo(tmp_path)
    all_records = build.build_all_records(inputs)
    by_group: dict[str, set[str]] = {}
    for record in all_records:
        if record["source"] in ("coco_captions_2017", "llava_instruct_150k") and record["split"] != "evaluation_only":
            by_group.setdefault(record["group_key"], set()).add(record["split"])
    shared_groups = [g for g, splits in by_group.items() if len(splits) > 1]
    assert shared_groups == []


def test_apply_pilot_train_cap_drops_only_overflow_pilot_train_buckets() -> None:
    # group_key "0" hashes to bucket 2705 (< PILOT_TRAIN_BUCKET_CEILING=5727,
    # so it is kept); group_key "1" hashes to bucket 8030 (>= 5727, so it is
    # the deterministic, evidence-backed overflow this cap drops). Both
    # buckets fall inside pilot_train's own [863, 10000) range, so this
    # exercises the cap itself, not the pilot_train/other-split boundary.
    kept_record = {"split": "pilot_train", "group_key": "0", "id": "kept"}
    dropped_record = {"split": "pilot_train", "group_key": "1", "id": "dropped"}
    other_split_record = {"split": "diagnostic", "group_key": "1", "id": "other-split"}

    result = build._apply_pilot_train_cap([kept_record, dropped_record, other_split_record])

    assert kept_record in result
    assert dropped_record not in result
    # Records outside pilot_train are never considered by the cap, even if
    # their own group_key would otherwise land above the ceiling.
    assert other_split_record in result
