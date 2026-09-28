from __future__ import annotations

import json
import zipfile

import yaml

from comppareto.data import build, coco, diffusiondb


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

    kept_part_id = diffusiondb.kept_part_ids()[0]
    diffusiondb_cache = tmp_path / "metadata.jsonl"
    with diffusiondb_cache.open("w", encoding="utf-8") as handle:
        for i in range(200):
            row = {
                "image_name": f"ddb-{i:06d}.png",
                "prompt": f"prompt {i}",
                "part_id": kept_part_id,
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
    assert (
        metrics["paired_core"]["diagnostic_d1_paired_record_count"]
        <= metrics["paired_core"]["diagnostic_total_record_count"]
    )
    assert metrics["media"]["metadata_admitted_records"] == len(all_records)
    assert metrics["near_duplicates"]["scoped_splits"] == list(("diagnostic", "pilot_validation", "pilot_meta"))

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


def test_shard_rows_splits_at_the_declared_byte_ceiling() -> None:
    rows = [{"record_id": f"r{i:05d}", "payload": "x" * 100} for i in range(1000)]
    one_row_bytes = len(build._encode_row(rows[0]))
    max_bytes = one_row_bytes * 10  # force several small shards deterministically
    shards = build._shard_rows(rows, max_bytes=max_bytes)
    assert sum(len(shard) for shard in shards) == len(rows)
    assert [row for shard in shards for row in shard] == rows
    for shard in shards[:-1]:
        assert len(shard) <= 10


def test_shard_rows_never_returns_empty_list_even_for_no_rows() -> None:
    assert build._shard_rows([]) == [[]]


def test_write_manifests_writes_pilot_train_shard_index(tmp_path) -> None:
    inputs = _make_repo(tmp_path)
    all_records = build.build_all_records(inputs)
    by_split = build.split_records(all_records)
    output_dir = tmp_path / "configs"
    counts = build.write_manifests(by_split, output_dir)

    index_path = output_dir / "pilot_train.shards.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["total_records"] == counts["pilot_train"]
    assert index["shards"][0]["path"] == "pilot_train.jsonl"
    assert sum(shard["record_count"] for shard in index["shards"]) == counts["pilot_train"]

    # pilot_train.jsonl (shard 0) must always exist at the literal path the
    # acceptance contract requires, even when there is exactly one shard.
    assert (output_dir / "pilot_train.jsonl").exists()
    for shard in index["shards"]:
        assert (output_dir / shard["path"]).exists()
        actual_bytes = (output_dir / shard["path"]).read_bytes()
        assert len(actual_bytes) == shard["bytes"]


def test_run_media_availability_sample_patches_only_sampled_records(tmp_path) -> None:
    inputs = _make_repo(tmp_path)
    all_records = build.build_all_records(inputs)

    def _fake_probe(_path: str) -> dict:
        return {"available": True, "bytes": 123, "sha256": "deadbeef"}

    fake_probes = {"coco_train2017": _fake_probe, "coco_val2017": _fake_probe, "diffusiondb_2m": _fake_probe}
    patched, results = build.run_media_availability_sample(all_records, probes=fake_probes)

    assert len(results) > 0
    assert all(r["available"] for r in results)
    materialized = [r for r in patched if r["image"]["media_materialized"]]
    assert len(materialized) == len(results)
    for record in materialized:
        assert record["image"]["media_sha256"] == "deadbeef"
        assert record["image"]["media_bytes"] == 123

    by_split = build.split_records(patched)
    counts = {name: len(rows) for name, rows in by_split.items()}
    metrics = build.compute_metrics(patched, counts, media_probe_results=results)
    assert metrics["media"]["media_materialized_records"] == len(results)
    assert metrics["media"]["media_availability_sample_size"] == len(results)
    assert metrics["media"]["media_availability_sample_confirmed_available"] == len(results)
