from __future__ import annotations

import json

from comppareto.data import diffusiondb
from comppareto.data.ids import group_bucket


def _row(image_name: str, *, image_nsfw=0.0, prompt_nsfw=0.0, part_id=1) -> dict:
    return {
        "image_name": image_name,
        "prompt": f"prompt for {image_name}",
        "part_id": part_id,
        "seed": 42,
        "cfg": 7.0,
        "sampler": 8,
        "width": 512,
        "height": 512,
        "image_nsfw": image_nsfw,
        "prompt_nsfw": prompt_nsfw,
    }


def _first_kept_name(n: int = 2000) -> str:
    for i in range(n):
        name = f"img-{i:06d}.png"
        if group_bucket(name, num_buckets=diffusiondb.KEEP_MODULUS) < diffusiondb.KEEP_THRESHOLD:
            return name
    raise AssertionError("no kept name found in range")


def _first_dropped_name(n: int = 2000) -> str:
    for i in range(n):
        name = f"img-{i:06d}.png"
        if group_bucket(name, num_buckets=diffusiondb.KEEP_MODULUS) >= diffusiondb.KEEP_THRESHOLD:
            return name
    raise AssertionError("no dropped name found in range")


def _write_cache(path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_iter_records_keeps_only_hash_selected_rows(tmp_path) -> None:
    kept_name = _first_kept_name()
    dropped_name = _first_dropped_name()
    cache = tmp_path / "metadata.jsonl"
    _write_cache(cache, [_row(kept_name), _row(dropped_name)])
    records = list(diffusiondb.iter_records(cache))
    names = {r["group_key"] for r in records}
    assert kept_name in names
    assert dropped_name not in names


def test_iter_records_excludes_rows_failing_safety_filter(tmp_path) -> None:
    kept_name = _first_kept_name()
    cache = tmp_path / "metadata.jsonl"
    _write_cache(cache, [_row(kept_name, image_nsfw=0.9, prompt_nsfw=0.9)])
    records = list(diffusiondb.iter_records(cache))
    assert records == []


def test_iter_records_excludes_rows_with_missing_nsfw_scores(tmp_path) -> None:
    kept_name = _first_kept_name()
    cache = tmp_path / "metadata.jsonl"
    row = _row(kept_name)
    row["image_nsfw"] = None
    _write_cache(cache, [row])
    records = list(diffusiondb.iter_records(cache))
    assert records == []


def test_iter_records_shape(tmp_path) -> None:
    kept_name = _first_kept_name()
    cache = tmp_path / "metadata.jsonl"
    _write_cache(cache, [_row(kept_name, part_id=3)])
    (record,) = list(diffusiondb.iter_records(cache))
    assert record["role"] == "D3_generation"
    assert record["source"] == diffusiondb.SOURCE_NAME
    assert record["license_tag"] == diffusiondb.LICENSE_TAG
    assert record["task_directions"] == ["t2i"]
    assert record["image"]["source_relative_path"] == f"images/part-000003.zip::{kept_name}"
