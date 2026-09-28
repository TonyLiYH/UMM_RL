from __future__ import annotations

import json

from comppareto.data import diffusiondb


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


def _write_cache(path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _a_kept_part_id() -> int:
    kept = diffusiondb.kept_part_ids()
    assert kept, "expected at least one kept part id"
    return kept[0]


def _a_dropped_part_id() -> int:
    kept = set(diffusiondb.kept_part_ids())
    for part_id in range(1, diffusiondb.PART_MODULUS + 1):
        if part_id not in kept:
            return part_id
    raise AssertionError("expected at least one dropped part id")


def test_kept_part_ids_matches_declared_modulus_and_threshold() -> None:
    kept = diffusiondb.kept_part_ids()
    assert all(1 <= part_id <= diffusiondb.PART_MODULUS for part_id in kept)
    assert kept == sorted(kept)
    assert len(kept) == len(set(kept))
    # Every kept id must independently satisfy _keep_part, and every
    # non-kept id must not -- kept_part_ids() is just an enumeration of
    # the same pure function, not a separate source of truth.
    for part_id in kept:
        assert diffusiondb._keep_part(part_id)
    dropped_sample = [p for p in range(1, diffusiondb.PART_MODULUS + 1) if p not in kept][:5]
    for part_id in dropped_sample:
        assert not diffusiondb._keep_part(part_id)


def test_iter_records_keeps_rows_only_from_kept_parts(tmp_path) -> None:
    kept_part = _a_kept_part_id()
    dropped_part = _a_dropped_part_id()
    cache = tmp_path / "metadata.jsonl"
    _write_cache(
        cache,
        [
            _row("kept-part-row.png", part_id=kept_part),
            _row("dropped-part-row.png", part_id=dropped_part),
        ],
    )
    records = list(diffusiondb.iter_records(cache))
    names = {r["group_key"] for r in records}
    assert "kept-part-row.png" in names
    assert "dropped-part-row.png" not in names


def test_iter_records_keeps_every_safety_passing_row_in_a_kept_part(tmp_path) -> None:
    """Once a part is selected, no additional row-level hash thinning applies."""
    kept_part = _a_kept_part_id()
    cache = tmp_path / "metadata.jsonl"
    rows = [_row(f"row-{i:04d}.png", part_id=kept_part) for i in range(25)]
    _write_cache(cache, rows)
    records = list(diffusiondb.iter_records(cache))
    assert len(records) == 25


def test_iter_records_excludes_rows_failing_safety_filter(tmp_path) -> None:
    kept_part = _a_kept_part_id()
    cache = tmp_path / "metadata.jsonl"
    _write_cache(cache, [_row("x.png", part_id=kept_part, image_nsfw=0.9, prompt_nsfw=0.9)])
    records = list(diffusiondb.iter_records(cache))
    assert records == []


def test_iter_records_excludes_rows_with_missing_nsfw_scores(tmp_path) -> None:
    kept_part = _a_kept_part_id()
    cache = tmp_path / "metadata.jsonl"
    row = _row("x.png", part_id=kept_part)
    row["image_nsfw"] = None
    _write_cache(cache, [row])
    records = list(diffusiondb.iter_records(cache))
    assert records == []


def test_iter_records_shape(tmp_path) -> None:
    kept_part = _a_kept_part_id()
    cache = tmp_path / "metadata.jsonl"
    _write_cache(cache, [_row("x.png", part_id=kept_part)])
    (record,) = list(diffusiondb.iter_records(cache))
    assert record["role"] == "D3_generation"
    assert record["source"] == diffusiondb.SOURCE_NAME
    assert record["license_tag"] == diffusiondb.LICENSE_TAG
    assert record["task_directions"] == ["t2i"]
    assert record["image"]["source_relative_path"] == f"images/part-{kept_part:06d}.zip::x.png"
    assert record["image"]["metadata_admitted"] is True
    assert record["image"]["media_materialized"] is False
    assert record["image"]["media_verified_available"] is None
    assert record["training_constraints"]["restricted_as_training_target"] is False


def test_iter_records_is_deterministic_across_calls(tmp_path) -> None:
    kept_part = _a_kept_part_id()
    cache = tmp_path / "metadata.jsonl"
    _write_cache(cache, [_row(f"row-{i:04d}.png", part_id=kept_part) for i in range(10)])
    first = list(diffusiondb.iter_records(cache))
    second = list(diffusiondb.iter_records(cache))
    assert first == second
