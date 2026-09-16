"""Orchestrate the five frozen pilot manifests and admission run evidence.

Reads the three admitted sources' raw metadata (never full media), builds
every D1-D4 record via ``comppareto.data.{coco,llava,diffusiondb}``, checks
schema validity and split-disjointness via ``comppareto.data.{records,dedup}``,
runs a small preregistered real media-availability sample
(``comppareto.data.media_check``), and writes:

* ``configs/data/posttraining-v1/diagnostic.jsonl``
* ``configs/data/posttraining-v1/pilot_train.jsonl`` (shard 0) plus, if the
  full ``pilot_train`` split does not fit in one sub-50MB shard,
  ``pilot_train.shard{N}.jsonl`` (N >= 1) and a companion
  ``pilot_train.shards.json`` hash-addressed index -- see
  :func:`_shard_rows`.
* ``configs/data/posttraining-v1/{pilot_validation,pilot_meta}.jsonl``
* ``configs/data/posttraining-v1/evaluation_only.yaml``
* ``runs/data-admission-posttraining-v1/{manifest.json,metrics.json,notes.md}``

This module performs no model inference, no gradient computation, and no
training of any kind; it only reads text/JSON/Parquet-derived metadata,
performs a bounded number of real HTTP(S) reads of already-admitted media
for verification, and writes deterministic manifests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from . import coco, diffusiondb, llava, media_check, near_dup
from .dedup import cross_split_duplicate_groups, duplicate_record_ids, evaluation_records_in_training
from .records import validate_record

JSONL_SPLITS = ("diagnostic", "pilot_train", "pilot_validation", "pilot_meta")

#: Soft per-shard byte ceiling for ``pilot_train`` (see :func:`_shard_rows`).
#: 40MB leaves comfortable headroom under both GitHub's 100MB hard push
#: limit and its 50MB "large file" warning threshold, so a single shard
#: growing by a normal amount between rebuilds will not suddenly cross
#: either limit.
PILOT_TRAIN_SHARD_MAX_BYTES = 40_000_000


@dataclass(frozen=True)
class BuildInputs:
    coco_zip: Path
    llava_json: Path
    diffusiondb_cache: Path


def _all_training_pool_records(inputs: BuildInputs) -> Iterator[dict[str, Any]]:
    yield from coco.iter_train_records(inputs.coco_zip)
    yield from llava.iter_records(inputs.llava_json)
    yield from diffusiondb.iter_records(inputs.diffusiondb_cache)


def _all_evaluation_records(inputs: BuildInputs) -> Iterator[dict[str, Any]]:
    yield from coco.iter_evaluation_records(inputs.coco_zip)


def build_all_records(inputs: BuildInputs) -> list[dict[str, Any]]:
    records = list(_all_training_pool_records(inputs)) + list(_all_evaluation_records(inputs))
    errors: list[str] = []
    for record in records:
        errors.extend(validate_record(record))
    if errors:
        raise ValueError("schema validation failed:\n" + "\n".join(errors[:50]))
    return records


def run_media_availability_sample(
    all_records: list[dict[str, Any]],
    *,
    probes: dict[str, media_check.ProbeFn] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Probe a small preregistered real sample and patch its media fields.

    Returns ``(patched_all_records, probe_results)``. When ``probes`` is
    omitted, :data:`comppareto.data.media_check.DEFAULT_PROBES` is used,
    which performs real network reads -- pass an explicit (fake) ``probes``
    dict to exercise this function without network access, as the test
    suite does.
    """
    sample = media_check.select_sample(all_records)
    results = media_check.run_media_checks(
        sample, probes=probes if probes is not None else media_check.DEFAULT_PROBES
    )
    patched = media_check.apply_probe_results(all_records, results)
    return patched, results


def split_records(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in (*JSONL_SPLITS, "evaluation_only")}
    for record in records:
        by_split[record["split"]].append(record)
    return by_split


def _encode_row(row: dict[str, Any]) -> bytes:
    # Compact separators (no space after "," / ":") -- pure
    # serialization-density change, not a schema or content change. Every
    # record's field set, values, and byte-for-byte content (aside from
    # JSON whitespace) are unchanged; see ``reports/T260/failure-ledger.md``.
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _shard_rows(
    rows: list[dict[str, Any]], *, max_bytes: int = PILOT_TRAIN_SHARD_MAX_BYTES
) -> list[list[dict[str, Any]]]:
    """Split ``rows`` (already in a fixed, deterministic order) into shards.

    Deterministic and content-only: a shard boundary falls exactly where
    adding the next row would push the *current* shard's encoded byte total
    past ``max_bytes`` -- never influenced by row count alone, wall-clock
    time, or any model outcome. Rerunning against the same frozen input
    rows always produces the same shard boundaries.
    """
    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for row in rows:
        row_bytes = len(_encode_row(row))
        if current and current_bytes + row_bytes > max_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(row)
        current_bytes += row_bytes
    if current or not shards:
        shards.append(current)
    return shards


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as handle:
        for row in rows:
            encoded = _encode_row(row)
            handle.write(encoded)
            digest.update(encoded)
            size += len(encoded)
    return size, digest.hexdigest()


def _write_pilot_train_shards(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write ``pilot_train`` as shard files plus a hash-addressed index.

    Local review item 5: a single 86-161MB monolithic ``pilot_train.jsonl``
    either brushes against or exceeds GitHub's 100MB per-blob push limit.
    Shard 0 is still written to the exact literal path
    ``pilot_train.jsonl`` the acceptance contract requires to exist;
    any additional shards go to ``pilot_train.shard{N}.jsonl`` (N >= 1).
    ``pilot_train.shards.json`` is the external hash-addressed index the
    review asked for: every shard's path, record count, byte size, and
    sha256 -- sufficient to verify any shard's integrity, or reconstruct
    the full split by concatenation in index order, without ever needing
    a single monolithic file on disk.
    """
    shard_groups = _shard_rows(rows)
    shard_meta: list[dict[str, Any]] = []
    for index, shard_rows in enumerate(shard_groups):
        shard_path = output_dir / ("pilot_train.jsonl" if index == 0 else f"pilot_train.shard{index}.jsonl")
        size, sha256 = _write_jsonl(shard_path, shard_rows)
        shard_meta.append(
            {
                "index": index,
                "path": shard_path.name,
                "record_count": len(shard_rows),
                "bytes": size,
                "sha256": sha256,
            }
        )
    index_path = output_dir / "pilot_train.shards.json"
    index_path.write_text(
        json.dumps(
            {
                "split": "pilot_train",
                "total_records": len(rows),
                "shard_max_bytes": PILOT_TRAIN_SHARD_MAX_BYTES,
                "shard_count": len(shard_meta),
                "shards": shard_meta,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_manifests(by_split: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name in JSONL_SPLITS:
        rows = sorted(by_split[split_name], key=lambda r: r["record_id"])
        if split_name == "pilot_train":
            _write_pilot_train_shards(rows, output_dir)
        else:
            path = output_dir / f"{split_name}.jsonl"
            _write_jsonl(path, rows)
        counts[split_name] = len(rows)

    evaluation_rows = sorted(by_split["evaluation_only"], key=lambda r: r["record_id"])
    evaluation_path = output_dir / "evaluation_only.yaml"
    evaluation_payload = {
        "split": "evaluation_only",
        "guarantee": (
            "Sourced exclusively from COCO val2017, the dataset's own official "
            "held-out partition disjoint from train2017 by construction; never "
            "used in pilot_train/pilot_validation/pilot_meta/diagnostic."
        ),
        "record_count": len(evaluation_rows),
        "records": evaluation_rows,
    }
    evaluation_path.write_text(
        yaml.safe_dump(evaluation_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    counts["evaluation_only"] = len(evaluation_rows)
    return counts


def compute_metrics(
    all_records: list[dict[str, Any]],
    counts: dict[str, int],
    *,
    media_probe_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    d1_records = [r for r in all_records if r["role"] == "D1_paired"]
    bidirectional_ok = bool(d1_records) and all(
        {"i2t", "t2i"} <= set(r["task_directions"]) for r in d1_records
    )
    duplicate_ids = duplicate_record_ids(all_records)

    diagnostic_records = [r for r in all_records if r["split"] == "diagnostic"]
    diagnostic_d1_records = [r for r in diagnostic_records if r["role"] == "D1_paired"]

    exact_text_dup_groups = near_dup.exact_normalized_text_duplicate_groups(all_records)
    near_dup_pairs = near_dup.near_duplicate_pairs(all_records)

    media_probe_results = media_probe_results or []
    media_probed_available = sum(1 for r in media_probe_results if r.get("available"))
    materialized_count = sum(1 for r in all_records if r["image"]["media_materialized"])

    return {
        "run_id": "data-admission-posttraining-v1",
        "provenance": {
            "admitted_sources": ["coco_captions_2017", "llava_instruct_150k", "diffusiondb_2m"],
            "rejected_sources": ["journeydb"],
            "admitted_sources_without_terms": 0,
        },
        "splits": {
            "counts": counts,
            "cross_split_duplicate_groups": len(cross_split_duplicate_groups(all_records)),
            "evaluation_records_in_training": evaluation_records_in_training(all_records),
        },
        "paired_core": {
            "bidirectional_mapping_verified": bidirectional_ok,
            "d1_record_count": len(d1_records),
            # Local review item 8: report D1-paired examples within
            # `diagnostic` separately from the split's total row count
            # (`splits.counts.diagnostic`, which also includes D2/D3 rows).
            "diagnostic_d1_paired_record_count": len(diagnostic_d1_records),
            "diagnostic_total_record_count": len(diagnostic_records),
        },
        "pilot": {
            "usable_task_records": counts["pilot_train"],
            "total_task_records": sum(counts[name] for name in JSONL_SPLITS),
        },
        "duplicate_record_ids": len(duplicate_ids),
        "media": {
            # Local review item 1: metadata_admitted (every record) vs.
            # media_materialized (only the records whose bytes were
            # actually fetched and hashed this run) are tracked as
            # separate, auditable counts here -- not just prose.
            "metadata_admitted_records": sum(1 for r in all_records if r["image"]["metadata_admitted"]),
            "media_materialized_records": materialized_count,
            "media_availability_sample_size": len(media_probe_results),
            "media_availability_sample_confirmed_available": media_probed_available,
        },
        "near_duplicates": {
            # Local review item 7, scoped per `comppareto.data.near_dup`'s
            # module docstring to diagnostic/pilot_validation/pilot_meta.
            "scoped_splits": list(near_dup.NEAR_DUP_SPLITS),
            "exact_normalized_text_duplicate_group_count": len(exact_text_dup_groups),
            "shingle_jaccard_near_duplicate_pair_count": len(near_dup_pairs),
            "jaccard_threshold": near_dup.JACCARD_THRESHOLD,
        },
        "resources": {
            "gpu_hours": 0,
            "model_inference_calls": 0,
            "training_steps": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-zip", required=True, type=Path)
    parser.add_argument("--llava-json", required=True, type=Path)
    parser.add_argument("--diffusiondb-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metrics-out", required=True, type=Path)
    parser.add_argument("--media-probes-out", type=Path, default=None)
    parser.add_argument(
        "--media-check",
        choices=("run", "skip"),
        default="run",
        help=(
            "'run' (default) performs the real preregistered media-availability "
            "network sample (comppareto.data.media_check); 'skip' leaves every "
            "record's media_materialized=False for a fully offline/hermetic rerun."
        ),
    )
    args = parser.parse_args(argv)

    inputs = BuildInputs(
        coco_zip=args.coco_zip,
        llava_json=args.llava_json,
        diffusiondb_cache=args.diffusiondb_cache,
    )
    all_records = build_all_records(inputs)

    probe_results: list[dict[str, Any]] = []
    if args.media_check == "run":
        all_records, probe_results = run_media_availability_sample(all_records)

    by_split = split_records(all_records)
    counts = write_manifests(by_split, args.output_dir)
    metrics = compute_metrics(all_records, counts, media_probe_results=probe_results)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if args.media_probes_out is not None:
        args.media_probes_out.parent.mkdir(parents=True, exist_ok=True)
        args.media_probes_out.write_text(json.dumps(probe_results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
