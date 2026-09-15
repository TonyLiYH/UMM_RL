"""Orchestrate the five frozen pilot manifests and admission run evidence.

Reads the three admitted sources' raw metadata (never full media), builds
every D1-D4 record via ``comppareto.data.{coco,llava,diffusiondb}``, checks
schema validity and split-disjointness via ``comppareto.data.{records,dedup}``,
and writes:

* ``configs/data/posttraining-v1/{diagnostic,pilot_train,pilot_validation,
  pilot_meta}.jsonl``
* ``configs/data/posttraining-v1/evaluation_only.yaml``
* ``runs/data-admission-posttraining-v1/{manifest.json,metrics.json,notes.md}``

This module performs no model inference, no gradient computation, and no
training of any kind; it only reads text/JSON/Parquet-derived metadata and
writes deterministic manifests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from . import coco, diffusiondb, llava
from .dedup import cross_split_duplicate_groups, duplicate_record_ids, evaluation_records_in_training
from .records import validate_record

JSONL_SPLITS = ("diagnostic", "pilot_train", "pilot_validation", "pilot_meta")


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


def split_records(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in (*JSONL_SPLITS, "evaluation_only")}
    for record in records:
        by_split[record["split"]].append(record)
    return by_split


def write_manifests(by_split: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name in JSONL_SPLITS:
        path = output_dir / f"{split_name}.jsonl"
        rows = sorted(by_split[split_name], key=lambda r: r["record_id"])
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
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


def compute_metrics(all_records: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    d1_records = [r for r in all_records if r["role"] == "D1_paired"]
    bidirectional_ok = bool(d1_records) and all(
        {"i2t", "t2i"} <= set(r["task_directions"]) for r in d1_records
    )
    duplicate_ids = duplicate_record_ids(all_records)
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
        },
        "pilot": {
            "usable_task_records": counts["pilot_train"],
            "total_task_records": sum(counts[name] for name in JSONL_SPLITS),
        },
        "duplicate_record_ids": len(duplicate_ids),
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
    args = parser.parse_args(argv)

    inputs = BuildInputs(
        coco_zip=args.coco_zip,
        llava_json=args.llava_json,
        diffusiondb_cache=args.diffusiondb_cache,
    )
    all_records = build_all_records(inputs)
    by_split = split_records(all_records)
    counts = write_manifests(by_split, args.output_dir)
    metrics = compute_metrics(all_records, counts)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
