"""DiffusionDB manifest builder -- the D3 diverse text-to-image extension.

DiffusionDB (``poloclub/diffusiondb``, CC0-1.0) replaces the task's original
JourneyDB candidate, which requires accepting a gated, non-commercial-only,
no-redistribution usage agreement tied to a named individual's identity --
see ``reports/T260/source-license-audit.md`` for the full admission
decision. DiffusionDB is fully ungated and CC0, so both its prompts and its
image references may be used and redistributed as manifest metadata without
restriction.

This module reads a plain JSON Lines *cache* of the official
``metadata.parquet`` file's rows (one JSON object per row, produced once by
``python3 -m comppareto.data.prep_diffusiondb`` using ``pandas``/``pyarrow``,
lazily imported only inside that CLI's ``main()`` -- so the installed
``comppareto`` package itself never requires a Parquet toolchain, and
``.venv`` need not list ``pandas``/``pyarrow`` as dependencies) rather than
parsing Parquet directly in this module.

Deterministic subsampling: DiffusionDB has 2,000,000 rows in the frozen
random 2m metadata file; ingesting all of them into a single pilot manifest
would dominate the other two sources and bloat the frozen manifest well
beyond an auditable pilot subset. :func:`iter_records` keeps a row only when
``group_bucket(image_name, num_buckets=KEEP_MODULUS) < KEEP_THRESHOLD`` --
a pure function of the row's own immutable ``image_name``, never of any
model outcome -- and only after the row clears the recorded safety
thresholds on ``image_nsfw``/``prompt_nsfw``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .ids import group_bucket, stable_id
from .split import assign_split

LICENSE_TAG = "cc0-1.0"
SOURCE_NAME = "diffusiondb_2m"

#: Keep roughly KEEP_THRESHOLD / KEEP_MODULUS of rows (1/50 = 2%).
KEEP_MODULUS = 50
KEEP_THRESHOLD = 1

IMAGE_NSFW_MAX = 0.2
PROMPT_NSFW_MAX = 0.5


def _passes_safety_filter(row: dict[str, Any]) -> bool:
    image_nsfw = row.get("image_nsfw")
    prompt_nsfw = row.get("prompt_nsfw")
    if image_nsfw is None or prompt_nsfw is None:
        # DiffusionDB's own NSFW scorer leaves some rows unscored; without a
        # score we cannot clear the safety threshold, so exclude the row
        # rather than assume it is safe.
        return False
    return image_nsfw < IMAGE_NSFW_MAX and prompt_nsfw < PROMPT_NSFW_MAX


def _keep_by_hash(image_name: str) -> bool:
    return group_bucket(image_name, num_buckets=KEEP_MODULUS) < KEEP_THRESHOLD


def iter_records(cache_jsonl_path: Path) -> Iterator[dict[str, Any]]:
    """Yield deterministic D3 records from the DiffusionDB metadata cache."""
    with cache_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_name = row["image_name"]
            if not _keep_by_hash(image_name):
                continue
            if not _passes_safety_filter(row):
                continue
            split = assign_split(image_name)
            part_id = int(row["part_id"])
            yield {
                "record_id": stable_id("d3-diffusiondb-prompt", image_name),
                "source": SOURCE_NAME,
                "role": "D3_generation",
                "split": split,
                "group_key": image_name,
                "task_directions": ["t2i"],
                "license_tag": LICENSE_TAG,
                "source_native_id": image_name,
                "image": {
                    "source_dataset": "diffusiondb_2m",
                    "source_relative_path": f"images/part-{part_id:06d}.zip::{image_name}",
                },
                "text": {
                    "prompt": row["prompt"],
                    "part_id": part_id,
                    "seed": int(row["seed"]),
                    "cfg": float(row["cfg"]),
                    "sampler": row["sampler"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "image_nsfw": row["image_nsfw"],
                    "prompt_nsfw": row["prompt_nsfw"],
                },
            }
