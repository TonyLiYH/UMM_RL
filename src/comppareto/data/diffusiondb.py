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

Deterministic subsampling -- archive-fan-out-aware (revised 2026-09-16)
=========================================================================

The prior round's subsampling rule kept a row only by hashing its own
``image_name`` (``group_bucket(image_name, 50) < 1``), independent of which
``part-NNNNNN.zip`` archive it lives in. Measured against the real,
complete 2,000,000-row metadata cache, that rule touched 1,999 of
DiffusionDB's 2,000 part archives (1.24 TB) to retain only 18,339 rows --
an average of ~9 usable rows per ~620 MB archive. A reviewer correctly
flagged this as an impractical media-materialization fan-out: actually
fetching the referenced media would require downloading essentially the
*entire* corpus, exactly what this task's resource envelope forbids.

This module now filters in two stages, in this order:

1. **Archive-level selection** (:func:`_keep_part`): keep a
   ``part_id`` only when
   ``group_bucket(f"diffusiondb-part::{part_id}", num_buckets=PART_MODULUS)
   < PART_KEEP_THRESHOLD`` -- a pure function of the archive's own
   ``part_id`` alone, never of any model outcome. Measured against the
   real DiffusionDB ``images/`` file listing (2,000 parts, real per-file
   byte sizes fetched via the HF tree API, 1,242,204,022,360 bytes total),
   ``PART_KEEP_THRESHOLD = 32`` keeps exactly 25 of the 2,000 archives,
   totaling 15,587,973,934 bytes (~15.6 GB) -- a single bounded download
   if this plan were ever executed, comfortably under this task's 20 GB
   single-step ceiling, and nowhere near "the full source corpus."
2. **Row-level safety filter** (:func:`_passes_safety_filter`, unchanged):
   within a kept archive, every row is still required to clear the
   recorded ``image_nsfw``/``prompt_nsfw`` thresholds.

There is no longer a separate row-level *sampling* hash on top of the
archive selection: once an archive is selected, every safety-passing row
in it is kept, because that is what maximizes rows-per-archive-byte
(the actual scarce resource here). Measured against the real cache with
this rule: 25 archives yield 19,842 retained rows (more than the prior
round's 18,339, from 1,999 archives) -- see
``reports/T260/media-materialization-plan.md`` for the full derivation
and ``reports/T260/mixture-and-accounting.md`` Sec. "DiffusionDB archive
fan-out" for the before/after comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .ids import group_bucket, stable_id
from .records import UNRESTRICTED_TRAINING_CONSTRAINTS
from .split import assign_split

LICENSE_TAG = "cc0-1.0"
SOURCE_NAME = "diffusiondb_2m"

#: Total number of ``images/part-NNNNNN.zip`` archives in the official
#: DiffusionDB 2M-random-subset release (confirmed via the HF tree API,
#: ``part-000001.zip`` .. ``part-002000.zip``).
PART_MODULUS = 2000

#: Keep a part archive iff its hash bucket (out of ``PART_MODULUS``) is
#: below this threshold. 32 measures to exactly 25 real kept archives
#: (~15.6 GB total, real HF-listed sizes) -- see the module docstring and
#: reports/T260/media-materialization-plan.md for the measurement across
#: candidate thresholds (16/24/32/40/50) that led to this choice.
PART_KEEP_THRESHOLD = 32

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


def _keep_part(part_id: int) -> bool:
    """Return whether ``part_id``'s whole archive is in the kept subset.

    A pure function of the archive's own numeric id -- never of any model
    outcome, row content, or download-time observation.
    """
    return group_bucket(f"diffusiondb-part::{part_id}", num_buckets=PART_MODULUS) < PART_KEEP_THRESHOLD


def kept_part_ids() -> list[int]:
    """Return every ``part_id`` in ``[1, PART_MODULUS]`` that :func:`_keep_part` retains.

    Exposed for the media-materialization plan and archive-fan-out report,
    which need the exact archive id list independent of iterating the
    (large, CQ7-resident) metadata cache.
    """
    return [part_id for part_id in range(1, PART_MODULUS + 1) if _keep_part(part_id)]


def iter_records(cache_jsonl_path: Path) -> Iterator[dict[str, Any]]:
    """Yield deterministic D3 records from the DiffusionDB metadata cache."""
    with cache_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            part_id = int(row["part_id"])
            if not _keep_part(part_id):
                continue
            if not _passes_safety_filter(row):
                continue
            image_name = row["image_name"]
            split = assign_split(image_name)
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
                    "metadata_admitted": True,
                    "media_materialized": False,
                    "media_verified_available": None,
                    "media_sha256": None,
                    "media_bytes": None,
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
                "training_constraints": dict(UNRESTRICTED_TRAINING_CONSTRAINTS),
            }
