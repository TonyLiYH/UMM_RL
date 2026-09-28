"""One-off DiffusionDB Parquet -> JSON Lines cache conversion.

This module is intentionally *not* imported by :mod:`comppareto.data.diffusiondb`
and does not appear in ``pyproject.toml``'s dependency list: it lazily
imports ``pandas``/``pyarrow`` only inside :func:`main`, so importing this
module (as ``python -m compileall`` and pytest collection both do) never
requires those packages to be installed in ``.venv``. It is meant to be run
once, directly, with whatever Python already has a Parquet toolchain
available (this task used the host's system ``python3``), to produce the
plain-JSON-Lines cache that the tested, dependency-light manifest builder
in :mod:`comppareto.data.diffusiondb` reads.

Usage::

    python3 -m comppareto.data.prep_diffusiondb \\
        --input /path/to/metadata.parquet \\
        --output /path/to/metadata.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_COLUMNS = (
    "image_name",
    "prompt",
    "part_id",
    "seed",
    "cfg",
    "sampler",
    "width",
    "height",
    "image_nsfw",
    "prompt_nsfw",
)


def convert(input_path: Path, output_path: Path) -> int:
    """Convert ``input_path`` (Parquet) to ``output_path`` (JSON Lines).

    Returns the number of rows written. Deterministic: rows are written in
    the Parquet file's own row order, with no shuffling or filtering here
    (filtering is the manifest builder's job, applied at manifest-build
    time so the filter thresholds stay visible and auditable in
    ``comppareto.data.diffusiondb`` rather than silently baked into a cache).
    """
    import pandas as pd  # local import: keeps this module import-safe without pandas installed

    frame = pd.read_parquet(input_path, columns=list(_COLUMNS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            record = {
                "image_name": row.image_name,
                "prompt": row.prompt,
                "part_id": int(row.part_id),
                "seed": int(row.seed),
                "cfg": float(row.cfg),
                "sampler": int(row.sampler),
                "width": int(row.width),
                "height": int(row.height),
                "image_nsfw": None if pd.isna(row.image_nsfw) else float(row.image_nsfw),
                "prompt_nsfw": None if pd.isna(row.prompt_nsfw) else float(row.prompt_nsfw),
            }
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    written = convert(args.input, args.output)
    print(f"wrote {written} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
