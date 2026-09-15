"""LLaVA-Instruct-150K manifest builder -- the D2 understanding extension.

Every record's image is a COCO ``train2017`` file (the dataset's own ``id``
field is the zero-padded COCO image id), so this module reuses
:func:`comppareto.data.split.assign_split` on the *same* normalized group key
(``str(int(image_id))``) that :mod:`comppareto.data.coco` uses -- an image
shared between D1 and D2 is therefore always assigned to the same split,
which is what makes the audited group-disjointness guarantee hold across
sources, not just within one source.

Only a bounded preview of each multi-turn conversation is stored in the
manifest (the first human/assistant turn plus a turn count) to keep the
frozen manifest file size auditable; the full conversation is recovered at
load time from the frozen raw ``llava_instruct_150k.json`` file referenced
by hash in ``configs/data/posttraining-v1/sources.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .ids import stable_id
from .split import assign_split

LICENSE_TAG = "cc-by-4.0"
SOURCE_NAME = "llava_instruct_150k"
PREVIEW_CHAR_LIMIT = 240


def _first_turn_preview(conversations: list[dict[str, str]]) -> dict[str, Any]:
    human = next((turn["value"] for turn in conversations if turn.get("from") == "human"), "")
    assistant = next((turn["value"] for turn in conversations if turn.get("from") == "gpt"), "")
    human = human.replace("<image>", "").strip()
    return {
        "first_human_turn": human[:PREVIEW_CHAR_LIMIT],
        "first_assistant_turn": assistant[:PREVIEW_CHAR_LIMIT],
        "num_turns": len(conversations),
    }


def iter_records(json_path: Path) -> Iterator[dict[str, Any]]:
    """Yield deterministic D2 records built from ``llava_instruct_150k.json``.

    Iteration order follows the frozen file's own array order, which is
    itself a fixed artifact (identified by
    ``configs/data/posttraining-v1/sources.yaml``'s recorded sha256) -- no
    additional shuffling or re-sorting is applied, so results are
    reproducible byte-for-byte given the same input file.
    """
    with json_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    for entry in entries:
        image_id = str(int(entry["id"]))
        split = assign_split(image_id)
        native_key = entry["id"]
        yield {
            "record_id": stable_id("d2-llava-instruct", native_key),
            "source": SOURCE_NAME,
            "role": "D2_understanding",
            "split": split,
            "group_key": image_id,
            "task_directions": ["vqa"],
            "license_tag": LICENSE_TAG,
            "source_native_id": native_key,
            "image": {
                "source_dataset": "coco_train2017",
                "source_relative_path": f"train2017/{entry['image']}",
            },
            "text": _first_turn_preview(entry["conversations"]),
        }
