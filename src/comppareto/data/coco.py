"""COCO captions (2017) manifest builder -- the D1 paired semantic core.

Reads ``captions_{train,val}2017.json`` directly out of the official
``annotations_trainval2017.zip`` (no extraction to disk is required). Builds:

* D1 ``train2017``-derived records, split deterministically into
  ``diagnostic`` / ``pilot_train`` / ``pilot_validation`` / ``pilot_meta``
  by :func:`comppareto.data.split.assign_split` keyed on the COCO
  ``image_id`` -- the same group key :mod:`comppareto.data.llava` uses, so
  an image shared between D1 and D2 always lands in the same split.
* D4 evaluation-only records from the official, disjoint ``val2017`` split
  -- disjointness from ``train2017`` is guaranteed by COCO's own dataset
  construction, not by this project's hashing.

Each image contributes at most ``captions_per_image`` records (the
lowest-numbered official annotation ids, chosen deterministically) to keep
the frozen manifest a bounded, auditable pilot subset rather than the full
~600k-caption ingest.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterator
import zipfile

from .ids import stable_id
from .records import UNRESTRICTED_TRAINING_CONSTRAINTS
from .split import assign_split

TRAIN_ANNOTATION_MEMBER = "annotations/captions_train2017.json"
VAL_ANNOTATION_MEMBER = "annotations/captions_val2017.json"
LICENSE_TAG = "cc-by-4.0"
SOURCE_NAME = "coco_captions_2017"


def _load_captions(zip_path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            return json.load(handle)


def _group_by_image(payload: dict[str, Any]) -> tuple[dict[int, str], dict[int, list[tuple[int, str]]]]:
    filenames: dict[int, str] = {
        image["id"]: image["file_name"] for image in payload["images"]
    }
    captions: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        captions[annotation["image_id"]].append((annotation["id"], annotation["caption"]))
    for image_id in captions:
        captions[image_id].sort(key=lambda pair: pair[0])
    return filenames, captions


def iter_train_records(
    zip_path: Path,
    *,
    captions_per_image: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic D1 records built from COCO ``train2017`` captions."""
    payload = _load_captions(zip_path, TRAIN_ANNOTATION_MEMBER)
    filenames, captions = _group_by_image(payload)
    for image_id in sorted(filenames):
        group_key = str(image_id)
        split = assign_split(group_key)
        for caption_id, caption_text in captions.get(image_id, [])[:captions_per_image]:
            native_key = f"{image_id}:{caption_id}"
            yield {
                "record_id": stable_id("d1-coco-caption", native_key),
                "source": SOURCE_NAME,
                "role": "D1_paired",
                "split": split,
                "group_key": group_key,
                "task_directions": ["i2t", "t2i"],
                "license_tag": LICENSE_TAG,
                "source_native_id": native_key,
                "image": {
                    "source_dataset": "coco_train2017",
                    "source_relative_path": f"train2017/{filenames[image_id]}",
                    "metadata_admitted": True,
                    "media_materialized": False,
                    "media_verified_available": None,
                    "media_sha256": None,
                    "media_bytes": None,
                },
                "text": {"caption": caption_text, "annotation_id": caption_id},
                "training_constraints": dict(UNRESTRICTED_TRAINING_CONSTRAINTS),
            }


def iter_evaluation_records(
    zip_path: Path,
    *,
    captions_per_image: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic D4 evaluation-only records from COCO ``val2017``."""
    payload = _load_captions(zip_path, VAL_ANNOTATION_MEMBER)
    filenames, captions = _group_by_image(payload)
    for image_id in sorted(filenames):
        group_key = str(image_id)
        for caption_id, caption_text in captions.get(image_id, [])[:captions_per_image]:
            native_key = f"{image_id}:{caption_id}"
            yield {
                "record_id": stable_id("d4-coco-eval-caption", native_key),
                "source": SOURCE_NAME,
                "role": "D4_evaluation",
                "split": "evaluation_only",
                "group_key": group_key,
                "task_directions": ["i2t", "t2i"],
                "license_tag": LICENSE_TAG,
                "source_native_id": native_key,
                "image": {
                    "source_dataset": "coco_val2017",
                    "source_relative_path": f"val2017/{filenames[image_id]}",
                    "metadata_admitted": True,
                    "media_materialized": False,
                    "media_verified_available": None,
                    "media_sha256": None,
                    "media_bytes": None,
                },
                "text": {"caption": caption_text, "annotation_id": caption_id},
                "training_constraints": dict(UNRESTRICTED_TRAINING_CONSTRAINTS),
            }
