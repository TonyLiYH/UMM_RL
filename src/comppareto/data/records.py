"""Common JSONL record schema shared by every D1-D4 manifest builder.

Every manifest emitted by this package is a sequence of these records
serialized one JSON object per line (JSON Lines). The schema is
intentionally source-agnostic: a validator downstream of this module never
needs to special-case COCO vs. LLaVA vs. DiffusionDB shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VALID_ROLES = frozenset(
    {"D1_paired", "D2_understanding", "D3_generation", "D4_diagnostic", "D4_evaluation"}
)
VALID_SPLITS = frozenset(
    {"diagnostic", "pilot_train", "pilot_validation", "pilot_meta", "evaluation_only"}
)
VALID_TASK_DIRECTIONS = frozenset({"i2t", "t2i", "vqa"})


@dataclass(frozen=True)
class ImageRef:
    """A reference to media by path/URL -- never embedded bytes."""

    source_dataset: str
    source_relative_path: str


@dataclass(frozen=True)
class Record:
    record_id: str
    source: str
    role: str
    split: str
    group_key: str
    task_directions: tuple[str, ...]
    license_tag: str
    source_native_id: str
    image: ImageRef
    text: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def validate_record(payload: dict[str, Any]) -> list[str]:
    """Return a list of schema-violation messages; empty means valid."""
    errors: list[str] = []
    required = (
        "record_id",
        "source",
        "role",
        "split",
        "group_key",
        "task_directions",
        "license_tag",
        "source_native_id",
        "image",
        "text",
    )
    for key in required:
        if key not in payload:
            errors.append(f"missing field: {key}")
    if errors:
        return errors
    if payload["role"] not in VALID_ROLES:
        errors.append(f"invalid role: {payload['role']!r}")
    if payload["split"] not in VALID_SPLITS:
        errors.append(f"invalid split: {payload['split']!r}")
    directions = payload["task_directions"]
    if not isinstance(directions, (list, tuple)) or not directions:
        errors.append("task_directions must be a non-empty list")
    else:
        for direction in directions:
            if direction not in VALID_TASK_DIRECTIONS:
                errors.append(f"invalid task_direction: {direction!r}")
    image = payload["image"]
    if not isinstance(image, dict) or not image.get("source_relative_path"):
        errors.append("image.source_relative_path must be a non-empty string")
    if not isinstance(payload["record_id"], str) or not payload["record_id"]:
        errors.append("record_id must be a non-empty string")
    if payload["split"] == "evaluation_only" and payload["role"] != "D4_evaluation":
        errors.append("evaluation_only split must carry role D4_evaluation")
    return errors
