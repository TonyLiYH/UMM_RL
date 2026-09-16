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

#: Shared default for sources with no project-level training-target
#: restriction (COCO captions, DiffusionDB prompts). LLaVA overrides this
#: with an explicit restricted decision -- see
#: :mod:`comppareto.data.llava`'s ``TRAINING_CONSTRAINTS``.
UNRESTRICTED_TRAINING_CONSTRAINTS: dict[str, Any] = {
    "restricted_as_training_target": False,
    "note": None,
}


@dataclass(frozen=True)
class ImageRef:
    """A reference to media by path/URL -- never embedded bytes.

    ``metadata_admitted`` is ``True`` for every record in a frozen
    manifest (the record's metadata -- path, ids, captions/prompts -- has
    been audited and admitted). ``media_materialized`` is a *separate*,
    independently auditable claim: it is ``True`` only for the small,
    deterministic verification sample whose actual image bytes were
    fetched and hashed this run (see
    ``comppareto.data.media_check``/``reports/T260/media-availability-check.md``).
    Most COCO/LLaVA/DiffusionDB rows are ``metadata_admitted=True`` but
    ``media_materialized=False`` -- their bytes were never downloaded,
    only planned (see
    ``reports/T260/media-materialization-plan.md``) -- and this module
    requires that distinction to be explicit in every record rather than
    left to prose.
    """

    source_dataset: str
    source_relative_path: str
    metadata_admitted: bool = True
    media_materialized: bool = False
    media_verified_available: bool | None = None
    media_sha256: str | None = None
    media_bytes: int | None = None


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
    training_constraints: dict[str, Any] = field(
        default_factory=lambda: {"restricted_as_training_target": False, "note": None}
    )

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
        "training_constraints",
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
    else:
        if "metadata_admitted" not in image or "media_materialized" not in image:
            errors.append(
                "image must carry explicit metadata_admitted/media_materialized flags"
            )
        elif not isinstance(image["metadata_admitted"], bool) or not isinstance(
            image["media_materialized"], bool
        ):
            errors.append("image.metadata_admitted/media_materialized must be booleans")
        elif image["media_materialized"] and (
            not image.get("media_sha256") or not image.get("media_bytes")
        ):
            errors.append(
                "media_materialized=True requires a recorded media_sha256 and media_bytes"
            )
        elif not image["media_materialized"] and (
            image.get("media_sha256") is not None or image.get("media_bytes") is not None
        ):
            errors.append(
                "media_materialized=False must not carry a media_sha256/media_bytes value"
            )
    if not isinstance(payload["record_id"], str) or not payload["record_id"]:
        errors.append("record_id must be a non-empty string")
    if payload["split"] == "evaluation_only" and payload["role"] != "D4_evaluation":
        errors.append("evaluation_only split must carry role D4_evaluation")
    constraints = payload["training_constraints"]
    if not isinstance(constraints, dict) or "restricted_as_training_target" not in constraints:
        errors.append(
            "training_constraints.restricted_as_training_target must be present "
            "(explicit, per-record project decision -- see reports/T260/"
            "source-license-audit.md Sec. LLaVA GPT-terms decision)"
        )
    elif not isinstance(constraints["restricted_as_training_target"], bool):
        errors.append("training_constraints.restricted_as_training_target must be a boolean")
    elif constraints["restricted_as_training_target"] and not constraints.get("note"):
        errors.append("a restricted training_constraints entry must carry an explanatory note")
    return errors
