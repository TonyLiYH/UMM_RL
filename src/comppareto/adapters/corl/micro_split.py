"""Schema validation for the x2x_rft_22k micro-split JSONL evidence file.

Stdlib-only so it can run in the lightweight local dev environment (no
torch/datasets required) as well as inside the GPU container.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "idx",
    "prompt",
    "qa_problem",
    "qa_solution",
    "qa_type",
    "image_path",
)

VALID_QA_TYPES = {"MC", "OE"}


class MicroSplitValidationError(ValueError):
    pass


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise MicroSplitValidationError(
                    f"{path}:{line_no}: invalid JSON: {error}"
                ) from error
    return records


def validate_records(
    records: list[dict[str, Any]],
    *,
    max_records: int = 32,
    require_unique_prompts: bool = True,
) -> dict[str, Any]:
    """Validate the micro-split contract: schema, size bound, uniqueness.

    Returns a small summary dict; raises MicroSplitValidationError on any
    violation.
    """
    if len(records) > max_records:
        raise MicroSplitValidationError(
            f"micro-split has {len(records)} records, exceeds the "
            f"resource-envelope bound of {max_records} unique records"
        )

    seen_prompts: set[str] = set()
    n_with_image = 0
    qa_type_counts: dict[str, int] = {}

    for i, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise MicroSplitValidationError(
                f"record {i} missing required fields: {missing}"
            )
        prompt = record["prompt"]
        if require_unique_prompts:
            if prompt in seen_prompts:
                raise MicroSplitValidationError(
                    f"record {i} has a duplicate prompt: {prompt!r}"
                )
            seen_prompts.add(prompt)
        if record.get("image_path"):
            n_with_image += 1
        qa_type = record.get("qa_type")
        qa_type_counts[qa_type] = qa_type_counts.get(qa_type, 0) + 1

    return {
        "num_records": len(records),
        "num_unique_prompts": len(seen_prompts),
        "num_with_image": n_with_image,
        "qa_type_counts": qa_type_counts,
    }
