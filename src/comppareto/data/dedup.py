"""Split-disjointness and near-duplicate isolation checks.

"Near-duplicate" here is defined at the group-key level (for example a COCO
``image_id`` or a DiffusionDB ``image_name``): two records are near-duplicate
candidates precisely when they share a group key, because they were either
derived from the same source image or are different annotations of the same
underlying media. The audit gate requires zero *cross-split* group-key
collisions -- a group key may legitimately appear many times (for example
five COCO captions of one image), but only ever within a single split.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def cross_split_duplicate_groups(records: Iterable[dict[str, Any]]) -> list[str]:
    """Return group keys that appear in more than one split.

    An empty return value is the pass condition required by
    ``tasks/contracts/T260.acceptance.yaml``
    (``splits.cross_split_duplicate_groups == 0``).
    """
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        splits_by_group[record["group_key"]].add(record["split"])
    return sorted(group for group, splits in splits_by_group.items() if len(splits) > 1)


def evaluation_records_in_training(records: Iterable[dict[str, Any]]) -> int:
    """Count evaluation-only group keys that also appear in a training split.

    Zero is the pass condition required by the acceptance contract
    (``splits.evaluation_records_in_training == 0``). Unlike
    :func:`cross_split_duplicate_groups`, this is deliberately scoped to
    training splits only (``pilot_train``, ``pilot_validation``,
    ``pilot_meta``, ``diagnostic``) so it directly encodes the pass/fail
    gate's "evaluation-only records must never enter training" requirement.
    """
    training_splits = {"pilot_train", "pilot_validation", "pilot_meta", "diagnostic"}
    eval_groups: set[str] = set()
    training_groups: set[str] = set()
    for record in records:
        if record["split"] == "evaluation_only":
            eval_groups.add(record["group_key"])
        elif record["split"] in training_splits:
            training_groups.add(record["group_key"])
    return len(eval_groups & training_groups)


def duplicate_record_ids(records: Iterable[dict[str, Any]]) -> list[str]:
    """Return any ``record_id`` values that were emitted more than once."""
    seen: dict[str, int] = defaultdict(int)
    for record in records:
        seen[record["record_id"]] += 1
    return sorted(record_id for record_id, count in seen.items() if count > 1)
