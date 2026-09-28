"""Stable, deterministic sample-ID scheme shared by every source builder.

A record's ``record_id`` must be a pure function of ``(namespace, native_key)``
so that rebuilding a manifest from the same frozen source metadata always
reproduces byte-identical IDs -- no timestamp, random seed, or filesystem
enumeration order may influence it.
"""

from __future__ import annotations

import hashlib

_DIGEST_CHARS = 16


def stable_id(namespace: str, native_key: str) -> str:
    """Return a deterministic record ID for ``native_key`` within ``namespace``.

    ``namespace`` should identify the source and role (for example
    ``"d1-coco-caption"``); ``native_key`` should identify the record within
    the source (for example ``"397133:37"`` for a COCO image/annotation
    pair). The result is stable across processes, machines, and Python
    versions because it is derived only from ``hashlib.sha256`` of the UTF-8
    encoding of the two inputs joined by a fixed separator.
    """
    if not namespace or not native_key:
        raise ValueError("namespace and native_key must both be non-empty")
    digest = hashlib.sha256(f"{namespace}::{native_key}".encode("utf-8")).hexdigest()
    return f"{namespace}-{digest[:_DIGEST_CHARS]}"


def group_bucket(group_key: str, *, num_buckets: int = 10_000) -> int:
    """Return a deterministic integer bucket in ``[0, num_buckets)`` for a group key.

    Used to assign every record sharing ``group_key`` (for example a COCO
    ``image_id``) to the same split bucket, independent of source, so that
    D1/D2 records built from the same underlying image never straddle a
    split boundary. This is a pure hash function of the key only -- it never
    looks at model gradients, losses, or any training outcome.
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    digest = hashlib.sha256(f"split-bucket::{group_key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets
