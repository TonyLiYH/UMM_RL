"""Preregistered, real media-availability probes (local review item 4).

The prior round validated only manifest *schema* (does ``image`` have a
non-empty ``source_relative_path``?), never whether the referenced media is
actually fetchable. This module adds a small, deterministic sample of real
network probes per admitted source, run *before* any manifest content is
inspected for probe-worthiness -- the selection rule
(:func:`select_sample`) is fixed here as code, independent of any probe
outcome, which is what makes it "preregistered" rather than "picked after
looking at results."

Three real probe functions are provided, one per admitted source's media
transport:

* :func:`probe_coco_image` -- plain HTTP GET against
  ``images.cocodataset.org`` (used by both D1/D4 COCO records and D2 LLaVA
  records, which reference the same ``train2017`` files).
* :func:`probe_diffusiondb_image` -- an HTTP Range read of the *specific*
  member inside a remote ``part-NNNNNN.zip`` archive, using
  :class:`HTTPRangeFile` to drive Python's stdlib ``zipfile`` module's own
  End-Of-Central-Directory/local-header parsing via minimal Range requests
  -- this reads only the one referenced image's bytes, never the archive's
  other ~500-2,500 members and never the whole archive.

Every probe returns a plain dict; on any failure (network error, 404,
truncated read, ...) the caller (:func:`run_media_checks`) records
``available=False`` with the exception message rather than raising, so one
unreachable file cannot abort the whole run.

This module performs no model inference or training -- only HTTP reads of
already-admitted, already-licensed source media.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Callable, Iterable
import urllib.request
import zipfile

from .ids import group_bucket

#: How many records to probe per distinct (source, split) pair present in
#: the manifest. Fixed here, before any manifest is built or any probe is
#: attempted.
SAMPLE_PER_SOURCE_SPLIT = 5

#: Bucket space used only to pick a deterministic, low-hash-valued sample
#: within each (source, split) group -- a pure function of each record's
#: own immutable ``record_id``, never of network reachability.
_SELECTION_BUCKETS = 1_000_000

COCO_IMAGE_BASE_URL = "http://images.cocodataset.org"
DIFFUSIONDB_RESOLVE_BASE_URL = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main"

ProbeFn = Callable[[str], dict[str, Any]]


def select_sample(
    records: Iterable[dict[str, Any]],
    *,
    per_group: int = SAMPLE_PER_SOURCE_SPLIT,
) -> list[dict[str, Any]]:
    """Deterministically select up to ``per_group`` records per (source, split).

    Selection rule, fixed before this module ever observes a real manifest
    or probe result: within each ``(source, split)`` pair, keep the
    ``per_group`` records whose ``group_bucket(record_id, num_buckets=
    _SELECTION_BUCKETS)`` is smallest. Ties break on ``record_id`` string
    order, also fixed and content-only. Never depends on whether a probe
    would succeed for that record.
    """
    grouped: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        key = (record["source"], record["split"])
        bucket = group_bucket(record["record_id"], num_buckets=_SELECTION_BUCKETS)
        grouped[key].append((bucket, record["record_id"], record))

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda triple: (triple[0], triple[1]))
        selected.extend(record for _, _, record in items[:per_group])
    return selected


class HTTPRangeFile:
    """A minimal seekable, read-only file-like object backed by HTTP Range requests.

    Feeding this into :class:`zipfile.ZipFile` lets zipfile's own
    End-Of-Central-Directory and local-file-header parsing drive exactly
    the Range requests it needs (typically: the last ~64KB for the EOCD +
    central directory, then one more Range read for the specific member's
    local header + compressed bytes) -- never a full-archive download.
    """

    def __init__(self, url: str, *, timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout
        self._pos = 0
        self._length = self._fetch_length()

    def _fetch_length(self) -> int:
        request = urllib.request.Request(self._url, method="HEAD")
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            length = response.headers.get("Content-Length")
        if length is None:
            raise OSError(f"server did not report Content-Length for {self._url}")
        return int(length)

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._length + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            end = self._length - 1
        else:
            end = min(self._pos + size, self._length) - 1
        if end < self._pos:
            return b""
        request = urllib.request.Request(
            self._url,
            headers={"Range": f"bytes={self._pos}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            data = response.read()
        self._pos += len(data)
        return data


def probe_coco_image(source_relative_path: str, *, timeout: float = 15.0) -> dict[str, Any]:
    """Fetch one COCO image over plain HTTP and hash the real bytes.

    ``source_relative_path`` is e.g. ``"train2017/000000000009.jpg"``.
    Uses ``http://`` (not ``https://``) because ``images.cocodataset.org``
    serves a TLS certificate for ``s3.amazonaws.com``, not its own
    hostname -- an ``https://`` GET fails TLS hostname verification even
    though the file is genuinely served (see
    ``reports/T260/failure-ledger.md``).
    """
    url = f"{COCO_IMAGE_BASE_URL}/{source_relative_path}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return {
        "available": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "probe_url": url,
    }


def probe_diffusiondb_image(source_relative_path: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Range-read one member out of a remote DiffusionDB part archive.

    ``source_relative_path`` is e.g.
    ``"images/part-000086.zip::272f29e8-...-...png"``.
    """
    archive_relative_path, separator, member = source_relative_path.partition("::")
    if not separator:
        raise ValueError(f"expected 'archive.zip::member' shape, got {source_relative_path!r}")
    url = f"{DIFFUSIONDB_RESOLVE_BASE_URL}/{archive_relative_path}"
    range_file = HTTPRangeFile(url, timeout=timeout)
    with zipfile.ZipFile(range_file) as archive:
        data = archive.read(member)
    return {
        "available": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "probe_url": url,
    }


#: Real network probes, keyed by each record's ``image.source_dataset``.
DEFAULT_PROBES: dict[str, ProbeFn] = {
    "coco_train2017": probe_coco_image,
    "coco_val2017": probe_coco_image,
    "diffusiondb_2m": probe_diffusiondb_image,
}


def run_media_checks(
    sample: Iterable[dict[str, Any]],
    *,
    probes: dict[str, ProbeFn],
) -> list[dict[str, Any]]:
    """Run a real (or injected, for tests) probe for every sampled record.

    Never raises: a probe exception is captured as
    ``{"available": False, "error": "..."}`` so one unreachable/renamed
    file cannot abort the run.
    """
    results: list[dict[str, Any]] = []
    for record in sample:
        source_dataset = record["image"]["source_dataset"]
        probe = probes.get(source_dataset)
        result: dict[str, Any] = {
            "record_id": record["record_id"],
            "source": record["source"],
            "split": record["split"],
            "source_dataset": source_dataset,
            "source_relative_path": record["image"]["source_relative_path"],
        }
        if probe is None:
            result.update({"available": False, "error": f"no probe registered for {source_dataset!r}"})
        else:
            try:
                result.update(probe(record["image"]["source_relative_path"]))
            except Exception as exc:  # noqa: BLE001 - deliberately broad: a probe must never abort the run
                result.update({"available": False, "error": f"{type(exc).__name__}: {exc}"})
        results.append(result)
    return results


def apply_probe_results(
    records: Iterable[dict[str, Any]],
    results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of ``records`` with probed records' ``image`` fields patched.

    Only the sampled records (identified by ``record_id``) are changed;
    every other record is returned unchanged. A successful probe sets
    ``media_materialized=True`` and records the real ``media_sha256``/
    ``media_bytes``; a failed probe leaves ``media_materialized=False`` but
    still records ``media_verified_available=False`` -- the distinction
    the schema (``comppareto.data.records``) requires between "we admitted
    this metadata" and "we independently confirmed the bytes exist."
    """
    result_by_id = {result["record_id"]: result for result in results}
    patched: list[dict[str, Any]] = []
    for record in records:
        result = result_by_id.get(record["record_id"])
        if result is None:
            patched.append(record)
            continue
        new_record = dict(record)
        new_image = dict(record["image"])
        if result.get("available"):
            new_image["media_materialized"] = True
            new_image["media_verified_available"] = True
            new_image["media_sha256"] = result["sha256"]
            new_image["media_bytes"] = result["bytes"]
        else:
            new_image["media_verified_available"] = False
        new_record["image"] = new_image
        patched.append(new_record)
    return patched
