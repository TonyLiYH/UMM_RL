from __future__ import annotations

from comppareto.data import media_check


def _record(record_id: str, source: str, split: str, source_dataset: str, path: str) -> dict:
    return {
        "record_id": record_id,
        "source": source,
        "split": split,
        "image": {
            "source_dataset": source_dataset,
            "source_relative_path": path,
            "metadata_admitted": True,
            "media_materialized": False,
            "media_verified_available": None,
            "media_sha256": None,
            "media_bytes": None,
        },
    }


def _sample_records() -> list[dict]:
    records = []
    for i in range(10):
        records.append(_record(f"coco-{i}", "coco_captions_2017", "pilot_train", "coco_train2017", f"train2017/{i}.jpg"))
    for i in range(3):
        records.append(_record(f"ddb-{i}", "diffusiondb_2m", "pilot_train", "diffusiondb_2m", f"images/part-{i:06d}.zip::{i}.png"))
    return records


def test_select_sample_caps_per_source_split_group() -> None:
    records = _sample_records()
    sample = media_check.select_sample(records, per_group=3)
    coco_ids = [r["record_id"] for r in sample if r["source"] == "coco_captions_2017"]
    ddb_ids = [r["record_id"] for r in sample if r["source"] == "diffusiondb_2m"]
    assert len(coco_ids) == 3
    assert len(ddb_ids) == 3  # only 3 exist, all selected


def test_select_sample_is_deterministic() -> None:
    records = _sample_records()
    first = media_check.select_sample(records)
    second = media_check.select_sample(records)
    assert [r["record_id"] for r in first] == [r["record_id"] for r in second]


def test_run_media_checks_records_success() -> None:
    records = [_record("r1", "coco_captions_2017", "pilot_train", "coco_train2017", "train2017/x.jpg")]

    def fake_probe(path: str) -> dict:
        assert path == "train2017/x.jpg"
        return {"available": True, "bytes": 10, "sha256": "abc"}

    results = media_check.run_media_checks(records, probes={"coco_train2017": fake_probe})
    assert results[0]["available"] is True
    assert results[0]["sha256"] == "abc"


def test_run_media_checks_captures_probe_exception_without_raising() -> None:
    records = [_record("r1", "coco_captions_2017", "pilot_train", "coco_train2017", "train2017/x.jpg")]

    def failing_probe(path: str) -> dict:
        raise OSError("boom")

    results = media_check.run_media_checks(records, probes={"coco_train2017": failing_probe})
    assert results[0]["available"] is False
    assert "boom" in results[0]["error"]


def test_run_media_checks_handles_missing_probe_registration() -> None:
    records = [_record("r1", "unknown_source", "pilot_train", "unknown_dataset", "x")]
    results = media_check.run_media_checks(records, probes={})
    assert results[0]["available"] is False
    assert "no probe registered" in results[0]["error"]


def test_apply_probe_results_patches_only_matching_records() -> None:
    records = [
        _record("r1", "coco_captions_2017", "pilot_train", "coco_train2017", "train2017/x.jpg"),
        _record("r2", "coco_captions_2017", "pilot_train", "coco_train2017", "train2017/y.jpg"),
    ]
    results = [{"record_id": "r1", "available": True, "bytes": 5, "sha256": "deadbeef"}]
    patched = media_check.apply_probe_results(records, results)
    by_id = {r["record_id"]: r for r in patched}
    assert by_id["r1"]["image"]["media_materialized"] is True
    assert by_id["r1"]["image"]["media_sha256"] == "deadbeef"
    assert by_id["r1"]["image"]["media_bytes"] == 5
    assert by_id["r1"]["image"]["media_verified_available"] is True
    assert by_id["r2"]["image"]["media_materialized"] is False
    assert by_id["r2"]["image"]["media_verified_available"] is None


def test_apply_probe_results_marks_failed_probe_as_unavailable_without_materializing() -> None:
    records = [_record("r1", "coco_captions_2017", "pilot_train", "coco_train2017", "train2017/x.jpg")]
    results = [{"record_id": "r1", "available": False, "error": "404"}]
    patched = media_check.apply_probe_results(records, results)
    assert patched[0]["image"]["media_materialized"] is False
    assert patched[0]["image"]["media_verified_available"] is False


def test_http_range_file_read_and_seek_semantics(monkeypatch) -> None:
    body = b"0123456789"

    class _FakeResponse:
        def __init__(self, data: bytes, headers: dict) -> None:
            self._data = data
            self.headers = headers

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        if request.get_method() == "HEAD":
            return _FakeResponse(b"", {"Content-Length": str(len(body))})
        range_header = request.headers.get("Range", "")
        assert range_header.startswith("bytes=")
        start_str, end_str = range_header[len("bytes="):].split("-")
        start, end = int(start_str), int(end_str)
        return _FakeResponse(body[start : end + 1], {})

    monkeypatch.setattr(media_check.urllib.request, "urlopen", fake_urlopen)
    range_file = media_check.HTTPRangeFile("http://example.invalid/f.bin")
    assert range_file._length == len(body)
    range_file.seek(2)
    assert range_file.read(3) == body[2:5]
    range_file.seek(-2, whence=2)
    assert range_file.read() == body[-2:]
