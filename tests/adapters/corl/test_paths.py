from __future__ import annotations

import importlib
from pathlib import Path


def _reload_paths(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from comppareto.adapters.corl import paths

    return importlib.reload(paths)


def test_defaults_point_at_local_ssd_not_ceph(monkeypatch):
    monkeypatch.delenv("T710_ASSETS_ROOT", raising=False)
    monkeypatch.delenv("T710_RUN_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("T710_CORL_REPO_PATH", raising=False)
    from comppareto.adapters.corl import paths

    paths = importlib.reload(paths)
    # /dockerdata is the verified local-NVMe mount on the GPU container;
    # execution assets/caches must never default onto ceph (/apdcephfs_*).
    assert str(paths.ASSETS_ROOT).startswith("/dockerdata")
    assert str(paths.RUN_OUTPUT_ROOT).startswith("/dockerdata")
    assert str(paths.CORL_REPO_PATH).startswith("/dockerdata")


def test_env_override(monkeypatch, tmp_path: Path):
    paths = _reload_paths(monkeypatch, T710_ASSETS_ROOT=str(tmp_path / "assets"))
    assert paths.ASSETS_ROOT == tmp_path / "assets"
    assert paths.janus_model_dir() == tmp_path / "assets" / "Janus-Pro-1B"
    assert paths.mpnet_model_dir() == tmp_path / "assets" / "all-mpnet-base-v2"
    assert paths.micro_split_jsonl() == tmp_path / "assets" / "micro-split.jsonl"


def test_run_dir_variants(monkeypatch, tmp_path: Path):
    paths = _reload_paths(monkeypatch, T710_RUN_OUTPUT_ROOT=str(tmp_path / "runs"))
    assert paths.run_dir("upstream_exact") == tmp_path / "runs" / "upstream_exact"
    assert paths.run_dir("corrected_candidate") == tmp_path / "runs" / "corrected_candidate"
