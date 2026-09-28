"""Tests for comppareto.adapters.janus_pro_r1.env (T720).

Runs on a CPU-only, torch-free development machine: ``require_cuda()`` is
asserted to raise ``RuntimeError`` genuinely (not mocked) -- either because
torch is not installed here at all, or because it is installed but no CUDA
device is visible, both of which this dev machine satisfies.
"""

from __future__ import annotations

import pytest

from comppareto.adapters.janus_pro_r1 import env


def test_repo_root_points_at_the_real_checkout() -> None:
    root = env.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "vendor" / "janus-pro-r1").is_dir()


def test_vendor_root_default_matches_the_vendored_tree() -> None:
    root = env.vendor_root()
    assert root == env.repo_root() / "vendor" / "janus-pro-r1"
    assert (root / "janus-sft").is_dir()
    assert (root / "janus-rl").is_dir()


def test_vendor_root_honors_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("JANUS_PRO_R1_VENDOR_ROOT", str(tmp_path))
    assert env.vendor_root() == tmp_path


def test_sft_and_rl_source_roots_are_subdirectories_of_vendor_root() -> None:
    assert env.sft_source_root() == env.vendor_root() / "janus-sft"
    assert env.rl_source_root() == env.vendor_root() / "janus-rl"


def test_checkpoint_and_data_paths_default_under_local_ssd_scratch() -> None:
    # These defaults are not asserted to *exist* here (this dev machine has
    # no /dockerdata mount) -- only that every path lives under the same
    # local-SSD scratch root documented in reports/T720/first-report.md, so
    # a single JANUS_PRO_R1_RUN_ROOT-style override family is sufficient to
    # repoint the whole adapter at a different machine.
    assert str(env.base_checkpoint_root()).startswith("/dockerdata/t720-janus-pro-r1/")
    assert str(env.reward_checkpoint_root()).startswith("/dockerdata/t720-janus-pro-r1/")
    assert str(env.data_root()).startswith("/dockerdata/t720-janus-pro-r1/")
    assert str(env.sft_venv_python()).startswith("/dockerdata/t720-janus-pro-r1/")
    assert str(env.rl_venv_python()).startswith("/dockerdata/t720-janus-pro-r1/")


def test_env_var_override_repoints_base_checkpoint_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JANUS_PRO_R1_BASE_CHECKPOINT", "/tmp/some-other-place")
    assert str(env.base_checkpoint_root()) == "/tmp/some-other-place"


def test_require_cuda_raises_runtime_error_on_this_cpu_only_machine() -> None:
    with pytest.raises(RuntimeError):
        env.require_cuda()
