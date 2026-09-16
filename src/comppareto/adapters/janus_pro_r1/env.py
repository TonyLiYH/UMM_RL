"""Environment/path resolution for the Janus-Pro-R1 adapter (T720).

Every path here is env-var driven with a default that matches the actual
layout used on the H20-FoldUMM Taiji container for this task (see
``reports/T720/first-report.md`` section 6 and
``configs/janus-pro-r1/admission/*.md``): everything lives under
``/dockerdata/t720-janus-pro-r1/`` -- the container's local (non-ceph) SSD,
verified via ``mount``/``df`` to be ``xfs`` on ``/dev/mapper/gpu-gpu_volume``,
satisfying the frozen protocol's "Execute from verified local SSD"
requirement. No path here is a bare string baked into any *caller*; callers
must go through these functions so a single override
(``JANUS_PRO_R1_*`` env vars) repoints every consumer at once, e.g. for
testing against a scratch directory.

This module intentionally imports nothing beyond ``os``/``pathlib`` (no
torch) so it stays importable -- and its own tests runnable -- on a
CPU-only, torch-free development machine.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Root of the ``UMM_RL`` checkout (or worktree) this file lives under.

    ``env.py`` is at ``src/comppareto/adapters/janus_pro_r1/env.py``; the
    repository root is four parents up from that file.
    """

    return Path(__file__).resolve().parents[4]


def vendor_root() -> Path:
    """Root of the vendored ``wendell0218/Janus-Pro-R1`` source tree.

    Default matches ``configs/janus-pro-r1/admission/source-lock.yaml``'s
    ``code.vendored_at``.
    """

    override = os.getenv("JANUS_PRO_R1_VENDOR_ROOT")
    if override:
        return Path(override)
    return repo_root() / "vendor" / "janus-pro-r1"


def sft_source_root() -> Path:
    """Vendored upstream ``janus-sft/`` (SFT trainer, models, datasets)."""

    return vendor_root() / "janus-sft"


def rl_source_root() -> Path:
    """Vendored upstream ``janus-rl/`` (GRPO trainer, reward model)."""

    return vendor_root() / "janus-rl"


def base_checkpoint_root() -> Path:
    """Local-SSD copy of the ``deepseek-ai/Janus-Pro-7B`` base checkpoint.

    Default is the local-SSD copy made from the pre-existing, hash-verified
    H20-FoldUMM overlay-fs cache at ``/root/local_cache/models/Janus-Pro-7B``
    (see ``configs/janus-pro-r1/admission/source-lock.yaml``
    ``base_checkpoint.provenance_note``); the copy step itself re-verifies
    both shard sha256 hashes on the destination before this path is
    considered usable.
    """

    return Path(
        os.getenv(
            "JANUS_PRO_R1_BASE_CHECKPOINT",
            "/dockerdata/t720-janus-pro-r1/models/Janus-Pro-7B",
        )
    )


def reward_checkpoint_root() -> Path:
    """Local-SSD copy of the ``OpenGVLab/InternVL2_5-8B`` reward checkpoint."""

    return Path(
        os.getenv(
            "JANUS_PRO_R1_REWARD_CHECKPOINT",
            "/dockerdata/t720-janus-pro-r1/models/InternVL2_5-8B",
        )
    )


def data_root() -> Path:
    """Local-SSD root for downloaded/vendored SFT+GRPO data."""

    return Path(os.getenv("JANUS_PRO_R1_DATA_ROOT", "/dockerdata/t720-janus-pro-r1/data"))


def sft_venv_python() -> Path:
    """Interpreter of the separately pinned SFT venv (see environment-sft-lock.md)."""

    return Path(
        os.getenv(
            "JANUS_PRO_R1_SFT_VENV_PYTHON",
            "/dockerdata/t720-janus-pro-r1/venvs/sft/bin/python",
        )
    )


def rl_venv_python() -> Path:
    """Interpreter of the separately pinned RL/GRPO venv (see environment-rl-lock.md)."""

    return Path(
        os.getenv(
            "JANUS_PRO_R1_RL_VENV_PYTHON",
            "/dockerdata/t720-janus-pro-r1/venvs/rl/bin/python",
        )
    )


def run_output_root() -> Path:
    """Local-SSD scratch root for smoke-run checkpoints/logs (not synced to git).

    Only small, hashed summary artifacts derived from what is written here
    are copied into the git-tracked ``runs/janus-pro-r1-stack-v1/`` --
    checkpoints themselves are excluded by ``.gitignore``
    (``runs/*/checkpoints/``).
    """

    return Path(os.getenv("JANUS_PRO_R1_RUN_ROOT", "/dockerdata/t720-janus-pro-r1/runs"))


def require_cuda() -> None:
    """Raise ``RuntimeError`` unless a CUDA device is actually available.

    GPU-required Janus-Pro-R1 adapter code (the SFT/GRPO smoke runners) must
    never silently fall back to CPU -- a CPU run of a 7B-parameter model
    would not exercise the real code path and could produce misleading
    "it works" results. This is checked defensively at the top of every
    GPU-requiring entry point, in addition to whatever device placement
    torch itself would otherwise attempt.

    Raises even when torch itself is not installed (treated as "no CUDA
    available", the more conservative of the two readings) rather than
    letting an ``ImportError`` propagate -- this keeps the failure mode
    uniform and makes the function safely callable from a torch-free
    environment, which is what lets it be unit-tested on a CPU-only,
    torch-free development machine.
    """

    try:
        import torch  # local import: keep this module torch-free at import time
    except ImportError as exc:  # pragma: no cover - exercised via the "no torch" branch below
        raise RuntimeError(
            "PyTorch is not installed in this interpreter; GPU-required "
            "Janus-Pro-R1 adapter code refuses to run without CUDA."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this interpreter; GPU-required "
            "Janus-Pro-R1 adapter code refuses to silently fall back to CPU."
        )
