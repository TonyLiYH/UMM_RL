"""Environment-variable-driven path resolution for the T710 CoRL admission.

Follows the repo's dev-env-paths rule: large assets/caches live on CQ7,
code lives on CQ9, everything is overridable via env var with a sane
default, never a bare hard-coded path baked into logic.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: str) -> Path:
    return Path(os.getenv(var, default))


# CQ7: large, execution-local (GPU-container SSD) assets and outputs.
ASSETS_ROOT = _env_path(
    "T710_ASSETS_ROOT", "/dockerdata/t710-corl/assets"
)
RUN_OUTPUT_ROOT = _env_path(
    "T710_RUN_OUTPUT_ROOT", "/dockerdata/t710-corl/runs"
)
CORL_REPO_PATH = _env_path(
    # The importable root: contains open_r1/ (rewards, trainer, grpo_janus_unify.py)
    # and scripts/. Confirmed by live inspection of the materialized checkout under
    # /dockerdata/t710-corl/ULM-R1/corl on the H20-FoldUMM container.
    "T710_CORL_REPO_PATH", "/dockerdata/t710-corl/ULM-R1/corl"
)

# CQ7 durable outputs (survives container recycle; ceph-fuse, not local SSD --
# used only for copying *small* JSON/log evidence off the disposable SSD,
# never for the multi-GB checkpoints/caches themselves).
DURABLE_EVIDENCE_ROOT = _env_path(
    "T710_DURABLE_EVIDENCE_ROOT",
    "/apdcephfs_cq7/share_1447896/yihangli/outputs/T710-corl-admission",
)


def janus_model_dir() -> Path:
    return ASSETS_ROOT / "Janus-Pro-1B"


def mpnet_model_dir() -> Path:
    return ASSETS_ROOT / "all-mpnet-base-v2"


def micro_split_jsonl() -> Path:
    return ASSETS_ROOT / "micro-split.jsonl"


def micro_split_images_dir() -> Path:
    return ASSETS_ROOT / "micro-split-images"


def dataset_cache_dir() -> Path:
    return ASSETS_ROOT / "dataset-cache"


def run_dir(variant: str) -> Path:
    """variant: 'upstream_exact' or 'corrected_candidate'."""
    return RUN_OUTPUT_ROOT / variant
