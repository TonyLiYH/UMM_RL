# SenseNova-U1 admission — environment lock

Task-scoped environment description for the H20-FoldUMM GPU container venv used for T230's
admission smoke runs (`/dockerdata/t230-sensenova/venv`, local-SSD `python3.10 -m venv`, separate
from Show-o2's `/root/venvs/showo2` per `dev-env-paths.md`). Built directly from the official
repository's `requirements.txt` / `pyproject.toml` at the pinned commit
`f97964a6e54b0abf92aa2db849af4e942bb2ff08`, with no unofficial version substitution — every
package in the environment resolved cleanly to the versions the repository's own dependency
declarations request, with one documented optional-dependency omission (below).

## Install procedure (matches official `pyproject.toml` / `requirements.txt` verbatim)

```bash
python3.10 -m venv /dockerdata/t230-sensenova/venv
/dockerdata/t230-sensenova/venv/bin/pip install torch==2.8.0 torchvision==0.23.0 \
  --extra-index-url https://download.pytorch.org/whl/cu128
/dockerdata/t230-sensenova/venv/bin/pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
/dockerdata/t230-sensenova/venv/bin/pip install -e . --no-deps
```

No environment-repair or version-pin deviation from the official declarations was needed — unlike
T210/Show-o2, which required three corrective pins (`reports/T210` R1-R7). All top-level pins in
`requirements.txt` (`torch==2.8.0`, `torchvision==0.23.0`, `sentencepiece==0.2.1`, `pillow==12.0.0`,
`tqdm==4.67.1`, `packaging==25.0`) resolved exactly as declared; the open-range dependencies
(`transformers>=4.57.1,<6`, `accelerate>=1.1,<2`, `huggingface-hub>=0.34,<2`, `safetensors>=0.4.3,<1`,
`numpy>=1.24,<3`, `httpx>=0.27,<1`) resolved to the newest versions available on PyPI at install
time, shown below.

## Full verified environment (live install, H20-FoldUMM container, 2026-09-03)

| Package | Version |
|---|---|
| `torch` | `2.8.0+cu128` (CUDA available: confirmed via `torch.cuda.is_available() == True`) |
| `torchvision` | `0.23.0+cu128` |
| `transformers` | `5.16.1` |
| `accelerate` | `1.14.0` |
| `huggingface_hub` | `1.29.0` |
| `safetensors` | `0.8.0` |
| `sentencepiece` | `0.2.1` |
| `numpy` | `2.2.6` |
| `pillow` | `12.0.0` |
| `tqdm` | `4.67.1` |
| `packaging` | `25.0` |
| `httpx` | `0.28.1` |
| `sensenova_u1` (editable install of the repo itself) | `0.1.0` at `f97964a6e54b0abf92aa2db849af4e942bb2ff08` |
| CUDA driver / toolkit | driver `535.161.08`; container `nvcc` reports `release 12.4, V12.4.99` (torch ships its own bundled CUDA 12.8 runtime via `nvidia-*-cu12` wheels, independent of the container's system `nvcc`) |

Full `pip freeze` snapshot preserved durably outside Git:
- `/apdcephfs_cq7/share_1447896/yihangli/outputs/T230-sensenova-u1-admission/pip_freeze_20260903.txt`
  (sha256 `e06b3bcb5db531f5094ebda1f819a462ab2e8d22545156f97890349e9229c832`, 1208 bytes)

## Documented optional-dependency omission (non-blocking, per upstream's own fallback)

`flash-attn` (the `flash` extra, `flash-attn>=2.8,<3`) was **not installed**. The reference
environment documented in the repo's own `pyproject.toml` uses a CUDA-specific wheel
(`flash_attn 2.8.3+cu12torch28cxx11abitrue-cp311-*`) that PyPI does not host generically; installing
it requires a locally-built or hand-matched `.whl` for this exact `torch==2.8.0`/CUDA
12.8/`cp310` combination, which was not available in this pass. This is **not an unofficial
workaround** — the repository's own `pyproject.toml` states the model "transparently falls back to
torch SDPA when absent" (comment above the `flash` extra, referencing
`src/sensenova_u1/models/neo_unify/modeling_qwen3.py`). Both smokes in this admission therefore run
on the SDPA attention path. If a smoke fails specifically due to the missing flash-attn (unlike
T210/Show-o2, where flash-attn was hard-required with no SDPA fallback for the DiT), this will be
recorded explicitly in the failure ledger rather than silently patched.

## Scope note

This lock file governs the SenseNova-U1 inference venv used for T230's read-only admission smoke
tests only (understanding via `examples/vqa/inference.py`, generation via
`examples/t2i/inference.py`, and the static parameter-block/routed-overlap audit). It authorizes no
training and no U1.5 substitution; the `training/` subtree (derived from InternEvo) is out of
scope for this task per the frozen protocol.
