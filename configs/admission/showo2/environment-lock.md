# Show-o2 admission — repaired environment lock (T210 R7)

Task-scoped environment description for the H20-FoldUMM GPU container venv used for every T210
smoke run (`/root/venvs/showo2`, `python3 -m venv --system-site-packages`). This file exists
because the official `show-o2/build_env.sh` (pinned commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23`)
does not reproduce a working environment as-is on this platform — see
`reports/T210/environment-checkpoint-smoke.md` for the full narrative of the defect and its fix.
This file states the resulting constraints the official script needs on this platform; it does not
replace or fork the official script.

## Required additional constraints on top of `build_env.sh`

1. **Reassert `torch==2.5.1+cu124` after the CLIP install.** `build_env.sh` installs
   `torch==2.5.1` early, then later runs `pip3 install git+https://github.com/openai/CLIP.git`,
   which has an unpinned `torchvision` dependency. pip's resolver upgrades to `torchvision`
   requiring `torch==2.6.0`, silently uninstalling the pinned torch. Reinstall
   `torch==2.5.1` from `https://download.pytorch.org/whl/cu124` immediately after the CLIP step.
2. **Pin `torchvision==0.20.1+cu124`.** `build_env.sh` never pins torchvision at all — it only
   arrives transitively via CLIP. Without an explicit pin matching step 1's torch build, an
   ABI-mismatched torchvision is left in `dist-packages`.
3. **Pin `wandb==0.17.0`.** `build_env.sh`'s unpinned `pip3 install wandb` resolves to
   `wandb==0.29.0` on this platform, which both fails to import against the resolved
   `protobuf==4.25.9` and has removed `wandb.util.generate_id`, which both `inference_mmu.py` and
   `inference_t2i.py` call unconditionally. `0.17.0` is the newest release confirmed to still
   expose `generate_id` and to import cleanly against `protobuf==4.25.9`.

No other package required a version different from what `build_env.sh` naturally resolves to.

## Full verified environment (reconfirmed live 2026-08-28, R2 rerun container)

| Package | Version |
|---|---|
| `torch` | `2.5.1+cu124` |
| `torchvision` | `0.20.1+cu124` |
| `transformers` | `4.47.0` |
| `diffusers` | `0.31.0` |
| `clip` (openai/CLIP) | `1.0` |
| `flash_attn` | `2.8.3.post1` |
| `wandb` | `0.17.0` |
| `nvcc` / CUDA | `12.4.99` |

Full `pip freeze` snapshot (large third-party dependency list, not copied into this directory):
- durable copy: `/apdcephfs_cq7/share_1447896/yihangli/outputs/T210-showo2-admission/r2_runs/pip_freeze_20260828.txt`
  (sha256 `d5db5cb61a13e4a327399d005debe058f3e1f351cde5d7c50478a5c97198e1cf`, 5396 bytes)
- referenced from the R3 admission manifest as an `artifact`.

## Scope note

This lock file governs the Show-o2 inference venv used for T210's read-only admission smoke tests
only. It authorizes no training, joint or otherwise; T160/T170/T215/T300 remain out of scope for
this task.
