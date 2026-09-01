# T210 — Environment and checkpoint smoke

Stage 2 of the T210 execution protocol (see `docs/plans/showo2-first-attempt.md`): build the
official Show-o2 dependency stack inside the GPU container (H20-FoldUMM), download the pinned
checkpoints, and verify both import and run correctly on GPU before touching any task path.

All commands below ran inside the H20-FoldUMM container (`taiji_client exec`), per the project's
GPU-execution rule. No training was started; nothing outside `configs/admission/showo2/`,
`runs/admission-showo2/`, `reports/T210/` was written inside the repo. The venv itself and its
build/download scripts live outside the repo, under
`/apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/` (scratch, not a repo artifact).

## Checkpoints

Downloaded via `huggingface_hub.snapshot_download`, pinned to the exact commit SHAs recorded in
`reports/T210/first-report.md`, into
`/apdcephfs_cq7/share_1447896/yihangli/models/pretrained/showo2/`:

| Component | Repo ID | Resolved revision | SHA-256 of main weight file |
|---|---|---|---|
| Show-o2 1.5B | `showlab/show-o2-1.5B` | `07ec16589d4fc5422a74dddbbc4b2cd11e551039` | `a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c` |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| SigLIP so400m | `google/siglip-so400m-patch14-384` | `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` | `ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a` |
| Wan2.1 VAE | (already present locally) | n/a | see `first-report.md` |

Full file-level hash listing: `checkpoint_hashes.txt` alongside the downloaded files
(cjob log: `showo2_ckpt_download.log`, exit 0).

## Environment build: a defect in the official `build_env.sh`

The venv (`/root/venvs/showo2`, `python3 -m venv --system-site-packages`) was built by replicating
`show-o2/build_env.sh` from the pinned commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23`, one
`pip install` per line, in the script's own order.

**Finding (official-script defect, not a local bug):** the official script installs
`torch==2.5.1` first, then several packages later runs
`pip3 install git+https://github.com/openai/CLIP.git`. OpenAI's `CLIP` package depends on
`torchvision`, which has no pin in `build_env.sh`. pip's resolver picked the latest `torchvision`,
which in turn required `torch==2.6.0`, so pip **silently uninstalled the pinned `torch-2.5.1+cu124`
and installed `torch-2.6.0`** (plain PyPI build, not the `+cu124` index build) partway through the
official script. This is reproducible by anyone following `build_env.sh` verbatim — it is not
specific to this container or to any change made here.

The consequence: the resulting `torch==2.6.0`'s own `nvidia-cusparselt-cu12==0.6.2` requirement was
reported "already satisfied" from system `dist-packages`, but the actual `.so` lives at a
non-standard path (`/usr/local/lib/python3.10/dist-packages/cusparselt/lib/libcusparseLt.so.0`,
not the `nvidia/cusparselt/lib/` layout torch's loader expects), so `import torch` failed with
`ImportError: libcusparseLt.so.0: cannot open shared object file`. Every later step in the official
script that imports torch (including the `flash-attn` build) failed for this same downstream
reason — flash-attn's build failure was **not** a flash-attn-specific incompatibility.

**Resolution (restores the official pin, not a substitute):**
1. Reinstalled `torch==2.5.1` from `https://download.pytorch.org/whl/cu124` — the exact pinned
   version the official script asks for — as a corrective step after the CLIP install. Verified:
   `torch 2.5.1+cu124`, `torch.cuda.is_available() == True`.
2. Installed the torchvision release paired with `torch==2.5.1+cu124` (`torchvision==0.20.1+cu124`)
   to replace the ABI-mismatched copy CLIP's install had left in system `dist-packages`
   (`torchvision-0.21.0+cu124`, built against `torch==2.6.0`). The official script does not pin a
   torchvision version at all — it only gets one transitively via CLIP — so this is filling a gap
   in the official pin, not overriding it.
3. Reran `pip install flash-attn --no-build-isolation` with the fixed torch in place. It built and
   installed cleanly this time (`flash_attn==2.8.3.post1`), confirming the original failure was
   entirely a symptom of (1).

**Minor fidelity note:** my adaptation of `build_env.sh` added `--index-url
https://download.pytorch.org/whl/cu124` to the `torch==2.5.1` install line; the official script
just runs bare `pip3 install torch==2.5.1`. On this platform both resolve to the identical
`torch-2.5.1+cu124` wheel, so the outcome is unaffected, but the discrepancy is recorded here per
"record every external component."

**Not investigated / left as-is:** pip reports pre-existing conflicts against a system-wide `vllm`
installation (`vllm 0.8.5.post1 requires torch==2.6.0`, etc.) that predates this venv and belongs to
a different project's environment (see `reference_server_status.md`); it does not affect this venv
since `torch` and `torchvision` are shadowed locally by `--system-site-packages` precedence rules,
and `vllm`/`xformers`/`outlines`/`compressed-tensors`/`xgrammar` are not part of Show-o2's own
dependency set. No action taken.

## Final verified environment

```
torch          2.5.1+cu124   (cuda 12.4, cuda_available=True)
torchvision    0.20.1+cu124
transformers   4.47.0
diffusers      0.31.0
clip           1.0            (openai/CLIP, installs as `clip`)
flash_attn     2.8.3.post1
nvcc           12.4.99        (matches torch's CUDA build)
```

Full `pip freeze` snapshot: `/apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/pip_freeze.txt`
(outside the repo; available on request, not copied into `configs/admission/showo2/` since it is a
large third-party dependency list, not a Show-o2-specific artifact).

## GPU smoke

Ran inside H20-FoldUMM (8x H20 96GB):

```
GPU0: 0 MiB used, 0% util      <- free, used for this smoke test
GPU1-7: 325 MiB used, 100% util <- benign placeholder daemon (train2.py), not touched
```

`torch.cuda.is_available()==True`, `device_count()==8`. A 4096x4096 matmul on `cuda:0` completed
successfully with expected memory allocation (~168MB), confirming CUDA execution actually works
end-to-end (import succeeding is not sufficient proof by itself).

## Conclusion of this stage

No blocker. The official dependency pin is fully reproducible on H20-FoldUMM once the
`build_env.sh` CLIP-install ordering defect (a genuine upstream issue, documented above rather than
silently patched) is corrected by reasserting the pin. All three downloadable checkpoints are
present locally with recorded hashes; the Wan2.1 VAE was already present per `first-report.md`.
GPU execution (import + matmul) verified on `cuda:0`. Proceeding to stage 3 (understanding +
generation task-path smoke) per `docs/plans/showo2-first-attempt.md`; still no training authorized.
