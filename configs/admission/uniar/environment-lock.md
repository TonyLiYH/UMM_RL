# UniAR admission — environment lock (T240)

Task-scoped environment description for the H20-FoldUMM GPU container venv used for
every T240 smoke run (`/root/venvs/uniar`, final build: `python3 -m venv /root/venvs/uniar`,
**without** `--system-site-packages` — see defect #3 below for why the initial
`--system-site-packages` build had to be abandoned). This file records the exact
package versions verified working on this platform and the three non-obvious
dependency-resolution defects encountered while building the venv from the official
`pyproject.toml` pins, following T210's precedent of recording such defects explicitly
rather than silently patching around them.

## Official pins (from `pyproject.toml`, pinned revision `92d8718d4cf282254ae63a4944b07edba0ce7abf`)

- `torch==2.7.0`, `transformers==4.57.0`, `diffusers==0.37.1`, `accelerate>=1.4.0`
- `einops`, `safetensors`, `qwen-vl-utils`, `Pillow`, `torchvision`, `numpy`, `tqdm`
- `requests>=2.31`, `sentencepiece`, `protobuf`
- `flash-attn` (recommended, `--no-build-isolation`)

## Defects encountered and required additional constraints

1. **`transformers`/`diffusers`/`accelerate` install silently downgrades `torch`.**
   Installing `torch==2.7.0` first (from `https://download.pytorch.org/whl/cu126`)
   succeeds cleanly. However, the subsequent
   `pip install transformers==4.57.0 diffusers==0.37.1 "accelerate>=1.4.0" ...` step
   has an unpinned `torch>=2.0.0` dependency (via `accelerate`) that pip's resolver
   satisfies by *downgrading* the already-installed `torch==2.7.0+cu126` to
   `torch==2.6.0` (plain, no `+cu124`/`+cu126` local version suffix — pulled from
   PyPI's default index rather than the PyTorch CUDA wheel index), along with a full
   set of `nvidia-*-cu12` runtime libraries at older (12.4.x-class) versions.
   `torch.cuda.is_available()` still returned `True` after this downgrade (the
   container's driver is compatible with both builds), so the defect is easy to miss
   if only a smoke-level `import torch; torch.cuda.is_available()` check is run.
   **Fix**: reassert `pip install torch==2.7.0 --index-url
   https://download.pytorch.org/whl/cu126` and `pip install --force-reinstall
   --no-deps torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126`
   **after** the transformers/diffusers/accelerate install step, matching T210's
   R7 lesson for Show-o2's CLIP-triggered torch downgrade (same root cause class:
   an unpinned downstream dependency's own `torch` constraint overriding an
   earlier explicit pin).
2. **`flash-attn`'s compiled CUDA extension is ABI-locked to the `torch` build present
   at `pip install` time; reinstalling `flash-attn` without `--no-cache-dir` reuses a
   stale cached wheel built against the wrong `torch` ABI.** After fix #1 restored
   `torch==2.7.0+cu126`, a plain `pip uninstall flash-attn && pip install flash-attn
   --no-build-isolation` reinstalled from a previously-cached wheel (built during the
   earlier step, against the since-downgraded `torch==2.6.0`), producing at import
   time: `ImportError: .../flash_attn_2_cuda...so: undefined symbol:
   _ZN3c105ErrorC2ENS_14SourceLocationESs` (a torch C++ ABI symbol mismatch).
   **Fix**: `pip cache remove flash_attn` then `pip install flash-attn
   --no-build-isolation --no-cache-dir`, forcing a fresh source build against the
   now-correct `torch==2.7.0+cu126`. This built and imported cleanly.

3. **`python3 -m venv --system-site-packages` leaks a container-wide, pre-existing
   `xformers==0.0.29.post2` into the venv, and that `xformers` build is broken against
   the `triton==3.3.0` pinned transitively by `torch==2.7.0`.** `diffusers` optionally
   imports `xformers` if present (`diffusers.models.attention_processor`: `if
   is_xformers_available(): import xformers.ops`), and this import chain is reached
   simply by importing `diffusers.loaders.peft` (itself imported by
   `uniar.modeling_vision_decoder`, which `inference/generate.py` needs for
   `UniARVisualDecoder`). The container's system-level `xformers` triggers a Triton
   JIT-kernel-unrolling code path (`xformers/triton/vararg_kernel.py:
   unroll_varargs()`) that tries to mutate a `triton.JITFunction`'s `.src` attribute
   directly — an operation Triton 3.3.0 now hard-rejects
   (`AttributeError: Cannot set attribute 'src' directly. Use
   '_unsafe_update_src()'...`), crashing the entire `diffusers` import and, transitively,
   `inference/generate.py`. This is a defect in the **container's pre-existing
   system Python's `xformers`/`triton` combination**, not in UniAR or diffusers code,
   and `xformers` is not a UniAR dependency at all (absent from `pyproject.toml`/
   `requirements.txt`). **Fix**: rebuild the venv **without**
   `--system-site-packages` (`python3 -m venv /root/venvs/uniar`, fully isolated), so
   the broken system `xformers` is never importable inside the venv; `diffusers`
   correctly falls back to `xformers = None` / `is_xformers_available() == False` and
   imports cleanly. Confirmed via `python3 -c "from diffusers import
   SD3Transformer2DModel"` succeeding post-rebuild. This required repeating fixes #1
   and #2 above inside the freshly created isolated venv (the same
   transformers/diffusers/accelerate install step again downgrades `torch` to an even
   newer unpinned build — observed `torch==2.14.0`+CUDA13 wheels on the second build —
   and `flash-attn` again needs a `--no-cache-dir` rebuild afterward).

4. **A partial, targeted `pip uninstall` of only the CUDA13-named `nvidia-*` packages
   (left over from the `torch==2.14.0` episode in fix #3) did not fully clean the venv,
   leaving both unsuffixed CUDA13 packages (e.g. `nvidia-cublas 13.1.1.3`,
   `nvidia-cudnn-cu13 9.24.0.43`) and correctly-suffixed CUDA12.6 packages (e.g.
   `nvidia-cublas-cu12 12.6.4.1`) installed simultaneously.** This mixed toolchain did
   **not** break `torch` import or `torch.cuda.is_available()` (which still returned
   `True`), but broke the first real cuDNN-backed op: `inference/generate.py`'s SD3
   visual decoder failed at its `pos_embed` `Conv2d` step with `RuntimeError: cuDNN
   error: CUDNN_STATUS_NOT_INITIALIZED`. This was first misdiagnosed as GPU contention
   (a stale zombie process was found holding 23GB on GPU 0 and cleared), but the error
   recurred identically after that process exited, and was confirmed to be an
   environment defect (not UniAR-specific, not GPU-contention-specific, not a smoke
   fluke) via a minimal standalone repro with no UniAR code at all: `python3 -c "import
   torch; a=torch.randn(4,4,8,8,device=0); c=torch.nn.Conv2d(4,8,3,padding=1).to(0);
   print(c(a).shape)"`, which reproduced the identical `CUDNN_STATUS_NOT_INITIALIZED`
   error in isolation. **Fix**: purged *all* `nvidia-*`/`cuda-toolkit`/`cuda-bindings`
   packages (both suffixed and unsuffixed) plus `torch`/`torchvision`/`flash-attn`
   entirely, then reinstalled `torch==2.7.0` and `torchvision==0.22.0` **together in a
   single pinned command**
   (`pip install torch==2.7.0 torchvision==0.22.0 --index-url
   https://download.pytorch.org/whl/cu126`) rather than as two sequential commands, to
   prevent the resolver from drifting again, then rebuilt `flash-attn` fresh with
   `--no-cache-dir`. Verified via: `pip list | grep nvidia` showing only clean
   `-cu12`-suffixed packages; the standalone `Conv2d` repro succeeding (`conv2d ok
   torch.Size([4, 8, 8, 8])`); `flash_attn 2.8.3.post1` importing cleanly; and
   `from diffusers import SD3Transformer2DModel` importing cleanly.

No other package required a version different from what a straight
`pip install -e .` naturally resolves to, once the four fixes above are applied in
order (torch reassertion, flash-attn rebuild, dropping `--system-site-packages`, and
purging the stray unsuffixed/`-cu13` nvidia packages before reinstalling torch and
torchvision together in a single pinned command).

## Full verified environment (confirmed live 2026-09-03, H20-FoldUMM container)

| Package | Version |
|---|---|
| `torch` | `2.7.0+cu126` |
| `torchvision` | `0.22.0+cu126` |
| `transformers` | `4.57.0` |
| `diffusers` | `0.37.1` |
| `accelerate` | `1.14.0` |
| `flash_attn` | `2.8.3.post1` |
| `huggingface_hub` | `0.36.2` |
| `tokenizers` | `0.22.2` |
| `qwen_vl_utils` | `0.0.14` |
| CUDA (driver-visible via `torch.cuda.is_available()`) | `True` on H20 GPU 0 |

Full `pip freeze` snapshot saved on the container at `/root/venvs/uniar_pip_freeze.txt`
(not copied into this repo; large third-party dependency list, same pattern as T210).

## PyPI yank notice (informational, non-blocking)

pip flagged during install: `WARNING: The candidate selected for download or install
is a yanked version: 'transformers' candidate (version 4.57.0 ...) Reason for being
yanked: Error in the setup causing installation issues`. This is the exact version
pinned by the official `pyproject.toml`. The install completed and all imports
(`transformers.__version__ == "4.57.0"`) succeeded without observed issues during the
smokes run under this admission; recorded here for visibility, not treated as a
blocker since it is the upstream repository's own pin, not a substitution made by
this admission.

## Scope note

This lock file governs the UniAR inference venv used for T240's read-only admission
smoke tests only (`inference/chat.py`, `inference/generate.py`, and read-only
`named_parameters()` enumeration). It authorizes no training of any kind — neither
the released AR/GRPO training path nor (especially) the unreleased visual-decoder
training path, which remains undocumented by the upstream repository and is not
attempted here.
