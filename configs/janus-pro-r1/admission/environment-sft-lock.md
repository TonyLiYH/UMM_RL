# Janus-Pro-R1 SFT environment lock (T720)

Separately pinned from the RL environment below, per the frozen protocol
("Keep SFT and RL environments separately pinned when required").

## Upstream declared requirements

`vendor/janus-pro-r1/requirements-sft.txt` (pinned verbatim upstream, not
edited) is a broad, **unversioned-for-torch/transformers** list (36
packages: `albumentations`, `basicsr`, `boto3`, `clip`, `datasets==2.16.1`,
`deepspeed` (unversioned), `diffusers==0.21.4`, `easydict`, `fairscale`,
`huggingface-hub==0.25.2`, `kornia`, `lpips`, `open-clip-torch==2.7.0`,
`pytorch_lightning`, `taming-transformers-rom1504`, `tensorboard`, `timm`,
etc.). **Neither `torch` nor `transformers` is pinned in this file at all**
— it assumes a base image that already provides them. This is a correction
of an earlier draft of this document that incorrectly asserted a
`torch==2.5.1+cu121`/`transformers==4.38.2` pin; verified by directly
reading `requirements-sft.txt` (no `grep -c torch==` match).

A source-level import scan of `janus-sft/{models,datasets,trainer,utils}`
(`grep -rhoE '^\s*(import|from) ...'`) shows the code paths T720 actually
exercises (model construction, `train_setup`, `run_step`, dataset batching)
only import: `torch`, `torchvision`, `transformers`, `einops`, `timm`,
`PIL`, `numpy`, `easydict`, `attrdict`, `yaml`, `datasets`, and `deepspeed`
(the last only inside `trainer/utils/parameter.py:2,98`, purely for
`deepspeed.add_config_arguments(parser)` — an argparse CLI-flag extension,
never used to wrap or launch the model; `trainer_t2i.py` uses PyTorch's own
`torch.distributed.fsdp.FullyShardedDataParallel`, not the DeepSpeed
engine). None of `albumentations`/`basicsr`/`boto3`/`clip`/`diffusers`/
`fairscale`/`kornia`/`lpips`/`open-clip-torch`/`pytorch_lightning`/
`taming-transformers-rom1504` are imported anywhere on this path (they
belong to other, unrelated tooling/data-prep scripts bundled in the same
top-level requirements file).

## What the T720 SFT smoke actually installs

Built fresh under `/dockerdata/t720-janus-pro-r1/venvs/sft` on H20-FoldUMM
(local SSD, not the pre-existing broken `/root/venvs/janus_pro`), pinned to
`torch==2.6.0` / `transformers==4.50.0` (matching the RL side's own
explicit pins — see `environment-rl-lock.md` for why torch is bumped from
upstream's RL-side `2.5.1+cu121` to `2.6.0` — for a single consistent
CUDA/transformers ABI across both venvs, since upstream leaves the SFT side
unpinned) plus `easydict`,
`attrdict`, `einops`, `timm`, `sentencepiece`, `pillow`, `pyyaml`,
`datasets==2.16.1`. `albumentations`/`basicsr`/etc. are
deliberately **not** installed — they are not on the exercised code path
(see import scan above), and skipping them avoids `apex`/CUDA-extension
build risk unrelated to this smoke's scope.

**Correction to the import-scan claim above:** `deepspeed` *is* a genuine
hard install requirement, not merely an unused CLI-arg extension, despite
`trainer/utils/parameter.py:2,98`'s only functional use being
`deepspeed.add_config_arguments(parser)`. The reason: `trainer/__init__.py`
does `from trainer.utils import *` unconditionally at package-init time
(before any specific submodule like `trainer_t2i` is even reached), and
`trainer/utils/__init__.py` does `from .parameter import parse_args, ...`
unconditionally too — so merely importing `trainer.trainer_t2i.train_setup`
(the one function this smoke actually calls) transitively executes
`import deepspeed` as a side effect of Python's package-init semantics,
regardless of whether `deepspeed`'s functionality is ever invoked. Caught
by an actual `ModuleNotFoundError: No module named 'deepspeed'` on the
first real GPU invocation of `sft_smoke.py`, not by re-reading the source
more carefully in advance.

`deepspeed` is pinned to **`0.15.4`**, not the latest PyPI release
(`0.19.6`, which is what a bare `pip install deepspeed` resolves to): `pip
install deepspeed` (latest) imports without error at the top level but then
fails during `models/modeling_vlm.py`'s own transitive import chain with
`ValueError: infer_schema(func): Parameter partition_sizes has unsupported
type list[int]` raised from `torch/_library/infer_schema.py` — a newer
`deepspeed` release registers a custom op via `torch.library.custom_op`
using a builtin-generic type hint (`list[int]`) that this `torch==2.6.0`'s
stricter `infer_schema` parser rejects (it only accepts
`typing.List[int]`, not the PEP 585 builtin alias, at this torch version).
`deepspeed==0.15.4` (the same version upstream's own
`requirements-rl.txt` pins for the RL side, reused here for consistency)
predates that custom-op registration and imports cleanly. Verified
end-to-end: `python -c "from trainer.trainer_t2i import train_setup"`
succeeds inside the SFT venv with `deepspeed==0.15.4` installed.

Also newly discovered, and installed into **both** the SFT and RL venvs (not
SFT-specific, so documented once here and cross-referenced from
`environment-rl-lock.md`): `scipy==1.15.3` (from the shared local wheel
cache, cp310 build). Neither venv is `comppareto`-specific — but
`sft_smoke.py`/`grpo_smoke.py` are invoked as `python -m
comppareto.adapters.janus_pro_r1.{sft_smoke,grpo_smoke}`, and Python's
`runpy` module-execution machinery always imports the full parent package
chain first, i.e. the top-level `src/comppareto/__init__.py` (outside
T720's `allowed_paths`, not editable), which unconditionally does `from
.quadratic import (...)`, and `quadratic.py` does `from scipy.optimize
import minimize` — a dependency of the broader `comppareto` package
entirely unrelated to the Janus-Pro-R1 adapter. Caught the same way as the
`deepspeed` gap: an actual `ModuleNotFoundError: No module named 'scipy'`
on the first real GPU invocation, not by static analysis alone. Installing
`scipy` (a small, pure-numeric package with no bearing on the Janus model
code itself) into both venvs is the correct fix — it does not require
editing any file outside `allowed_paths`.

## flash_attn: genuine hard requirement, unlike the RL side

`models/modeling_vlm.py`'s `MultiModalityCausalLM.__init__` (line 219)
hardcodes `language_config._attn_implementation = 'flash_attention_2'`
**unconditionally, with no `try/except` guard** (confirmed via
`grep -n "_attn_implementation\|flash_attn\|attn_implementation"` — a
single unconditional hit) before constructing
`LlamaForCausalLM(language_config)`. This is unlike the RL side's InternVL
code, which guards the same import and falls back to eager attention (see
`environment-rl-lock.md`). `flash_attn==2.7.4.post1+cu12torch2.6cxx11abiFALSE`
(exact ABI match for this venv's `torch==2.6.0+cu124`/cp310/cxx11abi=False,
identified via the container's own `torchinfo.py` check) was installed from
the shared local wheel cache
(`/apdcephfs_cq7/share_1447896/yihangli/tmp/vllm-wheels/`) directly by wheel
filename (a `pip install pkg==<version>` spec was rejected: pip normalizes
the wheel's local-version segment to lowercase per PEP 440,
`cxx11abiFALSE` -> `cxx11abifalse`, so the exact-cased version string in the
filename doesn't match pip's own normalized metadata — installing by file
path sidesteps that string comparison entirely). Verified with a real GPU
forward pass on GPU1 (`CUDA_VISIBLE_DEVICES=1`): constructed a tiny
`LlamaForCausalLM` with `_attn_implementation="flash_attention_2"`, ran a
bf16 autocast forward pass, asserted finite logits — passed
(`flash_attention_2 minimal LlamaForCausalLM forward OK, logits finite,
shape (1, 8, 32)`).

## attrdict / Python 3.10 compatibility shim

`models/{projector.py,modeling_vlm.py}` do `from attrdict import AttrDict`
on the real model-construction path (not an unused CLI-only import).
`attrdict==2.0.1` (the only PyPI release) itself does
`from collections import Mapping, MutableMapping, Sequence` at import time;
all three names were removed from the top-level `collections` namespace in
Python 3.10 (this venv's interpreter), moved to `collections.abc` only, so
a bare `import attrdict` raises `ImportError`. Fix: a small stdlib
compatibility shim, applied at the top of `sft_smoke.py`'s
`run_sft_smoke()` (and mirrored in the RL side's `grpo_smoke.py` — the same
`models` package is vendored into both `janus-sft` and `janus-rl`) *before*
`attrdict` or any vendored module is imported: re-alias every non-dunder
`collections.abc` name onto the top-level `collections` module if not
already present. No vendored file is edited; this is a process-level
interpreter compatibility patch, standard practice for this well-known
`attrdict`/3.10 incompatibility. Verified end-to-end in the SFT venv:
`attrdict shim OK: .../venvs/sft/lib/python3.10/site-packages/attrdict/__init__.py`,
followed by successful `torch`/`torchvision`/`transformers`/`easydict`/
`einops`/`timm`/`sentencepiece`/`PIL`/`yaml`/`datasets` imports.

## Deviations from the upstream full multi-GPU SFT path

1. **No `torch.distributed` / FSDP launch.** Upstream's
   `janus-sft/launch.py` spawns one process per visible GPU via raw
   `os.system("... python train.py --rank=... --world_size=...")` (a
   TCP-rendezvous `torch.distributed.init_process_group`, not `torchrun`
   and not the `deepspeed` launcher), and `TextToImageTrainer.__init__`
   hard-requires `cfg.common.use_fsdp` and wraps the model in
   `FSDP(..., HYBRID_SHARD)` across all ranks. T720's bounded single-GPU
   smoke instead loads `MultiModalityCausalLM` directly on one device
   (GPU1) with plain `torch.optim.AdamW`, reusing upstream's
   `train_setup()` (trainable/frozen split) and the exact `run_step()`
   loss/backward/clip/step sequence unmodified, just without the
   `FSDP`/`dist`/multi-process wrapper. This is a documented scope
   reduction for a "stack smoke," not a claim of reproducing the
   distributed training path.
2. **No `torch.distributed.init_process_group`.** `dist.get_rank()` /
   `dist.get_world_size()` calls in upstream code are bypassed by not
   invoking `launch.py`/`train.py`/`TextToImageTrainer` themselves; the
   adapter code calls the lower-level pieces
   (`MultiModalityCausalLM.from_pretrained`, `train_setup`,
   `build_optimizer`, the per-`task_type` batch construction and forward
   call) directly, single-process.

## Why this is sufficient for the smoke

The required checks are "SFT loss/backward/optimizer/scheduler/save/resume"
on ≤4 optimizer steps — none of which depend on multi-GPU sharding. Using
upstream's own `train_setup()` and per-task_type batch construction/forward
call keeps the exercised code authentically upstream; only the
distributed-launch scaffolding around it is swapped for a single-GPU
equivalent.
