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
`torch==2.5.1+cu121` / `transformers==4.50.0` (matching the RL side's own
explicit pins, for a single consistent CUDA/transformers ABI across both
venvs, since upstream leaves the SFT side unpinned) plus `easydict`,
`attrdict`, `einops`, `timm`, `sentencepiece`, `pillow`, `pyyaml`,
`datasets==2.16.1`. `deepspeed`/`albumentations`/`basicsr`/etc. are
deliberately **not** installed — they are not on the exercised code path
(see import scan above), and skipping them avoids `apex`/CUDA-extension
build risk unrelated to this smoke's scope.

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
