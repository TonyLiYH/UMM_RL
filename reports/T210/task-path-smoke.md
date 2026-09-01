# T210 — Understanding and generation task-path smoke

Stage 3 of the T210 execution protocol (`docs/plans/showo2-first-attempt.md`): run one official or
minimally-adapted understanding example and one generation example on the pinned `showo2-1.5B`
checkpoint, on GPU, inside H20-FoldUMM. No training was started or authorized; nothing outside
`configs/admission/showo2/`, `runs/admission-showo2/`, `reports/T210/` was written inside the repo.

## What ran, unmodified from official

Both smoke runs used the unmodified official scripts (`inference_mmu.py`, `inference_t2i.py`) and
the unmodified official config `configs/showo2_1.5b_demo_432x432.yaml`, against the pinned commit
`45a5a2de01d1ebd10cd5864d29310a76476cdf23` cloned to
`/apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/Show-o/`. No line of Show-o2
source code or config was edited. Adaptation was limited to environment variables and official
CLI-override arguments (both mechanisms the scripts themselves expose):

- `HF_HOME` pointed at a CQ7 cache dir (`/apdcephfs_cq7/share_1447896/yihangli/models/pretrained/hf_cache`)
  so `from_pretrained(repo_id)` calls populate the *default* HF cache rather than the flat
  `local_dir` layout used by the earlier `download_checkpoints.sh` (which `from_pretrained` cannot
  read directly). Network access via the container's `star_proxy` was used, letting `from_pretrained`
  do its own official download/caching exactly as the unmodified code intends.
- `WANDB_MODE=offline`, since both scripts call `wandb.init()` unconditionally and this container
  has no interactive wandb login.
- `Wan2.1_VAE.pth` placed via symlink into `show-o2/` (pointing at the existing, hash-verified CQ7
  copy at `Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth`) — this is the literal placement the official README
  instructs ("put it on the current directory"), not a path override.
- t2i smoke only: `dataset.params.validation_prompts_file` overridden to a single-line scratch
  prompt file (official CLI-override mechanism, same pattern as the README's own
  `batch_size=... guidance_scale=...` overrides) and `num_inference_steps=10` (README default: 50)
  to keep the smoke run fast. This is a speed-only deviation, recorded here per "record every
  external component" / minimal-adaptation discipline — it does not change model logic, weights,
  or the config's own defaults file.

## Two environment defects found and fixed (both upstream, both unpinned transitive deps)

Same category of issue as the `torch`/`torchvision` clobber recorded in
`environment-checkpoint-smoke.md`: `build_env.sh` leaves several dependencies of the scripts it
installs unpinned, and the versions pip resolves today are incompatible with the official inference
code.

1. **`wandb` + `protobuf`.** `build_env.sh` runs bare `pip3 install wandb` (no version pin), which
   today resolves to `wandb==0.29.0`. `wandb==0.29.0`'s generated `wandb/proto/wandb_telemetry_pb2.py`
   dispatches on the installed `google.protobuf` major version, but only has branches for majors
   `5`/`6`/`7` — not `4`, which is what `build_env.sh`'s dependency chain actually installs
   (`protobuf==4.25.9`, pulled in transitively, also unpinned by the official script). Both
   `inference_mmu.py` and `inference_t2i.py` do `import wandb` unconditionally at module load, so
   this broke both task paths before any model code ran.
   - First attempted fix: upgrade `protobuf` to `5.29.6` (the range wandb 0.29.0's dispatcher
     supports). This worked for wandb but is itself a new unpinned-conflict risk: `tensorflow==2.16.1`
     (also in `build_env.sh`, unpinned against protobuf) declares `protobuf<5.0.0dev` and would break
     under this fix even though neither inference script imports tensorflow directly.
   - **Final fix, adopted instead:** pin `wandb==0.17.0`, the newest wandb release confirmed (by
     direct probe) to still expose `wandb.util.generate_id` under the *existing* `protobuf==4.25.9` —
     the version the rest of the unpinned stack (`onnx`, `onnxruntime`, `tensorflow`) already expects.
     This avoids introducing a new conflict rather than trading one for another. Verified:
     `wandb.util.generate_id()` and a full offline `wandb.init()` → `wandb.log()` → `wandb.finish()`
     cycle succeed; `torch`/`torchvision`/`transformers`/`diffusers`/`clip`/`flash_attn`/`onnx`/
     `onnxruntime` all still import cleanly afterward.
2. **`wandb.util.generate_id` itself.** Independently of the protobuf dispatch bug, this attribute
   was removed from `wandb.util` in some release between `0.17.0` and `0.29.0`. `inference_mmu.py`
   and `inference_t2i.py` both call `wandb.util.generate_id()` at the top of `__main__`, using an API
   that no longer exists on whatever "latest wandb" `build_env.sh`'s unpinned install resolves to
   today. This is a genuine version-drift break in the official code path, not an environment
   misconfiguration; pinning `wandb==0.17.0` (above) resolves it as a side effect.

Both fixes are dependency-version corrections that restore compatibility for packages `build_env.sh`
itself installs unpinned — no Show-o2 source, config, or checkpoint was substituted or altered, and
both are fully recorded here per "record every external component" / "do not use unofficial fixes...
without local authorization." Updated `pip freeze` snapshot:
`/apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/pip_freeze.txt` (outside the repo).

## Newly discovered external component (not in `first-report.md`): safety checker

`inference_t2i.py` (generation path only — `inference_mmu.py` does not need this) unconditionally
loads:

```python
processor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
```

This was not recorded in `first-report.md`'s checkpoint/dependency inventory. Recorded now:

| Component | Repo ID | Resolved revision | Main weight file | SHA-256 | Cache size |
|---|---|---|---|---|---|
| SD safety checker | `CompVis/stable-diffusion-safety-checker` | `697f152c6b6183309a1509fb3589fcbaae7f34e5` | `model.safetensors` | `64b8393f1afd5a0c1ed2aa5f341fa7c08286839a48f3743162a76a2835c808bd` | 2.3GB |

**License open item:** the model card states "License: More information needed" — no license
identifier is published for this repo. Recorded as an open item (same treatment as the unresolved
`Wan2.1_VAE.pth` license note in `first-report.md`), not a blocker, since it is used strictly
read-only for inference-time output filtering and no redistribution is planned.

All four downloadable components (`show-o2-1.5B`, `Qwen2.5-1.5B-Instruct`, `siglip-so400m`, and now
the safety checker) are present in the default HF cache at
`/apdcephfs_cq7/share_1447896/yihangli/models/pretrained/hf_cache/hub/`, populated by the scripts'
own unmodified `from_pretrained(repo_id)` calls over the container's proxied network access — not by
manual placement. One transient `ChunkedEncodingError` occurred during the safety-checker download
(network blip through `star_proxy`); `huggingface_hub`'s own retry logic recovered it automatically
and the run completed with exit 0 — noted here as a flaky-network event, not a persistent defect.

## Understanding-path smoke (`inference_mmu.py`)

Command (env vars above, otherwise unmodified):
```
python3 inference_mmu.py config=configs/showo2_1.5b_demo_432x432.yaml \
    mmu_image_path=./docs/mmu/pexels-jane-pham-727419-1571673.jpg \
    question='Describe the image in detail.'
```
Ran to completion (exit 0). Model load (fp32 checkpoint, `use_safetensors=False`, ~22GB) took about
26 minutes on this container's storage backend; inference itself (VAE encode + `mmu_generate`) took
under a minute. The real model output (recovered from the wandb offline run's media caption, since
neither script prints to stdout):

> User: Describe the image in detail.
> Answer: The image captures a serene setting on a white lace tablecloth. Dominating the scene is a
> beige scarf, casually draped over the tablecloth. The scarf is adorned with the word "Inspire"
> written in a black cursive font, adding a touch of elegance to the scene. To the right of the
> scarf, a black choker necklace is casually placed. The choker is distinguished by a small pink
> flower pendant, adding a pop of color to the otherwise monochrome piece. On the left side of the
> scarf, a clear glass bottle with a gold cap rests. [...]

This confirms the understanding path runs end-to-end on GPU with the pinned checkpoint and produces
coherent, on-topic output — no correctness judgment is claimed beyond "the path works as wired."

## Generation-path smoke (`inference_t2i.py`)

Command (env vars above; `num_inference_steps` and `validation_prompts_file` overridden as noted):
```
python3 inference_t2i.py config=configs/showo2_1.5b_demo_432x432.yaml \
    batch_size=1 guidance_scale=5.0 num_inference_steps=10 \
    dataset.params.validation_prompts_file=<scratch single-line prompt file>
```
Prompt used: "A red bicycle leaning against a brick wall, photorealistic, natural daylight." Ran to
completion (exit 0) in about 26 seconds of sampling time (model/VAE/safety-checker already resident
from the same process's earlier loads). Output: one 432x432 PNG, produced by the transport/flow-
matching sampler, decoded by the Wan2.1 VAE, and passed through the safety checker with no NSFW flag
raised, logged to the wandb offline run's `media/images/`. This confirms the generation path runs
end-to-end on GPU including the previously-undocumented safety-checker dependency.

## Conclusion of this stage

No blocker. Both task paths (`inference_mmu.py`, `inference_t2i.py`) run to completion on GPU with
the pinned `showo2-1.5B` checkpoint, using only official code/config plus environment-variable and
official-CLI-override adaptation. Two upstream dependency-pin defects (`wandb`/`protobuf` version
incompatibility, `wandb.util.generate_id` removal) were found, root-caused, and corrected the same
way as the earlier `torch`/`torchvision` defect — by restoring compatibility among the stack's own
unpinned transitive dependencies, not by substituting any Show-o2 component. One previously
undocumented external checkpoint dependency (`CompVis/stable-diffusion-safety-checker`, generation
path only) is now recorded, including its unresolved license status as a non-blocking open item.
Proceeding to stage 4 (parameter-block registry draft) per `docs/plans/showo2-first-attempt.md`;
still no joint post-training authorized or started.
