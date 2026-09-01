# T210 failure ledger

## Unresolved anomalies (require local-reviewer decision)

### 1. Wan2.1 VAE license unresolved

- **Component**: `Wan2.1_VAE.pth`, sourced from `Wan-AI/Wan2.1-T2V-14B` per the Show-o2 README, but
  not resolved from a specific HF commit — it was already present locally (pre-existing copy at
  `/apdcephfs_cq7/share_1447896/yihangli/models/pretrained/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth`,
  sha256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`, confirmed identical
  across the two on-disk locations it exists at).
- **Issue**: no specific HF repo revision/commit was resolved for this file (it predates this
  task's own downloads), so its exact provenance chain is one level less precise than the other
  four components in `result-summary.md`'s table, all of which were resolved via a live
  `from_pretrained`/`snapshot_download` call this task performed.
- **Disposition**: not blocking — the file's identity is confirmed by hash, it is used read-only as
  a frozen encoder/decoder (never trained), and it was placed via a symlink into the Show-o2 working
  directory exactly as the official README instructs ("put it on the current directory"), not
  copied or modified.
- **Recommendation for local review**: if a stricter provenance chain is required before any
  further T210-successor work, re-resolve this file via a fresh `from_pretrained("Wan-AI/Wan2.1-T2V-14B", ...)` call under the same `HF_HOME` cache used for the other four components, recording
  the resulting revision hash. Not done in this task since the pre-existing hash-verified copy
  already satisfies the "official source" requirement.

### 2. `CompVis/stable-diffusion-safety-checker` license unspecified

- **Component**: `CompVis/stable-diffusion-safety-checker`, an undocumented dependency of the
  generation path discovered this stage (not present in the original `first-report.md` inventory —
  `inference_t2i.py` loads it unconditionally at lines 92-93, `first-report.md`'s audit did not
  trace this far into the generation script before GPU execution began).
- **Issue**: the HF model card states "License: More information needed" — no license identifier
  is published for this repo.
- **Disposition**: not blocking for the smoke test performed — the component is used strictly
  read-only for inference-time NSFW output filtering; no redistribution occurred or is planned.
- **Recommendation for local review**: if joint post-training or any redistribution of
  generation-path outputs is planned in a successor task, confirm whether this dependency's
  unspecified license permits that use case before proceeding; this task did not need to make that
  determination since it performed inference only.

## Resolution update (T210 R6, 2026-08-28)

The two license/provenance items above were re-investigated per local review R6. Findings below;
the original entries are left unmodified above per the no-edit-history convention.

### 1. Wan2.1 VAE — resolved

A live `huggingface_hub.HfApi().model_info("Wan-AI/Wan2.1-T2V-14B")` query (via the `star_proxy`
network path, not the offline SSD cache used for the smoke runs themselves) returned:

- **Resolved revision**: `a064a6c71f5be440641209c07bf2a5ce7a2ff5e4`
- **License**: `apache-2.0` (from `card_data.license`; repo tags include `license:apache-2.0`)

This matches the Apache-2.0 license already established for every other Show-o2 dependency in
`first-report.md`. The pre-existing local copy's hash
(`38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`) is unaffected by this query —
this only adds the missing revision/license provenance, it does not re-download or replace the
file. **No longer an open item.**

### 2. `CompVis/stable-diffusion-safety-checker` — constrained, not resolvable further

The same live query confirms the license field is still unset (`card_data.license is None`, no
`license:*` tag) — the "License: More information needed" status on the model card is current, not
stale. This cannot be resolved further from our side; per local review R6, it is instead formally
constrained as follows:

- **Defined as an optional, display-only dependency.** Reading `inference_t2i.py` (lines 92-93,
  201-212): the safety checker runs strictly *after* the model has produced its generated image
  (`images`/`pil_images`, line 198-199) and *only* gates what gets attached to the `wandb.log` call
  for human-facing display (`checked_images`, `has_nsfw_concept`). It does not feed back into the
  model, the transport/flow-matching sampler, or any tensor used for a scientific metric.
- **Scientific evaluation path that does not require it**: any downstream evaluation (e.g.
  computing FID/CLIP-score/other image-quality metrics) should read `images`/`pil_images` directly
  at line 198-199, before the safety-checker call at line 201, and never construct or import
  `StableDiffusionSafetyChecker`. This makes the unspecified-license dependency avoidable for any
  evaluation pipeline while leaving the unmodified official inference script's own display path
  untouched.
- **Disposition remains non-blocking** for the read-only smoke inference performed by this task,
  which used the official script unmodified (safety checker included, no redistribution of its
  output occurred). The above constraint is provided for any successor task (e.g. training-loop
  evaluation) that would otherwise need to decide whether to depend on it.

## Resolved issues (not outstanding — recorded for completeness)

### 1. `torch`/`torchvision` version clobber (environment defect)

- **Symptom**: `build_env.sh`'s unpinned install order let a later step silently replace the pinned
  CUDA-matched `torch`/`torchvision` build with an incompatible one.
- **Fix**: reinstalled the pinned `torch==2.5.1+cu124`/`torchvision==0.20.1+cu124`. Detailed in
  `environment-checkpoint-smoke.md`. Verified: all downstream packages (`transformers`, `diffusers`,
  `clip`, `flash_attn`, `onnx`, `onnxruntime`) still import; matmul smoke on `cuda:0` succeeded.

### 2. `wandb`/`protobuf` incompatibility + `wandb.util.generate_id` removal (environment defect)

- **Symptom**: `build_env.sh`'s unpinned `pip3 install wandb` resolved to `wandb==0.29.0`, whose
  protobuf-major-version dispatcher lacks a branch for the resolved `protobuf==4.25.9`, breaking
  `import wandb` outright; additionally, `wandb==0.29.0` has removed `wandb.util.generate_id`, which
  both `inference_mmu.py` and `inference_t2i.py` call unconditionally.
- **Fix**: pinned `wandb==0.17.0` (confirmed by direct probe to be the newest release still exposing
  `generate_id`, and independently compatible with the existing `protobuf==4.25.9` with no further
  version change needed). Detailed in `task-path-smoke.md`. Verified: full offline
  `wandb.init()`/`wandb.log()`/`wandb.finish()` cycle succeeds; both inference scripts run to
  completion afterward.

### 3. Transient network error during safety-checker download (non-blocking)

- **Symptom**: `ChunkedEncodingError(ProtocolError('Response ended prematurely'))` during the t2i
  smoke run's proxied download of `CompVis/stable-diffusion-safety-checker`.
- **Disposition**: self-resolved by `huggingface_hub`'s internal retry logic; the run completed with
  exit 0 and a fully populated cache regardless. Not a persistent defect — recorded for completeness
  only.

## Summary

- No task-path execution failure occurred: both `inference_mmu.py` and `inference_t2i.py` ran to
  completion on the first attempt after the environment defects above were fixed.
- Of the 2 original license-status open items: the Wan2.1 VAE item is now fully **resolved**
  (revision + Apache-2.0 license confirmed live, see "Resolution update (T210 R6)" above); the
  safety-checker item remains **formally constrained** (license genuinely unspecified upstream,
  not resolvable from our side) rather than resolved, with an optional-dependency definition and a
  safety-checker-free evaluation path now recorded for any successor task.
- 2 environment defects found and fixed via dependency-pin restoration (not unofficial substitution),
  both fully documented.
- No unofficial fix was applied to Show-o2 source, config, or checkpoint; no different checkpoint was
  substituted; no joint post-training was started, per the frozen protocol.
