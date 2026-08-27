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
- 2 unresolved license-status open items (Wan2.1 VAE, safety checker), both non-blocking for the
  inference-only work performed, both flagged for local-reviewer decision before any successor task
  that would redistribute outputs or begin training.
- 2 environment defects found and fixed via dependency-pin restoration (not unofficial substitution),
  both fully documented.
- No unofficial fix was applied to Show-o2 source, config, or checkpoint; no different checkpoint was
  substituted; no joint post-training was started, per the frozen protocol.
