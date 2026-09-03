# T240 failure ledger

## Unresolved anomalies (require local-reviewer decision)

### 1. No standalone root `LICENSE` file in the UniAR repository

- **Component**: the UniAR repository itself (`https://github.com/ShareLab-SII/UniAR`,
  pinned revision `92d8718d4cf282254ae63a4944b07edba0ce7abf`).
- **Issue**: license is declared only in `pyproject.toml`
  (`license = {text = "Apache-2.0"}`); the only `LICENSE` file physically present in
  the tree belongs to the vendored `train/rl/trl/` subtree (itself Apache-2.0-licensed
  HuggingFace TRL), not the UniAR-authored code.
- **Disposition**: not blocking — three independent apache-2.0 declarations exist
  (the `pyproject.toml` field, the `UniAR-RL` HF model card, and the `UniAR-SFT` HF
  model card), all consistent with each other.
- **Recommendation for local review**: if a stricter provenance chain is required
  before any successor task's redistribution of derived artifacts, request a
  standalone root `LICENSE` file be added upstream, or independently confirm via the
  GitHub API's license-detection field. Not done in this task since the three
  independent declarations already satisfy the "record license" requirement.

### 2. `sd3_pipeline/LICENSE.md` (Stability AI's SD3.5-medium terms) not independently re-fetched

- **Component**: `sd3_pipeline/` inside `ShareLab-SII/UniAR-RL` — a bundled
  third-party Stable Diffusion 3.5-medium pipeline (VAE + text encoders), governed by
  its own `LICENSE.md` file shipped inside the HF repo.
- **Issue**: this admission read the bundled `LICENSE.md` as shipped but did not
  independently re-fetch or diff it against Stability AI's own canonical release terms
  to confirm it has not been altered or is not stale relative to the upstream original.
- **Disposition**: not blocking — the pixel decoder (`sd3_transformer` +
  `sd3_pipeline`) is used strictly read-only/frozen as a fixed component of the
  generation smoke; no redistribution of decoder weights, fine-tuned derivatives, or
  generated pixel-decoder outputs beyond the single smoke PNG is planned by this
  admission.
- **Recommendation for local review**: if a successor task plans to redistribute
  decoder-derived outputs at scale, or to fine-tune the SD3 pixel decoder itself,
  independently confirm the bundled license terms against Stability AI's canonical
  release before proceeding.

## Resolved issues (not outstanding — recorded for completeness)

### 1. `torch` silently downgraded by transitive package installs (occurred twice, environment defect)

- **Symptom**: installing `transformers==4.57.0`/`diffusers==0.37.1`/
  `accelerate>=1.4.0` per `pyproject.toml` pulled in a different `torch` build than
  the one explicitly pinned (`torch==2.7.0+cu126`), silently replacing it.
- **Fix**: reinstalled the pinned `torch==2.7.0+cu126`/`torchvision==0.22.0+cu126`
  after each occurrence. Detailed in `configs/admission/uniar/environment-lock.md`.
  Verified: `torch.__version__`/`torchvision.__version__` confirmed pinned after each
  fix; downstream imports unaffected.

### 2. `flash-attn` ABI staleness after each torch reinstall (environment defect)

- **Symptom**: the previously-built `flash-attn` wheel's compiled ABI no longer
  matched the just-reinstalled torch build, breaking `import flash_attn` with an
  ABI-mismatch error.
- **Fix**: rebuilt `flash-attn` with `--no-cache-dir` (`--no-build-isolation`, per the
  official `pyproject.toml`'s recommended install) after every torch reassertion.
  Detailed in `environment-lock.md`. Verified: `import flash_attn` succeeded, and the
  generation smoke's attention layers ran without an ABI error on the final attempt.

### 3. Broken system-level `xformers`/`triton` leaking in via `--system-site-packages` (environment defect)

- **Symptom**: the venv was initially built with `--system-site-packages`, which let
  an incompatible system-wide `xformers` install shadow the venv's own package,
  causing an import-time failure unrelated to any UniAR-pinned dependency.
- **Fix**: dropped `--system-site-packages` entirely and rebuilt the venv fully
  isolated. Detailed in `environment-lock.md`. Verified: `pip show xformers`/`triton`
  inside the rebuilt venv resolved only to the venv's own installed versions, no
  system-level shadowing observed afterward.

### 4. Mixed CUDA12/CUDA13 `nvidia-*` packages breaking cuDNN initialization (environment defect)

- **Symptom**: despite `torch.cuda.is_available()==True`, the first real forward pass
  raised `CUDNN_STATUS_NOT_INITIALIZED`; traced to a stale mix of CUDA-12-built and
  CUDA-13-built `nvidia-*` wheel packages coexisting in the venv (likely from the
  torch-reinstall cycles above pulling in inconsistent transitive `nvidia-*` pins).
- **Fix**: purged every installed `nvidia-*` package and reinstalled `torch`+
  `torchvision` together in a single `pip install` command, letting pip's resolver
  select one consistent CUDA-12 `nvidia-*` set rather than layering reinstalls.
  Detailed in `environment-lock.md`. Verified: a matmul smoke on `cuda:0` succeeded,
  and both `inference/chat.py` and `inference/generate.py` completed with exit 0
  afterward with no further cuDNN error.

### 5. Transient generation-smoke retries before final success (non-blocking)

- **Symptom**: the generation smoke (`inference/generate.py`) required six attempts
  total across the four environment-defect fix cycles above before completing
  cleanly — each of the first five attempts failed on one of the four defects in
  sequence as it was discovered and fixed.
- **Disposition**: fully resolved by the fourth environment-defect fix (the mixed
  CUDA12/CUDA13 `nvidia-*` purge); the sixth attempt completed with exit 0 and
  produced a real, hash-verified 1.59MB PNG. Not a persistent defect — recorded for
  completeness, matching the "record every attempt, not just the successful one"
  convention already established by T210.

## Summary

- No task-path execution failure remains outstanding: both `inference/chat.py` and
  `inference/generate.py` ran to completion (exit 0) after the four environment
  defects above were found and fixed, on the final attempt of each.
- Of the 2 open license-provenance items, neither is blocking for this admission's
  read-only smoke inference; both are documented as open items for any successor
  task with a stricter redistribution/fine-tuning use case.
- 4 environment defects found and fixed via dependency-resolution corrections (torch
  reassertion x2 equivalent cycles, flash-attn ABI rebuild, dropping
  `--system-site-packages`, purging mixed-CUDA `nvidia-*` packages), all fully
  documented in `configs/admission/uniar/environment-lock.md` rather than silently
  patched.
- **The central finding this admission exists to establish — that visual-decoder
  (SD3 pixel-decoder) training code is unreleased by the official UniAR
  repository — is not a failure of this admission; it is the expected, correctly
  documented result.** No attempt was made to reimplement, patch around, or claim
  support for this unreleased path, matching the frozen protocol exactly. No
  unofficial fix was applied to any UniAR source, config, or checkpoint; no different
  checkpoint was substituted; no joint post-training or decoder-training was started.
