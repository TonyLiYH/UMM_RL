# Run note — `admission-uniar-v1`

Formal admission run for T240 (UniAR homogeneous-objective boundary-control admission).
`manifest.json` in this directory is the schema-conformant record; this note explains
what it summarizes and where the full narrative lives.

## What this run covers

Two read-only inference smoke passes, both sourced from local SSD
(`/dockerdata/t240-uniar/`) inside the H20-FoldUMM GPU container, against the
`ShareLab-SII/UniAR-RL` checkpoint (revision `6b02e4eee3d45b34f7f41e6218b6cc3c56332454`)
at the pinned official repository revision
`92d8718d4cf282254ae63a4944b07edba0ce7abf`:

1. **Understanding smoke** (`inference/chat.py`) — single-image VQA/captioning, the
   README's demo image URL, generated a coherent multi-sentence description.
   `exit_code=0`.
2. **Generation smoke** (`inference/generate.py`) — AR visual-token rollout (900/900
   tokens) followed by the frozen SD3 pixel decode, exactly the README-documented
   default invocation (960x960, prompt `"A cute anime girl."`, no CLI overrides),
   producing a real 1.59MB PNG. `exit_code=0`.
3. **Read-only AR trainable-block enumeration** — loaded the full checkpoint and
   inspected `named_parameters()` directly (no training run), confirming the
   provisional stage-1 reading in `reports/T240/first-report.md`.

No training of any kind was run — neither the released AR/GRPO recipe nor (especially)
the unreleased visual-decoder-training path. No UniAR source file was modified.

## Where the full evidence and narrative live

- **Source/license/checkpoint/scope audit (stage 1)**: `reports/T240/first-report.md`
  (committed and pushed before any download/GPU work, per the task's hard gate).
- **Storage preflight (stage 2)**: `configs/admission/uniar/storage-preflight.json`
  (`status: pass`, `filesystem_class: local`, XFS, ~9.76TB free at
  `/dockerdata/t240-uniar/hf_cache`).
- **Environment setup and defect record (stage 3)**:
  `configs/admission/uniar/environment-lock.md` — records four non-obvious
  dependency-resolution/environment defects encountered and fixed while building the
  UniAR inference venv (torch silently downgraded by transformers/diffusers/accelerate
  installs, twice; flash-attn ABI staleness after each torch reinstall; a broken
  system-level `xformers`/`triton` combination leaking in via
  `--system-site-packages`; and a mixed CUDA12/CUDA13 `nvidia-*` package state breaking
  cuDNN initialization despite `torch.cuda.is_available()==True`). Each is recorded
  with its exact symptom, root cause, and fix — none were silently patched.
- **Checkpoint download and hash verification (stage 4)**: 44GB actual download to
  `/dockerdata/t240-uniar/assets/UniAR-RL/` (full `sd3_pipeline` fp16-duplicate
  text-encoder shards excluded, ~30GB saved); sha256 of every `.safetensors`/
  `config.json` file recorded in
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/T240-uniar-admission/t240_checksums.txt`
  (referenced as manifest artifact `checksums-log`).
- **Understanding + generation smokes (stage 5)**: raw logs and the generated PNG
  bundled and hash-addressed as manifest artifacts `understanding-smoke-run-bundle`,
  `generation-smoke-run-bundle`, and `generation-output-image`.
- **AR trainable-block map (stage 6)**:
  `configs/admission/uniar/parameter-block-registry.yaml` — three trainable blocks
  (`shared_llm_backbone` 8,190,735,360 params / 85.08%, `understanding_private`
  656,160,240 / 6.82%, `generation_private` 780,178,432 / 8.10%; sum exactly equals
  the checkpoint's total 9,627,074,032 parameters), cross-checked against a direct
  grep of `train/rl/train_grpo.py`'s explicit `model.visual.requires_grad_(False)`
  freeze call to confirm which block is (and is not) updated by the released RL
  recipe.
- **Missing-code / unreleased-scope report**: `reports/T240/first-report.md`'s
  "Released vs. unreleased scope" section, reaffirmed unchanged in this run — the
  visual-decoder (SD3 pixel decoder) training path remains confirmed absent from the
  official repository (README's sole open TODO item), corroborated by an empty grep
  for any decoder-training script/config/trainer across the entire tree. Recorded as
  `scope.unreleased_decoder_training_recorded=true` in `metrics.json`, per the frozen
  protocol — not attempted, not patched around, not substituted.
- **Artifact reverification (stage 7)**:
  `configs/admission/uniar/artifact-verification.json` — 28/28 pass, 0 failed,
  covering all 21 checkpoint component files (on `/dockerdata`, verified inside the
  H20-FoldUMM container since only that container mounts the path) plus 7 run-evidence
  artifacts (checksums log, two parameter-registry logs, pip-freeze snapshot, two
  smoke-evidence tar bundles, one generated-image output — all on shared
  `/apdcephfs_cq7` storage, reachable directly from this checkout).
- **Exit-code/resource-summary evidence**: `runs/admission-uniar-v1/metrics.json`.

## Known limitations (documented, not silently omitted)

1. **Unreleased visual-decoder training remains genuinely unavailable upstream** —
   this is the expected, correctly-documented finding this task exists to surface,
   not a defect in this admission. No further investigation of this specific gap is
   warranted or attempted; see `reports/T240/failure-ledger.md` for the full
   derivation and cross-check against the repository tree.
2. **No standalone root `LICENSE` file in the UniAR repository itself** — license is
   declared in `pyproject.toml` (Apache-2.0) plus three independent HuggingFace model
   card declarations (repo metadata, `UniAR-RL` card, `UniAR-SFT` card); recorded as a
   minor open item, non-blocking.
3. **`sd3_pipeline/LICENSE.md` (Stability AI's SD3.5-medium terms)** was not
   independently re-fetched from Stability AI's own canonical release; the pixel
   decoder is used strictly read-only/frozen in this admission's smoke, so this is
   non-blocking.
4. **Batched multi-GPU generation (`inference/generate_batch.py`) and the released
   GRPO RL training recipe (`train/rl/train_grpo.py`) were not executed** — both are
   released, but exceed this admission's "admission smoke... only" resource-envelope
   scope; the RL recipe's trainable-block structure was nonetheless enumerated
   read-only (stage 6) as the task requires, without running any training step.
5. **This admission's checkpoint copy on `/dockerdata/t240-uniar/assets/UniAR-RL/` is
   simultaneously the download target and the execution path** (unlike T210's Show-o2,
   which had a pre-existing shared-storage `hf_cache` copy that was separately migrated
   to local SSD). There is therefore no separate `*-ssd-execution` sibling-artifact
   pair for the checkpoint components in this manifest — each component's single
   `/dockerdata` path is both the provenance and the execution copy, verified once.

## Status

`pass` — both task paths (`chat.py`, `generate.py`) executed to completion on H20 GPU
0 from local SSD, both `exit_code=0`, generation smoke produced a real, hash-verified
PNG output. AR trainable-block boundaries enumerated read-only from the loaded
checkpoint, exact parameter counts sum to the checkpoint total with no unclassified
parameter. Unreleased visual-decoder-training scope explicitly and accurately
recorded (`scope.unreleased_decoder_training_recorded=true`), matching the frozen
protocol: no attempt made to reimplement, patch around, or claim support for it.
Remote reverification of every declared manifest artifact
(`configs/admission/uniar/artifact-verification.json`): 28/28 pass, 0 failed. Storage
preflight (`configs/admission/uniar/storage-preflight.json`): `status: pass`,
`filesystem_class: local`. Total GPU wall-clock consumed across every job in this
admission (download, environment setup and four fix cycles, both smokes and their
retries, parameter-block enumeration): ~0.68 GPU-hours on 1x H20, well within the
task's 8 GPU-hour cap. Exit-code and resource-summary evidence:
`runs/admission-uniar-v1/metrics.json`.
