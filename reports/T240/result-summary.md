# T240 result summary

Task: `tasks/T240-uniar-admission.md`. Branch: `agent/T240-uniar-admission`. Source
revision audited: UniAR commit `92d8718d4cf282254ae63a4944b07edba0ce7abf` (recorded in
`first-report.md`).

## Current admission run (authoritative status)

The formal admission run is `runs/admission-uniar-v1/` — see
`runs/admission-uniar-v1/manifest.json` (`status: pass`, 28 hash/byte-addressed
artifacts, 21 covering the checkpoint's four independently-versioned components
(`ar_model`, `bsq_encoder`, `sd3_transformer`, `sd3_pipeline`), 7 covering run
evidence — checksums log, two parameter-registry logs, pip-freeze snapshot, two
smoke-evidence bundles, and the generated-image output) and
`runs/admission-uniar-v1/metrics.json` (exit-code evidence, resource metrics, and
scope-declaration fields). This is an admission/smoke run (one understanding pass,
one generation pass), not a training run; see "No training run" below for that
distinction. Unlike T210's Show-o2 (which had a pre-existing shared-storage copy
separately migrated to local SSD), UniAR-RL was downloaded directly to
`/dockerdata/t240-uniar/assets/UniAR-RL/`, so the download target and the execution
copy are the same path for every checkpoint artifact — no separate `*-ssd-execution`
sibling artifacts exist in this manifest (documented in `runs/admission-uniar-v1/notes.md`).

## What was built/run

- First report (audit, no GPU execution yet): `reports/T240/first-report.md` —
  committed and pushed (commit `42462c6a8af020802851bcdd9563998a9dab1e8e`) before any
  download or GPU work, per the task's hard gate. Established the repository/paper/
  license identity, the exact four-component checkpoint decomposition, and — the
  central finding this admission exists to establish — that visual-decoder (SD3
  pixel-decoder) training code is confirmed absent from the official repository
  (README's sole open TODO item: "Release visual decoder training code"), corroborated
  by an empty grep for any decoder-training script/config/trainer across the entire
  tree.
- Storage preflight: `configs/admission/uniar/storage-preflight.json` — `status: pass`,
  `filesystem_class: local` (XFS local block device at
  `/dockerdata/t240-uniar/hf_cache`, ~9.76TB free measured via live `df -T`).
- Environment setup and defect record: `configs/admission/uniar/environment-lock.md` —
  a fresh venv built inside the H20-FoldUMM container for UniAR's pinned dependency set
  (`torch==2.7.0`, `transformers==4.57.0`, `diffusers==0.37.1`, `accelerate>=1.4.0`,
  `flash-attn`), distinct from T210's Show-o2 environment with no overlap. Four
  environment defects found and fixed (all detailed below).
- Checkpoint download and hash verification: 21 component files, 46,895,768,687 bytes
  (~43.7GiB) downloaded to `/dockerdata/t240-uniar/assets/UniAR-RL/`, excluding
  redundant fp16-duplicate text-encoder shards inside `sd3_pipeline/` (~30GB saved,
  per the storage-preflight evaluation flagged as a possibility in `first-report.md`).
  Every retained file's sha256 recorded in `t240_checksums.txt` (manifest artifact
  `checksums-log`).
- Understanding + generation task-path smokes: both official entry points run
  verbatim with no unofficial modification —
  - `inference/chat.py`: single-image VQA/captioning on the README's demo image URL,
    generated a coherent multi-sentence description. `exit_code=0`.
  - `inference/generate.py`: AR visual-token rollout (900/900 tokens) followed by the
    frozen SD3 pixel decode, README-documented default invocation (960x960, prompt
    "A cute anime girl.", no CLI overrides), producing a real 1.59MB PNG
    (sha256 `ec73a92e64e1b71fa3320e34908546b4f81dbfb5df884d8a91fee1cb2aedff43`).
    `exit_code=0`. Required all four environment-defect fixes before succeeding.
- AR trainable-block map: `configs/admission/uniar/parameter-block-registry.yaml` —
  exact weight-level enumeration of `named_parameters()` on the loaded
  `UniARForConditionalGeneration` checkpoint (9,627,074,032 total params), split into
  three blocks that sum exactly to the total with no unclassified parameter:
  `shared_llm_backbone` (8,190,735,360 params, 85.08%, `model.language_model`+`lm_head`),
  `understanding_private` (656,160,240 params, 6.82%, `model.visual` — the BSQ/SigLIP
  vision tower, explicitly frozen by the released RL recipe's
  `model.visual.requires_grad_(False)` at `train/rl/train_grpo.py:79`),
  `generation_private` (780,178,432 params, 8.10%, `visual_decoder.0..3` +
  `output_layer_vistok` — the AR generation head, trainable and exercised by both SFT
  and the released GRPO recipe). Cross-checked against a direct grep of the training
  script rather than config schema alone, refining the stage-1 provisional reading.
- Missing-code report: incorporated into `first-report.md`'s "Released vs. unreleased
  scope" section and reaffirmed unchanged through every later stage — the
  visual-decoder training path remains confirmed absent from the official repository.
  Recorded as `scope.unreleased_decoder_training_recorded=true` in
  `runs/admission-uniar-v1/metrics.json`.
- No adapter code was written under `src/comppareto/adapters/uniar/` or
  `tests/adapters/uniar/` — none was required to satisfy this task's pass/fail gate
  (both smoke paths execute using unmodified official scripts); adapter code, if any,
  is scoped to a successor task's training-interface work, matching T210's precedent.
- No joint post-training run and no attempt to reimplement or patch around the
  unreleased visual-decoder-training path — neither was authorized, per the frozen
  protocol ("do not claim visual-decoder training support that the official
  repository does not release").

## Checkpoints, revisions, and licenses recorded

| Component | Repo ID / source | Resolved revision | License | Size |
|---|---|---|---|---|
| UniAR-RL (primary target) | `ShareLab-SII/UniAR-RL` | `6b02e4eee3d45b34f7f41e6218b6cc3c56332454` | apache-2.0 | 74.40 GB declared; ~43.7GiB actually downloaded (redundant fp16-duplicate text-encoder shards excluded) |
| UniAR-SFT (base, not downloaded — not required for this admission's smoke) | `ShareLab-SII/UniAR-SFT` | `b84157ed4968737cdee5db2db6bbb5375490fdbd` | apache-2.0 | 74.39 GB |
| `ar_model` component (within UniAR-RL) | 4 shards + config | see `t240_checksums.txt` | apache-2.0 (repo-level) | 19.25 GB |
| `bsq_encoder` component | within UniAR-RL | see `t240_checksums.txt` | apache-2.0 (repo-level) | 1.31 GB |
| `sd3_transformer` component (learned pixel-decoder DiT) | within UniAR-RL | see `t240_checksums.txt` | apache-2.0 (repo-level) | 4.95 GB |
| `sd3_pipeline` component (third-party SD3.5-medium; frozen) | within UniAR-RL | see `t240_checksums.txt` | governed by bundled `sd3_pipeline/LICENSE.md` (Stability AI), not independently re-fetched this pass — non-blocking, read-only/frozen use | subset retained, ~30GB excluded |

The repository itself: Apache-2.0 (`pyproject.toml`; no standalone root `LICENSE`
file — minor open item, non-blocking given three independent apache-2.0 declarations
across repo metadata and both HF model cards).

## Environment defects found and fixed (all upstream/transitive, none a UniAR source/checkpoint substitution)

1. **`torch` silently downgraded by transitive installs (occurred twice)** — installing
   `transformers`/`diffusers`/`accelerate` per `pyproject.toml` pulled in a different
   `torch` build than the one explicitly pinned, requiring reassertion of
   `torch==2.7.0+cu126`/`torchvision==0.22.0+cu126` after each occurrence.
2. **`flash-attn` ABI staleness after each torch reinstall** — the previously-built
   `flash-attn` wheel's compiled ABI no longer matched the reinstalled torch build;
   required a `--no-cache-dir` rebuild after every torch reassertion.
3. **Broken system-level `xformers`/`triton` leaking in via `--system-site-packages`** —
   the venv was initially built with `--system-site-packages`, which let an
   incompatible system-wide `xformers` install shadow the venv's own; fixed by
   dropping `--system-site-packages` entirely and rebuilding the venv isolated.
4. **Mixed CUDA12/CUDA13 `nvidia-*` packages breaking cuDNN initialization** — despite
   `torch.cuda.is_available()==True`, a stale mix of CUDA-12-built and CUDA-13-built
   `nvidia-*` wheel packages in the venv caused `CUDNN_STATUS_NOT_INITIALIZED` at first
   real forward pass; fixed by purging every `nvidia-*` package and reinstalling
   `torch`+`torchvision` together in one command to let pip resolve a consistent set.

All four are documented in full (symptom, root cause, fix, verification) in
`configs/admission/uniar/environment-lock.md`. None involved substituting or patching
any UniAR source file, config, or checkpoint — each is a dependency-resolution
correction for the official `pyproject.toml`'s own pinned/unpinned package set.

## GPU/hardware evidence

H20-FoldUMM container, 8x H20 96GB. GPU0 used for all download/environment/smoke/
enumeration work (`cuda:0`); no other GPU was touched. Both `inference/chat.py` and
`inference/generate.py` completed with exit 0; the generation smoke required six
attempts total across the four environment-defect fix cycles before succeeding
cleanly on real hardware with a real image output.

## Resource envelope

- **GPU-hours consumed**: ~0.69 (2,453 seconds summed across every per-cjob GPU job in
  this admission — download, two environment-setup passes, six defect-fix cycles,
  understanding smoke, six generation-smoke attempts, two parameter-block-enumeration
  passes — all on 1x H20 GPU 0), well within the task's 8 GPU-hour cap. Full
  methodology recorded in `runs/admission-uniar-v1/metrics.json`.
- **GPU count**: 1 (default, as required).
- **Checkpoint download**: 46,895,768,687 bytes (~43.7GiB).
- **Storage free at preflight**: 9,761,259,622,400 bytes (~9.09TiB) on the local SSD
  target filesystem.

## Conclusion

**Supports gate.** Both task paths execute end-to-end on GPU with the pinned
UniAR-RL checkpoint using only official code/config; every checkpoint component and
its revision is recorded and hash-verified; shared and private parameter blocks are
enumerated exactly at the weight level (sum exactly matches the checkpoint total,
9,627,074,032 params, no unclassified remainder) and cross-checked against the
released RL recipe's own freeze code. The distinctive finding this admission exists
to establish — that visual-decoder (SD3 pixel-decoder) training is confirmed
unreleased by the official repository — is accurately documented
(`scope.unreleased_decoder_training_recorded=true`) without any attempt to
reimplement, patch around, or claim support for it, matching the frozen protocol
exactly. Four upstream environment defects were found and fixed via
dependency-resolution corrections, all fully documented rather than silently patched.
The formal admission run (`runs/admission-uniar-v1/manifest.json`, `status: pass`)
records 28 hash/byte-addressed artifacts, remotely reverified this round
(`configs/admission/uniar/artifact-verification.json`, 28/28 pass, 0 failed) with
existence, byte size, and SHA-256 confirmed against the manifest's declared values.
No joint post-training and no unreleased-decoder-training attempt was started.
