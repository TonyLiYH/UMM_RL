# Run note — `admission-sensenova-u1-v1`

Formal admission run for T230 (SenseNova-U1 native pixel/MoT admission), targeting the released
`sensenova/SenseNova-U1-8B-MoT` checkpoint only (U1, not U1.5 — see `reports/T230/first-report.md`
for the frozen-protocol scope confirmation). `manifest.json` in this directory is the
schema-conformant record; this note explains what it summarizes and where the full narrative
lives.

## What this run covers

Two read-only inference smoke passes plus one static parameter/routing audit, all executed from
local SSD (`/dockerdata/t230-sensenova/`) inside the H20-FoldUMM GPU container:

1. `smoke_vqa` — understanding-pathway smoke via the official `examples/vqa/inference.py`
   (single VQA call against a sample image, SDPA attention, bfloat16).
2. `smoke_t2i` — generation-pathway smoke via the official `examples/t2i/inference.py`
   (single text-to-image call, 50 diffusion/flow-matching steps, SDPA attention, bfloat16).
3. `param_inspect` — CPU-side, weight-level parameter enumeration via the repository's own
   `scripts/inspect_model_params.py`, run directly against the downloaded checkpoint (not the
   documentation example) to confirm the parameter-block registry's figures on the actual weights.

No training, joint or otherwise, was run. No SenseNova-U1 source file was modified — both smokes
invoke the official example scripts verbatim with only CLI arguments (`--attn_backend sdpa`
because no matching flash-attn wheel was available; the repository's own `pyproject.toml`
documents this as a supported fallback, not an unofficial patch).

## Where the full evidence and narrative live

- **Repository/license/scope audit (pre-download)**: `reports/T230/first-report.md`.
- **Environment build**: `configs/admission/sensenova-u1/environment-lock.md` (fresh venv, no
  corrective pins needed, flash-attn omission documented as non-blocking).
- **Storage preflight**: `configs/admission/sensenova-u1/storage-preflight.json`
  (`status: pass`, `filesystem_class: local`, target path `/dockerdata/t230-sensenova/hf_cache`).
- **Parameter-block registry**: `configs/admission/sensenova-u1/parameter-block-registry.yaml`
  (`unassigned_trainable_parameters: 0`; three disjoint blocks — `shared`,
  `understanding_transformer`, `generation_transformer` — plus pathway rollups).
- **Routed-overlap / MoT-vs-MoE-gate audit and static assumption violation**:
  `runs/admission-sensenova-u1-v1/metrics.json`'s `routed_overlap` block, and
  `reports/T230/result-summary.md`.
- **Remote artifact reverification**:
  `configs/admission/sensenova-u1/artifact-verification.json` (15/15 pass, 0 failed).
- **Exit-code and resource-summary evidence**: `runs/admission-sensenova-u1-v1/metrics.json`.
- **Raw evidence files** (durable, outside Git):
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/T230-sensenova-u1-admission/` —
  `evidence/answer.txt`, `evidence/output.png`, `evidence/inspect_output.txt`,
  `pip_freeze_20260903.txt`, `SHA256SUMS_20260903.txt`.

## Structural difference from T210 (documented, not an oversight)

T210's manifest paired every model-weight artifact with both a shared-storage (`/apdcephfs_cq7`)
canonical copy and a local-SSD execution copy, because those checkpoints already existed on
CQ7-backed HuggingFace cache storage before being migrated to SSD for execution. **T230's
checkpoint has no such shared-storage canonical copy**: `sensenova/SenseNova-U1-8B-MoT` was
downloaded directly to `/dockerdata/t230-sensenova/checkpoints/SenseNova-U1-8B-MoT` via
`snapshot_download(local_dir=...)`, with no intermediate HF-cache-on-CQ7 step (confirmed via
`find` across `/apdcephfs_cq7/share_1447896/yihangli/models` — no SenseNova-U1 weights are
present there). Accordingly, `manifest.json`'s `artifacts` array lists each of the 10 checkpoint
files (8 safetensors shards, the safetensors index, and `config.json`) as a single artifact
pointing directly at its `/dockerdata` path, rather than the canonical+ssd-execution pair pattern
T210 used. This is a genuine difference in how the checkpoint was provisioned, not a missed
provenance step — per `dev-env-paths.md`, pretrained weights normally belong on CQ7, but no CQ7
copy of this specific checkpoint was created in this admission's provisioning flow; both smokes
executed exclusively from the verified local-SSD copy, satisfying the resource envelope's "weights
and caches must execute from verified local SSD" requirement.

## Known limitations (documented, not silently omitted)

1. **`flash-attn` not installed.** The reference environment in the repository's own
   `pyproject.toml` uses a CUDA-specific wheel not generically hosted on PyPI. The model
   transparently falls back to torch SDPA per the repository's own documented behavior (see
   `configs/admission/sensenova-u1/environment-lock.md`). Both smokes ran on the SDPA attention
   path; neither smoke failed or showed any attention-related error.
2. **Mixed understanding+generation forward pass is not exercised (and cannot be, on released
   code).** The released code explicitly raises `NotImplementedError` for the case where a single
   forward pass contains both understanding and generation tokens simultaneously
   (`src/sensenova_u1/models/neo_unify/modeling_qwen3.py`, referencing open upstream issue #207).
   This is a static, code-visible limitation of the released repository — exactly the kind of
   "static assumption violation" the task's routed-overlap audit is meant to capture and record,
   not something this admission is expected to work around or fix. Both required smokes (VQA-only,
   T2I-only) exercise only the two pure branches the code does support, and both completed with
   `exit_code 0`.
3. **No non-Apache third-party dependency was found** (unlike T210/Show-o2's separately-licensed
   VAE/safety-checker). The 8B-MoT checkpoint bundles its own vision/generation components under
   the `sensenova_u1` package. Both the repository (`LICENSE`) and the HF model card
   (`cardData.license`) report `apache-2.0`; no additional non-commercial/research-only clause was
   found.
4. **Same-process warm inference was not attempted** (matching T210's precedent): both official
   inference scripts are single-shot CLIs with no loop entry point; a custom warm-inference driver
   would risk diverging from the exact audited code path. Each smoke was measured as a single cold
   process.
5. **Parameter-inspection tool runs on CPU, not GPU** (confirmed by reading
   `scripts/inspect_model_params.py`: no `.to(cuda)`/`device_map` call). Its wall-clock
   contribution to `resources.gpu_hours` is a small, conservative estimate (3 seconds) rather than
   a directly profiled figure, since the tool does not touch the GPU; this does not affect the
   `<=12 GPU-hour` budget given the two GPU smokes alone total under 3 minutes.

## Status

`pass` — both required task-path smokes (`examples/vqa/inference.py`,
`examples/t2i/inference.py`) executed to completion from local SSD only inside the H20-FoldUMM
container, both with `exit_code 0` captured directly from the wrapper script's `$?`. Parameter-block
registry accounts for all 17.552B parameters across three disjoint blocks with
`unassigned_trainable_parameters: 0`. Routed-overlap audit records the released code's static MoT
boolean-mask routing mechanism, explicitly distinguishes it from the out-of-scope A3B MoE-gate
mechanism, and records one static assumption violation (`NotImplementedError` on mixed
understanding+generation forward passes, upstream issue #207) —
`routed_overlap.static_assumption_violations_recorded: true`. Remote reverification of every
declared manifest artifact (`configs/admission/sensenova-u1/artifact-verification.json`): 15/15
pass, 0 failed. Storage preflight
(`configs/admission/sensenova-u1/storage-preflight.json`): `status: pass`,
`filesystem_class: local`. Total measured GPU wall-clock across both smokes: 159 seconds
(≈0.045 GPU-hours), well within the 12 GPU-hour cap. No U1.5 asset was downloaded or exercised at
any stage.
