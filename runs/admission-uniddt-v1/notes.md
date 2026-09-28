# Run note — `admission-uniddt-v1`

Formal admission run for T220 (UniDDT deep-sharing admission). `manifest.json` in this directory is
the schema-conformant record; this note explains what it summarizes and where the full narrative
lives.

## What this run covers

Three read-only smoke stages, all executed inside the H20-FoldUMM GPU container on GPU0
(`CUDA_VISIBLE_DEVICES=0`), driven through a small external harness that calls
`app_uniddt.py`'s own `instantiate_class`/`load_model`/`Pipeline` functions directly (no Gradio
server, no source modification), matching the T210 `timing_wrapper.py` precedent:

1. `build_pipeline` — instantiates the VAE, denoiser (`DDT2`), both samplers, and loads the pinned
   `vlm_uniddt_512.ckpt` checkpoint into the denoiser (`load_model` reported 0 "Failed to copy"
   warnings).
2. `understanding` — one `Pipeline._und_forward` call (1 image in, 1 text response out).
3. `generation` — one `Pipeline._gen_forward` call (1 image out).

No training, joint or otherwise, was run. No UniDDT source file was modified.

## Where the full evidence and narrative live

- **First report** (pre-GPU-execution audit: repository/checkpoint/license/dependency inventory,
  entry points, resource estimate): `reports/T220/first-report.md`.
- **Cumulative result narrative** (smoke timings/VRAM, checkpoint/license table, environment
  defects found and fixed, GPU-hours accounting): `reports/T220/result-summary.md`.
- **Claim-by-claim cross-check** against `tasks/T220-uniddt-admission.md`'s research
  claim/objective/gate/frozen-protocol/resource-envelope language: `reports/T220/claim-check.md`.
- **Failure/anomaly ledger** (3 unresolved license open items requiring local-reviewer decision; 4
  resolved environment/tooling issues, each with symptom/root-cause/fix/verification):
  `reports/T220/failure-ledger.md`.
- **Storage preflight** (local-SSD confirmation): `configs/admission/uniddt/storage-preflight.json`.
- **Remote artifact reverification**: `configs/admission/uniddt/artifact-verification.json`.
- **Environment freeze** (exact package versions, the one gap found relative to
  `requirements.txt`): `configs/admission/uniddt/environment-lock.md`.
- **Weight-level block registry** (shared/frozen-I/O/generation-private split, produced from the
  loaded module's own `named_parameters()`, cross-checked against the forward-call graph):
  `configs/admission/uniddt/parameter-block-registry.yaml`.
- **Raw smoke output**: durable copy at
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/T220-uniddt-admission/smoke_metrics.json`
  (sha256 `2833bef3b4a2a31bbac6060f1a434c46ec557b135605ff7baf661ca82793fe64`), referenced as manifest
  artifact `smoke-metrics-raw`.
- **Raw block-registry enumeration** (including the empty `unassigned_parameters: []` list):
  durable copy at
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/T220-uniddt-admission/block_registry_raw.json`
  (sha256 `1d1f882552907010307ff58f82a253a97f92b99bd00938c4cbcba018feb0d00f`), referenced as manifest
  artifact `block-registry-raw`.
- **Exit-code/resource-summary/block-registry-summary evidence**:
  `runs/admission-uniddt-v1/metrics.json`.
- **Immutable provenance vs. SSD-execution copies**: following the T210 precedent, each of the
  three migrated checkpoint/asset components has its own local-SSD copy recorded as a separate
  sibling artifact (`*-ssd-execution` artifact IDs) rather than a nested field, since
  `schemas/run-manifest.schema.json` sets `additionalProperties: false` on artifact objects.

## Known limitations (documented, not silently omitted)

1. **Three license open items remain genuinely unresolved upstream, not merely unexamined by this
   task**: the `MCG-NJU/UniDDT` GitHub repository has no `LICENSE` file, its HF checkpoint carries
   no license tag, and the `diffusers/FLUX.1-vae` dependency's license is ambiguous between two
   differently-licensed FLUX variants whose config values are identical. None of these three items
   was resolved further during this run (no new evidence was sought beyond `first-report.md`'s
   original findings) — see `reports/T220/failure-ledger.md`'s "Unresolved anomalies" section for
   the full disposition and recommendation to local review on each.
2. **No `strace`/file-access-syscall-level evidence of zero shared-storage fallback was collected**
   this run (log-content-based evidence only: `HF_HOME` pointed at `/dockerdata/t220-uniddt/`,
   `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` set throughout, no error indicating a network or
   `/apdcephfs_cq7` access attempt was observed in the smoke harness's stdout/stderr) — matching the
   T210 R11 qualification for the same limitation.
3. **Only one sample per task path was run** (one understanding call, one generation call), matching
   this task's "admission smoke only" resource-envelope constraint; no determinism/repeat-run check
   (e.g. T210's `mmu_cold1`/`mmu_cold2` bit-identical-hash check) was performed this round.
4. **No adapter code was written** under `src/comppareto/adapters/uniddt/` or
   `tests/adapters/uniddt/` — the pass/fail gate does not require one (both paths already execute
   via unmodified official code driven by an external harness); any adapter code is deferred to a
   successor task.

## Status

`pass` — both task paths (`understanding`, `generation`) executed to completion on H20 GPU0 with
`exit_code: 0` in both cases, through the official unmodified `app_uniddt.py` code and the pinned
`vlm_uniddt_512.ckpt` checkpoint. The NoisyViT/LLM/decoder shared/private split is programmatically
exposed at the exact weight level (`unassigned_trainable_parameters: 0`). Remote reverification of
every declared manifest artifact (`configs/admission/uniddt/artifact-verification.json`): 9/9 pass,
0 failed. Storage preflight (`configs/admission/uniddt/storage-preflight.json`): `status: pass`,
`filesystem_class: local`. Total GPU wall-clock for the three smoke stages: 134.27s (≈0.0373
GPU-hours), well inside the 10-GPU-hour resource envelope. The three license open items above remain
unresolved and are explicitly flagged for local-reviewer attention before any onward use of this
admission's evidence beyond audit/reproduction — they do not, per this task's own gate wording,
block the `pass` status recorded here.
