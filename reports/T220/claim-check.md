# T220 claim check

Each row maps a claim or pass/fail-gate bullet from `tasks/T220-uniddt-admission.md` to the
evidence that supports or refutes it.

## Research claim

> "UniDDT provides a deep shared semantic path with a private diffusion decoder suitable for
> cross-architecture validation."

Supported at the weight level, not just the config-schema level. `configs/admission/uniddt/parameter-block-registry.yaml`
enumerates `named_parameters()` on the checkpoint-loaded `DDT2` module directly: the `shared` block
(`noisy_encoder` + `llm_backbone`, excluding always-frozen I/O) is 4,199,807,488 params (74.29% of
total) used identically by both `forward_gen` and `forward_und_prefill` before they branch, and the
`generation_private` block (`diffusion_decoder`) is 1,064,238,144 params (18.83% of total) — the
*only* trainable block in the release checkpoint's config, called only from the generation path. No
parameter in the loaded module was left unclassified (`unassigned_trainable_parameters: 0`). This is
exactly the "deep shared semantic path with a private diffusion decoder" the claim asserts, verified
against the actual loaded module graph rather than assumed from the README's prose.

## Objective

> "Audit and reproduce the public UniDDT understanding and generation paths under the common
> admission contract."

| Sub-objective | Status | Evidence |
|---|---|---|
| Audit official code/revision | **Done** | `first-report.md` (GitHub commit `d04e037c0e1011a64703ad97d7bc4993bb69eade`, read-only audit before any GPU work; repo later cloned and hash-verified at the identical revision) |
| Audit license | **Done, but unresolved upstream** | `first-report.md` records: no repository `LICENSE` file (404 on both candidate filenames and the GitHub license API), no checkpoint license on the HF model card, and an ambiguous FLUX VAE license (config matches both FLUX.1-dev non-commercial and FLUX.1-schnell apache-2.0). All three are explicitly flagged as open items requiring local-reviewer decision, not silently assumed permissive. `result-summary.md` confirms none were resolved further this stage. |
| Reproduce understanding path | **Done** | `smoke_metrics.json`: `understanding` stage, `exit_code: 0`, 19.95s, 1 image in / 1 text response out, via `Pipeline._und_forward` (official, unmodified) |
| Reproduce generation path | **Done** | `smoke_metrics.json`: `generation` stage, `exit_code: 0`, 78.72s, 1 image out, via `Pipeline._gen_forward` (official, unmodified) |

## Pass/fail gate

> "Both paths are reproducible and the NoisyViT/LLM/decoder split is programmatically exposed."

| Gate bullet | Status | Evidence |
|---|---|---|
| Both paths are reproducible | **Pass** | Both `understanding` and `generation` smoke stages exit 0 in `smoke_metrics.json`, driven through the official, unmodified `app_uniddt.py` `Pipeline` class (no fork, no reimplementation) |
| NoisyViT/LLM/decoder split is programmatically exposed | **Pass** | `configs/admission/uniddt/parameter-block-registry.yaml` — produced from the loaded module's own `named_parameters()`, not the config schema alone; `noisy_encoder` and `llm_backbone` (shared) vs. `diffusion_decoder` (generation-private) are distinct top-level submodules with disjoint parameter sets, and the split is cross-checked against `DDT2.forward_gen`/`forward_und_prefill`'s actual call graph |

Both gate bullets are satisfied. Given the three unresolved license open items (repository,
checkpoint, FLUX VAE — none of which the gate's own wording requires to be resolved, since the gate
concerns reproducibility and the shared/private split, not licensing), this claim-check does not
treat them as gate blockers, but flags them for explicit local-reviewer attention before any onward
use of this admission's evidence beyond audit/reproduction, per `first-report.md`'s own
recommendation.

## Frozen protocol

> "Use the official architecture/checkpoint and record all FLUX/Qwen external dependencies."

| Constraint | Status | Evidence |
|---|---|---|
| Use official architecture | **Honored** | `src-uniddt` is the official GitHub repo at the pinned commit, unmodified; `DDT2`, `app_uniddt.py`'s `instantiate_class`/`load_model`/`Pipeline` are all called as-is, never subclassed or patched |
| Use official checkpoint | **Honored** | `vlm_uniddt_512.ckpt` from `MCG-NJU/UniDDT` at HF revision `1d9541af2314873d77e398e515d8d5a93480be13`, hash-verified (sha256 `46cf922df1eb30880f97c4d00f9e2c3b7cccfe3c6632ba48c9e273eb1a2cc28c`) both at download and again via remote reverification |
| Record FLUX dependency | **Done** | `diffusers/FLUX.1-vae` at revision `da548cfb003bdeebaff6da0211fc8fbc67cb563a`, sha256 `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`, recorded in `first-report.md` and `result-summary.md` including its unresolved license status |
| Record Qwen dependency | **Done** | `Qwen/Qwen3-VL-4B-Instruct` tokenizer/processor at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` (full 4.4B-param backbone weights not separately loaded, since the release checkpoint already carries the LLM backbone weights — recorded explicitly, not silently assumed) |
| No alternate fork or reimplemented model | **Honored** | Both environment fixes (`einops` install, manual `refs/main` files) are dependency-gap fills and offline-cache-resolution fixes, not substitutions of any UniDDT source, config, or checkpoint; no modified/forked copy of any UniDDT file was created or executed |

## Resource envelope

> "one H20 GPU by default; at most ten H20-equivalent GPU-hours; admission smoke only; weights and
> caches must execute from verified local SSD."

| Constraint | Status | Evidence |
|---|---|---|
| One H20 GPU | **Honored** | `CUDA_VISIBLE_DEVICES=0` throughout; `smoke_metrics.json`'s `gpu_name: "NVIDIA H20"` confirms a single-GPU execution; GPU1-7's placeholder processes untouched |
| At most 10 GPU-hours | **Honored** | Total measured GPU wall-clock for the three core smoke stages: 134.27s ≈ 0.0373 GPU-hours (`runs/admission-uniddt-v1/metrics.json`), well under the 10-hour envelope |
| Admission smoke only | **Honored** | Exactly one understanding pass and one generation pass were executed; no training entry point (`main.py fit`, `main.sh`) was invoked |
| Weights/caches execute from verified local SSD | **Honored** | `configs/admission/uniddt/storage-preflight.json`: `status: pass`, `filesystem_class: local`, `filesystem_type: xfs`, confirmed via live `df -T` on `/dockerdata/t220-uniddt/`; both the venv and the HF cache used for execution live under `/dockerdata/t220-uniddt/` |

## Required deliverables checklist

| Deliverable | Path |
|---|---|
| Admission manifest | `first-report.md` (checkpoint/revision/license table) + `result-summary.md` (cumulative) + `runs/admission-uniddt-v1/manifest.json` (formal, schema-valid, `status: pass`) |
| Block map | `configs/admission/uniddt/parameter-block-registry.yaml` (weight-level, `unassigned_trainable_parameters: 0`) |
| Smokes | `smoke_metrics.json` (both `understanding`/`generation` `exit_code: 0`) + `runs/admission-uniddt-v1/metrics.json` |
| Report | `first-report.md`, `result-summary.md` |
| Failure ledger | `failure-ledger.md` |
| Storage preflight | `configs/admission/uniddt/storage-preflight.json` (`status: pass`, `filesystem_class: local`) |
| Remote artifact reverification | `configs/admission/uniddt/artifact-verification.json` (9/9 pass, 0 failed) |
| Environment lock | `configs/admission/uniddt/environment-lock.md` |

## Conclusion

**Supports gate.** Both pass/fail-gate bullets (reproducibility of both paths; programmatic
exposure of the NoisyViT/LLM/decoder split) are satisfied with direct weight-level and exit-code
evidence, not assumption. The three license open items (repository, checkpoint, FLUX VAE) are
explicitly recorded, unresolved, and non-blocking per this task's own frozen-protocol framing —
flagged here again for local-reviewer attention rather than silently dropped. No claim in this
report set overstates what was verified: every "pass" cited above traces to a specific hash,
exit code, or enumerated parameter count in a committed artifact.
