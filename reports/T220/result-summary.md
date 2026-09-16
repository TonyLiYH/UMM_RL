# T220 result summary

Task: `tasks/T220-uniddt-admission.md`. Branch: `agent/T220-uniddt-admission`. Source revision
audited: UniDDT commit `d04e037c0e1011a64703ad97d7bc4993bb69eade` (recorded in `first-report.md`).

## Current admission run (authoritative status)

The formal admission run is `runs/admission-uniddt-v1/` — see `runs/admission-uniddt-v1/manifest.json`
(`status: pass`) and `runs/admission-uniddt-v1/metrics.json` (smoke exit-code evidence, block
registry, and resource metrics). This is an admission/smoke run (one understanding pass, one
generation pass through `app_uniddt.py`'s `Pipeline` class, driven by an external harness), not a
training run — no training entry point (`main.py fit` / `main.sh`) was invoked at any point.

## What was built/run

- First report (audit, no GPU execution yet): `reports/T220/first-report.md`.
- Storage preflight: `configs/admission/uniddt/storage-preflight.json` — `status: pass`,
  `filesystem_class: local`, `filesystem_type: xfs`, confirmed via live `df -T` on
  `/dockerdata/t220-uniddt/`, run with `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` set.
- Dedicated venv built from the official `requirements.txt`
  (`/dockerdata/t220-uniddt/venv`, Python 3.10.12) — one gap found and fixed
  (`einops` missing from `requirements.txt` but imported unconditionally by a transitively-imported
  module; see `configs/admission/uniddt/environment-lock.md` and `failure-ledger.md`).
- Official repo cloned read-only to `/dockerdata/t220-uniddt/src-uniddt`, checked out at the pinned
  revision `d04e037c0e1011a64703ad97d7bc4993bb69eade`; no source file was modified.
- Checkpoint and dependency assets downloaded, hashed, and canonicalized to CQ7
  (`/apdcephfs_cq7/share_1447896/yihangli/models/pretrained/hf_cache/hub/`), then remotely
  reverified end-to-end (both the CQ7 canonical copy and the `/dockerdata` SSD execution copy):
  `configs/admission/uniddt/artifact-verification.json` — **9/9 checked, 9 passed, 0 failed**.
- Dual-path smoke via a small external harness
  (`/apdcephfs_cq9/share_1447896/yihangli/tmp/t220/smoke_harness.py`, not part of the UniDDT
  repository, drives `app_uniddt.py`'s own `instantiate_class`/`load_model`/`Pipeline` calls
  directly — no Gradio server, no source modification, matching the T210 `timing_wrapper.py`
  precedent):
  - `build_pipeline`: exit 0, 35.60s (loads `vlm_uniddt_512.ckpt`, the FLUX VAE, and both samplers;
    `load_model` reported 0 "Failed to copy" warnings, i.e. every parameter name in the
    instantiated module graph is present in the checkpoint and vice versa).
  - `understanding`: exit 0, 19.95s, 1 image in / 1 text response out (`Pipeline._und_forward`).
  - `generation`: exit 0, 78.72s, 1 image out (`Pipeline._gen_forward`).
  - Total wall-clock across all three stages: 134.27s. Peak VRAM: 21,455,294,464 bytes
    (~19.98GiB) on a single `NVIDIA H20`.
  - Full raw output: `smoke_metrics.json`, durable copy at
    `/apdcephfs_cq7/share_1447896/yihangli/outputs/T220-uniddt-admission/smoke_metrics.json`
    (sha256 `2833bef3b4a2a31bbac6060f1a434c46ec557b135605ff7baf661ca82793fe64`).
- Weight-level parameter-block registry: `configs/admission/uniddt/parameter-block-registry.yaml`,
  produced by enumerating `named_parameters()` on the checkpoint-loaded `DDT2` module (see
  `/apdcephfs_cq9/share_1447896/yihangli/tmp/t220/block_registry.py`) and cross-checked against the
  `forward_gen`/`forward_und_prefill` call graph:
  - `total_params`: 5,653,001,792. `total_trainable_params`: 1,064,238,144.
  - **`unassigned_trainable_parameters`: 0** — every named parameter in the loaded module falls
    into exactly one of the three declared blocks below; none were left unclassified.
  - `shared` (`noisy_encoder` + `llm_backbone`, excluding the two always-frozen I/O submodules):
    4,199,807,488 params (74.29% of total), 0 trainable — called identically by both task paths
    before they branch.
  - `frozen_llm_io` (`llm_backbone.embed_tokens`, `llm_backbone.lm_head`): 388,956,160 params
    (6.88% of total), 0 trainable — frozen unconditionally by `DDT2.__init__`'s `no_grad(...)`
    calls, independent of the release config's freeze flags; also named by
    `DDT2._fsdp_ignore_modules()`.
  - `generation_private` (`diffusion_decoder`): 1,064,238,144 params (18.83% of total), **all
    1,064,238,144 trainable** — the only block trainable in the release config
    (`freeze_diffusion_decoder: false`), called only from the generation path.
  - `frozen_non_trainable.flux1_vae`: the FLUX.1 VAE, loaded separately by `LatentAE`, not part of
    `DDT2.named_parameters()` and not trained.
  - This confirms, at the exact weight level (not just the config-schema-level reading in
    `first-report.md`), that the shared/private split declared by the release config
    (`freeze_noisy_encoder: true`, `freeze_llm_backbone: true`, `freeze_diffusion_decoder: false`)
    is programmatically exposed and auditable: the "shared" block (Noisy ViT + LLM backbone) is
    used identically by both understanding and generation forward calls, and the
    "generation-private" block (diffusion decoder) is the sole trainable component, matching the
    README's "duality post-training" description exactly.
  - Raw enumeration (including the empty `unassigned_parameters: []` list):
    `/apdcephfs_cq7/share_1447896/yihangli/outputs/T220-uniddt-admission/block_registry_raw.json`
    (sha256 `1d1f882552907010307ff58f82a253a97f92b99bd00938c4cbcba018feb0d00f`).
- No adapter code was written under `src/comppareto/adapters/uniddt/` or `tests/adapters/uniddt/` —
  none was required to satisfy this task's pass/fail gate (both paths execute using unmodified
  official code, driven only by an external harness); adapter code, if any, is scoped to a
  successor task's training-interface work.
- No joint/duality post-training run — none was authorized or attempted, per the frozen protocol.
  `runs/admission-uniddt-v1/` records exactly the smoke work above (one understanding pass, one
  generation pass, plus checkpoint-load), not a training run of any kind.

## Checkpoints, revisions, and licenses recorded (cumulative across all stages)

| Component | Repo ID / source | Resolved revision | Hash | Size | License |
|---|---|---|---|---|---|
| VLM-UniDDT 512 (denoiser: Noisy ViT + LLM backbone + diffusion decoder, all in one file) | `MCG-NJU/UniDDT` | `1d9541af2314873d77e398e515d8d5a93480be13` | sha256 `46cf922df1eb30880f97c4d00f9e2c3b7cccfe3c6632ba48c9e273eb1a2cc28c` | 22,636,060,254 bytes | **unspecified — open item, carried forward from `first-report.md`** |
| Qwen tokenizer/processor (tokenizer/config only; full backbone weights not separately loaded) | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | tokenizer.json sha256 `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`; config.json sha256 `edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d` | 7,032,403 + 1,505 bytes | `apache-2.0` |
| FLUX VAE (visual latent space, frozen, not trained) | `diffusers/FLUX.1-vae` | `da548cfb003bdeebaff6da0211fc8fbc67cb563a` | sha256 `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3` | 167,666,902 bytes | **unresolved (ambiguous between FLUX.1-dev non-commercial and FLUX.1-schnell apache-2.0) — open item, carried forward from `first-report.md`** |
| UniDDT source (`ddt2.py`, audited/instantiated, not modified) | `MCG-NJU/UniDDT` (GitHub) | `d04e037c0e1011a64703ad97d7bc4993bb69eade` | sha256 `bb7dda12a547d58642158fcba7b5caee23bfec65433169835afa4be649800c53` | 7,588 bytes | **no repository LICENSE file exists — open item, carried forward from `first-report.md`** |

None of the three license open items flagged in `first-report.md` were resolved further this
stage (no new evidence was sought this round beyond the original repository/model-card checks); all
three remain explicit, recorded, non-blocking open items for local review, exactly as flagged in
`first-report.md`.

## Environment defects found and fixed

1. **`einops` missing from `requirements.txt`.** `src/utils/packed_seqs/seqs.py` imports it
   unconditionally, and that module is imported transitively by `app_uniddt.py`. Without it, the
   very first `import app_uniddt` fails with `ModuleNotFoundError`. Fixed by installing
   `einops==0.8.2` (current PyPI release at audit time; no version pin specified upstream). This is
   a dependency-gap fill for a package the official `requirements.txt` itself omits, not a
   substitution of any UniDDT component. Detailed in `configs/admission/uniddt/environment-lock.md`
   and `failure-ledger.md`.
2. **Missing `refs/main` files for two HF repos fetched by explicit revision.** `huggingface_hub`'s
   offline-mode resolution (`HF_HUB_OFFLINE=1`) requires a `refs/main` file to resolve `revision=None`/
   `"main"` lookups even when the target snapshot was already fetched by its explicit commit hash;
   since both `diffusers/FLUX.1-vae` and `Qwen/Qwen3-VL-4B-Instruct` were downloaded by pinned
   revision (not "main"), this file was absent, causing an `OSError` claiming the repo "does not
   appear to have a file named config.json" even though the snapshot was present on disk. Fixed by
   manually writing the two `refs/main` files (containing the already-pinned revision hash) — no
   different revision or file content was substituted. Detailed in `failure-ledger.md`.

Neither defect is a substitution of any UniDDT component, checkpoint, or logic — both are gap-fills
for an upstream requirements omission and an offline-cache-resolution quirk, fully recorded rather
than silently patched.

## GPU/hardware evidence

H20-FoldUMM container, 8x H20 96GB. GPU0 kept exclusively for this task's smoke work
(`CUDA_VISIBLE_DEVICES=0`); GPU1-7's `train2.py` placeholder occupancy processes were never
touched, per `gpu-placeholder-mechanism`. All three smoke stages (`build_pipeline`, `understanding`,
`generation`) completed with `exit_code: 0` and no exception recorded in `smoke_metrics.json`'s
per-stage `error` field (all `null`). Peak VRAM across the run: 21,455,294,464 bytes (~19.98GiB),
well under the H20's 96GB per-device headroom. Total GPU wall-clock for the three core smoke stages:
134.27s (≈0.0373 GPU-hours on a single device) — see `runs/admission-uniddt-v1/metrics.json` for the
full resource accounting including venv-build/download time, which does not consume GPU compute.

## Conclusion

**Supports gate**, subject to local-reviewer disposition of the three carried-forward license open
items (repository, checkpoint, FLUX VAE — none resolved further this stage; see "Checkpoints,
revisions, and licenses recorded" above and `first-report.md`'s "Open items"). Both task paths
(understanding via `Pipeline._und_forward`, generation via `Pipeline._gen_forward`) execute
end-to-end on GPU through the official, unmodified `app_uniddt.py` code with the pinned
`vlm_uniddt_512.ckpt` checkpoint (`exit_code: 0` for both); every checkpoint/revision touched is
recorded; and the NoisyViT/LLM/decoder shared/private split is exposed and auditable at the exact
weight level (`unassigned_trainable_parameters: 0`; every parameter in the loaded module falls into
exactly one of the `shared`, `frozen_llm_io`, or `generation_private` blocks, matching the release
config's freeze flags and the forward-call graph). Two upstream environment gaps were found and
fixed via a dependency-gap fill and an offline-cache-resolution fix, both documented, neither a
substitution of any UniDDT component. All 9 declared artifacts were remotely reverified with
existence, byte size, and SHA-256 confirmed against the manifest's declared values
(`configs/admission/uniddt/artifact-verification.json`, 9/9 pass, 0 failed). No joint/duality
post-training was started or attempted.
