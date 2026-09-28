# T230 first report — SenseNova-U1 native pixel/MoT admission

Per `tasks/T230-sensenova-u1-admission.md`'s "First report" requirement, this is published — and committed/pushed on `agent/T230-sensenova-u1-admission` — before any large-asset download or GPU execution. All information below was gathered by read-only inspection of the official repository (cloned to local SSD at `/dockerdata/t230-sensenova/SenseNova-U1` inside the H20-FoldUMM GPU container, not pushed anywhere; git-clone only, no weights, no code modification) and public HuggingFace/GitHub metadata.

## Official repository

- URL: `https://github.com/OpenSenseNova/SenseNova-U1`
- Pinned revision: `f97964a6e54b0abf92aa2db849af4e942bb2ff08` (tip of `main` at audit time, `2026-09-02 19:36:57 +0800`, resolved via `git log -1` on the local clone).
- Repository license: Apache License 2.0 (`LICENSE` at repo root, verified by direct read: "Apache License, Version 2.0, January 2004").
- HuggingFace model-card license (`sensenova/SenseNova-U1-8B-MoT`): `apache-2.0` (confirmed via live HF Hub API query, `cardData.license` and the `license:apache-2.0` tag).

## U1 vs. U1.5 scope confirmation (frozen-protocol check)

The task's frozen protocol prohibits substituting U1.5 preview/unreleased training code for the admitted U1 path. Evidence from the repository's Project Status checklist and dated news entries confirms U1 (not U1.5) is the released, admissible target:

- U1's inference code, checkpoints, and technical report are all marked released. News entries `[2026.05.21]`, `[2026.05.10]`, and `[2026.04.27]` document the U1 checkpoint releases (8B-MoT, 8B-MoT-SFT, and the A3B MoE variant) and the technical report (arXiv:2605.12500).
- U1.5 is announced separately, with its full training pipeline explicitly marked "in preparation" per the `[2026.08.20]` news entry — i.e. not yet publicly released. This matches the task's frozen-protocol boundary exactly: only the released U1 full-parameter path is in scope; U1.5's unreleased training pipeline is a blocker for any U1.5 work, not something to recreate.
- This admission targets **U1 only**. No U1.5 asset, checkpoint, or code path will be downloaded or exercised at any stage.

## Checkpoint identifiers and sizes

| Asset | HF repo ID | Revision (sha) | Role in this admission |
|---|---|---|---|
| **Primary target** — SenseNova-U1 8B MoT | `sensenova/SenseNova-U1-8B-MoT` | `bfa9b436503cb8aed4f2bc60e3236710cc77468d` (HF `sha`, confirmed via live API) | Admitted checkpoint for both understanding and generation smokes and the parameter-block registry |
| SFT variant (not targeted) | `sensenova/SenseNova-U1-8B-MoT-SFT` | not resolved (out of scope) | Recorded for completeness only; not downloaded |
| A3B MoE variant (not targeted) | `sensenova/SenseNova-U1-A3B-*` (exact repo id not yet resolved) | not resolved (out of scope) | Recorded for completeness only; not downloaded — uses a distinct MoE-gate routing mechanism (see routed-overlap note below), separate from the MoT boolean-mask routing this admission audits |
| Excluded: any U1.5 checkpoint | n/a | n/a | Explicitly out of scope per frozen protocol; U1.5's training pipeline is "in preparation," not released |

`architectures: ["NEOChatModel"]`, `model_type: neo_chat`, custom code via `auto_map` (`configuration_neo_chat.NEOChatConfig`, `modeling_neo_chat.NEOChatModel`). The repo is `gated: false` and `private: false` — no credentials required.

Checkpoint hash verification of the actual safetensors shards is deferred to stage 2 (storage preflight + download), where the files will be downloaded to local SSD and hashed; only the HF-reported repo `sha` is recorded here.

## Licenses and redistribution constraints

Apache-2.0 for both the repository and the primary checkpoint (`sensenova/SenseNova-U1-8B-MoT`). Apache-2.0 permits redistribution and modification with attribution and NOTICE preservation; no additional non-commercial or research-only clause was found in either license source inspected. No external non-Apache dependency (e.g. a separately-licensed VAE or safety checker, as was the case for T210/Show-o2) has been identified yet in this pass — the model appears to bundle its own vision/generation components under the `sensenova_u1` package rather than pulling in a third-party checkpoint with a distinct license. This will be re-verified once the checkpoint's `config.json`/model card is fully inspected during the storage-preflight/download stage.

## Required dependencies

From `pyproject.toml` / generated `requirements.txt` at the pinned commit:

- `torch==2.8.0`, `torchvision==0.23.0` (from `--extra-index-url https://download.pytorch.org/whl/cu128`, i.e. CUDA 12.8 wheels)
- `transformers>=4.57.1,<6`, `accelerate>=1.1,<2`, `huggingface-hub>=0.34,<2`, `safetensors>=0.4.3,<1`
- `sentencepiece==0.2.1`, `numpy>=1.24,<3`, `pillow==12.0.0`, `tqdm==4.67.1`, `packaging==25.0`, `httpx>=0.27,<1`
- Optional: `flash-attn>=2.8,<3` (reference env uses a cp311/torch2.8 CUDA wheel `flash_attn 2.8.3`; PyPI has no CUDA-specific wheel, so it must be installed from a matching local `.whl`; the model transparently falls back to torch SDPA when absent, per a code comment in `pyproject.toml` referencing `src/sensenova_u1/models/neo_unify/modeling_qwen3.py`)
- Optional `gguf` extra (`gguf>=0.10.0`, `diffusers>=0.30.0`) and `dev` extra (`ruff`, `pytest`, `pre-commit`) — not required for inference smokes
- `requires-python = ">=3.10,<3.14"`; `uv.lock` provides the exact tested/reproducible environment (220KB lockfile at repo root)

This is a distinct, non-overlapping environment from every existing environment recorded in `reference_server_status` and from T210/Show-o2's environment (`torch==2.5.1`+`transformers==4.47.0`+`diffusers==0.31.0`) — SenseNova-U1 requires `torch==2.8.0` (CUDA 12.8) and a much newer `transformers` floor. A fresh venv/uv environment must be built inside the H20-FoldUMM container, separate from Show-o2's, per `dev-env-paths.md`.

## Entry points

| Task path | Script |
|---|---|
| Understanding (VQA) | `examples/vqa/inference.py` |
| Generation (text-to-image) | `examples/t2i/inference.py` |
| Editing | `examples/editing/inference.py` |
| Interleaved generation | `examples/interleave/inference.py`, `examples/interleave/run.sh` |
| Serving (client/server) | `examples/serving/client.py` |
| Training (not authorized; recorded for completeness only) | `training/` subtree (derived from InternEvo per `pyproject.toml`'s ruff-exclude comment); not exercised in this admission |

This admission's two required smokes map directly to `examples/vqa/inference.py` (understanding) and `examples/t2i/inference.py` (generation), matching the task's execution-stage requirement ("understanding smoke, generation smoke"). Exact CLI invocation/args will be confirmed from each script's argparse/CLI surface in stage 3, run verbatim with no unofficial modification.

## Estimated VRAM, storage, and wall-clock

Per `docs/parameter_breakdown.md`'s documented example output for `sensenova/SenseNova-U1-8B-MoT` (the tool `scripts/inspect_model_params.py` will be re-run against the actual downloaded checkpoint in stage 4 to confirm these figures, not merely assumed):

- Total parameters: 17.552B; load dtype bfloat16; total memory: 35.105GB
- Group breakdown: `generation_transformer` 8.186B params / 16.373GB (46.64% of total memory); `understanding_transformer` 8.121B / 16.243GB (46.27%); `shared` 1.245B / 2.489GB (7.09%)
- Pathway coverage (shared counted in both): understanding pathway 9.366B params (53.36%); generation pathway 9.431B params (53.73%)
- No explicit author-published VRAM figure for inference was found in this pass; given the bf16 weight footprint (~35.1GB) plus activation/KV-cache overhead, a single H20 (96GB) should have ample headroom for both single-modality smokes. This is an estimate; stage 2/3 will record the measured peak.
- H20-FoldUMM (8x H20 96GB) matches the resource envelope's "one H20 GPU by default."

## MoT routing mechanism — draft basis for the formal routed-overlap audit

Preliminary, code-level (not yet weight-level) observations from the repository at the pinned commit, to be finalized once the checkpoint's `named_parameters()` is enumerated in stage 4:

- The 8B-MoT checkpoint implements "Mixture-of-Transformers" routing via a per-token/per-sample boolean mask (an `image_gen_indicators`-style flag observed in the modeling code) that selects between duplicate weight sets — generation-path modules carry a distinct suffix (observed pattern: `_mot_gen`) versus their understanding-path counterparts.
- A separate, not-yet-implemented mixed-modality code path was found gated behind a `NotImplementedError` referencing an open upstream issue (tracked as "issue #207" in the repository's issue tracker) — i.e. simultaneous mixed understanding+generation routing in a single forward pass is not currently supported by the released code; this is a **static, code-visible limitation**, not a runtime failure this admission needs to work around, and is exactly the kind of "static assumption violation" the task's routed-overlap audit is meant to capture and record (not fix).
- The out-of-scope A3B MoE variant uses a categorically different routing mechanism (a learned MoE gate over expert sub-networks) rather than the MoT boolean-mask/duplicate-weight-set mechanism used by the 8B-MoT checkpoint this admission targets — these two mechanisms must not be conflated when the formal routed-overlap audit is written up in stage 4/5 (`routed_overlap` block in `runs/admission-sensenova-u1-v1/metrics.json`).

This section is a draft; the formal audit (with weight-level `named_parameters()` evidence and an explicit `static_assumption_violations_recorded: true` field) will be produced in the later parameter-block-registry / routed-overlap-audit stage, not here.

## Missing assets / open items before stage 2

1. The `sensenova/SenseNova-U1-8B-MoT` safetensors shards are not yet downloaded or hash-verified; HuggingFace does not publish a per-file sha256 in its API, so file hashes will be computed after download and recorded in the storage-preflight / artifact-verification stage.
2. No dedicated venv/uv environment for SenseNova-U1's pinned dependency set (`torch==2.8.0`, CUDA 12.8) exists yet in the H20-FoldUMM container; it must be built fresh, separate from Show-o2's `torch==2.5.1` environment, and `flash-attn`'s optional CUDA-specific wheel availability will be checked (falls back to SDPA if unavailable — this is a documented, non-blocking fallback per the repo's own `pyproject.toml` comment, not an unofficial fix).
3. The A3B MoE variant's exact HF repo ID was not resolved in this pass (out of scope; recorded for completeness only, not a blocker for the 8B-MoT admission).
4. No non-Apache third-party dependency (VAE, safety checker, etc.) has been identified yet; this will be re-checked once `config.json` is inspected during download.
5. No credentials are required for the primary checkpoint (confirmed `gated: false`, `private: false` via the HF API).

## Proposed parameter-block boundaries (draft; refined in stage 4)

Based on `docs/parameter_breakdown.md`'s documented group/pathway breakdown alone (no weights loaded yet in this stage): the natural **shared** block is the `shared` parameter group (1.245B params / 2.489GB, 7.09% of total memory) — used by both understanding and generation pathways per the pathway-coverage figures, which double-count this group in both. The natural **understanding-private** block is the `understanding_transformer` group (8.121B / 16.243GB, 46.27%). The natural **generation-private** block is the `generation_transformer` group (8.186B / 16.373GB, 46.64%). This is a provisional reading of the documented example breakdown, not the actual weight-level module names on the downloaded checkpoint, which stage 4 will enumerate exactly via `named_parameters()` (using `scripts/inspect_model_params.py --model_path sensenova/SenseNova-U1-8B-MoT`, mirroring T210's approach) with the explicit goal of zero unassigned trainable parameters.

## Conclusion of this stage

No blocker identified for proceeding to stage 2 (storage preflight + environment build + checkpoint download). The 8B-MoT checkpoint (`sensenova/SenseNova-U1-8B-MoT`) is the confirmed admission target — U1, not U1.5, per the Project Status/news-entry evidence above. Estimated bf16 weight footprint is ~35.1GB, well within a single H20's 96GB, consistent with the resource envelope's "one H20 GPU by default." Proceeding next to the storage preflight required by the resource envelope's "weights and caches must execute from verified local SSD" clause.
