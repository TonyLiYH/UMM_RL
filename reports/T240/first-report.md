# T240 first report — UniAR admission

Per `tasks/T240-uniar-admission.md`'s "First report" requirement, this is published
before any GPU execution and before any large asset download. All information below
was gathered by read-only inspection of the official repository (shallow clone at
`/tmp/uniar-audit`, not committed — same pattern as T210's `/tmp/show-o-audit`) and
public HuggingFace/GitHub metadata; no weights were downloaded and no code was
modified or reimplemented.

## Official repository

- URL: `https://github.com/ShareLab-SII/UniAR` (single canonical repo; the paper's
  reference implementation).
- Pinned revision: `92d8718d4cf282254ae63a4944b07edba0ce7abf` (tip of `main` at audit
  time, confirmed live via `git ls-remote origin HEAD main` — both `HEAD` and
  `refs/heads/main` resolve to this commit).
- Repository license: Apache-2.0 (`pyproject.toml`: `license = {text = "Apache-2.0"}`;
  a top-level standalone `LICENSE` file was not found at repo root — the only
  `LICENSE` file in the tree belongs to the vendored `train/rl/trl/` subtree, which
  is itself Apache-2.0-licensed HuggingFace TRL). The `pyproject.toml` declaration is
  treated as authoritative for the UniAR-authored code; this is recorded as a minor
  open item (no standalone root `LICENSE` file), not a blocker, since every
  HuggingFace model card below independently confirms `apache-2.0`.
- Paper: arXiv:2606.18249 ("Unified Multimodal Autoregressive Modeling with Shared
  Context — Visual Tokenizer is Key to Unification"), ICML 2026.
- Authors/affiliation: Fudan University / Shanghai Innovation Institute / Qwen Team,
  Alibaba Inc.

## Checkpoint identifiers and sizes

| Asset | HF repo ID | Revision (sha) | License | Declared repo size |
|---|---|---|---|---|
| UniAR-RL (primary target — RL/GRPO finetune of UniAR-SFT) | `ShareLab-SII/UniAR-RL` | `6b02e4eee3d45b34f7f41e6218b6cc3c56332454` | apache-2.0 | 74.40 GB |
| UniAR-SFT (base checkpoint UniAR-RL is finetuned from) | `ShareLab-SII/UniAR-SFT` | `b84157ed4968737cdee5db2db6bbb5375490fdbd` | apache-2.0 | 74.39 GB |

Both are public, non-gated (`"gated": false` confirmed via the HF API for both repo
IDs), single canonical revisions with no variant proliferation (unlike T210's Show-o2,
UniAR ships exactly one size per checkpoint — no separate "1.5B/7B" family). This
admission targets **UniAR-RL** as the primary checkpoint (RL is the paper's
recommended/best-performing checkpoint; UniAR-SFT is its untuned base and is not
separately required for the boundary-control smoke).

Each checkpoint bundles four independently-versioned components in one HF repo,
matching the task's "record the visual tokenizer, decoder, and AR checkpoint
revisions separately" requirement — the repo-level commit sha is shared, but each
component is a distinct sub-tree with its own `config.json`/weights and will be
recorded as separate sibling manifest artifacts (per T210 R10's lesson that
`schemas/run-manifest.schema.json` forbids nested provenance objects):

| Component | Role | Size (from `UniAR-RL`) |
|---|---|---|
| `ar_model` (repo root: `model-0000{1..4}-of-00004.safetensors` + `config.json`) | Unified AR model: Qwen3-VL-8B-scale backbone (36 layers, hidden=4096) + `output_layer_vistok` head + `visual_decoder` (4-layer transformer stack, part of the AR head, **not** the pixel decoder — see naming-collision note below) | 19.25 GB (4 shards) |
| `bsq_encoder/` | BSQ discrete visual tokenizer: SigLIP-scale ViT (27 layers, hidden=1152) + lookup-free Binary Spherical Quantization (`bsq_dim=64`) | 1.31 GB |
| `sd3_transformer/` | Custom SD3 DiT transformer with SigLIP/BSQ-feature conditioning (`SD3Transformer2DModelWithSigLIP`) — the learned part of the pixel decoder | 4.95 GB |
| `sd3_pipeline/` | Full third-party Stable Diffusion 3.5-medium pipeline (VAE + 3 text encoders in multiple redundant precision/format copies: fp32+fp16 safetensors, `.safetensors`/sharded/`ComfyUI`-style `text_encoders/` variants) — frozen, off-the-shelf, dominates total checkpoint size | ~48.3 GB |

Checkpoint hash verification of each component's actual blob sha256 is deferred to
stage 2 (storage preflight / download), matching T210's precedent (HF's API does not
publish a per-file sha256 ahead of download; the file's own hash will be computed
after download and recorded in the run manifest).

## Licenses and redistribution constraints

Apache-2.0 across every declared asset (repo `pyproject.toml`, `UniAR-RL` model card,
`UniAR-SFT` model card). Apache-2.0 permits redistribution and modification with
attribution and NOTICE-preservation; no additional non-commercial/research-only
clause was found on either HF model card. The bundled `sd3_pipeline/` component is a
third-party asset (Stability AI's SD3.5-medium); its own `sd3_pipeline/LICENSE.md`
(bundled inside the HF repo, not independently re-fetched from Stability AI's
canonical release in this pass) governs that specific sub-tree and is recorded as an
open item — non-blocking, since it is used strictly read-only/frozen as a fixed pixel
decoder and no redistribution of decoder weights is planned by this admission.

## Released vs. unreleased scope — the distinctive gate for this admission

This is the central finding this admission exists to establish, and it is read
directly from the repository, not inferred:

**README.md "News" section**: `[2026/06] Code and model weights released.`
**README.md "TODO" section (verbatim, the only TODO item in the file)**:
`- [ ] Release visual decoder training code.`

Cross-checked against the actual file tree (`/tmp/uniar-audit`), which confirms this
literally — there is no training script, config, or trainer anywhere that touches
`sd3_transformer`, `SD3Transformer2DModelWithSigLIP`, or
`StableDiffusion3PipelineWithSigLIP`:

- `grep -rl "sd3_transformer\|SD3Transformer2DModelWithSigLIP"` across the repo
  returns only two files: `uniar/modeling_vision_decoder.py` (the model/pipeline
  *class definitions* and inference-only `__call__`/`decode` methods, every entry
  point decorated `@torch.no_grad()`) and `train/rl/reward_server/decode/decoder.py`
  (which *instantiates* the frozen decoder inside the RL reward pipeline purely to
  render images for reward scoring — it never calls `.train()`, computes no loss
  against the decoder, and never backprops through it).
- No `train_decoder.py`, no `finetune_sd3.py`, no decoder-specific training config,
  and no decoder-training section in any of the four `docs/*.md` files
  (`inference.md`, `evaluation.md`, `reward_servers.md`, `training_rl.md`) exists
  anywhere in the tree.
- The only "Training" section in `README.md` is titled "Reinforcement Learning" and
  describes GRPO training of the **AR model** only (rollout + reward + policy
  update); it explicitly does not claim decoder training.

**This is the expected, correctly-documented finding this task is designed to
surface — not a task failure.** Per the frozen protocol ("do not claim visual-decoder
training support that the official repository does not release"), this admission
will not attempt to write, patch, or substitute decoder-training code. It will be
recorded as a limitation via `scope.unreleased_decoder_training_recorded=true` in
`runs/admission-uniar-v1/metrics.json`, matching the pre-existing note already on
file at `docs/surveys/related-work.md` line 12 ("AR/GRPO training public;
visual-decoder training remains unreleased") — this audit independently confirms
that note rather than merely repeating it.

### What IS released and will be exercised by this admission

| Capability | Entry point | Status |
|---|---|---|
| Image understanding (VQA/captioning) inference | `inference/chat.py` | Released, will be smoke-tested |
| Image generation (T2I) inference — AR visual-token rollout + frozen SD3 pixel decode | `inference/generate.py` | Released, will be smoke-tested |
| Batched multi-GPU generation / benchmark eval | `inference/generate_batch.py` | Released, out of scope for this admission's smoke (single-sample smoke suffices per the task's resource envelope) |
| AR model RL training (GRPO, rollout + multi-reward stack: HPSv2 / GenEval / OCR / unified VLM-judge) | `train/rl/train_grpo.py` + `train/rl/trl/trl/trainer/grpo_trainer_uniar.py` (vendored/customized TRL) | Released as a multi-node decode-server/reward-server/training-node architecture; **not executed** by this admission (training, not inference, and exceeds the resource envelope's "admission smoke... only" scope) but its trainable-block structure will be enumerated in stage 4 (`configs/admission/uniar/parameter-block-registry.yaml`) as required by the task |
| Visual-decoder (SD3 pixel decoder) training | none — no script exists in the repo | **Not released.** Explicitly listed as the sole open TODO. This admission will document, not attempt to fill, this gap |

### Naming-collision note (important for correctly reading `parameter-block-registry.yaml` in stage 4)

`UniARForConditionalGeneration.visual_decoder` (an `nn.ModuleList` of 4
`Qwen3VLTextDecoderLayer` instances, gated by `config.visual_transformer_decoder`,
weight-present in the checkpoint's `ar_model` shards) is **not** the pixel decoder.
It is a small trainable transformer stack that refines LLM hidden states immediately
before the `output_layer_vistok` BSQ-bit-prediction head — part of the AR model's
generation head, differentiable, and exercised during both SFT and RL training of the
AR model. The actual pixel decoder is the separate `UniARVisualDecoder` class
(wrapping `sd3_transformer` + `sd3_pipeline`), which is frozen and inference-only.
These two same-named-adjacent concepts must not be conflated in the stage-4
trainable-block map: `model.visual_decoder` (AR head, trainable, released with
training code) vs. `UniARVisualDecoder` (pixel decoder, frozen, training code
unreleased).

## Required dependencies

From `pyproject.toml` (`pip install -e .` for inference; `pip install -e ".[train]"`
adds `deepspeed>=0.16, datasets>=3.0, aiohttp>=3.9`):

- `torch==2.7.0`, `transformers==4.57.0`, `diffusers==0.37.1`, `accelerate>=1.4.0`
- `einops`, `safetensors`, `qwen-vl-utils`, `Pillow`, `torchvision`, `numpy`, `tqdm`
- `requests>=2.31`, `sentencepiece`, `protobuf`
- `flash-attn` (recommended, `--no-build-isolation`, matching-CUDA-toolchain build
  requirement — same class of risk flagged in T210's environment lock)

Requirements stated by the README: Python 3.12, CUDA 12.1+, GPU with >=24GB VRAM for
inference. This is a distinct pinned set from T210's Show-o2 environment
(`torch==2.5.1`/`transformers==4.47.0`/`diffusers==0.31.0`) — no overlap, a fresh venv
must be built inside the H20-FoldUMM container per `dev-env-paths.md`, matching T210's
pattern of one dedicated venv per admission task.

## Entry points

| Task path | Script | Notes |
|---|---|---|
| Image understanding | `inference/chat.py --model_path <ckpt> --image <url_or_path> --prompt <text>` | Single image + text question, one generated text response |
| Image generation | `inference/generate.py --model_path <ckpt> --prompt <text> --output_path <png>` | AR visual-token rollout (`generate_visual`) followed by `UniARVisualDecoder.decode` (frozen SD3 pixel decode) |
| Batch generation (benchmark eval) | `inference/generate_batch.py` | Multi-GPU via `accelerate`; not required for this admission's smoke |
| RL training (AR model only) | `train/rl/train_grpo.py` (uses `UniARGRPOTrainer` from vendored `trl`) | Not executed; trainable-block structure enumerated read-only in stage 4 |

Both smoke entry points (`chat.py`, `generate.py`) are exactly the README-documented
commands, to be run verbatim in stage 5, no unofficial modification.

## Estimated VRAM, storage, and wall-clock (to be measured, not assumed, in stage 2)

- Weight storage: 74.40 GB (UniAR-RL, bf16 as shipped per `config.json`
  `"torch_dtype": "bfloat16"`) if the full checkpoint (including the ~48.3GB
  `sd3_pipeline/` redundant-precision text-encoder bundle) is kept as-is. The task's
  smoke only requires the `ar_model` + `bsq_encoder` + `sd3_transformer` +
  a **single** precision/format variant of `sd3_pipeline` (VAE + 3 text encoders);
  redundant fp16-duplicate / `text_encoders/`-ComfyUI-style copies inside
  `sd3_pipeline/` can be excluded from the local-SSD execution copy to reduce
  footprint — this will be evaluated precisely against measured `df` free space in
  stage 2's storage preflight.
- README states >=24GB VRAM for inference; the model's own backbone (36-layer,
  hidden=4096, ~8B-parameter-class Qwen3-VL LLM) plus a frozen SD3.5-medium decoder
  is a substantially larger footprint than Show-o2-1.5B's ~1.5B-parameter LLM
  backbone; single-GPU inference on an H20 96GB should have ample headroom
  regardless, matching the resource envelope's "one H20 GPU by default."
- No official multi-GPU requirement for the two smoke entry points (`chat.py`,
  `generate.py`); both instantiate a single model on one device.

## Missing assets / open items before stage 2

1. No standalone root `LICENSE` file in the UniAR repo itself (license only declared
   in `pyproject.toml` + HF model cards); recorded as a minor open item, non-blocking
   given three independent apache-2.0 declarations (repo metadata, `UniAR-RL` card,
   `UniAR-SFT` card).
2. `sd3_pipeline/LICENSE.md` (Stability AI's SD3.5-medium terms, bundled inside the
   HF repo) not independently re-fetched from Stability AI's own canonical release in
   this pass; non-blocking since the pixel decoder is used strictly read-only/frozen.
3. Neither checkpoint's component blobs are hash-verified yet — deferred to stage 2
   per the standard admission protocol (HF's API does not expose per-file sha256
   pre-download).
4. No dedicated conda/venv environment for UniAR's pinned dependency set exists yet
   in any running container; a fresh venv must be built inside H20-FoldUMM, with
   `flash-attn` build again being the most likely source of an environment-smoke
   failure (as it was for T210) — this will be recorded explicitly if it occurs, not
   silently patched with an unofficial fix.
5. **Visual-decoder training code is confirmed absent from the official repository**
   (README's own open TODO, corroborated by an empty grep across the entire tree).
   This is the expected finding this admission exists to document, not an open risk
   to resolve — no further investigation of this specific gap is warranted; stage 6
   will produce the formal missing-code report.

## Proposed parameter-block boundaries (draft; refined in stage 4 from loaded `named_parameters()`)

Based on `config.json` alone (no weights loaded yet): the Qwen3-VL-derived 36-layer
text backbone (`hidden_size=4096`) plus its token embedding/`lm_head` are the natural
**shared** block (serves both text generation/understanding and, via
`image_gen_step`'s `_skip_final_norm` prenorm-exit path, the visual-token rollout);
the BSQ vision tower (27-layer SigLIP-scale ViT, `vision_config`, frozen at RL-train
time per `train_grpo.py`'s `model.visual.requires_grad_(False)`) is the
**understanding-private** block (also used read-only for image-conditioned encoding
during generation/editing, but not updated by RL); `output_layer_vistok` (visual-token
BSQ-bit prediction head) plus the 4-layer `visual_decoder` transformer stack
(`config.visual_transformer_decoder=True`, `Qwen3VLTextDecoderLayer` instances) are
the natural **generation-private** block — trainable, exercised by both SFT and the
released GRPO RL recipe. The frozen SD3 pixel decoder (`sd3_transformer` +
`sd3_pipeline`) sits entirely outside the AR model's `named_parameters()` and is not
a candidate parameter block for the boundary-control comparison; it is inference
infrastructure only. This is a provisional reading of the config schema and top-level
`__init__`, not a full enumeration by loaded module name, which stage 4 will produce
exactly.

## Conclusion of this stage

No license/source blocker identified for proceeding to stage 2 (storage preflight,
environment setup, checkpoint download). The admitted scope is exactly and only the
officially released paths: understanding inference (`chat.py`), generation inference
(`generate.py`), and read-only enumeration of the AR model's trainable blocks
(including the AR-side `visual_decoder` head, not to be confused with the frozen
pixel `UniARVisualDecoder`). Visual-decoder (SD3 pixel-decoder) training is confirmed
unreleased by the official repository and will be recorded as a documented limitation,
not attempted, matching this task's frozen protocol.
