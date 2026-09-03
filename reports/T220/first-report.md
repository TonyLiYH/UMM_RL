# T220 first report — UniDDT admission

Per `tasks/T220-uniddt-admission.md`'s "First report" requirement, this is
published before any GPU execution or large-asset download. All information
below was gathered by read-only inspection of the official GitHub repository
(fetched file-by-file via the GitHub API/raw content, not cloned locally) and
public HuggingFace/arXiv metadata; no weights were downloaded and no code was
modified.

## Official repository

- URL: `https://github.com/MCG-NJU/UniDDT`.
- Pinned revision: `d04e037c0e1011a64703ad97d7bc4993bb69eade` (tip of `main`
  at audit time, resolved via `git ls-remote`).
- Paper: "UniDDT: Unifying Multimodal Understanding and Generation with
  Decoupled Diffusion Transformer", Wang, Li, Chen, Gao, Teng, Wang.
  arXiv:2606.16255. ECCV 2026 Spotlight.
- **Repository license: none found.** No `LICENSE` or `LICENSE.md` file
  exists at the repo root (`git/trees/<rev>?recursive=1` lists 114 objects,
  none named `LICENSE*`; direct `raw.githubusercontent.com` fetches of both
  candidate filenames return HTTP 404). `github.com/repos/MCG-NJU/UniDDT`'s
  license API also returns 404 ("Not Found"). This is a genuine upstream gap,
  not a fetch failure — the repository is public with 11 stars, 1 fork, and
  an active issue tracker, so the omission appears to be an authoring
  oversight rather than an intentionally withheld license. **Flagged as the
  primary open item for local-reviewer decision before any onward use beyond
  read-only admission smoke** (see "Open items" below).

## Checkpoint identifiers and sizes

| Asset | HF repo ID | Revision (sha) | File | Size (bytes) |
|---|---|---|---|---|
| VLM-UniDDT 512 (admission target) | `MCG-NJU/UniDDT` | `1d9541af2314873d77e398e515d8d5a93480be13` | `vlm_uniddt_512.ckpt` | 22,636,060,254 |
| VLM-UniDDT 1024 (not targeted this admission) | `MCG-NJU/UniDDT` | `1d9541af2314873d77e398e515d8d5a93480be13` | `vlm_uniddt_1024.ckpt` | 22,636,061,324 |

Both checkpoints share one architecture (Qwen3-VL-4B backbone, 24-layer/
1024-dim Noisy ViT, 16+4-layer/1536-dim diffusion decoder) and are ~22.6GB
each of fp32 EMA weights (repo total storage reported by the HF API:
45,305,676,010 bytes, matching the sum of the two files). The 512 checkpoint
is the smaller-resolution demo target and is recommended for this admission's
smoke stage; the 1024 checkpoint "continues from" the 512 one per the
README and is not separately required. HuggingFace does not publish a
per-file sha256 via its metadata API; the file's own hash will be computed
after download and recorded in the stage-2 admission manifest, matching the
T210 precedent.

The repo's `.gitignore` (`*.ckpt`, `*.safetensors`) confirms checkpoints are
never committed to Git and are Hub-only artifacts, consistent with the model
card's file listing.

**Checkpoint license: also unspecified.** The `MCG-NJU/UniDDT` HF model repo
carries no `license` field in its card metadata (`tags: ["region:us"]` only)
and is not gated (`"gated": false`). This compounds the repository-license
gap above — neither the code nor the released weights carry an explicit
license.

## Required external dependencies (recorded per frozen protocol)

The README states the released checkpoint already contains the LLM backbone
and Noisy ViT weights, so at inference time only two external components are
pulled from the Hub, not the full Qwen3-VL-4B checkpoint:

| Component | HF repo ID | Revision (sha) | License | Gated | Purpose |
|---|---|---|---|---|---|
| Qwen tokenizer/processor | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | `apache-2.0` (HF card) | `false` | tokenizer + chat template only; the full 4.4B-parameter backbone weights are not separately loaded because the release checkpoint already carries them (`llm_backbone.ckpt_path: null` in the release config) |
| Visual latent space (FLUX VAE) | `diffusers/FLUX.1-vae` | `da548cfb003bdeebaff6da0211fc8fbc67cb563a` | **unresolved** | `false` | `AutoencoderKL.from_pretrained` target in `src/models/autoencoder/latent.py`; 83.8M BF16 params, ~168MB |

**FLUX VAE license is a second unresolved open item**, distinct from and
compounding the repo/checkpoint-license gap above: the `diffusers/FLUX.1-vae`
repacking repo itself declares no `license` field in its card metadata
(only `tags: ["diffusers","safetensors","endpoints_compatible","region:us"]`).
Its `config.json` (`scaling_factor: 0.3611`, `shift_factor: 0.1159`,
`latent_channels: 16`) matches the published FLUX.1 VAE configuration exactly,
but that configuration is shared verbatim between `black-forest-labs/FLUX.1-dev`
(license `other` / `flux-1-dev-non-commercial-license`, gated) and
`black-forest-labs/FLUX.1-schnell` (license `apache-2.0`, gated), so config
values alone cannot disambiguate which upstream license this specific
VAE-only export inherits. Resolving this exactly would require downloading
and hash-comparing the VAE weights against both official sources — deferred
to a later stage per this task's "first report: metadata work only" scope,
and recorded here as an explicit open item rather than assumed permissive.

Training-only dependencies (not required for the admission smoke target,
recorded for completeness per the frozen protocol):

- `facebookresearch/dinov2` (`git clone` into `./torch_hub/`) — REPA
  alignment-loss teacher, used only by `training_repa.py`/`training_joint.py`'s
  `DINOv2` encoder during joint training. Not needed for inference smoke; its
  own license (Apache-2.0 for code, CC-BY-NC 4.0 for some released weight
  variants per Meta's repo) was not independently reverified this pass and is
  out of scope since no training is authorized.
- `djghosh13/geneval` and `TencentQQGYLab/ELLA` (DPG-Bench) — official
  evaluators used only for `main.py predict`-based benchmark scoring, not for
  the admission smoke target (Gradio-equivalent single-sample
  understanding/generation calls). Not needed for this admission.

## Required Python environment

`requirements.txt` (repo root):

```
torch==2.5.1, torchvision==0.20.1, torchaudio==2.5.1
lightning==2.5.0.post0, omegaconf==2.3.0, jsonargparse[signatures]==4.41.0
diffusers==0.35.0, bitsandbytes==0.46.0, accelerate==1.10.0
transformers==4.57.0, pydantic==1.10.9
```

Plus, per the README's install step, `gradio<4` (the demo app,
`app_uniddt.py`, uses the Gradio 3.x `Blocks` API). This is a distinct,
non-overlapping environment from every other environment recorded in
`reference_server_status`, including T210's Show-o2 venv: both pin
`torch==2.5.1`, but UniDDT requires `diffusers==0.35.0` /
`transformers==4.57.0`, materially newer than Show-o2's
`diffusers==0.31.0` / `transformers==4.47.0`. The H20-FoldUMM container's
system Python (3.10.12) currently has `torch==2.6.0+cu124` and
`transformers==4.52.1` installed system-wide — neither matches — confirming
a fresh venv is required, per `dev-env-paths.md`.

`flash-attn` is not in `requirements.txt`; the model instead uses
`torch.nn.attention.flex_attention` (stdlib PyTorch, `torch>=2.5`), avoiding
Show-o2's build-from-source `flash-attn` risk entirely.

## Entry points

| Task path | Script | Notes |
|---|---|---|
| Understanding (captioning at a chosen noise level) and generation (T2I) | `app_uniddt.py` | Single Gradio app; leave the reference-image input empty for T2I, upload an image for captioning. Both branches share one `Pipeline` class (`_und_forward`, `_gen_forward`) and one loaded `DDT2` denoiser. |
| Benchmark scoring (GenEval/DPG-Bench) | `main.py predict -c configs_uniddt/vlm_uniddt_{512,1024}.yaml --ckpt_path <path>` | Requires local GenEval/DPG-Bench metadata; not required for the admission smoke target. |
| Training (all 3 stages; not authorized) | `main.sh configs_uniddt/vlm_uniddt_512.yaml` → `main.py fit` | Recorded for completeness only. |
| Checkpoint bf16 re-export (optional storage optimization) | `tools/convert_weight/export_release_ckpt.py --ckpt_path ... --dtype bfloat16` | Not required; fp32 weights are used for reported GenEval/DPG numbers per the README. |

For a non-interactive, scriptable admission smoke (no Gradio server), stage 3
will drive `app_uniddt.py`'s `Pipeline.__call__`/`_und_forward`/`_gen_forward`
methods directly via a small external harness (imports the official module,
touches no Show-o2/UniDDT source, matching the T210 `timing_wrapper.py`
precedent) rather than launching the Gradio server itself.

## Proposed shared/private/frozen block boundaries (draft; refined in stage 4)

Read directly from `src/models/transformer/uniddt/ddt2.py`'s `DDT2` module
(the top-level denoiser wiring all three components) and the release configs
(`configs_uniddt/vlm_uniddt_512.yaml`), no weights loaded yet:

- **Shared**: `noisy_encoder` (`Qwen3VLVisionModel`-derived "Noisy ViT",
  24-layer, 1024-dim, timestep-conditioned via AdaLN-zero) and
  `llm_backbone` (`Qwen3VLTextModel`, 36-layer, hidden 2560). Both are called
  identically inside `forward_gen` (generation) and `forward_und_prefill`/
  `forward_und_decoding` (understanding) — `DDT2.forward_gen` and
  `DDT2.forward_und_prefill` both invoke `self.noisy_encoder(...)` then
  `self.llm_backbone(...)` with the same signature before branching. This
  matches the README's framing ("the Noisy ViT encoder and the LLM backbone
  jointly handle semantic perception for *both* understanding and
  generation").
- **Generation-private**: `diffusion_decoder` (`FlattenDiT`, 16-layer main +
  4-layer condition-refiner, 1536-dim). Called only from `forward_gen`/
  `forward_train`'s `gen_step=True` branch; never invoked by
  `forward_und_prefill` or `forward_und_decoding`.
  Conditioned only on the LLM backbone's refined hidden states
  (`condition_hidden_states=llm_hidden_states`), matching the README's
  "decouples diffusion decoding from text decoding" claim structurally.
- **Frozen, always** (independent of the release config's freeze flags):
  `llm_backbone.embed_tokens` and `llm_backbone.lm_head` — `DDT2.__init__`
  calls `no_grad(...)` on both unconditionally, even when
  `freeze_llm_backbone=False`.
  `AutoencoderKL` (FLUX VAE) — loaded separately in `LatentAE`, used only for
  `encode`/`decode`, never part of the denoiser's `named_parameters()`.
- **Release-config freeze state** (the checkpoint's actual trained
  configuration, from `configs_uniddt/vlm_uniddt_512.yaml`):
  `freeze_noisy_encoder: true`, `freeze_llm_backbone: true`,
  `freeze_diffusion_decoder: false` — i.e. the released checkpoint's final
  training stage only updated the generation-private diffusion decoder,
  consistent with the README's stage-3 "duality post-training" description
  (freeze understanding, train only the diffusion decoder). This is a
  provisional reading of the module graph, not the weight-level
  `named_parameters()` enumeration, which stage 4 will produce exactly once
  the checkpoint is loaded.

## Storage, VRAM, and compute estimate (to be measured, not assumed, in stage 2)

- **Download footprint**: ~22.6GB (`vlm_uniddt_512.ckpt`, the sole large
  asset needed — no separate LLM backbone checkpoint download required) +
  ~168MB (FLUX VAE) + a few MB (Qwen3-VL-4B-Instruct tokenizer/processor
  files only, not its 4.4B-parameter weights) ≈ 23GB total. This is
  materially smaller than T210's Show-o2 footprint (~37GB across four
  separate components), because the entire trained architecture ships in one
  file.
- **VRAM**: no explicit figure published by the authors. The release config
  uses `precision: bf16-mixed` and the Gradio pipeline runs under
  `torch.autocast(dtype=torch.bfloat16)`. Qwen3-VL-4B's own HF listing shows
  4,437,815,808 BF16 parameters for the backbone; UniDDT's total (backbone +
  Noisy ViT + diffusion decoder) is therefore plausibly in the 5-6B range,
  well under H20's 96GB per-GPU headroom regardless of the estimate's
  precision. Measured, not assumed, in stage 2.
- **GPU-hours**: given the single-file checkpoint, `flex_attention` (no
  build-from-source dependency), and the T210 precedent (SSD-sourced load
  ~9s for a similarly-sized fp32 checkpoint), the full download + venv build
  + dual-path smoke is estimated at 3-6 GPU-hours, well inside the 10-hour
  envelope.

## Local execution environment (verified live, H20-FoldUMM container)

- 8× H20 96GB, driver `535.161.08`, CUDA toolkit `12.4.99` (`nvcc`).
- `/dockerdata` (local SSD, `xfs`, local block device, verified via
  `df -T`): 9.0TiB total, ~15GB used (by the unrelated T210/T230 admission
  caches), effectively empty for this task's ~23GB footprint.
- System Python `3.10.12`; system-wide `torch==2.6.0+cu124`,
  `transformers==4.52.1` — a fresh venv is required (see above).

## Open items before stage 2 (recorded, not silently resolved)

1. **No repository license** (`MCG-NJU/UniDDT`, confirmed 404 on
   `LICENSE`/`LICENSE.md` and the GitHub license API) — primary open item,
   requires local-reviewer decision on acceptable use scope before anything
   beyond read-only admission smoke.
2. **No checkpoint license** (`MCG-NJU/UniDDT` HF model card has no `license`
   tag) — compounds item 1.
3. **FLUX VAE license ambiguous**: `diffusers/FLUX.1-vae` declares no
   license; its config matches both the non-commercial `FLUX.1-dev` and the
   Apache-2.0 `FLUX.1-schnell` VAE identically, and disambiguating requires a
   weight-level hash comparison deferred to a later stage.
4. **`vlm_uniddt_1024.ckpt`'s own hash and the 512 checkpoint's hash** are
   not yet verified against any authoritative value (HF publishes no
   per-file sha256); both will be computed post-download and recorded.
5. **No dedicated venv for this pinned dependency set exists yet** in any
   running container; must be built fresh inside H20-FoldUMM.
6. **DINOv2's exact weight license was not independently reverified** this
   pass (GitHub API rate-limited mid-audit); non-blocking since no training
   is authorized and DINOv2 is not required for the inference smoke target.

None of these is treated as a stage-2 blocker under this task's frozen
protocol — they are recorded, unresolved-but-non-blocking open items,
matching the disposition T210 gave its own two open license items (Wan2.1
VAE, safety checker), which local review accepted with documented
limitations. Given items 1-3 concern the *primary* model and its *primary*
visual-latent dependency (a materially larger share of the released
artifact than T210's peripheral open items), this report explicitly
recommends local review treat them as requiring an explicit decision before
any onward use of this admission's evidence beyond audit/reproduction.

## Conclusion of this stage

No storage, availability, or resource-envelope blocker identified for
proceeding to stage 2 (environment build + checkpoint smoke) with the 512
checkpoint. The unresolved license items above are recorded per the
"Automated submission gate" and are not, per T210 precedent, a stage-2
blocker on their own; they are carried forward into every subsequent report
in this task and flagged for explicit local-reviewer attention given their
centrality to this specific admission.
