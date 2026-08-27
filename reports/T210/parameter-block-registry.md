# T210 — Parameter-block registry (draft)

Stage 4 of the T210 execution protocol: enumerate the actually-loaded `showlab/show-o2-1.5B`
checkpoint's `named_parameters()` and refine the provisional shared/understanding-private/
generation-private boundary sketched in `first-report.md` ("Proposed parameter-block boundaries
(draft)") into an exact, weight-level map. Machine-readable form:
`configs/admission/showo2/parameter-block-registry.yaml`. No training was started; this is an
audit artifact only.

## Method

Loaded `models.Showo2Qwen2_5.from_pretrained("showlab/show-o2-1.5B", use_safetensors=False)` (the
same call `inference_mmu.py`/`inference_t2i.py` make) using config
`configs/showo2_1.5b_demo_432x432.yaml` (sha256
`d9f754ce8bdaf3a96cb6862782b51c781e2b0d3099bf37e15d244048c0559982`), then iterated
`model.named_parameters()`, grouping by top-level submodule name and cross-checking each group's
actual data flow against `models/modeling_showo2_qwen2_5.py`'s `forward()`/`generate()`/`t2i_generate()`
methods (not just the name string) to decide shared vs. private. Script:
`/apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/dump_named_parameters.sh`
(outside the repo, per `allowed_paths`). Total: 3,063,740,640 parameters, matching the checkpoint
load with no missing/unexpected keys.

## Block map

| Block | Top-level modules | Params | % of total | Role |
|---|---|---:|---:|---|
| **shared — LLM backbone** | `showo` (`showo.model.*`, `showo.lm_head`) | 1,776,267,776 | 58.0% | Qwen2.5-1.5B-Instruct transformer + LM head. Runs identically for both task paths on the fused token stream. |
| **shared — fusion junction** | `fusion_proj` | 6,493,824 | 0.2% | `nn.Sequential` over `cat([image_embeds_und, image_embeds_gen], dim=-1)` (`modeling_showo2_qwen2_5.py:221,326`). Structurally a junction consuming both private branches' outputs, feeding the shared backbone — not private to either. |
| **understanding-private** | `image_embedder_und`, `position_embedding`, `und_trans` | 397,141,792 | 13.0% | Patchify (`image_embedder_und`) + SigLIP positional embedding (`position_embedding`) + 8-layer transformer encoder adapted from SigLIP's vision encoder with its final layer removed (`und_trans`, matches config's `num_und_trans_layers=8`). Computed whenever an image is present; feeds only into `fusion_proj`. |
| **generation-private** | `image_embedder_gen`, `time_embed`, `time_embed_proj`, `diff_proj`, `diffusion_head_a`, `diffusion_head_b` | 883,837,248 | 28.8% | Generation-path patchify (`image_embedder_gen`); flow-matching timestep embedding injected into the shared backbone's input stream (`time_embed`/`time_embed_proj`); post-backbone projection into the diffusion head's width (`diff_proj`); 10-block refiner transformer (`diffusion_head_a`, matches config's `num_refiner_layers=10`) and final adaLN output layer (`diffusion_head_b`) predicting the flow-matching velocity `v_pred`. Only exercised when sampling/denoising an image (t2i path); absent from the pure-understanding forward pass. |

Sum check: 1,776,267,776 + 6,493,824 + 397,141,792 + 883,837,248 = 3,063,740,640 — matches
`TOTAL_PARAMS` exactly, confirming no top-level module was missed.

**Frozen, non-trainable, outside `named_parameters()`:** the Wan2.1 VAE (`models.WanVAE`, loaded
from `Wan2.1_VAE.pth`, sha256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`)
is instantiated separately from `Showo2Qwen2_5` and used only to encode images to latents /decode
latents to images — it is not part of the model object whose `named_parameters()` was enumerated
above, and is not intended to be trained.

## Corrections to the stage-1 provisional reading

The provisional reading in `first-report.md` was directionally correct but had two boundary
mis-calls, now corrected against the actual code:

1. **`fusion_proj` was not previously called out** as its own category — it is neither purely
   shared (it has its own private weights, unlike the backbone which is reused verbatim) nor purely
   private to one branch (it structurally requires both `image_embeds_und` and `image_embeds_gen`
   as input). Registered here as a third category, "shared — fusion junction," rather than folding
   it into the LLM-backbone block.
2. **The timestep-embedding chain (`time_embed`, `time_embed_proj`) was not mentioned** in the
   stage-1 draft at all — it was found only by tracing `forward()`/`t2i_generate()`, not from the
   config schema. It is generation-private: `time_embed_proj`'s output is added into the shared
   backbone's `input_embeds` only at image-token offsets when `config.add_time_embeds` is set
   (`modeling_showo2_qwen2_5.py:224-232`), and `time_embeds` also conditions `diffusion_head_a`'s
   adaLN layers directly — both uses are generation-only.

Everything else in the stage-1 draft (und-transformer layer count, refiner layer count, VAE frozen)
matched the weight-level enumeration exactly.

## Conclusion of this stage

Parameter-block boundaries are now enumerated at the weight level, not inferred from config schema
alone, and cross-checked against actual forward-pass data flow rather than name-string heuristics
alone (catching `fusion_proj`'s dual-branch dependency and the previously-unlisted timestep-embedding
chain). Registry published as both a narrative report (this file) and a machine-readable draft
(`configs/admission/showo2/parameter-block-registry.yaml`) for the local review side to accept or
amend. No joint post-training has been started; per the frozen protocol, block-freezing/unfreezing
decisions for any future training run remain a local-authorization decision, not something this
draft prescribes.
