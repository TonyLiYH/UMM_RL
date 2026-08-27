# T210 first report — Show-o2 admission

Per `tasks/T210-showo2-admission.md`'s "First report" requirement, this is published before any GPU execution. All information below was gathered by read-only inspection of the official repository (shallow clone at `/tmp/show-o-audit`, not committed) and public HuggingFace/GitHub metadata; no weights were downloaded and no code was modified.

## Official repository

- URL: `https://github.com/showlab/Show-o` (Show-o2 lives in the `show-o2/` subdirectory of the same repository, not a separate repo).
- Pinned revision: `45a5a2de01d1ebd10cd5864d29310a76476cdf23` (tip of `main` at audit time, resolved via `git ls-remote`).
- Repository license: Apache License 2.0 (`LICENSE` at repo root).
- Model card license (HuggingFace): `apache-2.0` for both `showlab/show-o2-1.5B` and its declared dependencies below.

## Checkpoint identifiers and sizes

| Asset | HF repo ID | Revision (sha) | Reported size |
|---|---|---|---|
| Show-o2 1.5B (primary target for this admission) | `showlab/show-o2-1.5B` | `07ec16589d4fc5422a74dddbbc4b2cd11e551039` | 22.65GB total repo storage (single `pytorch_model.bin`, fp32) |
| LLM backbone | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 10.21GB repo storage |
| Vision/CLIP-like encoder | `google/siglip-so400m-patch14-384` | `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` | 3.51GB repo storage |
| 3D causal VAE | `Wan-AI/Wan2.1-T2V-14B` (file `Wan2.1_VAE.pth` only) | not resolved from a specific HF commit; **already present locally** | 507,609,880 bytes; sha256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` (local copy at `/apdcephfs_cq7/share_1447896/yihangli/models/pretrained/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth`) |

Other declared variants not targeted by this admission: `show-o2-1.5B-HQ`, `show-o2-7B`, `show-o2-1.5B-w-video-und`, `show-o2-7B-w-video-und`. This admission scopes to the 1.5B base checkpoint, matching the smallest declared VRAM/compute footprint and the `configs/showo2_1.5b_demo_432x432.yaml` demo config.

Checkpoint hash verification of `pytorch_model.bin` itself is deferred to stage 2 (environment/checkpoint smoke), where the file will actually be downloaded and hashed; only its declared repo-level size is recorded here.

## Licenses and redistribution constraints

Apache-2.0 across every declared asset above (repo, `show-o2-1.5B`, `Qwen2.5-1.5B-Instruct`, `siglip-so400m-patch14-384`). Apache-2.0 permits redistribution and modification with attribution and a NOTICE-preservation requirement; no additional non-commercial or research-only clause was found in any of the four license/model cards inspected. `Wan2.1_VAE.pth`'s specific license was not independently re-verified in this audit (it ships from a separate `Wan-AI/Wan2.1-T2V-14B` repo); this is recorded as a minor open item, not a blocker, since the file is used read-only as a fixed decoder and no redistribution is planned.

## Required dependencies

The show-o2-specific dependency set (`show-o2/build_env.sh`) differs from the top-level repo `requirements.txt` (which targets the older, discrete-token Show-o v1) and must be used instead:

- `torch==2.5.1`, `transformers==4.47.0`, `diffusers==0.31.0`, `accelerate==0.23.0`, `deepspeed==0.15.3`, `timm==1.0.12`, `huggingface-hub==0.24.0`
- `flash-attn` (installed with `--no-build-isolation`, i.e. requires a matching CUDA toolchain at build time)
- `einops`, `decord`, `sentencepiece`, `omegaconf`, `torchdiffeq`, `segment_anything`, `lightning==2.4.0`, `onnxruntime==1.20.1`, `av==12.0.0`, `moviepy`, `tensorflow==2.16.1`, `pandas`, `pyarrow==11.0.0`, `jsonlines`, plus `git+https://github.com/openai/CLIP.git`

This is a distinct, non-overlapping environment from every existing environment recorded in `reference_server_status` (`aigc-exp`, `state-reward`, `avgen-*`, `bl-vabench`, `bl-avbench`) — none pin `torch==2.5.1`+`transformers==4.47.0`+`diffusers==0.31.0` together. A new conda environment must be built inside the H20-FoldUMM container per `dev-env-paths.md`.

## Entry points

| Task path | Script | Demo config |
|---|---|---|
| Understanding (image VQA/captioning) | `show-o2/inference_mmu.py` | `configs/showo2_1.5b_demo_432x432.yaml` (or the `_7b_` variant) |
| Understanding (video) | `show-o2/inference_mmu_vid.py` | `configs/showo2_1.5b_demo_video_understanding.yaml` |
| Generation (text-to-image) | `show-o2/inference_t2i.py` | `configs/showo2_1.5b_demo_1024x1024.yaml` / `_432x432.yaml` / `_512x512.yaml` |
| Mixed-modality | `show-o2/inference_mixed_modality.py` | `configs/showo2_1.5b_demo_432x432_mixed_modal.yaml` |
| Training (not authorized; recorded for completeness only) | `show-o2/train_stage_one.py`, `train_stage_two.py`, `train_mixed_modality_simple.py` | `showo2_1.5b_stage_1_a/b.yaml`, `showo2_1.5b_stage_2_a/b/c.yaml` |

Example commands (from the README, to be run verbatim in stage 3, no unofficial modification):

```
python3 inference_mmu.py config=configs/showo2_1.5b_demo_432x432.yaml \
    mmu_image_path=./docs/mmu/<image>.jpg question='<question>'

python3 inference_t2i.py config=configs/showo2_1.5b_demo_1024x1024.yaml \
    batch_size=4 guidance_scale=7.5 num_inference_steps=50
```

The `showo2_1.5b_demo_432x432.yaml` config (inspected directly) confirms both task paths share one `Showo2` model (`model.showo.model_name`, `hidden_size=1536`, `num_und_trans_layers=8`, `num_refiner_layers=10`) backed by `Qwen2.5-1.5B-Instruct`, one `siglip-so400m` CLIP-style encoder (`clip_latent_dim=1152`), and one `wan21` VAE (`image_latent_dim=16`) — i.e. one shared backbone with task-specific heads/transformer layers, matching the "shared/private blocks" framing this admission must audit in stage 4.

## Estimated VRAM, storage, and wall-clock (to be measured, not assumed, in stage 2)

- Weight storage: ~22.65GB (show-o2-1.5B, fp32 as shipped) + 10.21GB (Qwen2.5-1.5B-Instruct) + 3.51GB (siglip) + 0.47GB (Wan2.1 VAE, already local) ≈ 37GB on disk if all fp32 copies are kept; the config declares `weight_type: "bfloat16"` for inference, so runtime VRAM should be well under this once weights are cast, plausibly in the 10-20GB range for the 1.5B variant at 432x432 given the modest `hidden_size=1536` backbone. This is an estimate; stage 2 will record the measured peak.
- No explicit VRAM figure is published by the authors for any variant.
- H20-FoldUMM (8x H20 96GB) has ample headroom for the 1.5B variant on a single GPU regardless of the estimate's accuracy.

## Missing assets / open items before stage 2

1. `showlab/show-o2-1.5B`'s `pytorch_model.bin` is not yet downloaded or hash-verified against a manifest-declared value (HuggingFace does not publish a per-file sha256 in its API; the file's own hash will be computed after download and recorded in stage 2's admission manifest).
2. `Wan2.1_VAE.pth` is already present locally at two paths (`Wan2.1-I2V-14B-480P/` and `Wan2.2-I2V-A14B/`), both presumably identical; stage 2 will confirm both hashes match and record which copy is used.
3. No dedicated conda/venv environment for show-o2's pinned dependency set exists yet in any running container; must be built fresh inside H20-FoldUMM per `build_env.sh`, with `flash-attn` build requiring a matching CUDA toolchain — this is the most likely source of an environment-smoke failure and will be recorded explicitly if it occurs, not silently patched with an unofficial fix (per the frozen protocol's "do not use unofficial fixes... without local authorization").
4. `Wan2.1_VAE.pth`'s license was not independently re-verified from its own source repo card in this pass (noted above, non-blocking).
5. No credentials are required for any of the four assets (all public, non-gated on HuggingFace, confirmed via `"gated": false` in the API response for `show-o2-1.5B`).

## Proposed parameter-block boundaries (draft; refined in stage 4)

Based on `configs/showo2_1.5b_demo_432x432.yaml` alone (no weights loaded yet): the `Qwen2.5-1.5B-Instruct` transformer backbone and its embedding/output layers are the natural **shared** block (used by both understanding and generation, per the "unified learning" framing in the README); `num_und_trans_layers=8` understanding-specific transformer layers and the `siglip` encoder are the natural **understanding-private** block; `num_refiner_layers=10` plus the flow-matching/transport head (`transport.*` config block) and VAE-adjacent projection layers are the natural **generation-private** block; the `Wan2.1` VAE itself is frozen (not trainable, used only as an encoder/decoder). This is a provisional reading of the config schema, not the weight-level module names, which stage 4 will enumerate exactly via `named_parameters()` on the loaded model.

## Conclusion of this stage

No blocker identified for proceeding to stage 2 (environment + checkpoint smoke). The 1.5B variant, 432x432 demo config, is the recommended admission target given the resource envelope in `docs/plans/showo2-first-attempt.md` section 7 ("first report: CPU and metadata work only; smoke target: one GPU where supported").
