# UniDDT admission — environment lock

Task-scoped environment description for the H20-FoldUMM GPU container venv used for every T220
smoke run (`/dockerdata/t220-uniddt/venv`, `python3 -m venv` from the container's system
`/usr/bin/python3` (3.10.12), on local SSD per `dev-env-paths.md`). This venv follows the
official `requirements.txt` in `MCG-NJU/UniDDT` (revision
`d04e037c0e1011a64703ad97d7bc4993bb69eade`) with one addition (below); it does not fork or
reimplement any UniDDT source.

## Required additional constraint beyond `requirements.txt`

1. **Install `einops`.** `requirements.txt` does not list `einops`, but
   `src/utils/packed_seqs/seqs.py` imports it unconditionally
   (`import einops`), and that module is imported transitively by
   `app_uniddt.py` (`Pipeline`, `_und_forward`, `_gen_forward`) as well as
   by every diffusion-sampling entry point. Without it, the very first
   `import app_uniddt` fails with `ModuleNotFoundError: No module named
   'einops'`. `einops==0.8.2` (current PyPI release at audit time) was
   installed and confirmed sufficient; no version pin is specified upstream.

No other package required a version different from what `requirements.txt` specifies.

## Full verified environment (confirmed live 2026-09-03, H20-FoldUMM container)

| Package | Version |
|---|---|
| `torch` | `2.5.1+cu124` |
| `torchvision` | `0.20.1+cu124` |
| `torchaudio` | `2.5.1+cu124` |
| `lightning` | `2.5.0.post0` |
| `omegaconf` | `2.3.0` |
| `jsonargparse` | `4.41.0` |
| `diffusers` | `0.35.0` |
| `bitsandbytes` | `0.46.0` |
| `accelerate` | `1.10.0` |
| `transformers` | `4.57.0` |
| `pydantic` | `1.10.9` |
| `gradio` | `3.50.2` (`gradio<4` pin from README) |
| `einops` | `0.8.2` (not in `requirements.txt`; see above) |
| `huggingface_hub` | `0.36.2` |
| `nvcc` / CUDA | `12.4.99` |
| GPU | `NVIDIA H20` (96GB) |

No `flash-attn` is required: the model uses `torch.nn.attention.flex_attention`
(stdlib PyTorch >=2.5), confirmed by `import` succeeding and both smoke paths
completing without any flash-attn-related error.

Full `pip freeze` snapshot (large third-party dependency list, not copied into this directory):
- durable copy: `/apdcephfs_cq7/share_1447896/yihangli/outputs/T220-uniddt-admission/pip_freeze_20260903.txt`
  (sha256 `67bebadacf3268e185a929da8a40a070cbc334a5eafc3c91489821ed4edd1742`)
- referenced from the admission run manifest as an `artifact`.

## Scope note

This lock file governs the UniDDT inference venv used for T220's read-only admission smoke tests
only (understanding + generation via `app_uniddt.py`'s `Pipeline` class, driven by an external
harness, no Gradio server, no source modification). It authorizes no training (joint,
"duality post-training", or otherwise); the repository's `main.py fit` / `main.sh` training
entry points remain out of scope for this task.
