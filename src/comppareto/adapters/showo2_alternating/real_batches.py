"""Real (non-synthetic) MMU/T2I batch construction for Show-o2-1.5B (T216).

Traces the reference training pipeline in the audited, T210-approved Show-o2
library source at ``SHOWO2_LIB`` below *exactly* (see
``reports/T216/first-report.md`` section 7, "fixed batches throughout"):

    raw image -> image_transform(432) -> WanVAE.sample() -> raw latent
    -> Transport.sample()/path_sampler.plan() -> (t, xt, ut)
    -> format_sequence_und / format_sequence_gen_qwen2_5 -> token sequence
    -> omni_attn_mask_naive -> attention_mask
    -> model.forward(text_tokens=..., image_latents=xt, t=..., image_labels=ut
       or None, text_labels=... or None, ...) -> (logits, loss_ntp) or
       (logits, loss_flow)

The library at ``SHOWO2_LIB`` is used only as a read-only dependency via
``sys.path`` -- nothing from it is copied into this repo (per the first
report's explicit constraint).

Both tasks are built from the *same* single fixed real demo image, encoded
once through the frozen Wan2.1 VAE, then passed through two independent
(but each individually deterministic-given-the-master-seed) transport
samples -- one per task. Both batches are constructed exactly once by the
caller (``run_k1.py``) and reused, unmodified, for every row/protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Library + asset paths (read-only dependency; never copied into this repo)
# ---------------------------------------------------------------------------
SHOWO2_LIB = "/apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/Show-o/show-o2"
DEMO_IMAGE = f"{SHOWO2_LIB}/docs/mmu/pexels-jane-pham-727419-1571673.jpg"
VAE_WEIGHTS = "/dockerdata/t210-showo2/assets/Wan2.1_VAE.pth"
LLM_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"

RESOLUTION = 432
# The model prepends one extra "slot" for the time embedding to every image
# span when add_time_embeds=True (true in this checkpoint's config.json).
# The checkpoint's own image_latent_height*width == 729 real image embeds;
# format_sequence_{und,gen_qwen2_5} need the *span length including that
# slot*, i.e. 730 -- see Showo2Qwen2_5.forward()'s "length - 1" comments.
_BASE_NUM_IMAGE_TOKENS = 729
NUM_IMAGE_TOKENS = _BASE_NUM_IMAGE_TOKENS + 1
MAX_SEQ_LEN = 1024
UND_MAX_T0 = 1.0  # forces xt == x1 (clean latent) for the understanding task

MMU_PROMPT = "Describe this image briefly."
T2I_PROMPT = "A photo of a dog on a grassy field."


def _ensure_lib_on_path() -> None:
    if SHOWO2_LIB not in sys.path:
        sys.path.insert(0, SHOWO2_LIB)


def build_text_tokenizer() -> Tuple[Any, Dict[str, int]]:
    _ensure_lib_on_path()
    from models.misc import get_text_tokenizer

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        LLM_MODEL_PATH, add_showo_tokens=True, return_showo_token_ids=True,
        llm_name="qwen2_5",
    )
    return text_tokenizer, showo_token_ids


def build_vae(device: str = "cuda") -> Any:
    _ensure_lib_on_path()
    from models import WanVAE

    return WanVAE(vae_pth=VAE_WEIGHTS, dtype=torch.float32, device=device)


def build_transport() -> Any:
    _ensure_lib_on_path()
    from transport import create_transport

    return create_transport(
        path_type="Linear",
        prediction="velocity",
        snr_type="lognorm",
        do_shift=True,
        seq_len=NUM_IMAGE_TOKENS,
    )


def encode_demo_image(vae: Any, device: str) -> torch.Tensor:
    """Load the single fixed real demo image and encode it once via the VAE.

    Returns the raw (un-noised) latent, shape (1, 16, 54, 54), fp32.
    Wan2.1 VAE spatial downsample factor is 8 (dim_mult=[1,2,4,4] => 3
    downsample2d stages => 2**3 == 8); 432 / 8 == 54. This raw 54x54 latent
    is then patch-embedded (patch_size=2) inside the model down to the
    27x27 == 729 token grid that image_latent_height/width and
    image_position_ids are sized for.
    """
    _ensure_lib_on_path()
    from datasets.utils import image_transform

    img = Image.open(DEMO_IMAGE).convert("RGB")
    pixel = image_transform(img, resolution=RESOLUTION)  # (3, 432, 432) fp32 in [-1, 1]
    pixel_values = pixel.unsqueeze(0).unsqueeze(2).to(device).float()  # (1, 3, 1, 432, 432)
    with torch.no_grad():
        image_latents = vae.sample(pixel_values)  # (1, 16, 1, 54, 54)
        if pixel_values.shape[2] == 1:
            image_latents = image_latents.squeeze(2)  # (1, 16, 54, 54)
    return image_latents


def build_mmu_batch(
    text_tokenizer: Any,
    showo_token_ids: Dict[str, int],
    transport: Any,
    x1_raw: torch.Tensor,
    device: str,
) -> Dict[str, Any]:
    """Build the fixed MMU (understanding) batch: real image + real caption
    text, ``image_labels=None`` so ``model.forward`` takes the loss_ntp-only
    branch."""
    _ensure_lib_on_path()
    from datasets.utils import format_sequence_und

    pad_id = text_tokenizer.pad_token_id
    text_tokens_list = text_tokenizer(MMU_PROMPT, add_special_tokens=False).input_ids

    text_tokens, text_labels, modality_positions, text_mask, image_mask = format_sequence_und(
        text_tokens_list,
        showo_token_ids["bos_id"], showo_token_ids["eos_id"],
        showo_token_ids["boi_id"], showo_token_ids["eoi_id"],
        pad_id, showo_token_ids["img_pad_id"],
        NUM_IMAGE_TOKENS, MAX_SEQ_LEN,
    )

    # und_max_t0=1.0 forces t == 1.0 exactly (verified NaN-safe: the
    # intermediate 1/t term in Transport.time_shift evaluates to 0/inf == 0,
    # not NaN) -> xt == x1 (clean, un-noised latent) for the MMU task.
    t, x0, x1 = transport.sample(x1_raw, UND_MAX_T0)
    t, xt, _ut = transport.path_sampler.plan(t, x0, x1)

    modality_positions_b = modality_positions.unsqueeze(0).to(device)

    _ensure_lib_on_path()
    from models import omni_attn_mask_naive

    block_mask = omni_attn_mask_naive(
        1, MAX_SEQ_LEN, modality_positions_b, device,
    ).to(torch.bfloat16)

    return dict(
        text_tokens=text_tokens.unsqueeze(0).to(device),
        image_latents=xt.to(device),
        t=t.to(device=device, dtype=torch.bfloat16),
        attention_mask=block_mask,
        text_masks=text_mask.unsqueeze(0).to(device),
        image_masks=image_mask.unsqueeze(0).to(device),
        text_labels=text_labels.unsqueeze(0).to(device),
        image_labels=None,
        modality_positions=modality_positions_b,
        output_hidden_states=True,
        max_seq_len=MAX_SEQ_LEN,
        device=device,
    )


def build_t2i_batch(
    text_tokenizer: Any,
    showo_token_ids: Dict[str, int],
    transport: Any,
    x1_raw: torch.Tensor,
    device: str,
) -> Dict[str, Any]:
    """Build the fixed T2I (generation) batch: real prompt text + the same
    real image's VAE latent as the flow-matching target. ``text_labels=None``
    so ``model.forward`` takes the loss_flow-only branch (text is never
    modeled for t2i pairs, per the library's own comment)."""
    _ensure_lib_on_path()
    from datasets.utils import format_sequence_gen_qwen2_5, remove_prefix

    pad_id = text_tokenizer.pad_token_id
    prompt = remove_prefix(T2I_PROMPT)
    text_tokens_list = text_tokenizer(prompt, add_special_tokens=False).input_ids
    system_tokens = [[], [], []]  # empty system prompt -> system_token_len == 0

    text_tokens, _text_labels_unused, modality_positions, text_mask, image_mask = format_sequence_gen_qwen2_5(
        text_tokens_list, system_tokens,
        showo_token_ids["bos_id"], showo_token_ids["eos_id"],
        showo_token_ids["boi_id"], showo_token_ids["eoi_id"],
        pad_id, showo_token_ids["img_pad_id"],
        NUM_IMAGE_TOKENS, MAX_SEQ_LEN, system_token_len=0,
    )

    # Full lognorm+shift sampling over [0, 1] (no max_t0 restriction) -> a
    # genuine noised latent xt and velocity target ut = x1 - x0.
    t, x0, x1 = transport.sample(x1_raw, None)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)

    modality_positions_b = modality_positions.unsqueeze(0).to(device)

    _ensure_lib_on_path()
    from models import omni_attn_mask_naive

    block_mask = omni_attn_mask_naive(
        1, MAX_SEQ_LEN, modality_positions_b, device,
    ).to(torch.bfloat16)

    return dict(
        text_tokens=text_tokens.unsqueeze(0).to(device),
        image_latents=xt.to(device),
        t=t.to(device=device, dtype=torch.bfloat16),
        attention_mask=block_mask,
        text_masks=text_mask.unsqueeze(0).to(device),
        image_masks=image_mask.unsqueeze(0).to(device),
        text_labels=None,
        image_labels=ut.to(device),
        modality_positions=modality_positions_b,
        output_hidden_states=True,
        max_seq_len=MAX_SEQ_LEN,
        device=device,
    )


def build_fixed_batches(device: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build both fixed batches exactly once. Called after the master seed is
    set and the model is loaded, before any protocol/row loop.

    Returns (mmu_batch_kwargs, t2i_batch_kwargs), both ready to pass directly
    as ``model(**batch)``.
    """
    _ensure_lib_on_path()
    print("[real_batches] building text tokenizer...", flush=True)
    text_tokenizer, showo_token_ids = build_text_tokenizer()

    print("[real_batches] loading Wan2.1 VAE and encoding demo image...", flush=True)
    vae = build_vae(device)
    x1_raw = encode_demo_image(vae, device)
    del vae  # frozen, only needed once; drop it to free memory
    torch.cuda.empty_cache()

    transport = build_transport()

    print("[real_batches] building fixed MMU batch...", flush=True)
    mmu_batch = build_mmu_batch(text_tokenizer, showo_token_ids, transport, x1_raw, device)

    print("[real_batches] building fixed T2I batch...", flush=True)
    t2i_batch = build_t2i_batch(text_tokenizer, showo_token_ids, transport, x1_raw, device)

    return mmu_batch, t2i_batch
