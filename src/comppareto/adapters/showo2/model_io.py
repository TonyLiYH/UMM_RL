"""Real Show-o2 model loading and MMU/T2I loss closures (T215).

This module is import-guarded: it requires the real Show-o2 repo checkout,
its dedicated venv (``torch==2.5.1+cu124``, ``transformers==4.47.0``,
``diffusers==0.31.0``), and the local-SSD checkpoint/tokenizer/VAE cache
already migrated by T210 (``configs/admission/showo2/environment-lock.md``,
``configs/feasibility/showo2/storage-preflight.json``). It is NOT imported
by ``tests/adapters/showo2/`` and is only exercised by
``run_feasibility.py`` against the real GPU checkpoint.

Design notes (see ``reports/T215/first-report.md`` sections 1 and 4, and the
task's own progress record):

- ``protocols.py``'s ``LossFn`` interface is generic over single flat
  ``torch.Tensor`` objects for ``theta_s``/``theta_p``. The real Show-o2
  subspace blocks (``fusion_proj``, ``und_trans.layers[0]``,
  ``diffusion_head_a[0]``) are each multi-leaf-tensor ``nn.Module``
  submodules. :class:`LeafBlock` + :func:`flatten_block`/:func:`unflatten_block`
  bridge this: every leaf parameter tensor under a fixed, recorded dotted-name
  order is concatenated into one flat vector (the ``theta_s``/``theta_p``
  passed to/from ``protocols.py``), and :func:`call_with_overrides`
  substitutes the un-flattened per-leaf tensors back into the model via
  ``torch.nn.utils.stateless._reparametrize_module`` (the same primitive
  ``torch.func.functional_call`` uses internally) for the duration of one
  forward call -- the substituted tensors remain part of the autograd graph
  (no ``.data.copy_()`` anywhere), and the model's OWN stored
  ``nn.Parameter`` objects are swapped back untouched immediately on exit,
  so no persistent parameter update ever survives a loss-closure call by
  construction (independent of ``state.py``'s snapshot/restore, which is
  still run around every transition per the task's reversibility
  requirement, as defense in depth).
- Every parameter except the leaf tensors of the 3 declared subspace blocks
  is frozen (``requires_grad_(False)``) once, at load time
  (:func:`freeze_all_except_subspace`). The 3 blocks' OWN stored parameters
  are ALSO frozen (never touched via ``.grad``/optimizer directly) -- the
  differentiable ``theta_s``/``theta_p`` used by ``protocols.py`` are
  separate leaf tensors created by :func:`flatten_block`, outside the
  model's own parameter set.
- MMU loss closure calls the official ``model.forward_und_only(...)``
  (never touches ``diffusion_head_a``), reusing its returned ``loss_ntp``
  (``next_token_prediction``, official ``models/misc.py``) verbatim -- no
  loss formula is reimplemented here.
- T2I loss closure calls the official ``model.forward(...)`` with
  ``text_labels=None, image_labels=<ut>``, which is confirmed (source read,
  ``models/modeling_showo2_qwen2_5.py`` lines ~396-403) to return
  ``(logits, loss_flow)`` via the official ``velocity_prediction`` -- a
  faithful single-step differentiable T2I loss IS available through the
  official forward() API without any source modification, so T2I is
  implemented (not failed) here.
- Neither closure has ground-truth targets available from the asset
  manifest (first-report.md section 3 only lists images+questions / text
  prompts, no reference captions or reference target images). Both
  closures therefore build a fixed, seeded PSEUDO-TARGET once (frozen
  no_grad generation for MMU; the disjoint MMU image's VAE-encoded latent,
  run through the official transport sampler, for T2I) and hold it fixed
  for the rest of the diagnostic -- documented here and in
  ``runs/feasibility-showo2-v1/notes.md`` as a data-construction choice for
  exercising the gradient/FD numerics, NOT a claim about caption quality or
  prompt-image correspondence. No loss formula is invented either way: the
  scalar being differentiated is always the model's own returned
  ``loss_ntp``/``loss_flow``.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.utils.stateless import _reparametrize_module

from comppareto.adapters.showo2.protocols import LossFn

DEFAULT_SHOWO2_REPO_ROOT = (
    "/apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/Show-o/show-o2"
)
DEFAULT_DEMO_CONFIG_REL = "configs/showo2_1.5b_demo_432x432.yaml"

SHARED_BLOCK_PREFIX = "fusion_proj"
UND_PRIVATE_BLOCK_PREFIX = "und_trans.layers.0"
GEN_PRIVATE_BLOCK_PREFIX = "diffusion_head_a.0"


def _ensure_repo_on_path(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# ---------------------------------------------------------------------------
# Leaf-tensor flatten/unflatten bridge (protocols.py's flat theta_s/theta_p
# <-> the real multi-tensor Show-o2 subspace blocks).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafBlock:
    """A fixed, recorded-order enumeration of every leaf parameter tensor's
    dotted name and shape under one module prefix (e.g. ``fusion_proj`` or
    ``und_trans.layers.0``), used to flatten that block into (and back out
    of) the single flat ``torch.Tensor`` that ``protocols.py`` operates on.
    """

    prefix: str
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]

    @property
    def numel(self) -> int:
        total = 0
        for shape in self.shapes:
            n = 1
            for d in shape:
                n *= d
            total += n
        return total


def build_leaf_block(model: nn.Module, prefix: str) -> LeafBlock:
    """Enumerate every leaf parameter under ``prefix`` (exact match or
    ``prefix + "."``-prefixed), in ``named_parameters()`` order (stable
    across calls on the same, unmodified module).
    """

    names: list[str] = []
    shapes: list[tuple[int, ...]] = []
    for name, p in model.named_parameters():
        if name == prefix or name.startswith(prefix + "."):
            names.append(name)
            shapes.append(tuple(p.shape))
    if not names:
        raise ValueError(f"no parameters found under prefix {prefix!r}")
    return LeafBlock(prefix=prefix, names=tuple(names), shapes=tuple(shapes))


def flatten_block(model: nn.Module, block: LeafBlock) -> torch.Tensor:
    """Concatenate ``block``'s leaf tensors' CURRENT ``.data`` (detached)
    into one flat vector, in ``block.names`` order.
    """

    params = dict(model.named_parameters())
    parts = [params[name].detach().reshape(-1) for name in block.names]
    return torch.cat(parts)


def unflatten_block(flat: torch.Tensor, block: LeafBlock) -> dict[str, torch.Tensor]:
    """Split a flat vector back into ``{dotted_name: reshaped_tensor}`` per
    ``block``'s recorded names/shapes, preserving ``flat``'s autograd graph
    connectivity (pure ``.reshape`` views/slices, no ``.detach()``/``.data``).
    """

    if flat.numel() != block.numel:
        raise ValueError(
            f"flat tensor has {flat.numel()} elements, block {block.prefix!r} expects {block.numel}"
        )
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape in zip(block.names, block.shapes):
        n = 1
        for d in shape:
            n *= d
        out[name] = flat[offset : offset + n].reshape(shape)
        offset += n
    return out


def call_with_overrides(
    model: nn.Module,
    overrides: dict[str, torch.Tensor],
    method_name: str,
    call_kwargs: dict[str, Any],
) -> Any:
    """Call ``model.<method_name>(**call_kwargs)`` with ``overrides``'
    tensors substituted in place of the named parameters/buffers for the
    duration of this call only.

    Uses ``torch.nn.utils.stateless._reparametrize_module`` -- the exact
    primitive ``torch.func.functional_call`` builds on -- as a context
    manager: on ``__enter__`` it swaps the module's stored ``nn.Parameter``
    objects for ``overrides``' plain tensors (which may carry a live
    autograd graph back to ``theta_s``/``theta_p``); on ``__exit__`` it
    swaps the ORIGINAL parameters back untouched. The model's own
    parameters are therefore never mutated by this call, by construction.

    ``strict=False`` (the ``_reparametrize_module`` default) is used
    deliberately: ``overrides`` is intentionally a PARTIAL map covering only
    the 2 active subspace blocks' leaf tensors for this call, not every
    parameter/buffer in the whole model -- ``strict=True`` would instead
    require ``overrides`` to enumerate every one of them, which is neither
    needed nor intended here (confirmed via this function's own synthetic
    unit test, which reproduced and then resolved a ``strict=True``
    ``RuntimeError: Missing key(s)`` failure on a toy module with an
    un-overridden second submodule).
    """

    with _reparametrize_module(model, overrides):
        method = getattr(model, method_name)
        return method(**call_kwargs)


def freeze_all_except_subspace(model: nn.Module) -> None:
    """Freeze every parameter (``requires_grad_(False)``), including the 3
    declared subspace blocks' OWN stored parameters -- the differentiable
    ``theta_s``/``theta_p`` leaf tensors used by ``protocols.py`` are
    separate tensors (see :func:`flatten_block`), substituted in only for
    the duration of one forward call via :func:`call_with_overrides`, so no
    optimizer or ``.backward()`` call ever touches the model's own
    parameters directly.
    """

    for p in model.parameters():
        p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Model/tokenizer/VAE loading.
# ---------------------------------------------------------------------------


@dataclass
class ShowoEnv:
    model: Any
    vae: Any
    text_tokenizer: Any
    showo_token_ids: dict
    config: Any
    device: torch.device
    weight_type: torch.dtype
    hp: dict  # get_hyper_params(...) results, by name (see _hyper_params_dict)
    repo_root: str


def _hyper_params_dict(config, text_tokenizer, showo_token_ids) -> dict:
    from utils import get_hyper_params  # type: ignore

    (
        num_t2i_image_tokens,
        num_mmu_image_tokens,
        num_video_tokens,
        max_seq_len,
        max_text_len,
        image_latent_dim,
        patch_size,
        latent_width,
        latent_height,
        pad_id,
        bos_id,
        eos_id,
        boi_id,
        eoi_id,
        bov_id,
        eov_id,
        img_pad_id,
        vid_pad_id,
        guidance_scale,
    ) = get_hyper_params(config, text_tokenizer, showo_token_ids)
    return dict(
        num_t2i_image_tokens=num_t2i_image_tokens,
        num_mmu_image_tokens=num_mmu_image_tokens,
        num_video_tokens=num_video_tokens,
        max_seq_len=max_seq_len,
        max_text_len=max_text_len,
        image_latent_dim=image_latent_dim,
        patch_size=patch_size,
        latent_width=latent_width,
        latent_height=latent_height,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        boi_id=boi_id,
        eoi_id=eoi_id,
        bov_id=bov_id,
        eov_id=eov_id,
        img_pad_id=img_pad_id,
        vid_pad_id=vid_pad_id,
        guidance_scale=guidance_scale,
    )


def load_env(
    *,
    repo_root: str = DEFAULT_SHOWO2_REPO_ROOT,
    demo_config_rel: str = DEFAULT_DEMO_CONFIG_REL,
    device: str = "cuda:0",
    weight_type: torch.dtype = torch.float32,
) -> ShowoEnv:
    """Load the real Show-o2 model/tokenizer/VAE using the official code
    path (``Showo2Qwen2_5.from_pretrained``, mirroring
    ``inference_mmu.py``/``inference_t2i.py``/``train_stage_two.py``
    exactly), then freeze every parameter except the 3 declared subspace
    blocks (whose own stored parameters are frozen too -- see
    :func:`freeze_all_except_subspace`).

    ``weight_type`` defaults to fp32 (not the demo config's bf16) because
    the diagnostic's finite-difference check and ``create_graph=True``
    rerun-response chain need fp32 numerical precision (this mirrors
    T210's admission environment, which is fp32-capable, and does not
    change the pinned checkpoint/tokenizer/VAE assets themselves).
    """

    _ensure_repo_on_path(repo_root)

    from omegaconf import OmegaConf  # type: ignore

    from models import Showo2Qwen2_5, WanVAE  # type: ignore
    from models.misc import get_text_tokenizer  # type: ignore
    from utils import path_to_llm_name  # type: ignore

    config = OmegaConf.load(os.path.join(repo_root, demo_config_rel))
    dev = torch.device(device)

    vae = WanVAE(
        z_dim=16,
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_type,
        device=dev,
    )

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    model = Showo2Qwen2_5.from_pretrained(
        config.model.showo.pretrained_model_path, use_safetensors=False
    ).to(dev)
    model.to(weight_type)
    model.eval()

    # Mirror inference_mmu.py/inference_t2i.py's time-embedding token-count
    # adjustment (applied once, in place, before any hyperparameter reads).
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    hp = _hyper_params_dict(config, text_tokenizer, showo_token_ids)

    freeze_all_except_subspace(model)

    return ShowoEnv(
        model=model,
        vae=vae,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        config=config,
        device=dev,
        weight_type=weight_type,
        hp=hp,
        repo_root=repo_root,
    )


# ---------------------------------------------------------------------------
# MMU batch construction + loss closure.
# ---------------------------------------------------------------------------


def _load_and_encode_image(env: ShowoEnv, image_path: str) -> torch.Tensor:
    from datasets.utils import image_transform  # type: ignore
    from torchvision.datasets.folder import default_loader  # type: ignore

    image = default_loader(image_path).convert("RGB")
    resolution = int(env.config.dataset.preprocessing.resolution)
    px = image_transform(image, resolution=resolution).to(env.device).unsqueeze(0)
    with torch.no_grad():
        image_latents = env.vae.sample(px.unsqueeze(2)).squeeze(2).to(env.weight_type)
    return image_latents


@torch.no_grad()
def _generate_mmu_pseudo_caption(
    env: ShowoEnv, image_latents: torch.Tensor, question: str, seed: int
) -> str:
    """Frozen, no_grad greedy generation (mirrors ``inference_mmu.py``'s
    tested call path exactly, including its own manual embedding
    construction and ``model.mmu_generate``) used ONLY to produce a fixed
    pseudo-target caption for the differentiable NTP loss below -- this
    generation's own computation graph is discarded (``torch.no_grad``);
    it is never part of the diagnostic's measured gradients.
    """

    from models import omni_attn_mask_naive  # type: ignore

    torch.manual_seed(seed)
    hp = env.hp
    model = env.model
    dev = env.device

    image_embeds_und = model.image_embedder_und(image_latents)
    image_embeds_gen = model.image_embedder_gen(image_latents)
    image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
    image_embeds_und = model.und_trans(image_embeds_und)["last_hidden_state"]
    image_embeds = model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

    sys_prompt_ids = env.text_tokenizer(
        "system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False
    )["input_ids"]
    role_a = env.text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_b = env.text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    input_ids = env.text_tokenizer(question, add_special_tokens=False).input_ids
    text_tokens_a = torch.tensor([hp["bos_id"]] + sys_prompt_ids + role_a, device=dev)[None, :]
    text_tokens_b = torch.tensor(
        [hp["boi_id"], hp["eoi_id"]] + input_ids + role_b, device=dev
    )[None, :]
    text_embeds_a = model.showo.model.embed_tokens(text_tokens_a)
    text_embeds_b = model.showo.model.embed_tokens(text_tokens_b)

    time_embeds = model.time_embed(torch.tensor([[1.0]], device=dev), text_embeds_a.dtype)
    if hasattr(model, "time_embed_proj"):
        time_embeds = model.time_embed_proj(time_embeds)
    input_embeds = torch.cat(
        [text_embeds_a, text_embeds_b[:, :1], time_embeds, image_embeds, text_embeds_b[:, 1:]],
        dim=1,
    ).to(env.weight_type)
    modality_positions = torch.tensor(
        [text_tokens_a.shape[1] + 2, hp["num_mmu_image_tokens"]]
    )[None, None, :].to(dev)

    attention_mask = omni_attn_mask_naive(
        B=input_embeds.size(0),
        LEN=input_embeds.size(1),
        modalities=modality_positions,
        device=dev,
        inverted=True,
    ).to(input_embeds.dtype)

    output_tokens = model.mmu_generate(
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        top_k=1,
        max_new_tokens=64,
        eos_token=env.text_tokenizer.eos_token_id,
    )
    output_tokens = torch.stack(output_tokens).squeeze()[None]
    text = env.text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return text.strip() or "An image."


IGNORE_INDEX = -100


def build_mmu_batch(
    env: ShowoEnv, image_path: str, question: str, *, caption_seed: int
) -> dict[str, Any]:
    """Build one MMU training-loss batch (``forward_und_only`` kwargs),
    mirroring ``datasets/mmu_dataset.py``'s
    ``MMUDataset.format_multi_sequence_und_qwen2_5`` sequence-construction
    convention (data plumbing, not a loss formula) for a single
    system-prompt + one-turn conversation, with a fixed pseudo-target
    caption generated once via the official, T210-tested
    ``model.mmu_generate`` call path (see
    :func:`_generate_mmu_pseudo_caption`).
    """

    hp = env.hp
    dev = env.device
    tok = env.text_tokenizer

    image_latents = _load_and_encode_image(env, image_path)
    caption = _generate_mmu_pseudo_caption(env, image_latents, question, caption_seed)

    sys_ids = tok("system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False)[
        "input_ids"
    ]
    role_a = tok("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_b = tok("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    q_ids = tok(question, add_special_tokens=False, max_length=512, truncation=True)["input_ids"]
    target_ids = tok(
        caption + tok.eos_token, add_special_tokens=False, max_length=512, truncation=True
    )["input_ids"]

    num_img = hp["num_mmu_image_tokens"]
    img_span = [hp["boi_id"]] + [hp["img_pad_id"]] * num_img + [hp["eoi_id"]]
    source_ids = sys_ids + role_a + img_span + q_ids + [hp["eos_id"]] + role_b

    text_tokens = [hp["bos_id"]] + source_ids + target_ids
    text_labels = [IGNORE_INDEX] * (1 + len(source_ids)) + list(target_ids)

    max_seq_len = int(hp["max_seq_len"])
    text_labels = text_labels + [IGNORE_INDEX] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [hp["pad_id"]] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens[:max_seq_len], device=dev)[None, :]
    text_labels = torch.tensor(text_labels[:max_seq_len], device=dev)[None, :]

    # offset = index of the first img_pad_id token (right after boi_id);
    # length = num_img (the time-embed slot at `offset` plus (num_img - 1)
    # real image-patch slots at offset+1 .. offset+num_img-1), matching
    # forward_und_only's internal scatter loop exactly (see module docstring).
    offset = 1 + len(sys_ids) + len(role_a) + 1
    modality_positions = torch.tensor([[offset, num_img]], device=dev)[None, :, :]

    from models import omni_attn_mask_naive  # type: ignore

    attention_mask = omni_attn_mask_naive(
        B=1, LEN=max_seq_len, modalities=modality_positions, device=dev, inverted=True
    ).to(env.weight_type)

    t = torch.tensor([1.0], device=dev, dtype=env.weight_type)

    return dict(
        text_tokens=text_tokens,
        image_latents=image_latents,
        t=t,
        attention_mask=attention_mask,
        text_labels=text_labels,
        modality_positions=modality_positions,
        max_seq_len=max_seq_len,
        device=dev,
    )


def make_mmu_loss_fn(env: ShowoEnv, shared_block: LeafBlock, private_block: LeafBlock) -> LossFn:
    """MMU ``LossFn(theta_s, theta_p, batch) -> loss_ntp`` via the official
    ``model.forward_und_only(...)`` (never touches ``diffusion_head_a``).
    ``batch`` is a pre-built kwargs dict from :func:`build_mmu_batch`.
    """

    model = env.model

    def loss_fn(theta_s: torch.Tensor, theta_p: torch.Tensor, batch: object) -> torch.Tensor:
        overrides: dict[str, torch.Tensor] = {}
        overrides.update(unflatten_block(theta_s, shared_block))
        overrides.update(unflatten_block(theta_p, private_block))
        assert isinstance(batch, dict)
        _logits, loss_ntp = call_with_overrides(model, overrides, "forward_und_only", batch)
        return loss_ntp

    return loss_fn


# ---------------------------------------------------------------------------
# T2I batch construction + loss closure.
# ---------------------------------------------------------------------------


def build_t2i_batch(
    env: ShowoEnv, prompt: str, pseudo_target_image_path: str, *, seed: int
) -> dict[str, Any]:
    """Build one T2I training-loss batch (``forward`` kwargs with
    ``text_labels=None, image_labels=<ut>``), using the official
    ``datasets/utils.py::format_sequence_gen_qwen2_5`` sequence builder and
    the official ``transport.sample`` + ``transport.path_sampler.plan``
    flow-matching batch construction (mirroring
    ``train_stage_two.py``'s ``prepare_latents_and_labels`` closure) --
    no loss formula is reimplemented; ``velocity_prediction`` inside the
    official ``model.forward(...)`` computes ``loss_flow`` from the
    ``image_latents``/``t``/``image_labels`` this function builds.

    No reference target image is available in the asset manifest for the
    T2I prompts (first-report.md section 3 only lists prompts). Per this
    module's docstring, ``pseudo_target_image_path`` supplies a real image
    (VAE-encoded, then transport-sampled) purely to exercise the flow-
    matching loss's numerics -- NOT a claim of prompt-image correspondence.
    """

    from transport import create_transport  # type: ignore
    from datasets.utils import format_sequence_gen_qwen2_5  # type: ignore

    hp = env.hp
    dev = env.device
    tok = env.text_tokenizer

    x1 = _load_and_encode_image(env, pseudo_target_image_path)  # (1, C, H, W)

    transport = create_transport(
        path_type=env.config.transport.path_type,
        prediction=env.config.transport.prediction,
        loss_weight=env.config.transport.loss_weight,
        train_eps=env.config.transport.train_eps,
        sample_eps=env.config.transport.sample_eps,
        snr_type=env.config.transport.snr_type,
        do_shift=env.config.transport.do_shift,
        seq_len=hp["num_t2i_image_tokens"],
    )

    torch.manual_seed(seed)
    with torch.no_grad():
        t, x0, x1_ = transport.sample(x1, None)
        t, xt, ut = transport.path_sampler.plan(t, x0, x1_)

    text_tokens_list = tok(prompt, add_special_tokens=False)["input_ids"][: hp["max_text_len"]]
    text_tokens, text_labels, modality_positions, text_mask, image_mask = (
        format_sequence_gen_qwen2_5(
            text_tokens_list,
            None,
            hp["bos_id"],
            hp["eos_id"],
            hp["boi_id"],
            hp["eoi_id"],
            hp["pad_id"],
            hp["img_pad_id"],
            hp["num_t2i_image_tokens"],
            hp["max_seq_len"],
            0,
        )
    )

    text_tokens = text_tokens[None, :].to(dev)
    modality_positions = modality_positions[None, :, :].to(dev)
    image_mask = image_mask[None, :].to(dev)

    from models import omni_attn_mask_naive  # type: ignore

    attention_mask = omni_attn_mask_naive(
        B=1, LEN=hp["max_seq_len"], modalities=modality_positions, device=dev, inverted=True
    ).to(env.weight_type)

    return dict(
        text_tokens=text_tokens,
        image_latents=xt.to(env.weight_type),
        t=t.to(env.weight_type),
        attention_mask=attention_mask,
        image_masks=image_mask,
        text_labels=None,
        image_labels=ut.to(env.weight_type),
        modality_positions=modality_positions,
        max_seq_len=hp["max_seq_len"],
        device=dev,
    )


def make_t2i_loss_fn(env: ShowoEnv, shared_block: LeafBlock, private_block: LeafBlock) -> LossFn:
    """T2I ``LossFn(theta_s, theta_p, batch) -> loss_flow`` via the official
    ``model.forward(...)`` (``text_labels=None`` -> the
    ``elif image_labels is not None`` branch, returning ``(logits,
    loss_flow)``). ``batch`` is a pre-built kwargs dict from
    :func:`build_t2i_batch`. Note ``und_trans.layers[0]`` (the MMU private
    block) is NOT overridden here -- it remains frozen at its loaded value,
    per the task's "additionally freeze und_trans.layers[0] ... since it
    participates in T2I's forward graph but isn't T2I's declared private
    tensor" requirement (freezing is the default state set once by
    :func:`freeze_all_except_subspace`; this closure simply never includes
    it in ``overrides``).
    """

    model = env.model

    def loss_fn(theta_s: torch.Tensor, theta_p: torch.Tensor, batch: object) -> torch.Tensor:
        overrides: dict[str, torch.Tensor] = {}
        overrides.update(unflatten_block(theta_s, shared_block))
        overrides.update(unflatten_block(theta_p, private_block))
        assert isinstance(batch, dict)
        _logits, loss_flow = call_with_overrides(model, overrides, "forward", batch)
        return loss_flow

    return loss_fn


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Deep-ish copy of a batch dict for the same-batch diagnostic variant
    (never substituted for the disjoint-batch result -- see
    ``run_feasibility.py``): tensors are cloned (not shared storage) so any
    accidental in-place mutation in one protocol call cannot leak into
    another; non-tensor leaves are ``copy.deepcopy``-ed.
    """

    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = copy.deepcopy(v)
    return out
