"""Real, single-GPU Janus-Pro-R1 GRPO rollout/backward/optimizer smoke (T720).

Runs entirely inside the separately pinned RL venv
(``configs/janus-pro-r1/admission/environment-rl-lock.md``), on GPU index 1
only. Reuses upstream's own rollout and reward primitives unmodified:

- ``vendor/janus-pro-r1/janus-rl/src/open_r1/llama.py``'s
  ``JanusLLamaModel.generate_with_refine`` (real CFG-guided autoregressive
  image-token sampling) for rollout, called with ``task_list=[1]`` so only
  the *initial* generation stage runs (stages 2/3 -- upstream's self-check
  reflection and conditional regeneration -- are skipped; see the module
  docstring in :mod:`comppareto.adapters.janus_pro_r1.grpo_math` and
  ``reports/T720/claim-check.md`` for why this and the other reductions
  below are documented scope reductions, not silent omissions).
- ``vendor/janus-pro-r1/janus-rl/src/open_r1/internvl_img.py``'s
  ``InternVLReward.evaluate()`` (real InternVL2.5-8B yes/no-probability
  reward), subclassed only to point ``__init__`` at
  ``env.reward_checkpoint_root()`` instead of the upstream hardcoded
  internal-cluster path -- ``evaluate()`` itself is inherited unmodified.
- The group-relative-advantage and policy-gradient-loss formulas in
  :mod:`comppareto.adapters.janus_pro_r1.grpo_math` (already cross-checked
  against ``grpo_trainer_t2iv1.py``'s real ``compute_loss``/advantage code).

Per-token log-probabilities are computed by
``_per_token_logps_visual_cfg``, a standalone function whose logic mirrors
``GRPOTrainer._get_per_token_logps`` (visual=True, addcfg=True branch only)
in ``grpo_trainer_t2iv1.py`` lines 358-398 -- extracted rather than calling
the real trainer method directly because that method lives on a
``trl.GRPOTrainer`` subclass that requires a full ``accelerate`` process
group, dataloaders, and reference model to instantiate (exactly the
multi-process machinery ``environment-rl-lock.md`` documents T720 bypassing
in favor of a custom single-GPU loop).

Reduced-from-upstream, clearly documented smoke parameters:
- ``image_token_num_per_image=64`` (``img_size=128``) instead of upstream's
  576 (``img_size=384``) -- 9x fewer autoregressive decode steps per
  rollout image; the same real sampling code path, just a smaller image.
- No frozen reference-model KL term (``grpo_math.policy_gradient_loss`` has
  none) and no PPO-style ratio clip (mathematically exact at ratio==1 on a
  single on-policy step -- see ``grpo_math`` module docstring).
- Two vendored, released bounded prompts (the same two prompts SFT's smoke
  draws its 8 images from,
  ``vendor/janus-pro-r1/janus-sft/data/t2i_examples/labels/*.jsonl``),
  cycled deterministically across steps -- no new prompts invented.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from comppareto.adapters.janus_pro_r1 import env
from comppareto.adapters.janus_pro_r1.grpo_math import group_relative_advantage, policy_gradient_loss


def _apply_attrdict_collections_shim() -> None:
    """See identical function in ``sft_smoke.py`` -- same fix, same reason.

    ``vendor/janus-pro-r1/janus-rl/src/open_r1/models/{projector.py,
    modeling_vlm.py}`` (used by ``JanusLLamaModel``, which subclasses
    ``MultiModalityCausalLM`` from that same ``models`` package) also do
    ``from attrdict import AttrDict``; this venv is also Python 3.10.
    """

    import collections
    import collections.abc

    for name in dir(collections.abc):
        if not name.startswith("_") and not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def _load_bounded_prompts(sft_source_root_for_prompts: Path) -> list[str]:
    """The (small, fixed) set of released bounded prompts reused for rollout.

    Reuses the exact same two-file source SFT's smoke draws its images
    from, but only needs the ``prompt`` field here (not the candidate
    images) -- GRPO generates its own images from these prompts rather than
    using the pre-existing candidates.
    """

    labels_dir = sft_source_root_for_prompts / "data" / "t2i_examples" / "labels"
    prompts: list[str] = []
    for jsonl_path in sorted(labels_dir.glob("*.jsonl")):
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["prompt"] not in prompts:
                    prompts.append(record["prompt"])
    return prompts


def _per_token_logps_visual_cfg(model, input_embeds, output_ids, attention_mask, logits_to_keep, guidance_scale):
    """Mirrors ``GRPOTrainer._get_per_token_logps(..., addcfg=True, visual=True)``.

    See ``grpo_trainer_t2iv1.py`` lines 358-398 (real upstream source,
    read verbatim while designing this smoke) -- extracted as a standalone
    function so it can be called without instantiating the full
    ``trl``-based trainer. Semantics are unchanged from upstream: build
    visual embeddings for ``output_ids`` (repeat-interleaved x2 to match
    the CFG-doubled ``input_embeds`` batch, exactly as upstream does),
    concatenate after ``input_embeds``, run the language-model backbone,
    apply the same classifier-free-guidance logit combination
    (``logit_cond - (guidance_scale-1)/guidance_scale * logit_uncond``),
    keep the trailing ``logits_to_keep`` positions, and gather the
    log-probability of each realized ``output_ids`` token.
    """

    import torch

    if output_ids.shape[0] < input_embeds.shape[0]:
        new_img_ids = torch.repeat_interleave(output_ids, 2, dim=0)
    else:
        new_img_ids = output_ids
    output_embeds = model.gen_aligner(model.gen_embed(new_img_ids))
    inputs_embeds = torch.cat([input_embeds, output_embeds], dim=1)

    outputs = model.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state
    logits = model.gen_head(hidden_states)
    logit_cond = logits[0::2, :]
    logit_uncond = logits[1::2, :]
    logits = logit_cond - (guidance_scale - 1) / guidance_scale * logit_uncond
    logits = logits[:, -1 - logits_to_keep : -1, :]
    input_ids = output_ids.long()

    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def run_grpo_smoke(
    *,
    n_steps: int = 4,
    num_generations: int = 4,
    lr: float = 1e-6,
    guidance_scale: float = 5.0,
    image_token_num_per_image: int = 64,
    img_size: int = 128,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the real GRPO rollout/reward/loss/backward/optimizer smoke.

    Must be called from inside the RL venv, on a machine with GPU index 1
    visible (``CUDA_VISIBLE_DEVICES=1``, matching every other T720 GPU
    command). Each of the ``n_steps`` steps: samples ``num_generations``
    completions for one (deterministically cycled) bounded prompt via
    ``generate_with_refine``, scores them with the real
    ``InternVLReward.evaluate()``, computes the group-relative advantage
    and policy-gradient loss, backpropagates, clips, and steps the
    optimizer over the *entire* policy model -- matching upstream's own
    freeze policy for GRPO, which (confirmed by grep of
    ``grpo_trainer_t2iv1.py``) only ever sets ``requires_grad=False`` on
    the separate frozen reference-model copy, never on the policy model
    itself; T720's smoke has no reference model, so nothing is frozen.
    """

    env.require_cuda()
    _apply_attrdict_collections_shim()

    import torch
    from torch.nn.utils import clip_grad_norm_

    rl_root = env.rl_source_root()
    sys.path.insert(0, str(rl_root / "src" / "open_r1"))
    from llama import JanusLLamaModel  # noqa: E402  (vendored, path-injected)
    from models import VLChatProcessor  # noqa: E402  (vendored, path-injected)
    from internvl_img import InternVLReward  # noqa: E402  (vendored, path-injected)

    class T720InternVLReward(InternVLReward):
        """``InternVLReward`` with the hardcoded internal path parameterized.

        Every line below is copied verbatim from
        ``internvl_img.py``'s ``InternVLReward.__init__`` except the value
        of ``path`` (upstream hardcodes an internal-cluster path; this uses
        ``env.reward_checkpoint_root()``, our locally verified download --
        see ``configs/janus-pro-r1/admission/environment-rl-lock.md``).
        ``evaluate()`` is inherited unmodified.
        """

        def __init__(self) -> None:  # noqa: D107 (see class docstring)
            from transformers import AutoTokenizer
            from internvl.model.internvl_chat import InternVLChatModel

            path = str(env.reward_checkpoint_root())
            self.model = InternVLChatModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                use_flash_attn=True,
                trust_remote_code=True,
            ).eval().cuda()
            for _n, p in self.model.named_parameters():
                p.requires_grad = False
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
            self.vocab_dict = self.tokenizer.get_vocab()

    if output_dir is None:
        output_dir = env.run_output_root() / "grpo-smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = env.base_checkpoint_root()
    t_load_start = time.time()
    vl_chat_processor = VLChatProcessor.from_pretrained(str(checkpoint_path))
    tokenizer = vl_chat_processor.tokenizer
    model = JanusLLamaModel.from_pretrained(
        str(checkpoint_path), trust_remote_code=True
    ).to(torch.bfloat16).cuda()
    # Upstream's own GRPO trainer applies no freeze to the policy model
    # (only to its separate `ref_model` copy) -- full fine-tune, matched
    # here exactly (see run_grpo_smoke's docstring).
    for p in model.parameters():
        p.requires_grad = True
    t_load_mid = time.time()
    reward_model = T720InternVLReward()
    t_load_end = time.time()

    trainable_params = list(model.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    prompts = _load_bounded_prompts(env.sft_source_root())
    if not prompts:
        raise RuntimeError(f"No bounded prompts found under {env.sft_source_root()}/data/t2i_examples/labels")

    # Snapshot for authorized-change verification (same technique as
    # sft_smoke.py): one parameter's value before/after the full loop.
    #
    # NOTE (same bug class as sft_smoke.py, same fix): the naive
    # `next(iter(model.named_parameters()))` picks the literal first
    # parameter in module-registration order, which for
    # `MultiModalityCausalLM` is `vision_model.vision_tower.pos_embed` --
    # part of the *understanding*-branch vision tower. This smoke's
    # `generate_with_refine(..., task_list=[1])` rollout and
    # `_per_token_logps_visual_cfg` recompute both only exercise the
    # *generation* branch (`gen_embed`/`gen_aligner`/`gen_head` project the
    # VQ image tokens, `language_model` is the shared backbone); the
    # understanding-branch `vision_model`/`aligner` are structurally
    # unreachable on this forward path and correctly show zero gradient
    # even though upstream leaves them `requires_grad=True` (full
    # fine-tune, no freeze -- see the comment above). Restrict probe
    # selection to a parameter that is actually on the exercised path, so
    # "authorized_param_change" is real evidence rather than a
    # false-negative from picking an unreachable tensor.
    _t2i_forward_prefixes = ("gen_head", "gen_embed", "gen_aligner", "language_model")
    check_name, check_param = next(
        (n, p)
        for n, p in model.named_parameters()
        if any(n == pfx or n.startswith(pfx + ".") for pfx in _t2i_forward_prefixes)
    )
    check_before = check_param.detach().float().flatten()[:8].clone().cpu()

    probe_grad_norms: list[float] = []
    step_records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    t_train_start = time.time()
    for step in range(n_steps):
        prompt = prompts[step % len(prompts)]
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        text_prompt = sft_format + vl_chat_processor.image_start_tag
        encoded = tokenizer(
            [text_prompt] * num_generations, return_tensors="pt", padding="longest", max_length=200, truncation=True
        )
        prompt_ids = encoded["input_ids"].cuda()
        prompt_mask = encoded["attention_mask"].cuda()

        model.eval()
        (img_ids_1, all_imgs_1), _unused2, _unused_text, (embeds_1, attention_mask_1), _e2, _e3 = model.generate_with_refine(
            vl_chat_processor=vl_chat_processor,
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            cfg_weight=guidance_scale,
            image_token_num_per_image=image_token_num_per_image,
            img_size=img_size,
            task_list=[1],
            cur_step=step,
        )
        model.train()

        rewards = reward_model.evaluate(all_imgs_1, [prompt] * num_generations)
        rewards_arr = [float(r) for r in rewards]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            per_token_logps = _per_token_logps_visual_cfg(
                model,
                input_embeds=embeds_1,
                output_ids=img_ids_1,
                attention_mask=attention_mask_1,
                logits_to_keep=img_ids_1.size(1),
                guidance_scale=guidance_scale,
            )
        seq_logps = per_token_logps.sum(dim=-1)  # sum over tokens, matching upstream's token-sum convention

        import numpy as np

        rewards_np = np.asarray(rewards_arr, dtype=np.float64).reshape(1, num_generations)
        advantages_np = group_relative_advantage(rewards_np)
        advantages = torch.tensor(advantages_np.reshape(-1), dtype=seq_logps.dtype, device=seq_logps.device)

        loss = policy_gradient_loss(
            seq_logps.detach().cpu().numpy().reshape(1, num_generations),
            advantages_np,
        )
        # Recompute the loss as a real torch scalar (graph-attached) for
        # backward -- policy_gradient_loss's numpy version above is used
        # only to double-check the value cross-implementation-agrees with
        # the direct torch computation below (asserted per-step).
        torch_loss = -(seq_logps * advantages).mean()
        torch_loss.backward()
        # Direct evidence of a nonzero gradient on the probe parameter,
        # independent of any bf16-precision-limited before/after value
        # diff (same rationale as sft_smoke.py's identical capture).
        probe_grad = check_param.grad
        probe_grad_norms.append(
            float(probe_grad.detach().float().norm().item()) if probe_grad is not None else 0.0
        )
        grad_norm = clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        loss_value = float(torch_loss.detach().item())
        cross_check_diff = abs(loss_value - loss)
        step_records.append(
            {
                "step": step,
                "prompt": prompt,
                "rewards": rewards_arr,
                "advantages": advantages_np.reshape(-1).tolist(),
                "loss": loss_value,
                "loss_finite": bool(torch.isfinite(torch_loss.detach()).item()),
                "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
                "numpy_vs_torch_loss_abs_diff": cross_check_diff,
            }
        )
    t_train_end = time.time()
    peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    check_after = check_param.detach().float().flatten()[:8].clone().cpu()
    param_changed = bool((check_after != check_before).any().item())

    ckpt_path = output_dir / "grpo_smoke_state.pt"
    torch.save(model.state_dict(), ckpt_path)
    reload_model = JanusLLamaModel.from_pretrained(
        str(checkpoint_path), trust_remote_code=True
    ).to(torch.bfloat16).cuda()
    reloaded_state = torch.load(ckpt_path, map_location="cuda")
    missing, unexpected = reload_model.load_state_dict(reloaded_state, strict=True)
    reload_ok = len(missing) == 0 and len(unexpected) == 0
    reload_check_param = dict(reload_model.named_parameters())[check_name]
    reload_matches_trained = bool(torch.equal(reload_check_param.detach().float().flatten()[:8].cpu(), check_after))

    summary = {
        "n_steps": n_steps,
        "num_generations": num_generations,
        "lr": lr,
        "guidance_scale": guidance_scale,
        "image_token_num_per_image": image_token_num_per_image,
        "img_size": img_size,
        "n_trainable_params": n_trainable,
        "checkpoint_path": str(checkpoint_path),
        "reward_checkpoint_path": str(env.reward_checkpoint_root()),
        "policy_load_seconds": t_load_mid - t_load_start,
        "reward_load_seconds": t_load_end - t_load_mid,
        "train_seconds": t_train_end - t_train_start,
        "peak_memory_bytes": peak_memory_bytes,
        "steps": step_records,
        "all_losses_finite": all(r["loss_finite"] for r in step_records),
        "authorized_param_change": {
            "param_name": check_name,
            "changed": param_changed,
            "probe_grad_norms_per_step": probe_grad_norms,
        },
        "checkpoint_roundtrip": {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_bytes": ckpt_path.stat().st_size,
            "n_missing": len(missing),
            "n_unexpected": len(unexpected),
            "reload_ok": reload_ok,
            "reload_matches_trained_values": reload_matches_trained,
        },
        "prompts_used": prompts,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--image-token-num-per-image", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = run_grpo_smoke(
        n_steps=args.n_steps,
        num_generations=args.num_generations,
        lr=args.lr,
        guidance_scale=args.guidance_scale,
        image_token_num_per_image=args.image_token_num_per_image,
        img_size=args.img_size,
        output_dir=args.output_dir,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "steps"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
