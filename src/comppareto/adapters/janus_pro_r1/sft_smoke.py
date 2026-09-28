"""Real, single-GPU Janus-Pro-R1 SFT optimizer smoke (T720).

Runs entirely inside the separately pinned SFT venv
(``configs/janus-pro-r1/admission/environment-sft-lock.md``), on GPU index 1
only. Reuses upstream's own ``train_setup()`` (trainable/frozen split) and
the exact loss/backward/clip/step sequence from
``vendor/janus-pro-r1/janus-sft/trainer/trainer_t2i.py``'s ``run_step()``,
but bypasses ``TextToImageTrainer`` itself (which hard-requires FSDP across
multiple ranks -- see ``environment-sft-lock.md`` deviation #1) in favor of
loading ``MultiModalityCausalLM`` directly on a single device with a plain
``torch.optim.AdamW``.

Data: the 8 vendored, released bounded examples under
``vendor/janus-pro-r1/janus-sft/data/t2i_examples/{labels,images}`` (4 from
``janus_0.7.jsonl`` + 4 from ``flux_0.7.jsonl``, one shared prompt each,
task_type=0 -- pure text-to-image generation loss, matching the frozen
protocol's "use released bounded examples or deterministic subsets";
nothing here is downloaded or invented). Batch size 2, 4 optimizer steps,
covering every one of the 8 images exactly once in a fixed, deterministic
order (no shuffling, no randomness anywhere in this module).

This module is only ever imported/run inside the SFT venv -- it does real
``import torch`` at call time (not at module import time), so importing
*this file itself* stays safe on a torch-free CPU dev machine; only calling
``run_sft_smoke()`` requires torch+CUDA (enforced via ``env.require_cuda()``
before anything else happens).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from comppareto.adapters.janus_pro_r1 import env
from comppareto.adapters.janus_pro_r1.param_inventory import classify_parameters


def _apply_attrdict_collections_shim() -> None:
    """Re-alias collections.abc names onto ``collections`` before attrdict.

    ``vendor/janus-pro-r1/janus-sft/models/{projector.py,modeling_vlm.py}``
    do ``from attrdict import AttrDict`` on the real model-construction
    path. ``attrdict==2.0.1`` (the only PyPI release) itself does
    ``from collections import Mapping, MutableMapping, Sequence`` at import
    time; all three names were removed from the top-level ``collections``
    namespace in Python 3.10 (this venv's interpreter), moved to
    ``collections.abc`` only. See
    ``configs/janus-pro-r1/admission/environment-sft-lock.md`` for the full
    justification -- this is a process-level stdlib compatibility shim, not
    an edit to any vendored file.
    """

    import collections
    import collections.abc

    for name in dir(collections.abc):
        if not name.startswith("_") and not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def _load_bounded_examples(sft_source_root: Path) -> list[dict[str, Any]]:
    """The 8 vendored (prompt, image_path) pairs used for this smoke.

    Reads both released ``t2i_examples/labels/*.jsonl`` files (one line
    each); every ``data[*].img_path`` entry becomes one training example,
    all sharing that file's single prompt. Order is the on-disk order of
    ``sorted(glob(...))`` then list order within each file -- fixed and
    reproducible.
    """

    labels_dir = sft_source_root / "data" / "t2i_examples" / "labels"
    examples: list[dict[str, Any]] = []
    for jsonl_path in sorted(labels_dir.glob("*.jsonl")):
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prompt = record["prompt"]
                for candidate in record["data"]:
                    examples.append(
                        {
                            "prompt": prompt,
                            "image_path": sft_source_root / candidate["img_path"],
                            "source_jsonl": jsonl_path.name,
                            "candidate_id": candidate["id"],
                        }
                    )
    return examples


def run_sft_smoke(
    *,
    n_steps: int = 4,
    batch_size: int = 2,
    lr: float = 1e-5,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the real SFT optimizer smoke and return a JSON-serializable summary.

    Must be called from inside the SFT venv, on a machine with GPU index 1
    visible (``CUDA_VISIBLE_DEVICES=1`` set by the caller, matching every
    other T720 GPU command -- see ``reports/T720/first-report.md`` section
    6). Raises ``RuntimeError`` via ``env.require_cuda()`` if CUDA is not
    available, rather than silently running on CPU.
    """

    env.require_cuda()
    _apply_attrdict_collections_shim()

    import torch
    from torch.nn.utils import clip_grad_norm_

    sft_root = env.sft_source_root()
    sys.path.insert(0, str(sft_root))
    from models import MultiModalityCausalLM, VLChatProcessor  # noqa: E402  (vendored, path-injected)
    from trainer.trainer_t2i import train_setup  # noqa: E402  (vendored, path-injected)

    if output_dir is None:
        output_dir = env.run_output_root() / "sft-smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = env.base_checkpoint_root()
    t_load_start = time.time()
    vl_chat_processor = VLChatProcessor.from_pretrained(str(checkpoint_path))
    tokenizer = vl_chat_processor.tokenizer
    model = MultiModalityCausalLM.from_pretrained(
        str(checkpoint_path), trust_remote_code=True
    ).to(torch.bfloat16).cuda()
    train_setup(model)
    t_load_end = time.time()

    inventory = classify_parameters(model.named_parameters())
    if inventory.requires_grad_mismatches:
        raise RuntimeError(
            "train_setup() produced requires_grad values that disagree with "
            f"DEFAULT_TRAINABLE_PREFIXES: {inventory.requires_grad_mismatches}"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    examples = _load_bounded_examples(sft_root)
    n_needed = n_steps * batch_size
    if len(examples) < n_needed:
        raise RuntimeError(
            f"Need {n_needed} bounded examples for {n_steps} steps x "
            f"batch_size {batch_size}, but only found {len(examples)} under "
            f"{sft_root / 'data' / 't2i_examples'}."
        )
    examples = examples[:n_needed]

    from PIL import Image
    from torchvision import transforms

    def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
        # Verbatim from vendor/janus-pro-r1/janus-sft/datasets/t2i_dataset.py
        import numpy as np

        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])

    gen_transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    input_token_max_len = 300  # matches trainer_t2i.py's input_token_max_len[0]
    max_prompt_length = 200  # matches t2i_dataset.py's self.max_prompt_length

    def build_example(ex: dict[str, Any]) -> dict[str, Any]:
        image = Image.open(ex["image_path"]).convert("RGB")
        image_tensor = gen_transform(image)
        conversation = [
            {"role": "<|User|>", "content": ex["prompt"]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag
        input_ids = tokenizer.encode(
            prompt, return_tensors="pt", max_length=max_prompt_length, truncation=True
        ).squeeze(0)
        return {"input_ids": input_ids, "image": image_tensor}

    def build_batch(batch_examples: list[dict[str, Any]]) -> dict[str, Any]:
        built = [build_example(ex) for ex in batch_examples]
        bsz = len(built)
        batched_input_ids = torch.full(
            (bsz, input_token_max_len), vl_chat_processor.pad_id
        ).long()
        batched_attention_mask = torch.zeros((bsz, input_token_max_len)).long()
        for k, item in enumerate(built):
            ids = item["input_ids"]
            seq_len = len(ids)
            batched_attention_mask[k, -seq_len:] = 1
            batched_input_ids[k, -seq_len:] = ids
        image1 = torch.stack([item["image"] for item in built], dim=0)
        return {
            "input_ids": batched_input_ids.cuda(),
            "attention_mask": batched_attention_mask.cuda(),
            "image1": image1.to(torch.bfloat16).cuda(),
            "task_type": 0,
        }

    # Snapshot one trainable tensor's pre-training value to prove an
    # authorized (trainable-prefixed) parameter actually changed, and one
    # frozen tensor's value to prove it did NOT change -- both checked
    # against real numbers below, not asserted blindly.
    #
    # The probe MUST be a parameter genuinely on this batch's forward path.
    # `train_setup()`'s trainable-prefix set includes "aligner" (the
    # understanding-branch projector, see DEFAULT_TRAINABLE_PREFIXES'
    # docstring), but `MultiModalityCausalLM.forward(..., task_type=0)`
    # (`vendor/janus-pro-r1/janus-sft/models/modeling_vlm.py:264-287`) only
    # ever calls `prepare_embedding(image1)` with its default `gen_image=True`,
    # which routes through `gen_vision_model`/`gen_embed`/`gen_aligner`, NOT
    # `vision_model`/`aligner` (those are only reached when `gen_image=False`,
    # i.e. the understanding task_type==1 branch this smoke never runs). An
    # earlier version of this probe picked the first trainable parameter by
    # `named_parameters()` order regardless of prefix and happened to land on
    # `aligner.layers.0.weight`, which is real and correctly trainable but
    # structurally unreachable by this forward pass -- it always shows zero
    # gradient and therefore never visibly changes, which would have looked
    # like a broken optimizer smoke despite training working correctly. Fixed
    # by restricting the probe to prefixes this task_type actually exercises.
    _t2i_forward_prefixes = ("gen_head", "gen_embed", "gen_aligner", "language_model")
    trainable_name, trainable_param = next(
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad
        and any(n == pfx or n.startswith(pfx + ".") for pfx in _t2i_forward_prefixes)
    )
    frozen_name, frozen_param = next(
        (n, p) for n, p in model.named_parameters() if not p.requires_grad
    )
    trainable_before = trainable_param.detach().float().flatten()[:8].clone().cpu()
    frozen_before = frozen_param.detach().float().flatten()[:8].clone().cpu()

    step_records: list[dict[str, Any]] = []
    probe_grad_norms: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    t_train_start = time.time()
    for step in range(n_steps):
        batch = build_batch(examples[step * batch_size : (step + 1) * batch_size])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            loss = model(**batch)
        loss.backward()
        # Direct evidence the probe parameter itself received a nonzero
        # gradient this step -- independent of whether the subsequent
        # optimizer.step() produces a bf16-representable (visibly different)
        # value. AdamW's per-parameter update magnitude is ~lr (1e-5 here)
        # regardless of raw gradient scale, and bf16 has ~7 mantissa bits
        # (~0.4% relative precision), so a handful of 1e-5-scale steps can
        # legitimately round back to the identical bf16 value even though the
        # optimizer genuinely acted on the parameter every step -- this
        # capture makes that distinction checkable rather than assumed.
        probe_grad = trainable_param.grad
        probe_grad_norms.append(
            float(probe_grad.detach().float().norm().item()) if probe_grad is not None else 0.0
        )
        grad_norm = clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        loss_value = float(loss.detach().item())
        step_records.append(
            {
                "step": step,
                "loss": loss_value,
                "loss_finite": bool(torch.isfinite(loss.detach()).item()),
                "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
            }
        )
    t_train_end = time.time()
    peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    trainable_after = trainable_param.detach().float().flatten()[:8].clone().cpu()
    frozen_after = frozen_param.detach().float().flatten()[:8].clone().cpu()
    trainable_changed = bool((trainable_after != trainable_before).any().item())
    frozen_unchanged = bool((frozen_after == frozen_before).all().item())

    # Checkpoint save/reload round-trip: only the trainable-parameter subset
    # is persisted (keeps the smoke's checkpoint small -- megabytes, not the
    # full ~14GB bf16 model -- while still exercising a genuine save/load
    # cycle on the parameters this smoke actually trained).
    trainable_names = set(inventory.trainable.names)
    trainable_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable_names
    }
    ckpt_path = output_dir / "sft_smoke_trainable_state.pt"
    torch.save(trainable_state, ckpt_path)

    reload_model = MultiModalityCausalLM.from_pretrained(
        str(checkpoint_path), trust_remote_code=True
    ).to(torch.bfloat16).cuda()
    train_setup(reload_model)
    reloaded_state = torch.load(ckpt_path, map_location="cuda")
    missing, unexpected = reload_model.load_state_dict(reloaded_state, strict=False)
    missing_outside_trainable = [m for m in missing if m in trainable_names]
    reload_ok = len(missing_outside_trainable) == 0 and len(unexpected) == 0

    reload_check_param = dict(reload_model.named_parameters())[trainable_name]
    reload_matches_trained = bool(
        torch.equal(
            reload_check_param.detach().float().flatten()[:8].cpu(), trainable_after
        )
    )

    summary = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "lr": lr,
        "checkpoint_path": str(checkpoint_path),
        "load_seconds": t_load_end - t_load_start,
        "train_seconds": t_train_end - t_train_start,
        "peak_memory_bytes": peak_memory_bytes,
        "steps": step_records,
        "all_losses_finite": all(r["loss_finite"] for r in step_records),
        "param_inventory": inventory.to_dict(),
        "authorized_param_change": {
            "trainable_param_name": trainable_name,
            "trainable_changed": trainable_changed,
            "trainable_probe_grad_norms_per_step": probe_grad_norms,
            "frozen_param_name": frozen_name,
            "frozen_unchanged": frozen_unchanged,
        },
        "checkpoint_roundtrip": {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_bytes": ckpt_path.stat().st_size,
            "n_missing_outside_trainable": len(missing_outside_trainable),
            "n_unexpected": len(unexpected),
            "reload_ok": reload_ok,
            "reload_matches_trained_values": reload_matches_trained,
        },
        "examples_used": [
            {"source_jsonl": ex["source_jsonl"], "candidate_id": ex["candidate_id"]}
            for ex in examples
        ],
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = run_sft_smoke(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "steps"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
