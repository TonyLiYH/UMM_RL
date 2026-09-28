"""T710 bounded GPU optimizer smoke for the CoRL / Janus-Pro-1B Unified-GRPO stack.

This is the heavy driver. It is intentionally NOT imported by anything else in
this package (``__init__.py`` explicitly keeps torch/trl/corl/janus imports
out of module scope) so the rest of ``comppareto.adapters.corl`` stays
importable on a CPU-only dev box. Every third-party import used here is
inside function bodies.

Design (see configs/corl/admission/discrepancy-lock.yaml D9-D14 for the
paper-vs-code audit this driver's instrumentation is built to exercise):

- Uses the *unmodified* upstream ``JanusProUnifiedGRPOTrainer`` and the real
  upstream default reward set (``t2i_bid_cycle_reward t2i_ti_sim qa_accuracy
  format`` -> ``[T2ICycleConsistencyReward, t2i_match_reward,
  common_qa_accuracy_reward, format_reward]``), exactly matching
  ``corl/scripts/corl_unified.sh``. No CoRL source file is modified.
- Two variants, per D11 and mandatory check #8:
    * ``upstream_exact``: ``model_ckpt_dir`` left at the literal upstream
      default placeholder ``"XXX/checkpoint/"``. Expected to raise inside
      ``T2ICycleConsistencyReward.load_external_model`` at trainer
      construction time (the actual D11 defect reproduced live, not merely
      read from source) -- this is a valid, recorded failure, not a bug in
      this driver.
    * ``corrected_candidate``: ``model_ckpt_dir`` pointed at the real local
      ``all-mpnet-base-v2`` download (an argument-level CLI switch, not a
      source patch). Expected to construct and train successfully; this is
      the variant whose numbers populate ``runs/corl-admission-v1/metrics.json``'s
      ``smoke.*`` gate fields.
- num_generations reduced from the official default of 8 to 2 (still a valid
  GRPO group of >=2 for variance/advantage computation) and
  gradient_accumulation_steps reduced from 4 to 1, to fit the task's <=8
  optimizer steps / <=32 unique records / <=16 GPU-hour envelope while still
  exercising the identical code path end to end (generation, both reward
  branches, joint T2I+MM2T loss, backward, optimizer step).
- Instrumentation for the 8 mandatory checks is implemented as targeted,
  version-agnostic monkeypatches of bound methods on the *real* trainer
  instance (``training_step`` for check #2, ``_get_per_token_logps`` for
  check #3) rather than a second, separately-reimplemented pipeline, so the
  numbers recorded are measured from the exact code path that actually ran,
  not from a hand-rolled approximation of it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from comppareto.adapters.corl import param_policy, paths
from comppareto.adapters.corl.micro_split import load_records, validate_records


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_dataset(records: list[dict[str, Any]]):
    """Mirror grpo_janus_unify.py's make_conversation_joint, plus PIL image load
    from the micro-split's image_path (the dataset's real image field is named
    ``real_image``, per source-lock.yaml's materialization note)."""
    from datasets import Dataset
    from PIL import Image

    rows = []
    for rec in records:
        prompt_text = rec["prompt"].strip()
        qa_problem = rec["qa_problem"]
        rows.append(
            {
                "prompt": [
                    {"role": "<|User|>", "content": prompt_text},
                    {"role": "<|Assistant|>", "content": ""},
                ],
                "qa_prompt": [
                    {"role": "<|User|>", "content": f"<image_placeholder>\n{qa_problem}"},
                    {"role": "<|Assistant|>", "content": ""},
                ],
                "qa_problem": qa_problem,
                "qa_solution": rec["qa_solution"],
                "qa_type": rec["qa_type"],
                "real_image": Image.open(rec["image_path"]).convert("RGB"),
            }
        )
    return Dataset.from_list(rows)


def _param_snapshot(model) -> dict[str, float]:
    """name -> float32 L2 norm, for cheap before/after change detection."""
    snapshot = {}
    for name, p in model.named_parameters():
        snapshot[name] = float(p.detach().float().norm().cpu())
    return snapshot


def _diff_changed_names(before: dict[str, float], after: dict[str, float]) -> list[str]:
    changed = []
    for name, before_val in before.items():
        after_val = after.get(name)
        if after_val is None:
            continue
        if before_val != after_val:
            changed.append(name)
    return changed


def _build_task_args(model_ckpt_dir: str, dataset_cache_dir: str):
    from corl.open_r1.grpo_janus_unify import GRPOScriptArguments

    return GRPOScriptArguments(
        dataset_name="mm-vl/x2x_rft_22k",  # unused: we pass an in-memory Dataset directly
        reward_funcs=["t2i_bid_cycle_reward", "t2i_ti_sim", "qa_accuracy", "format"],
        task_format="unify",
        mm2t_format="qa",
        unify_advantage=False,
        unify_reward=True,
        model_ckpt_dir=model_ckpt_dir,
        dataset_cache_dir=dataset_cache_dir,
    )


def _reward_funcs():
    from corl.open_r1.rewards import reward_funcs_registry

    keys = ["t2i_bid_cycle_reward", "t2i_ti_sim", "qa_accuracy", "format"]
    return [reward_funcs_registry[k] for k in keys]


def run_variant(
    variant: str,
    *,
    janus_dir: str,
    model_ckpt_dir: str,
    dataset_cache_dir: str,
    dataset,
    max_steps: int,
    num_generations: int,
    output_dir: str,
) -> dict[str, Any]:
    import torch
    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.utils import selective_log_softmax

    from corl.open_r1.trainer.grpo_trainer_unified import JanusProUnifiedGRPOTrainer

    result: dict[str, Any] = {"variant": variant, "started_at": _now_iso()}

    task_args = _build_task_args(model_ckpt_dir, dataset_cache_dir)
    reward_funcs = _reward_funcs()

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        beta=0.0,
        num_generations=num_generations,
        # trl's GRPOConfig requires effective_batch_size (per_device * grad_accum *
        # num_processes) to be evenly divisible by num_generations; with a single
        # GPU and no accumulation the simplest valid choice is
        # per_device_train_batch_size == num_generations (each step samples
        # `num_generations` prompts, each producing `num_generations`
        # completions).
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        max_prompt_length=1024,
        max_completion_length=576,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        seed=0,
    )

    construction_t0 = time.time()
    try:
        trainer = JanusProUnifiedGRPOTrainer(
            model=janus_dir,
            reward_funcs=reward_funcs,
            args=grpo_config,
            train_dataset=dataset,
            task_args=task_args,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: this IS the D11 probe
        result["construction_exit"] = "raised"
        result["construction_error_type"] = type(exc).__name__
        result["construction_error_message"] = str(exc)
        result["construction_traceback"] = traceback.format_exc()
        result["construction_elapsed_seconds"] = time.time() - construction_t0
        result["finished_at"] = _now_iso()
        return result

    result["construction_exit"] = "ok"
    result["construction_elapsed_seconds"] = time.time() - construction_t0

    model = trainer.model

    # ---- Check #1: enumerate requires_grad / optimizer membership --------
    named_params = [(n, p.requires_grad, p.numel()) for n, p in model.named_parameters()]
    authorization = param_policy.summarize_authorization(named_params)
    result["check1_parameter_authorization"] = authorization.to_dict()

    before_snapshot = _param_snapshot(model)

    # ---- Check #2 instrumentation: frozen-head gradient-flow probe -------
    grad_flow_probe: dict[str, Any] = {}
    orig_training_step = trainer.training_step

    def patched_training_step(*args, **kwargs):
        out = orig_training_step(*args, **kwargs)
        if not grad_flow_probe:
            lm_grad_norms = [
                float(p.grad.detach().float().norm().cpu())
                for n, p in model.named_parameters()
                if n.startswith("language_model.") and p.grad is not None
            ]
            gen_head_any_grad = any(p.grad is not None for p in model.gen_head.parameters())
            gen_embed_any_grad = any(p.grad is not None for p in model.gen_embed.parameters())
            vision_model_any_grad = any(
                p.grad is not None for p in model.vision_model.parameters()
            )
            grad_flow_probe["num_language_model_params_with_grad"] = len(lm_grad_norms)
            grad_flow_probe["language_model_grad_norm_mean"] = (
                statistics.mean(lm_grad_norms) if lm_grad_norms else None
            )
            grad_flow_probe["language_model_grad_norm_max"] = (
                max(lm_grad_norms) if lm_grad_norms else None
            )
            grad_flow_probe["language_model_any_nonzero_grad"] = (
                any(v > 0.0 for v in lm_grad_norms) if lm_grad_norms else False
            )
            # Frozen leaves never populate .grad themselves (requires_grad=False),
            # even though gen_head sits between language_model and the T2I loss
            # in the forward graph -- gradient still flows THROUGH it back to
            # language_model, it is just not accumulated ON it.
            grad_flow_probe["gen_head_leaf_grad_is_none"] = not gen_head_any_grad
            grad_flow_probe["gen_embed_leaf_grad_is_none"] = not gen_embed_any_grad
            grad_flow_probe["vision_model_leaf_grad_is_none"] = not vision_model_any_grad
        return out

    trainer.training_step = patched_training_step

    # ---- Check #3 instrumentation: image-token next-token alignment probe
    alignment_probe: dict[str, Any] = {}
    orig_get_per_token_logps = trainer._get_per_token_logps

    def patched_get_per_token_logps(model_arg, **kwargs):
        out = orig_get_per_token_logps(model_arg, **kwargs)
        if not alignment_probe and kwargs.get("t2i_discrete_img_ids") is not None:
            try:
                with torch.no_grad():
                    t2i_discrete_img_ids = kwargs["t2i_discrete_img_ids"].to(
                        kwargs["t2i_inputs_ids"].dtype
                    )
                    _, t2i_logits = model_arg(
                        mm2t_input_ids=kwargs["mm2t_input_ids"],
                        mm2t_images_seq_mask=kwargs["mm2t_images_seq_mask"],
                        mm2t_pixel_values=kwargs["mm2t_pixel_values"],
                        mm2t_images_emb_mask=kwargs["mm2t_images_emb_mask"],
                        mm2t_attention_mask=kwargs["mm2t_attention_mask"],
                        mm2t_logits_to_keep=kwargs["mm2t_logits_to_keep"] + 1,
                        t2i_input_ids=kwargs["t2i_inputs_ids"],
                        t2i_attention_mask=kwargs["t2i_attention_mask"],
                        t2i_discrete_img_ids=t2i_discrete_img_ids,
                        t2i_logits_to_keep=kwargs["t2i_logits_to_keep"],
                        task="unify",
                    )
                    t2i_logits = t2i_logits / trainer.temperature
                    correct_logp = selective_log_softmax(
                        t2i_logits, t2i_discrete_img_ids
                    ).mean().item()
                    shifted_ids = torch.roll(t2i_discrete_img_ids, shifts=1, dims=1)
                    shifted_logp = selective_log_softmax(t2i_logits, shifted_ids).mean().item()
                    alignment_probe["t2i_mean_logp_correct_alignment"] = correct_logp
                    alignment_probe["t2i_mean_logp_shifted_by_one_misalignment"] = shifted_logp
                    alignment_probe["alignment_confirmed"] = correct_logp > shifted_logp
            except Exception as exc:  # noqa: BLE001
                alignment_probe["error"] = repr(exc)
                alignment_probe["error_traceback"] = traceback.format_exc()
        return out

    trainer._get_per_token_logps = patched_get_per_token_logps

    # ---- Check #6 instrumentation: capture per-step metrics before the
    # trainer's own log() call clears its internal self._metrics accumulator.
    metrics_by_step: list[dict[str, float]] = []
    orig_log = trainer.log

    def patched_log(logs, *log_args, **log_kwargs):
        # trainer's own log() merges self._metrics into the logged dict
        # internally and clears self._metrics afterward -- so the metrics
        # must be read here, from the live accumulator, before orig_log runs
        # and wipes it, not from the small `logs` arg (loss/grad_norm/lr
        # only) passed in by the outer training loop.
        snapshot: dict[str, float] = {
            k: v for k, v in logs.items() if isinstance(v, (int, float))
        }
        raw = getattr(trainer, "_metrics", None)
        if isinstance(raw, dict):
            for key, values in raw.items():
                if isinstance(values, dict):
                    continue  # mode-keyed (e.g. {"train": {...}}) -- flatten below
                if isinstance(values, (list, tuple)) and values:
                    finite = [v for v in values if isinstance(v, (int, float)) and v == v]
                    if finite:
                        snapshot[key] = statistics.mean(finite)
                elif isinstance(values, (int, float)):
                    snapshot[key] = values
            for mode_key, mode_val in raw.items():
                if isinstance(mode_val, dict):
                    for key, values in mode_val.items():
                        if isinstance(values, (list, tuple)) and values:
                            finite = [
                                v for v in values if isinstance(v, (int, float)) and v == v
                            ]
                            if finite:
                                snapshot[key] = statistics.mean(finite)
        metrics_by_step.append(snapshot)
        return orig_log(logs, *log_args, **log_kwargs)

    trainer.log = patched_log

    # ---- Run the bounded training loop ------------------------------------
    train_t0 = time.time()
    train_exit = "ok"
    train_error = None
    try:
        trainer.train()
    except Exception as exc:  # noqa: BLE001
        train_exit = "raised"
        train_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    train_elapsed = time.time() - train_t0

    result["train_exit"] = train_exit
    result["train_error"] = train_error
    result["train_elapsed_seconds"] = train_elapsed
    result["global_step"] = int(trainer.state.global_step)
    result["check2_grad_flow_probe"] = grad_flow_probe
    result["check3_alignment_probe"] = alignment_probe

    # ---- Check #6: reward ranges / zero-variance groups / effective tokens
    result["check6_raw_metrics_by_step"] = metrics_by_step
    reward_series: dict[str, list[float]] = {}
    for step_dict in metrics_by_step:
        for key, val in step_dict.items():
            if not key.startswith("rewards/") and key not in (
                "reward_unified",
                "reward_t2i",
                "reward_mm2t",
            ):
                continue
            if val == val:  # drop NaN
                reward_series.setdefault(key, []).append(val)
    reward_summary = {}
    for key, finite_vals in reward_series.items():
        reward_summary[key] = {
            "min": min(finite_vals),
            "max": max(finite_vals),
            "mean": statistics.mean(finite_vals),
            "n": len(finite_vals),
            "zero_variance": len(set(finite_vals)) <= 1,
        }
    result["check6_reward_range_summary"] = reward_summary
    result["check6_clipping_note"] = (
        "compute_loss's per-token ratio is (logps - old_logps).exp() with no "
        "torch.clamp and epsilon_low/epsilon_high are stored but never read in "
        "compute_loss (confirmed by full source re-read); clipping structurally "
        "cannot occur in this trainer regardless of ratio magnitude."
    )

    # ---- Check #1/#8 authorization diff after training ---------------------
    if train_exit == "ok":
        after_snapshot = _param_snapshot(model)
        changed_names = _diff_changed_names(before_snapshot, after_snapshot)
        n_unauthorized, unauthorized_names = param_policy.count_unauthorized_changes(
            changed_names
        )
        result["changed_param_count"] = len(changed_names)
        result["unauthorized_changed_param_count"] = n_unauthorized
        result["unauthorized_changed_param_names"] = unauthorized_names

        # ---- Checkpoint save/reload ------------------------------------
        ckpt_dir = str(Path(output_dir) / "smoke-checkpoint")
        try:
            trainer.save_model(ckpt_dir)
            from transformers import AutoModelForCausalLM

            reloaded = AutoModelForCausalLM.from_pretrained(
                ckpt_dir, trust_remote_code=True, torch_dtype=torch.bfloat16
            ).to(model.device)
            reloaded.eval()
            example = dataset[0]
            fwd_ok = True
            fwd_err = None
            try:
                from PIL import Image as _Image  # noqa: F401

                probe_inputs = trainer.processing_class(
                    conversations=[example["qa_prompt"]],
                    images=[[example["real_image"]]],
                    force_batchify=True,
                ).to(model.device)
                with torch.no_grad():
                    embeds = reloaded.prepare_inputs_embeds(**probe_inputs)
                    out = reloaded.language_model.model(
                        inputs_embeds=embeds, attention_mask=probe_inputs.attention_mask
                    )
                    logits_finite = bool(torch.isfinite(out.last_hidden_state).all().item())
            except Exception as exc:  # noqa: BLE001
                fwd_ok = False
                fwd_err = repr(exc)
                logits_finite = False
            result["checkpoint_reload_pass"] = fwd_ok and logits_finite
            result["checkpoint_reload_error"] = fwd_err
            del reloaded
        except Exception as exc:  # noqa: BLE001
            result["checkpoint_reload_pass"] = False
            result["checkpoint_reload_error"] = repr(exc)
    else:
        result["changed_param_count"] = None
        result["unauthorized_changed_param_count"] = None
        result["checkpoint_reload_pass"] = False
        result["checkpoint_reload_error"] = "training did not complete"

    result["finished_at"] = _now_iso()
    return result


def run_reference_immutability_probe(
    *,
    janus_dir: str,
    model_ckpt_dir: str,
    dataset_cache_dir: str,
    dataset,
    output_dir: str,
) -> dict[str, Any]:
    """Check #7, isolated: beta>0 so a ref_model is actually constructed;
    one optimizer step; checksum ref_model params before/after."""
    import torch
    from trl.trainer.grpo_config import GRPOConfig

    from corl.open_r1.trainer.grpo_trainer_unified import JanusProUnifiedGRPOTrainer

    result: dict[str, Any] = {"started_at": _now_iso()}
    task_args = _build_task_args(model_ckpt_dir, dataset_cache_dir)
    reward_funcs = _reward_funcs()

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        beta=0.1,
        num_generations=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=1,
        max_prompt_length=1024,
        max_completion_length=576,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        seed=0,
    )

    try:
        trainer = JanusProUnifiedGRPOTrainer(
            model=janus_dir,
            reward_funcs=reward_funcs,
            args=grpo_config,
            train_dataset=dataset,
            task_args=task_args,
        )
    except Exception as exc:  # noqa: BLE001
        result["exit"] = "construction_raised"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        return result

    if trainer.ref_model is None:
        result["exit"] = "no_ref_model_constructed"
        result["ref_model_is_none"] = True
        return result

    before = {n: float(p.detach().float().norm().cpu()) for n, p in trainer.ref_model.named_parameters()}
    try:
        trainer.train()
        train_exit = "ok"
        train_error = None
    except Exception as exc:  # noqa: BLE001
        train_exit = "raised"
        train_error = repr(exc)
    after = {n: float(p.detach().float().norm().cpu()) for n, p in trainer.ref_model.named_parameters()}

    changed = [n for n in before if before[n] != after.get(n)]
    result["exit"] = train_exit
    result["train_error"] = train_error
    result["ref_model_is_none"] = False
    result["ref_model_changed_param_count"] = len(changed)
    result["ref_model_immutable"] = len(changed) == 0
    result["global_step"] = int(trainer.state.global_step)
    result["finished_at"] = _now_iso()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--num-records", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-reference-probe",
        action="store_true",
        help="Skip the isolated beta>0 check #7 probe (still required for the gate normally).",
    )
    args = parser.parse_args(argv)

    if args.max_steps < 1 or args.max_steps > 8:
        raise SystemExit("max_steps must be in [1, 8] per the T710 resource envelope")
    if args.num_records > 32:
        raise SystemExit("num_records must be <= 32 per the T710 resource envelope")

    records = load_records(paths.micro_split_jsonl())
    validate_records(records, max_records=32)
    records = records[: args.num_records]

    dataset = _build_dataset(records)
    janus_dir = str(paths.janus_model_dir())

    overall: dict[str, Any] = {
        "started_at": _now_iso(),
        "num_records_used": len(records),
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
    }

    import torch

    overall["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        overall["gpu_name"] = torch.cuda.get_device_name(0)

    t0 = time.time()

    overall["upstream_exact"] = run_variant(
        "upstream_exact",
        janus_dir=janus_dir,
        model_ckpt_dir="XXX/checkpoint/",
        dataset_cache_dir="XXX/data/cache/",
        dataset=dataset,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        output_dir=str(paths.run_dir("upstream_exact")),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    overall["corrected_candidate"] = run_variant(
        "corrected_candidate",
        janus_dir=janus_dir,
        model_ckpt_dir=str(paths.ASSETS_ROOT),
        dataset_cache_dir=str(paths.dataset_cache_dir()),
        dataset=dataset,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        output_dir=str(paths.run_dir("corrected_candidate")),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_reference_probe:
        overall["reference_immutability_probe"] = run_reference_immutability_probe(
            janus_dir=janus_dir,
            model_ckpt_dir=str(paths.ASSETS_ROOT),
            dataset_cache_dir=str(paths.dataset_cache_dir()),
            dataset=dataset,
            output_dir=str(paths.run_dir("corrected_candidate")) + "-refprobe",
        )

    if torch.cuda.is_available():
        overall["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(0))

    overall["total_wallclock_seconds"] = time.time() - t0
    overall["finished_at"] = _now_iso()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(overall, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: v for k, v in overall.items() if not isinstance(v, dict)}, indent=2))
    print(f"Full evidence written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
