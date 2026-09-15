# T250 — Training interface audit (static, source-level, no GPU)

Per execution stage 4: official training/resume paths inspected statically
from source. No GPU load/resume smoke was run for any candidate — static
source evidence was judged sufficient for every required dimension, including
resume-restoration granularity (the historically hardest dimension to resolve
without a live run). See `first-report.md` for the rationale for skipping GPU
smokes and `decision-matrix.md` for how each dimension below was weighted.

## Candidate A — SenseNova-U1 (`training/` subtree of `OpenSenseNova/SenseNova-U1`)

**Trainer/optimizer**: custom framework derived from InternEvo
(`sensenovalm/core`), `HybridZeroOptimizer` (ZeRO-1) as the optimizer wrapper;
Intern Sequence Parallel (ISP) for sequence/tensor parallelism; weight-sharding
via `wp_size` (FSDP-like); pipeline parallel disabled by default
(`pp_size=1`) in shipped configs.

**Resume mechanism** — read `sensenovalm/checkpoint/checkpoint_manager.py`
(`try_load_internevo_ckpt`) directly:
- Calls `load_model_checkpoint(...)`, `load_optimizer_checkpoint(...)`,
  `load_scheduler(...)`, and restores `TrainState`/`load_context` (step
  counters, dataloader position for the "streaming-resumable data loading
  with checkpoint state" the training README advertises).
- Optimizer-state restoration is gated by `ckpt.load_optimizer` (a config
  flag defaulting to enabled in shipped configs) and an explicit
  `content=("model","sampler","optimizer")` tuple passed at the call site —
  i.e., **weights + optimizer state + sampler/dataloader position** are all
  restored when resuming, not merely the model weights.
- **Verdict: full-state resume (model, optimizer, scheduler, step/dataloader
  counters).** This is the strongest resume guarantee among the three
  candidates and is confirmed at the code level, not inferred from framework
  reputation alone.

**Configuration/data interfaces**: env-var-driven `torchrun` launchers
(`shell/train_u1/{8B,A3B,U1.5_8B}.sh`); dataset described via a meta JSON
(`mm_data_path`) listing `{root, annotation, repeat_time, task}` per dataset;
five `type_id` task categories (`mm_t2i`, `mm_it2i`, `mm_interleave_gen`,
`mm_interleaved`, `multimodal`) with **per-task loss bucketing already
implemented** or logging — directly supports "sequential-task-batch support
at one frozen shared version" as required by the audit dimensions, since
understanding and generation batches can be mixed within one `mm_data_path`
meta JSON and monitored independently.

**Minimum/recommended GPU memory**: shipped `8B.sh` launcher requires 1 node
× 8 GPUs × 80GB HBM minimum (`wp=8 × tp=1 × pp=1 = 8` ranks at
`seq_len=28672`, `num_imgs=144`); reducible by lowering `seq_len`/`num_imgs`/
`wp_size` but not reducible to fit this audit's 2-GPU envelope without config
changes not yet attempted. This does not block the *audit* (no training is
run here) but is a material planning fact for T270.

**Shared/private/routed parameter classification**: `docs/parameter_breakdown.md`
in the official repo publishes an exact, reproducible split for the MoT
architecture: `shared` group (`language_model.model.embed_tokens.weight` +
`language_model.lm_head.weight`) = 1.245B params (7.09%);
`understanding_transformer` = 8.121B (46.27%); `generation_transformer` =
8.186B (46.64%); total 17.552B params. Pathway totals: understanding path
≈9.366B (53.36%), generation path ≈9.431B (53.73%). This is the most
precisely quantified and independently reproducible parameter-block registry
of the three candidates (published with a documented inspection script).

**Missing dependencies**: `flash-attn` build extra required for shipped
configs (`use_flash_attn=True`); Apex / `grouped_gemm` / `deeplink_ext`
accelerator kernels are configuration-specific and must be built separately.
No missing tokenizer/VAE/reward-model dependency identified for the SFT
checkpoint's own forward/training path (reward models are only needed for
Stage 5 RL, which is explicitly out of scope for this audit and for T270's
starting point).

**License**: Apache-2.0 for the training code and (per the inference-repo
license file referenced from the training README) the released weights;
some files under `sensenovavl/model/sensenovavl_moe_chat/` are MIT-licensed
(derived from InternVL, attribution headers present) — a permissive,
compatible combination.

## Candidate B — Show-o2-1.5B (`showlab/Show-o`, `show-o2/` subdir)

**Trainer/optimizer**: HuggingFace `accelerate`-based training
(`accelerate launch`, DeepSpeed ZeRO-2 config for the downstream
mixed-modality path); standard `AdamW` optimizer instantiated directly in
`train_stage_one.py` (no custom optimizer wrapper).

**Resume mechanism** — read `train_stage_one.py` resume block directly
(lines ~264-320):
- On resume, the script scans the output directory for `checkpoint-<step>`
  subdirectories, picks the latest, and restores **model weights only** via
  `model.load_state_dict(...)` (or the `accelerate`-wrapped equivalent).
- `global_step` and `first_epoch` are **reconstructed from the checkpoint
  directory name** (parsing the trailing integer), not read back from any
  serialized optimizer/scheduler state file.
- A **fresh** `AdamW` optimizer and a **fresh** LR scheduler (cosine/warmup)
  are instantiated after the weight-load — optimizer momentum/variance
  buffers and scheduler warmup/decay position are **not** restored.
- **Verdict: weights-only resume.** This is a materially weaker guarantee
  than SenseNova-U1's (no optimizer/scheduler state carried across a
  resume), and must be stated plainly per the task's honesty mandate — it is
  a genuine limitation of this candidate as a post-training starting point
  if long-horizon resumable training is required, though it does not by
  itself disqualify Show-o2 as a *starting point* for a **fresh** joint
  post-training run (which would initialize a new optimizer/scheduler
  regardless of the source checkpoint's own resume fidelity).

**Configuration/data interfaces**: YAML configs under `configs/`
(`showo2_1.5b_demo_*.yaml` for inference; `train_showo2_1.5b_stage{1,2}.sh`
launchers for training); Stage-1 data is jsonl `{path, prompt}` pairs; Stage-2
data follows LLaVA-OneVision/DenseFusion annotation conventions.
`frozen_params` config key allows freezing named submodules
(`image_embedder_und`, `und_trans`, `showo`, `position_embedding` shown in
the downstream mixed-modality example) — directly usable for shared/private
parameter partitioning in a joint post-training design, though this is a
manual allow-list, not an automatically-derived registry like SenseNova-U1's.

**Minimum/recommended GPU memory**: T210's own measured smoke used a single
H20 with ~13.9-14.9GB allocated for inference; the shipped training launcher
targets one node × 8 GPUs, but (unlike SenseNova-U1) this is a convenience
default for the `accelerate` multi-GPU config, not a hard topology
requirement — the 1.5B model is far smaller and could plausibly run training
on fewer/smaller GPUs, though this has not been verified for training
specifically (only inference was smoke-tested by T210).

**Shared/private/routed parameter classification**: reuse
`configs/admission/showo2/parameter-block-registry.yaml` from T210's
admission wholesale (no need to re-derive).

**Missing dependencies**: T210 already identified and fixed a
`wandb==0.17.0` pin requirement and a `wandb.util.generate_id` removal issue;
also flagged `CompVis/stable-diffusion-safety-checker` as an external
dependency with an open ("More information needed") license status — carried
forward as a pre-existing open item, not new to this audit.

**License**: Apache-2.0 (confirmed by T210's accepted admission).

## Candidate C — UniDDT (`MCG-NJU/UniDDT`)

**Trainer/optimizer**: PyTorch Lightning (`lightning.pytorch.cli.LightningCLI`
in `main.py`); optimizer/scheduler configuration is delegated to Lightning's
standard `configure_optimizers()` hook pattern (framework-managed, not a
custom wrapper).

**Resume mechanism**:
- `main.py` reads a `ckpt_path` key from the CLI config
  (`is_resume = self._get(self.config_init, "ckpt_path", ...)`) and passes it
  through to Lightning's `Trainer.fit(ckpt_path=...)` call path.
- PyTorch Lightning's documented default behavior for `ckpt_path` restores
  model weights, optimizer state, LR scheduler state, and
  epoch/global-step/loop position — a full-state resume guarantee **at the
  framework level**.
- **Caveat (explicitly flagged, not glossed over):** I read `main.py`'s
  top-level wiring but did **not** locate and fully read UniDDT's own
  `Trainer`/`LightningModule` subclasses (e.g. any custom
  `on_load_checkpoint`/`on_save_checkpoint` override) to rule out a
  project-specific override that could silently narrow this guarantee (for
  example, EMA-weight handling or the dual diffusion-decoder/LLM-backbone
  split could plausibly need custom checkpoint hooks that I have not
  verified). Recorded as "framework-level guarantee, not independently
  verified against UniDDT's own trainer subclasses" — an honest partial
  finding rather than an inferred pass.

**Configuration/data interfaces**: Lightning CLI YAML configs; three-stage
pipeline (Warmup, Joint training, Duality post-training) with per-stage
config files (based on README's "Architecture and training" section).

**Minimum/recommended GPU memory**: not independently determined in this
audit (README does not appear to publish an explicit minimum-hardware table
comparable to SenseNova-U1's or a measured smoke comparable to Show-o2's/
T210's); recorded as an open item, not fabricated.

**Shared/private parameter classification**: architecturally well-defined
(shared Noisy-ViT + Qwen3-VL-4B LLM backbone vs. private
`dit_decoder.py`/`jit_decoder.py` diffusion decoder), but no published exact
parameter-count breakdown comparable to SenseNova-U1's
`docs/parameter_breakdown.md` was found — the split is qualitatively
auditable from the code/architecture but not quantitatively pre-published.

**License: UNRESOLVED / MISSING.** No `LICENSE` or `LICENSE.md` file exists
in the `MCG-NJU/UniDDT` GitHub repository root (confirmed via direct `curl`
requests returning HTTP 404 for both filenames), and the Hugging Face Hub
API response for the model (`huggingface.co/api/models/MCG-NJU/UniDDT`)
carries no `license:` tag. This is a code/API-confirmed gap, not an
assumption, and independently disqualifies UniDDT under the decision rule's
"permits the intended research use under recorded licenses" criterion
regardless of its otherwise-competitive resume/architecture properties. See
`failure-ledger.md`.
