# T210 result summary

Task: `tasks/T210-showo2-admission.md`. Branch: `agent/T210-showo2-admission`. Source revision
audited: Show-o2 commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23` (recorded in `first-report.md`).

## What was built/run

- First report (audit, no GPU execution yet): `reports/T210/first-report.md`.
- Environment + checkpoint smoke: `reports/T210/environment-checkpoint-smoke.md` — GPU environment
  built on H20-FoldUMM, checkpoint(s) downloaded and hash-verified, one matmul smoke on `cuda:0`.
  Fixed one upstream defect (`build_env.sh`'s unpinned torch/torchvision install clobbered by a
  later step; restored the official pinned versions).
- Understanding + generation task-path smoke: `reports/T210/task-path-smoke.md` —
  `inference_mmu.py` and `inference_t2i.py` both run to completion on GPU with the pinned
  `showlab/show-o2-1.5B` checkpoint, using only official code/config plus environment-variable and
  official-CLI-override adaptation (`HF_HOME` redirection, `WANDB_MODE=offline`, a symlinked
  `Wan2.1_VAE.pth`, a reduced `num_inference_steps` for smoke speed). Fixed a second upstream
  defect (`build_env.sh`'s unpinned `wandb` install, incompatible with the resolved `protobuf`
  version and missing an API the inference scripts call).
- Parameter-block registry draft: `reports/T210/parameter-block-registry.md` and machine-readable
  `configs/admission/showo2/parameter-block-registry.yaml` — exact weight-level enumeration of
  `named_parameters()` on the loaded model (3,063,740,640 params), cross-checked against forward-pass
  data flow, refining and correcting the stage-1 provisional config-schema-only reading.
- No adapter code was written under `src/comppareto/adapters/showo2/` or `tests/adapters/` — none
  was required to satisfy this task's pass/fail gate (both paths execute using unmodified official
  scripts); adapter code, if any, is scoped to a successor task's training-interface work.
- No training run, no `runs/admission-showo2/` artifact — none was authorized or attempted, per the
  frozen protocol ("do not begin joint post-training").

## Checkpoints, revisions, and licenses recorded (cumulative across all stages)

| Component | Repo ID / source | Resolved revision | Hash | Size | License |
|---|---|---|---|---|---|
| Show-o2 LLM+heads | `showlab/show-o2-1.5B` | see `environment-checkpoint-smoke.md` | see `environment-checkpoint-smoke.md` | ~22GB (fp32) | Apache-2.0 (repo `LICENSE`) |
| LLM tokenizer/backbone | `Qwen/Qwen2.5-1.5B-Instruct` | see `first-report.md` | see `first-report.md` | — | Apache-2.0 |
| Vision encoder (und-private source) | `google/siglip-so400m-patch14-384` | see `first-report.md` | see `first-report.md` | — | Apache-2.0 |
| 3D causal VAE (frozen) | `Wan-AI/Wan2.1-T2V-14B` (`Wan2.1_VAE.pth` only) | not resolved from a specific HF commit; pre-existing local copy | sha256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` | 507,609,880 bytes | **not resolved — open item, see failure-ledger.md** |
| Safety checker (generation-path only; newly discovered this stage) | `CompVis/stable-diffusion-safety-checker` | `697f152c6b6183309a1509fb3589fcbaae7f34e5` | sha256 `64b8393f1afd5a0c1ed2aa5f341fa7c08286839a48f3743162a76a2835c808bd` (`model.safetensors`) | 2.3GB (cache) | **unspecified on model card ("More information needed") — open item, see failure-ledger.md** |

## Environment defects found and fixed (both upstream, both unpinned transitive deps)

1. `build_env.sh`'s unpinned `pip3 install torch`/a later step clobbering the pinned CUDA build —
   fixed by reinstalling the pinned `torch==2.5.1+cu124`/`torchvision==0.20.1+cu124`. Detailed in
   `environment-checkpoint-smoke.md`.
2. `build_env.sh`'s unpinned `pip3 install wandb` resolving to `wandb==0.29.0`, incompatible with
   the resolved `protobuf==4.25.9` (missing dispatcher branch) and missing `wandb.util.generate_id`
   (removed upstream) — fixed by pinning `wandb==0.17.0`. Detailed in `task-path-smoke.md`.

Both are dependency-version restorations for packages the official build script itself installs
unpinned, not substitutions of any Show-o2 component, checkpoint, or logic — consistent with "do
not use unofficial fixes... without local authorization," and fully recorded rather than silently
patched.

## GPU/hardware evidence

H20-FoldUMM container, 8x H20 96GB. GPU0 kept exclusively free of the benign `train2.py` placeholder
occupancy and used for all smoke work (`cuda:0`); GPU1-7's placeholder processes were never touched.
Matmul smoke (stage 2), `inference_mmu.py` (~26 min model load + <1 min inference), and
`inference_t2i.py` (~26s sampling once model/VAE/safety-checker already resident) all completed
with exit 0.

## Conclusion

**Supports gate.** Both task paths execute end-to-end on GPU with the pinned checkpoint using only
official code/config; every checkpoint/revision touched is recorded, including one newly-discovered
external component this stage; shared/private parameter blocks are enumerated exactly at the weight
level and are auditable. Two upstream environment defects were found and fixed via dependency-pin
restoration, both documented. Two license-status open items remain unresolved (see
`failure-ledger.md`) and are flagged for local-reviewer decision rather than resolved unilaterally.
No joint post-training was started or attempted.
