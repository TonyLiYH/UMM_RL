# T230 result summary

Task: `tasks/T230-sensenova-u1-admission.md`. Branch: `agent/T230-sensenova-u1-admission`. Source
revision audited: SenseNova-U1 commit `f97964a6e54b0abf92aa2db849af4e942bb2ff08` (recorded in
`first-report.md`). Admission target: **`sensenova/SenseNova-U1-8B-MoT`** only (U1, not U1.5 — see
`first-report.md`'s frozen-protocol scope confirmation).

## Current admission run (authoritative status)

The formal admission run is `runs/admission-sensenova-u1-v1/` — see
`runs/admission-sensenova-u1-v1/manifest.json` (`status: pass`, 15 hash/byte-addressed artifacts)
and `runs/admission-sensenova-u1-v1/metrics.json` (smoke exit-code evidence, block-registry
rollup, routed-overlap audit, and resource metrics). This is an admission/smoke run (one VQA pass,
one T2I pass, one CPU-side parameter enumeration), not a training run — no training script was run
or authorized.

Measured figures (H20-FoldUMM container, GPU0, local SSD `/dockerdata/t230-sensenova/`, `xfs`
local block device confirmed via `configs/admission/sensenova-u1/storage-preflight.json`):

- **Model load time**: ~5.55s (understanding smoke) / ~4.80s (generation smoke), both cold
  processes from local SSD.
- **Peak GPU memory**: understanding smoke ~32.98GiB allocated / ~33.22GiB reserved; generation
  smoke ~34.78GiB allocated / ~35.75GiB reserved. Both comfortably within a single H20's 96GB.
- **GPU wall-clock / GPU-hours**: 73s (VQA) + 86s (T2I) + ~3s (CPU-side parameter inspection,
  conservative estimate since the tool never touches the GPU) = 162s total, ≈0.045 GPU-hours —
  well within the 12 GPU-hour cap.
- **Checkpoint footprint**: 35,104,947,297 bytes (~32.7GiB) across 8 safetensors shards + index +
  config, downloaded directly to `/dockerdata/t230-sensenova/checkpoints/SenseNova-U1-8B-MoT`
  (no CQ7 shared-storage canonical copy exists for this checkpoint — see "Structural difference
  from T210" in `runs/admission-sensenova-u1-v1/notes.md` for why).
- **Durable evidence footprint** (raw logs/outputs, CQ7 copy): 5,329,360 bytes.

## What was built/run

- First report (audit, no GPU execution yet): `reports/T230/first-report.md` — repository/license
  audit, U1-vs-U1.5 scope confirmation, checkpoint identifiers, dependency inventory, entry points,
  and a draft MoT-routing observation, all committed/pushed before any download or GPU work.
- Storage preflight: `configs/admission/sensenova-u1/storage-preflight.json` — `status: pass`,
  `filesystem_class: local`, target `/dockerdata/t230-sensenova/hf_cache` on local `xfs`, ~9.9TB
  free.
- Environment build: `configs/admission/sensenova-u1/environment-lock.md` — fresh venv built
  directly from the official `pyproject.toml`/`requirements.txt` at the pinned commit; every
  top-level pin resolved exactly as declared, no corrective pin was needed (unlike T210, which
  required three). One documented, non-blocking optional-dependency omission: `flash-attn` (no
  generically-hosted PyPI wheel for this exact torch/CUDA/Python combination); the model falls
  back to SDPA per the repository's own documented behavior.
- Understanding + generation task-path smoke: `runs/admission-sensenova-u1-v1/metrics.json`'s
  `smoke` block — `examples/vqa/inference.py` and `examples/t2i/inference.py` both run to
  completion on GPU with the pinned `sensenova/SenseNova-U1-8B-MoT` checkpoint, using only
  official code and CLI arguments (`--attn_backend sdpa`, `--dtype bfloat16`, `--profile`), no
  source modification.
- Parameter-block registry: `configs/admission/sensenova-u1/parameter-block-registry.yaml` — exact
  weight-level enumeration of `named_parameters()` on the loaded checkpoint (17,552,340,992 total
  params), classified via the repository's own `DEFAULT_GROUPS` rules
  (`src/sensenova_u1/utils/param_count.py`), independently re-verified to confirm zero unassigned
  or catch-all-grouped parameters.
- Routed-overlap / MoT-vs-MoE-gate audit: `runs/admission-sensenova-u1-v1/metrics.json`'s
  `routed_overlap` block — documents the boolean per-token routing mask (`image_gen_indicators`)
  that selects between duplicate weight sets, explicitly distinguishes this MoT mechanism from the
  out-of-scope A3B variant's learned MoE gate, and records one static assumption violation (see
  below).
- No adapter code was written under `src/comppareto/adapters/sensenova_u1/` or `tests/adapters/` —
  none was required to satisfy this task's pass/fail gate (both paths execute using unmodified
  official scripts).
- No joint post-training run — none was authorized or attempted, per the frozen protocol.

## Checkpoints, revisions, and licenses recorded

| Component | Repo ID / source | Resolved revision | Hash (sha256, aggregate via `SHA256SUMS_20260903.txt`) | Size | License |
|---|---|---|---|---|---|
| SenseNova-U1 8B MoT (admission target) | `sensenova/SenseNova-U1-8B-MoT` | HF sha `bfa9b436503cb8aed4f2bc60e3236710cc77468d` | see `configs/admission/sensenova-u1/artifact-verification.json` (per-shard sha256, 10/10 pass) | 35,104,947,297 bytes | Apache-2.0 (repo `LICENSE` + HF `cardData.license`) |
| SFT variant (not targeted) | `sensenova/SenseNova-U1-8B-MoT-SFT` | not resolved (out of scope) | n/a | n/a | not checked (out of scope) |
| A3B MoE variant (not targeted) | `sensenova/SenseNova-U1-A3B-*` | not resolved (out of scope) | n/a | n/a | not checked (out of scope); uses a categorically different MoE-gate routing mechanism, not conflated with this admission's MoT audit |
| Excluded: any U1.5 checkpoint | n/a | n/a | n/a | n/a | out of scope per frozen protocol; U1.5 training pipeline "in preparation," not released |

## Routed-overlap audit — MoT boolean-mask routing and its static assumption violation

The 8B-MoT checkpoint routes each token through one of two duplicate weight sets inside the Qwen3
language-model backbone via a boolean mask (`image_gen_indicators`): understanding-path modules
versus `_mot_gen`-suffixed generation-path modules. This is a **hard, non-learned boolean split**,
categorically different from the out-of-scope A3B variant's **learned MoE gate** over expert
sub-networks — the two are documented separately and never conflated.

The released code's `forward()` methods
(`src/sensenova_u1/models/neo_unify/modeling_qwen3.py`) only implement the two pure cases (all
tokens understanding, or all tokens generation). The mixed case explicitly raises:

```
raise NotImplementedError(
    "The mixed und/gen forward path is not yet validated (issue #207): known "
    "issues are fixed, but it has no parity test and no production caller. "
    "Split the sequence at token-type boundaries and use forward_und / forward_gen."
)
```

This is recorded as the required static assumption violation
(`routed_overlap.static_assumption_violations_recorded: true` in
`runs/admission-sensenova-u1-v1/metrics.json`) — a documented, code-visible limitation of the
released repository, not a runtime failure of this admission's required smokes. Both required
smokes (VQA-only understanding, T2I-only generation) each exercise exactly one pure branch and
completed with `exit_code 0`.

## Environment defects found and fixed

None. Unlike T210, which required three corrective dependency-pin restorations, every pinned
dependency in SenseNova-U1's `requirements.txt`/`pyproject.toml` resolved exactly as declared with
no environment repair needed (see `configs/admission/sensenova-u1/environment-lock.md`).

## GPU/hardware evidence

H20-FoldUMM container, 8x H20 96GB. GPU0 used for all smoke work (`CUDA_VISIBLE_DEVICES=0`),
GPU1-7's placeholder occupancy untouched. Both `examples/vqa/inference.py` and
`examples/t2i/inference.py` completed with `exit_code 0`, captured directly from the wrapper
script's `$?` immediately after each subprocess invocation (direct, not indirect, exit-code
evidence — see `metrics.json`'s `exit_code_evidence` fields).

## Conclusion

**Supports gate.** The released U1 8B-MoT checkpoint executes both required task paths
end-to-end on GPU with only official code/config; every checkpoint/revision touched is recorded;
the parameter-block registry accounts for all 17.552B parameters at the weight level across three
disjoint blocks (`unassigned_trainable_parameters: 0`); the routed-overlap audit documents the
MoT boolean-mask routing mechanism, explicitly distinguishes it from the A3B MoE-gate mechanism,
and records the one static assumption violation the released code itself flags. The formal
admission run (`runs/admission-sensenova-u1-v1/manifest.json`, `status: pass`) records 15
hash/byte-addressed artifacts, all remotely reverified this round
(`configs/admission/sensenova-u1/artifact-verification.json`, 15/15 pass, 0 failed). No U1.5 asset
was downloaded or exercised at any stage. No joint post-training was started or attempted.
