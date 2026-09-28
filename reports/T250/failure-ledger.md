# T250 — Failure ledger

This ledger records genuine, verifiable gaps and open items discovered during
the audit, per the task's honesty mandate ("say so plainly rather than
guessing"). None of these block the overall task's pass/fail gate, since the
gate requires only that every candidate have an evidence-backed or explicitly
unknown stage label and that at least one candidate qualify as a reproducible
starting point — both are satisfied (see `decision-matrix.md`).

## 1. UniDDT — missing license (disqualifying for UniDDT specifically)

- **What**: No `LICENSE` or `LICENSE.md` file exists in the `MCG-NJU/UniDDT`
  GitHub repository root; the Hugging Face Hub API response for the model
  carries no `license:` tag.
- **How verified**: direct `curl` requests to
  `https://raw.githubusercontent.com/MCG-NJU/UniDDT/main/LICENSE` and
  `.../LICENSE.md` both returned HTTP 404; `https://huggingface.co/api/models/MCG-NJU/UniDDT`
  response inspected for a `license` field — absent.
- **Impact**: UniDDT is excluded as both primary and fallback candidate per
  the decision rule's "permits the intended research use under recorded
  licenses" requirement. This is an external blocker specific to UniDDT, not
  a flaw in the audit methodology, and does not affect the other two
  candidates.
- **Resolution path (not pursued in this task)**: contact the UniDDT authors
  for a license clarification, or check for a license statement in the arXiv
  paper's PDF (not the HTML abstract) or supplementary material — out of
  scope for this audit given time/resource constraints and since two other
  candidates already satisfy the pass/fail gate.

## 2. Show-o2 — no explicit checkpoint-card stage statement

- **What**: No official Show-o2 documentation states outright that the
  released `showlab/show-o2-1.5B` checkpoint is "the Stage-2 SFT output."
  The stage label used in this audit (SFT-equivalent) is the best-supported
  conclusion from pipeline structure (only two supervised stages exist, no
  RL/preference stage exists anywhere in the code or docs) plus T210's
  functional GPU smoke evidence, not a direct publisher statement.
- **How verified**: full read of `show-o2/README.md`'s Training section and
  the arXiv 2506.15564 abstract; no stage-completion statement found for the
  released weights specifically (only for the training *procedure* in the
  abstract).
- **Impact**: lower confidence label (moderate, vs. high for SenseNova-U1)
  but does not disqualify Show-o2 — it is recorded as "moderate confidence,
  hedged" in `checkpoint-stage-evidence.md` and `candidates.yaml`, per the
  honesty mandate, rather than asserted as certain.

## 3. Show-o2 — resume restores weights only, not optimizer/scheduler state

- **What**: `train_stage_one.py`'s resume logic (lines ~264-320) restores
  model weights and reconstructs an approximate step count from the
  checkpoint directory name, but instantiates a **fresh** `AdamW` optimizer
  and a **fresh** LR scheduler after resume — optimizer momentum/variance and
  scheduler warmup/decay state are lost across a resume.
- **How verified**: direct read of the official `train_stage_one.py` source
  (fetched from `raw.githubusercontent.com/showlab/Show-o/main/show-o2/train_stage_one.py`).
- **Impact**: a real, material limitation of Show-o2 as the fallback
  candidate if T270 needs long-horizon resumable training from an
  intermediate Show-o2-based checkpoint. Does not block using Show-o2 as a
  fresh starting point (new optimizer/scheduler state would be initialized
  regardless). Stated plainly in `training-interface-audit.md` and factored
  into the decision matrix's "Resume restores full state" dimension (scored
  1/2, vs. 2/2 for SenseNova-U1).

## 4. UniDDT — stage-completion ambiguity between two official sources

- **What**: the GitHub README's prose names three stages (Warmup, Joint
  training, Duality post-training) but the arXiv 2606.16255 abstract's
  Figure 1 caption (fetched via ar5iv HTML) names only two ("Warmup" and
  "Joint training"). Whether the released checkpoint has completed the
  README's third stage is not explicitly confirmed by either source.
- **How verified**: side-by-side comparison of the two official texts; no
  further primary source (e.g., a model card changelog) was found to resolve
  the discrepancy.
- **Impact**: recorded as a hedged/unknown stage label for UniDDT in
  `checkpoint-stage-evidence.md`. Moot for the final selection outcome, since
  UniDDT is independently excluded by item 1 (missing license) regardless of
  its exact stage.

## 5. UniDDT — resume-hook override not independently verified

- **What**: `main.py` wires a `ckpt_path` config key through to PyTorch
  Lightning's `Trainer.fit(ckpt_path=...)`, which by default restores full
  training state (model, optimizer, scheduler, step/loop counters). However,
  I did not fully read UniDDT's own `LightningModule`/`Trainer` subclasses to
  rule out a project-specific checkpoint-hook override (e.g. for EMA-weight
  handling or the dual diffusion-decoder/LLM-backbone split) that could
  narrow this guarantee.
- **How verified**: read `main.py`'s top-level CLI wiring only; did not
  locate/read the full trainer subclass implementation within the time spent
  on this audit.
- **Impact**: recorded as "framework-level guarantee, not independently
  verified against UniDDT's own trainer subclasses" in
  `training-interface-audit.md`. Moot for the final selection outcome (item 1
  already excludes UniDDT).

## 6. SenseNova-U1 — "OPD" terminology not expanded in any fetched source

- **What**: the official README's changelog groups "OPD training" with
  "Multi-Expert RL" as occurring after Unified SFT, but no fetched document
  spells out what OPD stands for.
- **How verified**: full-text search of the README, the training README, and
  the arXiv HTML for "OPD" — found only in the changelog line, never expanded.
- **Impact**: does not change the stage label (OPD is named as occurring
  *after* SFT, consistent with — and not weakening — the "-SFT precedes
  preference/RL" conclusion). Recorded as an open terminology gap only, not
  guessed at.

## 7. No GPU smoke executed for this task — SUPERSEDED 2026-09-16

- **What (original, 2026-09-15)**: per execution stage 6 ("run minimal
  load/resume-interface smokes only where static evidence is insufficient"),
  no GPU smoke was run for any of the three candidates in this task. Static
  source-level evidence was judged sufficient to resolve the
  resume-restoration-granularity dimension for all three candidates.
- **Superseded**: 2026-09-16 local review (see the task file's "Local review
  requirements") judged static evidence insufficient specifically for
  SenseNova-U1-8B-MoT-SFT and required a real GPU smoke on the SFT checkpoint
  itself. That smoke was executed: one pure-understanding and one
  pure-generation forward+backward pass on the pinned SFT checkpoint, loaded
  from verified local SSD, plus an optimizer+scheduler+resume-metadata
  construction verified not to mutate weights. Full detail in
  `reports/T250/training-interface-audit.md`'s Revision addendum;
  `runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json` has the
  numeric evidence. GPU-hours consumed: ~0.01 of the 4-hour cap. Show-o2 and
  UniDDT's static-only audits were judged sufficient by local review and are
  unchanged.

## 8. SenseNova-U1 — smallest-feasible-H20-topology for T270 is a derived projection, not a live 2-GPU run

- **What**: local-review item 5 asked for the smallest feasible H20 topology
  for T270's intended trainable subspace. This audit measured the real
  single-H20 footprint of a load+forward+backward smoke (peak ~59.5GB) and
  separately derived, via exact meta-device parameter counts and standard
  bf16-weight/fp32-AdamW-state arithmetic, that a `generation_private`-only
  subspace plausibly fits 2xH20 with optimizer-state sharding, while
  full-parameter fine-tuning of both trainable groups does not fit 2xH20
  without further sharding/offload.
- **How verified**: the single-H20 measurement is real (executed). The
  2xH20 projection is arithmetic (weights + grads + optimizer-state bytes
  summed from exact measured parameter counts), not an actual 2-GPU
  distributed run — this task's envelope permits but does not require
  spending GPU-hours on a distributed-training-topology smoke for a
  selection audit that does not itself authorize training.
- **Impact**: recorded as "derived/hedged" in `reports/T250/claim-check.md`.
  Does not block this task's pass/fail gate or the primary-candidate
  decision; a follow-up 2-GPU sharded-optimizer smoke can be requested for
  T270 specifically if the actual training run needs tighter confirmation
  before committing to a resource request.
