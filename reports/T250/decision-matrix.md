# T250 — Weighted decision matrix

## Weighting rationale

Weights follow the decision rule's own ordering and the task's stated
priority (a reproducible, honestly-evidenced starting point over raw
capability). Each dimension is scored 0-2 (0 = fails/unknown-and-blocking,
1 = partial/hedged, 2 = strong pass) except License, which is a hard gate
(0 = disqualifying regardless of other scores, matching the decision rule's
"permits the intended research use under recorded licenses" requirement).

| Dimension | Weight | Rationale |
|---|---|---|
| License (hard gate) | gate | Decision rule requirement; a real, unresolved gap is disqualifying, not merely penalized |
| Precedes preference/RL stage | 3 | Central research claim of this task |
| Executes both task paths | 3 | Explicit decision-rule requirement; "earlier is not better if unusable" |
| Resume restores weights+optimizer+scheduler+counters | 2 | Needed for long-horizon T270 training; explicitly audited per task instructions |
| Auditable shared/private/routed ownership | 2 | Needed for T270's parameter-block design |
| Fits H20 resource envelope (for this audit) | 1 | Only this audit's envelope; T270's own envelope is separately re-checked at that time |
| Headroom for further post-training | 1 | Qualitative, secondary to the above |

## Scores

| Dimension | SenseNova-U1-8B-MoT-SFT | Show-o2-1.5B | UniDDT |
|---|---|---|---|
| License | Apache-2.0 (+ MIT-derived files, attributed) — **pass** | Apache-2.0 (T210-confirmed) — **pass** | **No LICENSE file, no HF license tag — gate fails** |
| Precedes preference/RL stage | 2 — explicit README+paper statement, named Stage 5 RL (Flow-GRPO) named as later | 2 — no RL stage exists in pipeline at all; Stage-2 SFT is terminal supervised stage | 2 — no RL stage in pipeline either way (moot given license gate) |
| Executes both task paths | 2 — GPU-verified in the 2026-09-16 revision: real forward+backward smoke on the pinned SFT checkpoint itself (understanding_loss=9.830, generation_loss=5.436), gradients cleanly confined to each path's owned parameter group | 2 — T210 GPU-verified both `inference_mmu.py` and `inference_t2i.py` end-to-end with coherent output | 2 (architecturally dual-path per README; not independently GPU-verified) — moot given license gate |
| Resume restores full state | 2 — code-confirmed (`try_load_internevo_ckpt`: model+optimizer+scheduler+counters) | 1 — code-confirmed weights-only; optimizer/scheduler NOT restored | 1-2 — Lightning framework default is full-state, but not independently verified against UniDDT's own trainer subclasses — moot given license gate |
| Auditable parameter ownership | 2 — exact published breakdown (1.245B shared / 8.121B U / 8.186B G of 17.552B total) with reproducible inspection script | 2 — reuses T210's accepted `parameter-block-registry.yaml` | 1 — architecturally clear but no published exact parameter counts — moot given license gate |
| Fits audit's H20 envelope | 2 — real single-H20 smoke measured peak ~59.5GB of 96GB (2026-09-16 revision); derived that a `generation_private`-only T270 subspace plausibly fits 2xH20 with optimizer-state sharding, replacing the unverified 8x80GB-default assumption | 2 — smallest footprint, T210 measured ~14GB single-GPU already | not scored — moot given license gate |
| Headroom for post-training | 2 — 17.6B dense+MoT, ample headroom | 1 — 1.5B, less headroom but still workable | not scored — moot given license gate |

## Outcome

- **UniDDT is excluded** at the license hard gate. This is independent of and
  prior to any capability/architecture scoring — per the decision rule, a
  candidate that does not "permit the intended research use under recorded
  licenses" cannot be selected as primary or fallback, regardless of how
  favorably it might otherwise score. This exclusion is evidence-based (HTTP
  404 on both `LICENSE`/`LICENSE.md`, no `license:` tag on the HF Hub API),
  not a convenience-driven omission, and is logged in `failure-ledger.md`.
- **Primary recommendation: SenseNova-U1-8B-MoT-SFT.** Strongest, most
  explicit, official stage evidence of any candidate (README states outright
  that SFT precedes RL/OPD); the only candidate with a fully code-confirmed
  full-state resume (model + optimizer + scheduler + step/dataloader
  counters); the only candidate with an exact, reproducible, published
  shared/private parameter breakdown; native five-task-type support with
  per-task loss bucketing, directly supporting sequential-task-batch training
  at one frozen shared version. **2026-09-16 revision:** this is no longer
  documentation-only — the pinned SFT checkpoint itself was loaded from
  verified local SSD and exercised with real pure-understanding and
  pure-generation forward+backward smokes (losses 9.830/5.436, gradients
  cleanly confined to each path's owned parameter group), a real
  optimizer+scheduler+resume-metadata construction was verified to not mutate
  weights, and the smallest-feasible-H20-topology was measured/derived
  (peak ~59.5GB on one H20; `generation_private`-only training plausibly fits
  2xH20 with optimizer-state sharding) — directly replacing what was
  previously this candidate's only weakness (the shipped launcher's
  unverified 8x80GB default). No blocking limitation was found.
- **Fallback: Show-o2-1.5B.** Already-accepted admission (T210) with the only
  candidate-level **GPU-executed** functional evidence of both task paths
  working end-to-end; smallest footprint, best fit for a constrained GPU
  envelope, highest reproducibility certainty (environment already pinned and
  fixed by T210). Its honestly-disclosed weakness: the official resume path
  restores weights only, not optimizer/scheduler state — a real limitation if
  T270 needs to resume a long-horizon run from an intermediate Show-o2-based
  checkpoint, though it does not block using Show-o2 as a **fresh** starting
  point (which would initialize new optimizer/scheduler state regardless of
  the source checkpoint's own internal resume fidelity). Its stage label is
  also less definitively sourced than SenseNova-U1's (inferred from pipeline
  structure and T210's functional smoke, not from explicit model-card text).
  Unchanged by this revision — SenseNova-U1 remains primary since all six
  revision items now pass with real evidence (local-review item 7).

## Pass/fail gate check (per task file)

- Every candidate has an evidence-backed or explicitly-hedged stage label:
  **satisfied** (see `checkpoint-stage-evidence.md`).
- Inference admission separated from training readiness: **satisfied** —
  Show-o2's T210 evidence is explicitly labeled inference-only in this audit;
  its training/resume path was independently statically audited here rather
  than assumed from the inference smoke.
- At least one candidate classified as a reproducible starting point:
  **satisfied** — two candidates (SenseNova-U1, Show-o2) qualify; UniDDT does
  not, for a documented, verifiable reason.
- No persistent parameter update: **satisfied** — no `.step()` was ever
  called on any optimizer in this task; 0 optimizer steps, 0 dataset-scale
  runs. A real GPU smoke (forward+backward only) was executed in the
  2026-09-16 revision consuming ~0.01 GPU-hours (against a 4-hour cap); no
  weights were mutated (verified by sha256 fingerprint, see
  `training-interface-audit.md`).
