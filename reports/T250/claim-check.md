# T250 — Claim check

Every load-bearing claim in this task's deliverables, with its evidence
source. Claims without a direct citation are flagged as hedged/inferred in
the source report itself (see `checkpoint-stage-evidence.md` and
`failure-ledger.md`) rather than asserted here as certain.

| Claim | Source evidence | Status |
|---|---|---|
| SenseNova-U1 `-SFT` checkpoints precede Multi-Expert RL/OPD training | `OpenSenseNova/SenseNova-U1/README.md` "Models" section, direct quote | Verified — direct official statement |
| SenseNova-U1's Stage 5 is explicit RL (Flow-GRPO + reward models) | arXiv 2605.12500, "Training Procedure" section (ar5iv HTML) | Verified — direct paper text |
| SenseNova-U1 resume restores model + optimizer + scheduler + counters | `sensenovalm/checkpoint/checkpoint_manager.py`, `try_load_internevo_ckpt` (direct source read) | Verified — code-level |
| SenseNova-U1 shared/private parameter split (1.245B/8.121B/8.186B of 17.552B) | `docs/parameter_breakdown.md` in official repo | Verified — official published figures, not independently recomputed by this audit |
| SenseNova-U1 shipped launcher needs 1 node x 8 GPUs x 80GB minimum | `training/README.md` hardware table | Verified — official doc |
| Show-o2 checkpoint is Apache-2.0, checkpoint hash as recorded | T210's accepted admission (`configs/admission/showo2/`, `reports/T210/environment-checkpoint-smoke.md`) | Verified — reused prior accepted evidence, not re-derived |
| Show-o2 both task paths execute with coherent output on H20 | `reports/T210/task-path-smoke.md` (real GPU smoke, T210) | Verified — reused prior accepted evidence; NOT re-run in this task |
| Show-o2 released checkpoint = Stage-2 SFT-equivalent | Pipeline-structure inference (only 2 stages exist, no RL stage) + T210 functional smoke | **Hedged** — no explicit checkpoint-card statement; stated as moderate confidence |
| Show-o2 resume restores weights only, not optimizer/scheduler state | `train_stage_one.py` resume block, lines ~264-320 (direct source read) | Verified — code-level |
| UniDDT has no recorded license (no LICENSE file, no HF license tag) | Direct `curl` 404 on `LICENSE`/`LICENSE.md`; HF Hub API response inspected | Verified — direct HTTP/API check |
| UniDDT's Duality post-training is not preference/RL | `MCG-NJU/UniDDT/README.md` "Architecture and training" section + arXiv 2606.16255 abstract — no reward-model/RLHF/DPO terminology found | Verified by absence — no contradicting evidence found in either official source |
| UniDDT released checkpoint's exact stage (post-Joint vs. post-Duality) | README prose vs. arXiv figure caption — inconsistent | **Hedged / unresolved** — recorded as unknown, not guessed |
| UniDDT resume is full-state via PyTorch Lightning `ckpt_path` | `main.py` CLI wiring + PyTorch Lightning's documented default `Trainer.fit(ckpt_path=...)` behavior | **Partially verified** — framework-level default confirmed; UniDDT's own trainer-subclass overrides not independently checked |
| No GPU work / no large downloads were needed to reach these conclusions | This task's own execution log: 0 GPU-hours, 0 downloaded checkpoint weights | Verified — self-evident from the commands actually run (see `first-report.md`) |
| Repository test suite passes on this branch | `.venv/bin/python -m pytest -q` → `156 passed` | Verified — actually executed, see final validator output reported to the orchestrator |
| Repository-state check passes | `.venv/bin/python -m comppareto.repo_state.cli --root .` → `task_tree=pass tasks=32`, `run_manifests=pass manifests=<n>`, `research_state=pass` | Verified — actually executed |

## Claims explicitly NOT made

- This task does **not** claim UniDDT is technically inferior to the other
  two candidates — its architecture, resume support, and non-RL final stage
  are all comparably strong. It is excluded solely on the license gate, which
  is stated plainly rather than conflated with a capability judgment.
- This task does **not** claim SenseNova-U1's shipped launcher fits this
  audit's 2-GPU envelope — it explicitly does not, as shipped. This is
  recorded as a T270 planning consideration, not glossed over.
- This task does **not** claim training readiness has been demonstrated by a
  live run for any candidate — no persistent parameter update, optimizer
  step, or training-loop execution occurred in this task for any candidate.
