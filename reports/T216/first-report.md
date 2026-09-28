# T216 first report — Show-o2 compute-matched alternating-protocol diagnostic

Published before any GPU/container execution, per `tasks/T216-showo2-alternating-protocol-diagnostic.md`'s
"Execution stages" item 1 ("Commit and push a first report before GPU execution") and its "First
report" checklist. Everything below is derived from (a) accepted T210 evidence
(`reports/T210/*.md`, `configs/admission/showo2/*`), which this task reuses verbatim rather than
re-deriving, (b) read-only inspection of the official, already-audited Show-o2 library source at
`/apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/Show-o/show-o2/` (the same clone
T210 audited; used here only as a library dependency, never copied into this repo), and (c) the
formal protocol definitions in `docs/math/00-notation-and-problem.md` and
`docs/math/01-alternating-shared-private-update.md`. **No T215 code, adapter, or implementation is
imported or depended on anywhere in this task**, per the task's explicit "no dependency on T215
code or its failed full-rerun implementation" instruction; T215's public failure ledger is cited
below only as a risk-analysis reference (its branch is not merged, checked out, or read from at
runtime).

## 1. Model, checkpoint, and local-SSD execution layout (reused from accepted T210)

All of the following are the exact, accepted T210 facts; T216 does not re-derive them and does not
change them:

- Model: `showlab/show-o2-1.5B` (repo revision `07ec16589d4fc5422a74dddbbc4b2cd11e551039`), backbone
  `Qwen/Qwen2.5-1.5B-Instruct` (revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`), vision encoder
  `google/siglip-so400m-patch14-384` (revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`), VAE
  `Wan2.1_VAE.pth` (sha256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`).
- Config: `configs/showo2_1.5b_demo_432x432.yaml` (sha256
  `d9f754ce8bdaf3a96cb6862782b51c781e2b0d3099bf37e15d244048c0559982`), `hidden_size=1536`,
  `num_und_trans_layers=8`, `num_refiner_layers=10`, `weight_type: bfloat16`.
- Local-SSD execution layout (inside H20-FoldUMM): `HF_HOME=/dockerdata/t210-showo2/hf_cache`,
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, filesystem `xfs` on `/dev/mapper/gpu-gpu_volume`
  (local block device, classified `local` by `storage_preflight`, ~9.0TiB free). Python environment:
  `/root/venvs/showo2` (per `configs/admission/showo2/environment-lock.md`), pinned
  `torch==2.5.1+cu124`, `transformers==4.47.0`, `diffusers==0.31.0`, `flash_attn==2.8.3.post1`.
- Total parameters: 3,063,740,640, matching checkpoint load with no missing/unexpected keys
  (`reports/T210/parameter-block-registry.md`).
- GPU: this task uses **GPU index 3 only** inside H20-FoldUMM (indices 0/2 are reserved by other
  concurrent tasks; GPU 0 additionally carries the benign `train2.py` placeholder, never touched).

T216 will re-run `configs/alternating/showo2/storage-preflight.json` fresh (stage 4 below) rather
than assuming T210's snapshot is still valid, since container state can change between tasks; the
schema mirrors `configs/admission/showo2/storage-preflight.json` exactly (`status`,
`filesystem_class`, `capacity_bytes`, `free_bytes`, `environment`, `path`).

## 2. Selected shared/private module paths and parameter counts

Reusing the accepted T210 parameter-block registry (`configs/admission/showo2/parameter-block-registry.yaml`,
`reports/T210/parameter-block-registry.md`), T216 selects one **short-backward-path** subspace inside
each of the three registry blocks it touches, to keep every K=1 gradient evaluation cheap and to keep
the shared/private boundary declared at weight level (per the frozen protocol: "Use the T210
shared/private block registry to select one declared shared subspace and one private subspace per
task"):

| Role | Module path | Registry block | Estimated params | Rationale |
|---|---|---|---:|---|
| Shared (\(P_i\theta\), both tasks) | `showo.model.layers[27]` (last of 28 Qwen2.5 decoder layers) | shared — LLM backbone (1,776,267,776 total) | ≈45–50M (estimate; backbone total minus tied embedding ≈1.31B over 28 layers) | Nearest decoder layer to the LM head / output stream for both task paths ⇒ shortest backward path through the shared block; identical module instance is used by both the MMU forward and the T2I forward, so `P_i` for both tasks selects the same physical tensors. |
| Understanding-private (\(\phi_{und}\)) | `und_trans.layers[7]` (last of 8 SigLIP-derived encoder layers) | understanding-private (397,141,792 total) | ≈35–45M (estimate) | Downstream-most understanding-private layer; feeds `fusion_proj` directly. Note: `und_trans` is architecturally *upstream* of the shared backbone in the forward graph (image tokens are embedded before entering `showo`), so a backward pass through this block still requires traversing the shared backbone's computation graph — this is unavoidable given the architecture and is accounted for in the runtime estimate below, not hidden. |
| Generation-private (\(\phi_{gen}\)) | `diffusion_head_a[9]` (last of 10 refiner blocks) | generation-private (883,837,248 total) | ≈70–90M (estimate) | Refiner blocks run *after* the shared backbone (`diff_proj` sits between them), so this is a short, cheap backward path, symmetric in spirit to the shared-layer choice. |

Exact parameter counts (not estimates) will be recorded in `configs/alternating/showo2/resolved-config.yaml`
during stage 4, obtained the same way T210 did — enumerating `named_parameters()` on the loaded model
and summing the tensors under each selected module path — rather than being guessed at execution
time. The estimates above are principled (derived from registry block totals and layer counts) and
are declared now, before observing any run outcome, so the shared/private boundary itself is not
selected post-hoc from results.

`fusion_proj` (shared — fusion junction, 6,493,824 params) and the timestep-embedding chain
(`time_embed`, `time_embed_proj`, generation-private) are **not** selected as the shared/private
subspace for this diagnostic; they remain frozen (`requires_grad=False`) throughout, along with
every other parameter outside the three selected module paths. This keeps the experiment's
gradient/optimizer bookkeeping limited to exactly three small tensors groups, which is what makes
exact snapshot/restore and cheap K=1 gradient evaluations tractable inside the resource envelope.

## 3. Optimizer state inventory

- **Persistent optimizer**: none. The admitted checkpoint is a pretrained inference checkpoint, not
  a live training run; there is no pre-existing Adam momentum/variance state to snapshot or restore
  for the shared or private subspaces. This is declared explicitly rather than assumed.
- **Per-row private-step optimizer**: a **fresh** `torch.optim.AdamW` instance is constructed at the
  start of every \(A_i^K\) private-adaptation call (P1's control, P2's post-update private steps, and
  P3's pre-adaptation virtual steps), scoped to exactly the one private subspace's parameters, with
  hyperparameters taken verbatim from the official `configs/showo2_1.5b_stage_2_a.yaml` training
  config (`learning_rate_showo=5e-5`, `beta1=0.9`, `beta2=0.999`, `weight_decay=0.0`,
  `epsilon=1e-8`) — reused rather than invented, and identical across every control/treatment branch
  so the private-step budget and hyperparameters are matched exactly. "Fresh" means `step=0`,
  `exp_avg=0`, `exp_avg_sq=0` at the start of every row; this optimizer instance is discarded (not
  reused) at the end of the row, and only the fresh-state convention itself needs to be identical
  across rows, which it trivially is.
- **Shared update**: never uses an optimizer — the shared displacement is applied as a single
  literal tensor update \(\theta'=\theta+\eta d\) with a plain in-place add, no Adam state at all.
  This is a deliberate design choice (see §9) that avoids T215's specific MMU-NAN failure mode.
- **Snapshot/restore inventory** captured once before every row and restored after every row:
  shared-subspace parameter tensors (`.clone().detach()`), private-subspace parameter tensors
  (`.clone().detach()`), full RNG state (`torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`,
  `random.getstate()`, `numpy.random.get_state()`), and the fixed single-example batch tensors
  (already immutable/re-used, no data-order state to restore beyond re-using the identical tensors).
  There are no buffers (no BatchNorm-style running stats in the selected subspaces — Qwen2.5 decoder
  layers and the refiner/encoder blocks use LayerNorm/RMSNorm, which are parameter-only, no running
  buffers) and no gradient scaler (no mixed-precision `GradScaler` is used; the model runs in
  `bfloat16` per the admitted config, which does not require loss scaling).

## 4. Equations and pseudocode for P0–P3

Notation follows `docs/math/00-notation-and-problem.md`/`01-alternating-shared-private-update.md`
and the task file exactly: \(\theta\) = shared subspace, \(P_i\theta\) = shared coordinates as seen
by task \(i\) (identical tensors for both tasks here), \(\phi_i\) = task-\(i\) private subspace,
\(s_i=(\phi_i,\omega_i)\) = private state including optimizer state, \(A_i^K(P_i\theta;s_i)\) =
exactly \(K\) native AdamW private steps at fixed shared parameters, \(\pi_\phi\) = projection onto
the private parameters, \(g_i^{raw}=P_i^\top\nabla_{x_i}L_i\) = raw shared gradient for task \(i\).

For \(i\in\{und,gen\}\), \(L_{und}\) = `next_token_prediction` cross-entropy on the fixed MMU batch
(`text_labels`≠None, `image_labels`=None), \(L_{gen}\) = `velocity_prediction` masked MSE on the
fixed T2I batch (`image_labels`≠None, `text_labels`=None) — using the model's own official
single-task loss branches (see §6), never the mixed dual-loss branch.

```
snapshot0 = snapshot(theta_shared, phi_und, phi_gen)   # cloned tensors
rng0      = snapshot_rng()

# ---- P1: private-only control (also reused as the common control for P2/P3) ----
restore(snapshot0); restore_rng(rng0)
for i in {und, gen}:
    opt_i = fresh_AdamW(phi_i; lr=5e-5, betas=(0.9,0.999), eps=1e-8, wd=0.0)
    for k in range(K):
        loss_i = L_i(theta_shared, phi_i; batch_i)      # theta_shared == original, fixed
        loss_i.backward()
        opt_i.step(); opt_i.zero_grad()
    s_i_control_K = clone(phi_i)                          # = s_i^{0,K}
    L_i_control   = L_i(theta_shared, s_i_control_K; batch_i)   # final control loss, no grad
restore(snapshot0); restore_rng(rng0)                      # roll back private control steps

# ---- P0: simultaneous raw baseline ----
restore(snapshot0); restore_rng(rng0)
g_und_raw = grad_theta(L_und(theta_shared, phi_und; batch_und))     # at ORIGINAL state
g_gen_raw = grad_theta(L_gen(theta_shared, phi_gen; batch_gen))     # at ORIGINAL state
d = normalize(negotiate(g_und_raw, g_gen_raw))                       # any candidate direction
for eta in step_scales:
    theta_new = theta_shared + eta * d                                # single literal update
    for i in {und, gen}:                                              # private update does NOT see theta_new
        s_i_simul_K = A_i^K(theta_shared; phi_i)                      # identical to P1's control call
    L_i_simul = L_i(theta_new, s_i_simul_K; batch_i)
    restore(snapshot0); restore_rng(rng0)

# ---- P2: shared-then-private (SP) ----
restore(snapshot0); restore_rng(rng0)
g_und_raw, g_gen_raw = <same raw gradients as P0, recomputed at original state>
for d in {d_rawsum, d_normsum, d_pcgrad, d_mgda}:
    for eta in step_scales:
        theta_prime = theta_shared + eta * d
        for i in {und, gen}:
            s_i_dK = A_i^K(theta_prime; phi_i)            # K private steps AFTER shared update
        Delta_i_controlled = L_i(theta_prime, s_i_dK; batch_i) - L_i_control[i]   # vs P1
        restore(snapshot0); restore_rng(rng0)

# ---- P3: private-then-shared commit (virtual commit, per docs/math §"virtual commit") ----
restore(snapshot0); restore_rng(rng0)
for i in {und, gen}:
    s_i_bar = A_i^K(theta_shared; phi_i)                   # identical to P1's control call
g_und_commit = grad_theta( L_und(theta_shared, stopgrad(s_und_bar); batch_und) )
g_gen_commit = grad_theta( L_gen(theta_shared, stopgrad(s_gen_bar); batch_gen) )
restore(snapshot0); restore_rng(rng0)                       # virtual private transition restored
for d_commit in {d_rawsum_commit, d_mgda_commit}:            # mandatory PS directions
    for eta in step_scales:
        theta_pp = theta_shared + eta * d_commit
        for i in {und, gen}:
            s_i_post = A_i^K(theta_pp; phi_i)                # fresh K steps from ORIGINAL phi_i snapshot
        Delta_i_controlled = L_i(theta_pp, s_i_post; batch_i) - L_i_control[i]     # vs P1
        restore(snapshot0); restore_rng(rng0)
```

The exact attribution identity from `docs/math/01-alternating-shared-private-update.md`
(\(\Delta_i^{total}=\Delta_i^{private}+\Delta_i^{controlled}\)) will be reported for every mandatory
row: \(\Delta_i^{private} = L_i^{control}[i] - L_i(\theta,\phi_i^0)\) (private-only improvement over
the untouched original state) and \(\Delta_i^{controlled}\) as computed above; their sum must equal
the uncorrected total change \(L_i(\theta',s_i^{d,K}) - L_i(\theta,\phi_i^0)\) within float
tolerance, which is itself a cross-check on the bookkeeping, not merely a reported quantity.

P3 here implements exactly the **virtual commit** variant from `docs/math`'s three-way distinction
(persistent PS / virtual commit / commit-then-SP) — matching the task file's "The virtual
pre-adaptation state must not persist across candidate directions" and the frozen protocol's "No
full AdamW trajectory differentiation and no persistent joint training." Neither persistent PS nor
commit-then-SP is implemented; if time/budget remain after the mandatory rows this will be noted as
a documented scope limitation, not silently attempted.

## 5. Shared candidate directions

All raw gradients are flattened per-task vectors over the shared subspace only (the single decoder
layer's parameters, concatenated). Let \(g_1=g_{und}^{raw}\), \(g_2=g_{gen}^{raw}\) (or the commit
gradients for PS).

- **Raw SUM**: \(d=g_1+g_2\).
- **Normalized SUM**: \(d=g_1/\lVert g_1\rVert_2+g_2/\lVert g_2\rVert_2\) (guarded: if either norm is
  below a declared epsilon floor \(10^{-12}\), that task's term is treated as exactly zero rather
  than dividing by a near-zero norm, and this fallback is logged per-row).
- **PCGrad** (frozen order `und→gen`, mandatory; reversed order `gen→und` as a separately declared
  sensitivity check, not a third mandatory row): for ordered pair \((g_a,g_b)\), if
  \(g_a\cdot g_b<0\), \(g_a'=g_a-\frac{g_a\cdot g_b}{\lVert g_b\rVert^2}g_b\), else \(g_a'=g_a\);
  symmetrically for \(g_b'\); \(d=g_a'+g_b'\).
- **Exact two-task MGDA**: closed-form convex combination
  \(\alpha^\*=\mathrm{clip}\left(\frac{(g_2-g_1)\cdot g_2}{\lVert g_1-g_2\rVert^2},0,1\right)\)
  (with the degenerate \(g_1=g_2\) case handled as \(\alpha^\*=0.5\)), \(d=\alpha^\* g_1+(1-\alpha^\*)g_2\).

All four are mandatory for P0/P2. For P3, only raw-SUM-of-commit-gradients and MGDA-of-commit-gradients
are mandatory (per the task file: "PCGrad/normalized SUM for PS are optional only after mandatory
rows pass" — deferred, not attempted this round unless the mandatory rows pass with budget to spare).

**Direction normalization (declared common trust-region metric)**: every candidate direction \(d\)
above is rescaled to unit L2 norm (\(d_{unit}=d/\lVert d\rVert_2\)) *before* step scales are applied,
so that step scale \(\eta\) has the same geometric meaning (Euclidean displacement magnitude in the
shared subspace) across every negotiator. This is declared here, before any run, so it cannot be
picked post-hoc to favor one negotiator.

## 6. Step-scale grid

Pilot scale chosen from the official training config's shared-parameter learning rate
(`learning_rate_showo=5e-5`, `configs/showo2_1.5b_stage_2_a.yaml`) applied as the coefficient on the
unit-normalized direction — i.e. treating the pretrained optimizer's own per-step displacement
magnitude as the natural pilot, not a value chosen after seeing any T216 outcome. Three symmetric
logarithmic scales around it (decade steps):

\[
\eta \in \{5\times10^{-6},\ 5\times10^{-5},\ 5\times10^{-4}\}.
\]

This satisfies "at least three symmetric logarithmic step scales around a pilot scale selected
without observing treatment outcomes." If K=1 mandatory rows pass with budget remaining, additional
half-decade scales may be added, but the three above are the pre-registered minimum and will always
be reported regardless of outcome.

## 7. Task batches, seeds, RNG coupling, and data-order protocol

- **Understanding (MMU) batch**: batch size 1, reusing T210's exact admitted demo asset — the image
  at `docs/mmu/pexels-jane-pham-727419-1571673.jpg` (sha256
  `a128066cc9ef5f0aaa1d1ba72d6f73938e1be40b4f4b92f0afe9e7bc441a0671`) with a fixed caption target
  text used as `text_labels` (via the official `format_sequence_und` tokenization utility). Only the
  loss/gradient pipeline is exercised — this is a diagnostic, not a captioning-accuracy benchmark, so
  caption quality is irrelevant to the result.
- **Generation (T2I) batch**: batch size 1, a single fixed real image encoded through the frozen
  Wan2.1 VAE to obtain the clean latent \(x_1\), with a fixed text prompt as conditioning
  (`text_labels=None`), using `transport.sample`/`path_sampler.plan` to draw \((t,x_t,u_t)\) and the
  official `format_sequence_gen_qwen2_5` utility for sequence construction.
- **Fixed batches throughout**: the exact same single MMU example and single T2I example are used
  for every row of every protocol/direction/step-scale — trivially satisfies "same batches... for
  treatment and control branches" since there is only one example per task and no data loader.
- **RNG coupling**: one master RNG snapshot (`torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`,
  `random.getstate()`, `numpy.random.get_state()`) is captured once immediately after model load
  under a fixed master seed **216**. Before *every* row (control or treatment), RNG state is
  explicitly *restored* from this one saved master snapshot (not re-seeded via repeated
  `manual_seed()` calls) — this is a deliberate fix for T215's documented `ROLLBACK-RNG` failure
  (branches consuming different amounts of randomness before a re-seed call, causing byte-inexact
  restores). Restoring from an explicit saved snapshot guarantees both common random numbers across
  every branch and byte-exact RNG rollback after each row.
- **Determinism check**: one representative row (P0, raw-SUM direction, pilot step scale, MMU task)
  is executed twice under identical conditions (same restored master RNG snapshot both times) to
  confirm bit-identical losses/gradients, satisfying "repeated same-seed executions match declared
  deterministic tolerances."
- **Data order**: N/A beyond the above — single fixed example per task, no shuffling, no ordering
  degrees of freedom to protocol.

## 8. Expected gradient-evaluation count, memory, runtime, and GPU-hours

At \(K=1\), reusing shared computation across directions/step-scales wherever the protocol permits
(P1's control call is computed once per task and reused as the control baseline for P0/P2/P3; raw
per-task shared/private gradients at the original state are computed once per task and reused to
build all four P0/P2 directions):

| Stage | Forward+backward passes (both tasks combined) |
|---|---:|
| P1 control (shared once per task) | 2 |
| P0 (4 directions × 3 step scales; private steps reused from P1, only the shared+final losses vary, no extra backward per step-scale since the shared update is a literal add) | ≈2 raw-grad evals + 12 final-loss forward evals |
| P2 (4 directions × 3 step scales × K=1 private step each) | ≈24 fwd+bwd |
| P3 (pre-adaptation private steps reused from P1; 1 commit-gradient eval per task; 2 directions × 3 step scales × K=1 post-update private step) | ≈2 commit-grad evals + ≈12 fwd+bwd |
| Determinism replicate (1 row × 2 runs) | ≈2 fwd+bwd |

**Estimated total: on the order of 55–65 forward+backward passes** across both task paths for the
full K=1 mandatory sweep, plus a handful of loss-only forward evaluations. This is a pre-registered
upper-bound-style estimate, not a promise of the exact final count — the real count will be tallied
row-by-row in `runs/alternating-showo2-v1/metrics.json`'s resource-accounting section.

Per T210's measured figures (§R2/R5 of `reports/T210/r2-r5-ssd-rerun.md`), a full-model single
forward pass (batch size 1, this checkpoint, this GPU family) completes in single-digit seconds once
loaded from local SSD, with peak allocated memory ≈14GB (MMU) / ≈13GB (T2I) for *inference*. A
training-style forward+backward with only three small module subspaces `requires_grad=True` (all
other parameters frozen) is expected to add modest activation-memory overhead over the inference
figures (autograd must retain the intermediate activations needed for backward through the
trainable-subspace's path, plus the necessarily-full forward pass through the frozen backbone) — a
conservative estimate is peak allocated memory in the 20–30GB range, comfortably inside a single
H20's 96GB. Wall-clock: at a conservative 10–20s per fwd+bwd (including any autograd-graph
construction overhead beyond the measured inference numbers), 55–65 passes ≈ 10–22 minutes of GPU
time, i.e. **≈0.2–0.4 GPU-hours on one H20**, far inside the 8 GPU-hour envelope and leaving very
large headroom for K=3 if the K=1 gate passes (K=3 roughly triples only the private-step-related
passes, not the raw/commit-gradient passes, so the total would remain well under 1 GPU-hour). Actual
measured figures replace every estimate in this section in `runs/alternating-showo2-v1/metrics.json`
and `reports/T216/result-summary.md`.

## 9. Why this design structurally avoids T215's documented failure modes

T215 (a structurally related, currently-`blocked` task on the same model, whose branch/code T216
does not import or depend on) documented two genuine, non-infrastructure numerical findings in its
failure ledger (`origin/agent/T215-showo2-finite-response-feasibility:reports/T215/failure-ledger.md`,
read-only reference):

- **MMU-NAN**: root-caused to AdamW's `eps`-outside-sqrt convention combined with `create_graph=True`
  second-order differentiation, hitting a `d(sqrt(x))/dx` singularity at `x=grad_p^2=0` for a
  step=1 unrolled trajectory — a genuine K=1 numerical result of *differentiating through* an Adam
  step, not an infra bug.
- **T2I-FDMISS**: a finite-difference reference mismatch on 3/4 directions, a numerical-accuracy
  limitation of gradient-checking via finite differences on this model.
- **ROLLBACK-RNG**: RNG restore failing to be byte-exact because the driver re-seeded via repeated
  `torch.manual_seed()` calls rather than saving/restoring the actual generator state.

T216's protocol design is deliberately different in exactly the ways that matter for these three
findings: (1) T216 never uses `create_graph=True` or differentiates through an optimizer step —
every private adaptation \(A_i^K\) is a literal, first-order AdamW step applied normally (parameters
updated in place via `.step()`), and every shared "gradient" used to build a candidate direction is
an ordinary first-order `.backward()` at a fixed, non-differentiated state; there is no second-order
graph through Adam's `sqrt` anywhere in this design, so the specific MMU-NAN mechanism cannot occur
by construction (this is a structural argument, not a guarantee against all possible NaN sources —
any nonfinite result that does occur on real GPU execution will be reported honestly, per the task's
own "negative/indeterminate result is valid" policy, not hidden or explained away). (2) T216 does not
depend on finite-difference gradient checking at all — attribution rests entirely on the exact
compute-matched control identity (§4), not on FD approximation, so T2I-FDMISS's specific failure mode
is inapplicable. (3) T216's RNG protocol (§7) explicitly restores from one saved master snapshot
before every row, rather than re-seeding via `manual_seed()`, which is precisely the fix
ROLLBACK-RNG's own root-cause analysis calls for.

## 10. Exact commands and artifact paths (stages 4–8)

Local-SSD preflight and artifact verification (stage 4, read-only, no model load required for the
preflight check itself):

```bash
cd /apdcephfs_cq9/share_1447896/yihangli/workspace/UMM_RL/.worktrees/T216
script -qec "taiji_client exec -scfg '$SCFG' '$TASK_FLAG' '$INSTANCE_ID' bash -c '\
  cd /workspace/code && \
  CUDA_VISIBLE_DEVICES=3 /root/venvs/showo2/bin/python -m comppareto.repo_state.storage_preflight \
    --path /dockerdata/t210-showo2/hf_cache \
    --output configs/alternating/showo2/storage-preflight.json'" /dev/null 2>&1 | tr -d '\r'
```

K=1 protocol execution (stage 5, backgrounded per the container's `cjob.sh`/`setsid` convention,
never bare `&`):

```bash
script -qec "taiji_client exec -scfg '$SCFG' '$TASK_FLAG' '$INSTANCE_ID' bash -c '\
  cd /workspace/code && CUDA_VISIBLE_DEVICES=3 setsid /root/venvs/showo2/bin/python \
    -m comppareto.adapters.showo2_alternating.run_k1 \
    --config configs/alternating/showo2/resolved-config.yaml \
    --out runs/alternating-showo2-v1 \
    > /dockerdata/t210-showo2/t216_k1.log 2>&1 &'" /dev/null 2>&1 | tr -d '\r'
```

polled via repeated read-only `tail -n 40 /dockerdata/t210-showo2/t216_k1.log` calls through the same
`script -qec` wrapper, never a blocking foreground call.

Toy-model unit tests (stage 3, CPU-only, run locally in this worktree, no container/GPU needed):

```bash
cd /apdcephfs_cq9/share_1447896/yihangli/workspace/UMM_RL/.worktrees/T216
.venv/bin/python -m pytest -q tests/adapters/showo2_alternating/
```

Deliverable artifact paths (all inside `allowed_paths`):

- `configs/alternating/showo2/storage-preflight.json`, `artifact-verification.json`,
  `resolved-config.yaml`
- `reports/T216/first-report.md` (this file), `result-summary.md`, `claim-check.md`,
  `failure-ledger.md`
- `runs/alternating-showo2-v1/manifest.json`, `metrics.json`, `notes.md`
- `src/comppareto/adapters/showo2_alternating/` (snapshot/restore, negotiators, P0–P3 driver)
- `tests/adapters/showo2_alternating/` (toy-model protocol-ordering and rollback tests)

## 11. Resource envelope compliance (declared, to be verified against measured figures)

- GPUs: 1 (GPU index 3 only) — within "at most two H20 GPUs."
- GPU-hours: estimated ≈0.2–0.4 for K=1 (§8) — within "at most eight."
- Wall-clock: expected well under an hour of active execution — within "at most 24 wall-clock hours."
- \(K=3\): conditional on the K=1 gate passing per §"Pass/fail gate" in the task file; not run yet.
- No full-backbone unroll (only three small module subspaces are ever `requires_grad=True`); no
  persistent parameter update (every row restores from `snapshot0` before the next row begins, and
  the final on-disk checkpoint is never modified).

This report will be committed and pushed before any container/GPU command is issued, per the task's
hard gate.
