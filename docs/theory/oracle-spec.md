# T155 exact finite-response oracle: mathematical specification

Status: first report, published before implementation, per `tasks/T155-exact-finite-response-oracle.md`. Every symbol is coordinate-checked against `docs/theory/formulation.md` (rerun/commit semantics, \(P_i\), \(J_i\), \(C_i\), \(S_i\)) and `docs/theory/2026-08-27-theory-breakthrough-audit.md` §4/§6.1 (\(Z_k\), \(J_k\), \(B_k\), tangent-residual recurrence). This oracle is a mechanism-validation benchmark. It does not claim to approximate any real unified multimodal model.

## 1. Global state, blocks, and selectors

Global shared parameter \(\theta\in\mathbb R^p\) is partitioned into \(B\) contiguous blocks of widths \(w_1,\dots,w_B\), \(p=\sum_b w_b\), \(B\in[4,64]\). There are \(m\in[2,8]\) tasks. Task \(i\) touches a block subset \(\mathcal B_i\subseteq\{1,\dots,B\}\); its selector \(P_i\in\{0,1\}^{p_i\times p}\) stacks, in block order, the one-hot rows that copy each coordinate of every block in \(\mathcal B_i\). By construction:

- each row of \(P_i\) selects exactly one global coordinate (single lift);
- no global coordinate is selected twice **within the same task's** \(P_i\) (no-duplicate-coordinate contract);
- distinct tasks' selectors may select the same global coordinate (this is the overlap the graph families control; it is not a violation of the per-task contract).

\(x_i=P_i\theta\in\mathbb R^{p_i}\). Task-private parameters are \(\phi_i\in\mathbb R^{d_i}\), \(d_i\in[2,32]\). This exactly matches `QuadraticTask.selector` validation in `src/comppareto/quadratic.py` (each row sums to 1, each column sums to \(\le 1\)), which the oracle's own selector type re-implements independently (per-task, not reused, so that the multi-task graph-family generator in `src/comppareto/oracle/` is self-contained and does not require editing `quadratic.py`, which is outside T155's `allowed_paths`).

## 2. Train loss and meta loss

Fix task \(i\). Curvature blocks \(H_{xx}^i\in\mathbb R^{p_i\times p_i}\) (symmetric), \(H_{x\phi}^i\in\mathbb R^{p_i\times d_i}\), \(H_{\phi\phi}^i\in\mathbb R^{d_i\times d_i}\) (symmetric) are properties of the task's local quadratic model and are **shared between the train and meta objective** — curvature is a property of the local model around the current operating point, not of which data batch evaluates it. Linear/gradient coefficients differ between train and meta, because train and meta batches are independent draws:

\[
\ell_i(x_i,\phi_i)=a_i^\top x_i+\tfrac12x_i^\top H_{xx}^ix_i+x_i^\top H_{x\phi}^i\phi_i+b_i^\top\phi_i+\tfrac12\phi_i^\top H_{\phi\phi}^i\phi_i,
\]
\[
\ell_i^{\mathrm{meta}}(x_i,\phi_i)=a_i^{\mathrm{meta}\top}x_i+\tfrac12x_i^\top H_{xx}^ix_i+x_i^\top H_{x\phi}^i\phi_i+b_i^{\mathrm{meta}\top}\phi_i+\tfrac12\phi_i^\top H_{\phi\phi}^i\phi_i,
\]

with \((a_i,b_i)\) and \((a_i^{\mathrm{meta}},b_i^{\mathrm{meta}})\) drawn independently from the same generative distribution (§8), so the meta batch is genuinely disjoint from train, per `formulation.md` §2.2's \(\xi_{i,t}^{\mathrm{meta}}\).

Private regularization anchor \(\phi_i^0\in\mathbb R^{d_i}\) and strength \(\mu_i>0\) define, per `formulation.md` §2.3,
\[
J_i(x_i,\phi_i)=\ell_i(x_i,\phi_i)+\tfrac{\mu_i}{2}\|\phi_i-\phi_i^0\|^2.
\]
**Design decision:** the finite native SGD/momentum private updates descend \(J_i\) (the regularized train objective — this is what makes the private curvature controllable and positive definite, matching `QuadraticTask`'s convention that \(\mu\) is folded into `private_curvature`), while both the rerun-response and commit-response meta evaluations use the **unregularized** \(\ell_i^{\mathrm{meta}}\) (matching `formulation.md`'s \(F_i^{K,\mathrm{rerun}}\)/\(F_i^{K,\mathrm{commit}}\), which evaluate \(\ell_i\), not \(J_i\), at the meta batch). Reason for stating this explicitly: the task file's "quadratic losses with explicit private regularization" is satisfied by regularizing the *inner update objective*; regularizing the *outer evaluation* would silently change which operational object is being measured.

Write \(C_i=H_{\phi\phi}^i+\mu_iI\succ0\) (required, checked at construction, mirrors `CurvatureError`), \(c_i=b_i-\mu_i\phi_i^0\). The private gradient of \(J_i\) is
\[
\nabla_{\phi_i}J_i(x_i,\phi_i)=H_{\phi x}^ix_i+C_i\phi_i+c_i,\qquad H_{\phi x}^i=(H_{x\phi}^i)^\top.
\]

## 3. Graph-family construction

A family generator produces the task\(\times\)block incidence \(\mathrm{Inc}\in\{0,1\}^{m\times B}\), from which every \(P_i\) follows deterministically (union of blocks in row \(i\), in block order). Six families, each seeded and each required to realize a distinct point on the overlap/coupling range:

1. **disjoint** — blocks partitioned into \(m\) contiguous groups, one group per task; no two tasks share a block. Coupling rank across tasks is exactly 0; this is the "no negotiation needed" boundary control.
2. **full-overlap** — every task selects every block. Maximal coupling; the other boundary control.
3. **star** — a hub block set (first \(\lceil B/8\rceil\), at least 1) is selected by all tasks; the remaining blocks are partitioned disjointly into per-task spokes.
4. **chain** — blocks arranged on a line; task \(i\) (0-indexed) selects a sliding window of blocks with one block of overlap with task \(i+1\) (path topology in the bipartite task-block graph).
5. **partial** — Bernoulli incidence with a target overlap probability \(\rho_{\mathrm{ov}}\in[0.2,0.6]\), then repaired so every task selects \(\ge1\) block and at least one block is shared by \(\ge2\) tasks and at least one block is private to exactly one task (both regimes co-exist within one instance).
6. **random-sparse** — Bernoulli incidence with a sparsity probability \(p_{\mathrm{sp}}\in[0.05,0.2]\) (sparser than `partial`), repaired only for "every task selects \(\ge1\) block."

All six are validated post-generation: every generated \(P_i\) must pass the single-lift/no-duplicate contract (§1), and the realized incidence must match the family's defining property (e.g. `disjoint` asserts zero column has weight \(>1\); `full-overlap` asserts every column has weight \(m\)).

## 4. SGD state and recurrence

Fix task \(i\) inside one \(K\)-step response window, shared coordinate \(x_i\) held fixed throughout (per `formulation.md` §2.2, "\(K\) task-native private updates with the shared state held fixed"). Step size \(\eta_i>0\), noise draws \(\zeta_{i,0},\dots,\zeta_{i,K-1}\in\mathbb R^{d_i}\) (§8). State \(s_k=\phi_{i,k}\), transition
\[
\phi_{i,k+1}=T_{i,k}(\phi_{i,k},x_i;\zeta_{i,k})=\phi_{i,k}-\eta_i\big(H_{\phi x}^ix_i+C_i\phi_{i,k}+c_i+\zeta_{i,k}\big)=M_i\phi_{i,k}-\eta_i\big(H_{\phi x}^ix_i+c_i+\zeta_{i,k}\big),
\]
with \(M_i=I-\eta_iC_i\). This is the exact per-step Jacobian pair required by Theorem A (`2026-08-27-theory-breakthrough-audit.md` §4): \(J_k=M_i\) (constant across \(k\) since \(\eta_i\) and \(C_i\) are fixed for the window), \(B_k=-\eta_iH_{\phi x}^i\) (also constant). Both are exported per case even though they do not vary with \(k\) in this oracle, so that T160 can consume them directly as the "exact trajectory Jacobian."

**Analytic closed-form state.** With \(M_i=I-\eta_iC_i\),
\[
\phi_{i,K}=M_i^K\phi_{i,0}-\eta_i\sum_{j=0}^{K-1}M_i^{K-1-j}\big(H_{\phi x}^ix_i+c_i\big)-\eta_i\sum_{j=0}^{K-1}M_i^{K-1-j}\zeta_{i,j}.
\]
The constant-term geometric sum uses \(\sum_{l=0}^{K-1}M_i^l=(I-M_i^K)(\eta_iC_i)^{-1}\) (valid since \(C_i\succ0\), \(\eta_i>0\Rightarrow\eta_iC_i\) invertible); the noise term is a finite sum evaluated directly (noise varies per step, so no geometric shortcut is used or needed — this keeps the formula exact with no truncation).

**Analytic sensitivity.** \(Z_k=\partial\phi_{i,k}/\partial x_i\) obeys \(Z_{k+1}=M_iZ_k-\eta_iH_{\phi x}^i\), \(Z_0=0\) (the snapshot \(\phi_{i,0}\) is external to this window and independent of the candidate \(x_i\)). Closed form:
\[
Z_K=-\big(I-M_i^K\big)C_i^{-1}H_{\phi x}^i.
\]
(\(M_i^K\) and \(C_i^{-1}\) commute since \(M_i=I-\eta_iC_i\) shares eigenvectors with \(C_i\); the left-multiplication order relative to \(H_{\phi x}^i\) is forced by the recursion \(Z_{k+1}=M_iZ_k-\eta_iH_{\phi x}^i\), where \(M_i\) always multiplies \(Z_k\) on the left.)

## 5. Momentum state and recurrence

Heavy-ball momentum, buffer \(v_{i,k}\in\mathbb R^{d_i}\), momentum coefficient \(\beta_i\in[0,1)\), augmented state \(u_k=(\phi_{i,k},v_{i,k})\in\mathbb R^{2d_i}\):
\[
v_{i,k+1}=\beta_iv_{i,k}+\big(H_{\phi x}^ix_i+C_i\phi_{i,k}+c_i+\zeta_{i,k}\big),\qquad\phi_{i,k+1}=\phi_{i,k}-\eta_iv_{i,k+1}.
\]
Substituting gives the affine block form
\[
u_{k+1}=A_iu_k+b_i(x_i)+n_k,\qquad
A_i=\begin{pmatrix}I-\eta_iC_i&-\eta_i\beta_iI\\C_i&\beta_iI\end{pmatrix},\quad
b_i(x_i)=\begin{pmatrix}-\eta_i(H_{\phi x}^ix_i+c_i)\\H_{\phi x}^ix_i+c_i\end{pmatrix},\quad
n_k=\begin{pmatrix}-\eta_i\zeta_{i,k}\\\zeta_{i,k}\end{pmatrix}.
\]
This is again a constant-Jacobian window: \(J_k=A_i\), \(B_k=\mathrm dbdx_i=\begin{pmatrix}-\eta_iH_{\phi x}^i\\H_{\phi x}^i\end{pmatrix}\), exported per case for T160 exactly as in the SGD case.

**Analytic closed-form state.** \(u_K=A_i^Ku_0+\sum_{j=0}^{K-1}A_i^{K-1-j}b_i(x_i)+\sum_{j=0}^{K-1}A_i^{K-1-j}n_j\). \(A_i\) need not be symmetric or diagonalizable with a convenient closed inverse in general (it can be defective at \(\beta_i=0\) boundary or near-unstable spectra), so the constant-term sum is accumulated by direct repeated matrix multiplication (cheap: \(K\le10\), \(d_i\le32\)) rather than by a matrix-inverse shortcut. This keeps the closed form robust to the deliberately-unstable configurations in §8, where \((I-A_i)\) may be ill-conditioned or singular.

**Analytic sensitivity.** \(W_k=\partial u_k/\partial x_i\in\mathbb R^{2d_i\times p_i}\), \(W_{k+1}=A_iW_k+B_k\), \(W_0=0\); \(Z_K^\phi\) is the top \(d_i\) rows of \(W_K\) (the \(\phi\)-block of the sensitivity, used below exactly as \(Z_K\) is used for SGD).

## 6. Rerun-response and commit-response

Let \(\phi_i^K(x_i)\) denote either the SGD or momentum closed-form state at horizon \(K\) (§4/§5), evaluated with the noise sequence frozen at its seeded realization. By construction \(\phi_i^K(x_i)\) is an **exact affine function of \(x_i\)** for fixed noise (both recurrences are affine in \(x_i\) with time-invariant Jacobians). Consequently, since \(\ell_i^{\mathrm{meta}}\) is exactly quadratic in \((x_i,\phi_i)\) jointly, the composition
\[
F_i^{K,\mathrm{rerun}}(\theta)=\ell_i^{\mathrm{meta}}\big(x_i,\phi_i^K(x_i)\big),\qquad x_i=P_i\theta,
\]
is an **exact quadratic function of \(x_i\)** — not a local approximation. Substituting \(\phi_i^K(x_i)=Z_Kx_i+r_K\) (SGD; use \(Z_K^\phi\) and the corresponding affine remainder for momentum) and expanding:
\[
F_i^{K,\mathrm{rerun}}(\theta)=\mathrm{const}+\hat g_i^\top x_i+\tfrac12x_i^\top Q_i^Kx_i,
\]
\[
\hat g_i=a_i^{\mathrm{meta}}+H_{x\phi}^ir_K+Z_K^\top b_i^{\mathrm{meta}}+Z_K^\top H_{\phi\phi}^ir_K,\qquad
Q_i^K=H_{xx}^i+H_{x\phi}^iZ_K+Z_K^\top H_{\phi x}^i+Z_K^\top H_{\phi\phi}^iZ_K.
\]
The gradient at the current operating point \(x_i\) is
\[
\nabla F_i^{K,\mathrm{rerun}}=\big[a_i^{\mathrm{meta}}+H_{xx}^ix_i+H_{x\phi}^i\phi_i^K\big]+Z_K^\top\big[b_i^{\mathrm{meta}}+H_{\phi x}^ix_i+H_{\phi\phi}^i\phi_i^K\big],
\]
lifted to global coordinates once as \(P_i^\top\nabla F_i^{K,\mathrm{rerun}}\) (single lift, per `formulation.md` §3's "No second lift is applied").

The **commit-response** counterfactual (`formulation.md` §2.2) fixes \(\phi_i^K\) at the current point and evaluates a candidate \(x_i'\):
\[
F_i^{K,\mathrm{commit}}(\theta';\theta)=\ell_i^{\mathrm{meta}}(x_i',\phi_i^K(x_i)),\qquad\text{derivative at }x_i'=x_i:\quad
\nabla F_i^{K,\mathrm{commit}}=a_i^{\mathrm{meta}}+H_{xx}^ix_i+H_{x\phi}^i\phi_i^K.
\]
This yields an exact, exportable decomposition — the **compensation gap contributed by the finite private response**:
\[
\nabla F_i^{K,\mathrm{rerun}}-\nabla F_i^{K,\mathrm{commit}}=Z_K^\top\big[b_i^{\mathrm{meta}}+H_{\phi x}^ix_i+H_{\phi\phi}^i\phi_i^K\big].
\]
Per the frozen protocol, \(F_i^{K,\mathrm{rerun}}\) is the primary operational target that the oracle validates against; \(F_i^{K,\mathrm{commit}}\) is computed and reported as a separate labeled reference, never substituted for the rerun value in the pass/fail gate.

**Exact realized loss change.** For any candidate shared step \(d\) (i.e. \(\theta\to\theta+d\), \(x_i\to x_i+P_id\)), because \(F_i^{K,\mathrm{rerun}}\) is exactly quadratic in \(x_i\),
\[
\Delta F_i^{K,\mathrm{rerun}}(d)=\big(\nabla F_i^{K,\mathrm{rerun}}\big)^\top(P_id)+\tfrac12(P_id)^\top Q_i^K(P_id)
\]
holds with **zero truncation error** — this is an algebraic identity, not a second-order approximation, and is the basis of the direct-evaluation cross-check in §7(c).

## 7. Three independent reference methods

No autodiff library is available under `pyproject.toml` (`numpy`, `scipy`, `PyYAML`, `jsonschema` only; `pyproject.toml` is outside T155's `allowed_paths` so no new dependency may be added). All three references below are implemented from these primitives, as separate code paths from §4–§6's closed forms, so that a coding error in one is unlikely to be masked by the other:

**(a) Independently implemented reverse-mode differentiation over the literal unroll.** Forward pass stores every intermediate \(\phi_{i,0},\dots,\phi_{i,K}\) (and \(v_{i,0},\dots,v_{i,K}\) for momentum) using the literal per-step update formula of §4/§5 (not the closed-form matrix-power shortcut). Backward pass seeds the adjoint at \(k=K\) from \(\ell_i^{\mathrm{meta}}\)'s exact partials (\(\lambda_K=H_{x\phi}^{i\top}x_i+b_i^{\mathrm{meta}}+H_{\phi\phi}^i\phi_{i,K}\), plus the direct term \(a_i^{\mathrm{meta}}+H_{xx}^ix_i+H_{x\phi}^i\phi_{i,K}\) added once at the end), then propagates \(\lambda_k=M_i^\top\lambda_{k+1}\) (SGD) or the transpose of \(A_i\) (momentum) backward through the stored trajectory, accumulating \(\sum_k\lambda_{k+1}^\top B_k\) into \(dL/dx_i\). This is a genuinely separate implementation (forward-store-then-backward-accumulate) from the forward-only \(Z_k\)/\(W_k\) recursion of §4/§5, even though both compute the same exact mathematical quantity.

**(b) Central finite differences.** Using the *same* frozen noise realization, literally simulate the forward recursion (the unroll function, not the closed form) at \(x_i+hd\) and \(x_i-hd\) for a fixed probe direction \(d\), evaluate \(\ell_i^{\mathrm{meta}}\), and form \(\big(F(x_i+hd)-F(x_i-hd)\big)/(2h)\). Because \(F_i^{K,\mathrm{rerun}}\) is exactly quadratic (§6), central differencing has **zero truncation error** in exact arithmetic — the only error source is floating-point roundoff, which grows for very small \(h\) (catastrophic cancellation in the numerator) and is negligible for moderate \(h\). Step sizes \(h\in\{10^{-2},10^{-3},10^{-4},10^{-5},10^{-6},10^{-7}\}\) are swept and reported; the preregistered stability envelope is \(h\in[10^{-6},10^{-2}]\), required to match the analytic directional derivative \(\big(\nabla F_i^{K,\mathrm{rerun}}\big)^\top(P_id)\) to relative error \(\le10^{-6}\); \(h<10^{-6}\) is reported but not required to pass, since roundoff-dominated degradation there is expected and not a defect.

**(c) Direct loss evaluation.** Literally simulate the forward recursion at baseline \(\theta\) and at \(\theta+d\) for the same probe directions used for the run's Pareto references, evaluate \(\ell_i^{\mathrm{meta}}\) at both, and take the difference. This is compared against the exact closed-form \(\Delta F_i^{K,\mathrm{rerun}}(d)\) of §6, which is an algebraic identity for this model class — so the only expected discrepancy is floating-point roundoff, giving an unusually strict (near machine-precision) cross-check rather than an approximation-quality check.

## 8. Parameter ranges and seed policy

Dimensions are narrowed from the full task-file ranges for CPU runtime, preserving every required mechanism (reason stated per row):

| Parameter | Full range (task file) | Resolved range for this run | Reason |
|---|---|---|---|
| tasks \(m\) | 2–8 | \(\{2,3,4,6,8\}\), one value per family, covering both ends | small grid keeps the sweep enumerable while touching every family/size combination the gate cares about |
| blocks \(B\) | 4–64 | \(\{4,8,16,32,64\}\) tied to \(m\) per family | spans the full range without a full Cartesian product |
| block width \(w_b\) | unconstrained | \(\{1,2\}\) | keeps global dim \(p\le128\), cheap dense linear algebra |
| private dim \(d_i\) | 2–32 | \(\{2,4,8,16,32\}\), sampled per task | full range covered, not exhaustively combined with every other axis |
| \(K\) | \(\{1,3,5,10\}\) | all four, required on every family/optimizer cell | task file requires all four explicitly |
| optimizer | SGD, momentum (Adam-like stretch) | SGD; momentum with \(\beta\in\{0.5,0.9\}\) | Adam-like is explicitly a stretch deliverable gated on SGD+momentum passing first; deferred (see §11) |
| stability regime | stable, deliberately unstable | both, every family/optimizer/K cell | required boundary control; unstable regime uses spectral radius of \(M_i\) (SGD) or \(A_i\) (momentum) in \((1.05,2.0)\), moderate enough that \(K\le10\) does not overflow float64 |
| curvature condition number | controllable | log-spaced eigenvalues, condition number \(\in\{1,10,10^3\}\) | spans well-conditioned to ill-conditioned without extreme values that would need higher precision |
| coupling rank of \(H_{x\phi}^i\) | controllable | \(\mathrm{rank}\in\{1,\min(p_i,d_i)\}\) (low-rank and full-rank endpoints) | endpoints exercise both regimes cheaply |
| shared-gradient cosine (task pairs sharing a block) | controllable | targets \(\{-0.8,0,0.8\}\) realized by Gram–Schmidt construction of \(a_i\) restricted to the shared coordinates | covers conflicting, orthogonal, aligned |
| gradient scale ratio (task pairs) | controllable | targets \(\{1,10,100\}\) | spans balanced to heavily imbalanced |
| noise model | Gaussian, block-correlated | both, every cell | required by minimum model spec |
| seeds per cell | — | 3 | statistical assurance without combinatorial blow-up |

**Noise models.** (i) Gaussian: \(\zeta_{i,k}\sim\mathcal N(0,\sigma_i^2I)\), i.i.d. across \(k\). (ii) Block-correlated: the private dimension \(d_i\) is partitioned into sub-blocks of width \(\lceil d_i/4\rceil\); covariance \(\Sigma_i=\sigma_i^2\big((1-\rho_i)I+\rho_iG_i\big)\) where \((G_i)_{jl}=1\) iff coordinates \(j,l\) share a private sub-block, \(\rho_i\in\{0.3,0.7\}\); \(\zeta_{i,k}=\mathrm{chol}(\Sigma_i)\varepsilon_k\), \(\varepsilon_k\sim\mathcal N(0,I)\) i.i.d. across \(k\), fully deterministic given the seed.

**Seed policy.** One top-level config seed spawns independent `numpy.random.SeedSequence` child streams (via `.spawn`), labeled separately for: graph-family structure, curvature spectra, \(H_{x\phi}\) construction, gradient-angle/scale construction, per-task per-step noise, meta-batch coefficients, and finite-difference probe directions. This guarantees that changing one axis (e.g. adding a noise-model variant) does not perturb any other axis's realized values for a fixed top-level seed, and that every case is exactly reproducible from `(config_seed, case_index)`.

**Near-zero handling.** A reference quantity with norm \(\le10^{-10}\) is treated as near-zero; comparisons against it require absolute error \(\le10^{-11}\) instead of relative error. This is the literal boundary stated in the T155 pass/fail gate and is applied uniformly to state, hypergradient, and loss-change comparisons.

## 9. Numerical tolerances (pass/fail gate, restated with the near-zero rule)

| Comparison | Tolerance |
|---|---|
| analytic state (§4/§6 closed form) vs. independently unrolled state (literal step-by-step loop, §7's forward pass) | relative error \(\le10^{-10}\); absolute \(\le10^{-11}\) if reference norm \(\le10^{-10}\) |
| analytic hypergradient (§6) vs. independently implemented reverse-mode differentiation (§7a) | relative error \(\le10^{-9}\); absolute \(\le10^{-11}\) if reference norm \(\le10^{-10}\) |
| central finite-difference directional derivative (§7b) vs. analytic directional derivative | relative error \(\le10^{-6}\) for \(h\in[10^{-6},10^{-2}]\) (preregistered stability envelope); degradation outside this envelope is recorded, not gate-blocking |
| direct rerun-response loss change (§7c) vs. exact closed-form \(\Delta F_i^{K,\mathrm{rerun}}(d)\) (§6) | relative error \(\le10^{-9}\); absolute \(\le10^{-11}\) if reference norm \(\le10^{-10}\) |
| selector contract | exact (boolean check, no tolerance) |

All tolerance values above are taken verbatim from the task file's pass/fail gate, except the finite-difference envelope, which the gate leaves to be "preregistered" here; \(10^{-6}\) over \(h\in[10^{-6},10^{-2}]\) is chosen because §6 shows the true truncation error is exactly zero for this model class, so any tolerance looser than double-precision roundoff is a deliberately conservative margin, not evidence of model error. Deliberately unstable cases (§8) are still subject to the same tolerances on relative error — instability changes the magnitude of the trajectory, not the exactness of the algebra — except where a state or gradient overflows finite float64 range, in which case the case is marked `unstable-overflow` in the failure ledger rather than scored as a numeric mismatch.

## 10. Expected CPU time, memory, and output size

Grid size estimate: 6 families \(\times\) 2 optimizers (SGD, momentum with 2 \(\beta\) values counted together) \(\times\) 4 \(K\) values \(\times\) 2 stability regimes \(\times\) 3 seeds \(\approx\) 300–350 cases, each with \(p\le128\), \(d_i\le32\). Every case does \(O(K)\) dense matrix operations on matrices no larger than \(32\times32\) (or \(64\times64\) augmented for momentum) plus one finite-difference sweep of 6 step sizes. Expected wall time: low tens of seconds to a few minutes total on a single CPU core (no parallelism required at this scale); expected peak memory: well under 200 MB. Expected total output size: a per-case summary record (family, dimensions, seed, optimizer, hashes, pass/fail, errors) of a few KB each, totalling a few MB for the full sweep; a small selected subset (5–10 cases spanning families/optimizers/K, see §11) additionally persists full trajectories and Jacobians as the detailed reference dataset, adding at most a few more MB. These are expectations to be checked against measured numbers in the T155 completion report; if measured cost materially exceeds this estimate the run is throttled (fewer seeds, not fewer families or narrower ranges) and the throttling is recorded in the failure ledger.

## 11. Handoff artifacts for T160 and T170

Every case (whether it passes or fails) records, per the task file's artifact requirements: graph family; selector hashes (sha256 of each canonical \(P_i\) encoding); dimensions (\(m,B,w_b,d_i\)); seed; optimizer and horizon \(K\) (and \(\beta_i\) for momentum); realized curvature spectra and condition number; realized coupling rank; realized gradient cosine/scale versus target; noise model and parameters; stability regime and realized spectral radius; source revision; configuration hash; exact state, hypergradient, and loss-change references with their cross-check errors against all three §7 methods; pass/fail per check; output hash.

Because storing full per-step trajectories and Jacobians for the entire ~300-case sweep would be disproportionate to what T160 needs, the full sweep persists only the scalar/hash summary above, while a curated subset of 5–10 representative cases (spanning every family, both optimizers, all four \(K\) values, and both stability regimes) additionally persists the full detailed reference dataset that T160 §2 ("import and independently verify the accepted T155 exact references") is expected to consume directly: the complete state trajectory \(\phi_{i,0..K}\) (and \(v_{i,0..K}\)), the per-step Jacobian pair \((J_k,B_k)\) (constant in this oracle, still exported per step for forward compatibility with a future time-varying oracle), the sensitivity trajectory \(Z_{0..K}\) (or \(W_{0..K}\)), and the exact \(\hat g_i,Q_i^K\) pair. This scoping choice is stated here per the task file's "first report may narrow dimensions for runtime, but... must state the reason": the narrowing is in artifact volume, not in mechanism coverage, and every mechanism in the minimum model specification is still realized by at least one persisted case with full detail.

T170 (graph-localized robust descent and resource allocation) additionally needs, from this oracle, the boundary-control behavior of `disjoint` and `full-overlap` on coupling rank and hypergradient magnitude — both are already first-class families here (§3), so no separate artifact is required beyond the per-case summary.

Each case record also carries a `pareto_reference`: an independent exact/high-accuracy common-descent (Pareto) reference over the tasks' real lifted exact rerun-response gradients (the \(\hat g_i\) from §6, lifted through each task's real selector \(P_i\), not a random probe direction), including the active-set combined gradient, its KKT/projection residual, and disjoint/partial/full-overlap boundary behavior — see `src/comppareto/oracle/pareto.py`. This is a case-level artifact, not a per-task one, since the common-descent direction is defined jointly over all tasks sharing the global state.

The run manifest written by `sweep.py` is split into two files to satisfy `schemas/run-manifest.schema.json` (which requires a single JSON object, not an array): `manifest.json` holds the schema-valid top-level envelope (`schema_version`, `run_id`, `task_id`, `run_kind`, `source_revision`/`execution_revision`, `dirty`, `config_sha256`, `environment`, `status`, `result_files`, `artifacts`, `retry`), and `case-records.json` holds the flat per-case array described above (one record per `case_index`, each carrying its own `pareto_reference` and, for the detailed subset, per-task `detail`).

## 12. Deferred: Adam-like diagonal state

Per the frozen protocol, "a diagonal Adam-like state is a stretch deliverable and may start only after the SGD and momentum gates pass." Given the CPU budget estimated in §10 is already committed to the SGD+momentum sweep across six families, four horizons, two stability regimes, and two noise models, the Adam-like extension is explicitly deferred out of this execution pass. This is recorded here rather than silently dropped, per the task file's requirement to state ranges and mechanisms even when narrowed.
