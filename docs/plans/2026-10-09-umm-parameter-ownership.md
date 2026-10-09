# UMM post-training core: parameter ownership and plasticity

Status: research hypothesis and proposed protocol; no empirical result is claimed here.

## Core question

In a unified understanding-and-generation model, which **shared** parameter
groups support each capability, and does this ownership predict the effects of
pruning and restricted post-training? The intended result is a causal account
of when shared capacity enables transfer, when it causes interference, and
whether that account improves decisions at a fixed training budget.

Janus-Pro-1B is the first base model. Use the public CoRL code, data, reward
and evaluation path as the reproducible joint-GRPO environment. Reproducing a
semantically valid CoRL Unified-only reference remains an independent goal and
the admission gate for claims about post-training. A second architecture is a
later generality test, not a prerequisite for the first mechanism screen.

## Analysis unit and vocabulary to freeze before measurement

The main unit is a **structured, reversible group in the shared backbone**:
an attention head including its input/output projections, or an MLP
intermediate channel including both projections. Compare groups within layer
and type; measure both parameter count and actual inference cost. Exclude
task-private encoders, decoders, and heads from the ownership map. Record
their separate perturbation results as architecture controls. Do not call an
individual scalar parameter causally important because its magnitude or
gradient is large.

For group `g`, capability `c`, checkpoint `theta`, intervention `I_g`, and a
frozen evaluation set, define the intervention effect

`D_c(g) = quality_c(theta) - quality_c(I_g(theta))`.

`I_g` initially means a temporary zero gate on the complete group, with no
recovery training. For generation, fix prompts and sampling seeds and report
paired metric differences; for understanding, fix examples and decoding. A
larger positive `D` means removal damaged that capability. Negative values
mean improvement; values near zero are inconclusive until uncertainty is
measured. This is *checkpoint-, data-, metric-, and intervention-specific*
importance, not permanent parameter ownership. Recoverability after pruning
and usefulness for learning are separate outcomes.

Estimate uncertainty by paired resampling over evaluation items/prompts and
repeat across at least two disjoint calibration splits. Freeze a practical
damage threshold for each metric from evaluator noise and random-group
controls before inspecting the ownership map. Classify a group as important
for `c` only if its interval clears that threshold; unimportant only if its
interval lies inside a prespecified equivalence band. Everything else is
**unresolved**. Do not force every group into four boxes.

The four interpretable categories are understanding-selective, generation-
selective, shared-critical, and low-impact-for-this-test. The last name does
not mean generally redundant. Report the continuous `(D_U, D_G)` scatter and
layer/type distribution alongside categorical counts. Since groups interact,
also test a bounded set of paired and cumulative removals; single-group
effects cannot be added to predict a pruning curve.

For matched capability pairs, define a finer vector of effects: e.g. spatial
relation questions and spatial relation generation prompts describing the
same relation. This allows a test of concept-level shared support rather than
relying on one understanding and one generation aggregate score. The paired
set must be separate from CoRL training data and ownership calibration data.

## Causal experiment sequence

1. **Admit the platform.** Complete T711 semantic review, a real end-to-end
   corrected CoRL run, and a frozen Unified-only reproduction with exact
   checkpoint/evaluator provenance. Establish U-only and G-only headroom.
   These are gates, not evidence for ownership by themselves.
2. **Map and audit.** At the common pre-RL checkpoint, perturb shared groups
   on calibration splits. Measure U/G aggregate and paired capability slices,
   uncertainty, random and magnitude-based controls, repeated sampling, and
   stability across splits. Freeze group selection on calibration data only.
3. **Pruning test.** At equal removed parameter count and layer/type mix,
   compare selective categories with random and conventional importance
   ranking; report actual latency and memory as outcomes. Report immediate
   damage and, separately, recovery
   after the same CoRL training budget. Count actual shared and private
   parameter changes.
4. **Restricted post-training test.** Train U-only, G-only, and joint CoRL
   variants from the identical checkpoint with only prespecified shared
   groups open. Match trainable parameter count, task exposure, rollouts,
   reward calls, optimizer rule, search allowance, and GPU time. Include a
   random-group control, full-shared baseline, and private-only branch where
   private modules are trainable. Evaluate target gains, cross-task losses or
   gains, and paired concept transfer on held-out sets.
5. **Generalization.** Repeat only the decisive mapping/intervention tests on
   another admitted UMM. Compare shared backbone groups that have analogous
   roles; report architecture-specific effects explicitly.

The central prediction is that measured ownership should forecast both
pruning damage and the *direction* of cross-task effects from restricted
training better than random, magnitude, or raw-gradient baselines. This is a
hypothesis, not an assumption. Pruning dependence and learning plasticity may
diverge, which is itself a useful mechanistic result.

## Priority and decision gates

P0: CoRL semantic admission and Unified-only reproduction; group registry and
frozen U/G evaluators; a small stability/perturbation pilot. P1: locked
category-guided pruning and restricted post-training against matched controls.
P2: second architecture and large multi-seed confirmation after the pilot
identifies a reproducible signal.

Existing gradient negotiator and private-response experiments become
**conditional mechanism follow-ups**. Run them only if the ownership pilot
shows a concrete interference pattern that simple restricted updates do not
resolve. The prior theoretical programme remains archived/parallel research,
not an empirical claim required for this paper's first result. Do not launch
new GPU tasks or revise authorized task states solely from this planning note;
freeze detailed task contracts, budgets, and gates after the analysis-unit
decision with the user.

Advance from mapping to expensive training only if classification is stable
across calibration splits and perturbation controls reveal nontrivial U/G
variation. Advance to a practical ownership-guided method only if held-out
pruning or training improves a prespecified joint criterion at matched budget.
If categories are unstable or fail to predict outcomes, report that negative
result and test interaction/reorganization rather than retrofitting labels.
