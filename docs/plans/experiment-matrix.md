# Experiment matrix and baseline fairness

## Required method matrix

| Family | Method | Purpose |
|---|---|---|
| Oracles | base checkpoint; understanding-only; generation-only | Establish headroom and normalization |
| Isolation | private-only with shared frozen; shared-only with private frozen | Identify where adaptation capacity resides |
| Sampling | alternating batches; proportional and temperature sampling | Separate data scheduling from gradient geometry |
| Scalarization | static weights; uncertainty/GradNorm-style weights | Strong tuned joint-training baseline |
| Pareto/gradient | MGDA, PCGrad, CAGrad, Nash-MTL, FAMO, ConFIG | Standard multi-task optimization |
| Private-aware | ConsMTL; representation-space negotiation | Closest method family |
| Same-inner bilevel | FORUM/gMOBA/WC-MHGD or closest executable MOBLO; MGDA/normalized Chebyshev/Nash applied to the identical \(A_i^K\) hypergradients | Isolate compensation estimator from negotiation rule |
| Expert integration | task arithmetic/TIES, TATR, multi-teacher OPD when feasible | Alternative train-then-combine paradigm |
| Proposed | no compensation; stop-gradient; unroll-1/3/5; implicit/CG | Isolate each CompPareto component |

## Fairness protocol

- Every headline comparison is run under **both** Protocol A and Protocol B below; neither is silently substituted for the other.
- Give tuned scalarization and each adaptive baseline the same outer search budget as CompPareto.
- Select hyperparameters on validation metrics using worst-task normalized gain; never tune on final test metrics.
- Reuse data orders and random seeds across methods.
- Report method overhead separately from base forward/backward cost.
- Evaluate the last preregistered checkpoint and a validation-selected checkpoint for every method; do not choose different rules per method.

### Protocol A: equal total compute/data budget

| Accounted item | CompPareto | Baselines |
|---|---|---|
| Forward/backward evaluations | Inner adaptation, meta-gradient, acceptance evaluation, and shared update all count | May spend the same total evaluations on extra native task steps or shared updates |
| Training examples/rollouts | Inner and meta examples count even when the inner state is rolled back | Receive the same per-task sample/rollout budget |
| Private transition | Virtual: parameter and optimizer state are restored after estimating the response | Baselines may use the corresponding compute on virtual lookahead or additional training; both are reported |
| Stopping | Stop at the first of matched gradient-evaluation budget, measured FLOPs budget, or wall-clock cap | Same cap and hardware allocation |
| Search | GPU-hours for hyperparameter search count separately and are equalized | Same search GPU-hours |

Protocol A answers whether the method is worthwhile at equal cost. It is reported as two separate tables—A-FLOPs and A-Wallclock—not a compound “first cap reached” result. Before launch, every method receives a committed budget allocation file derived from `configs/budget_protocol.example.json`; unused budget cannot be reassigned after validation metrics are seen. FLOPs are estimated from profiler traces, and gradient evaluations remain the architecture-independent ledger.

### Protocol B: equal persistent shared-update budget

| Accounted item | CompPareto | Baselines |
|---|---|---|
| Shared updates | Same number and same scheduled shared learning rates | Same |
| Persistent private updates | Exactly \(K\) per shared update when the configured transition is committed | Every baseline receives exactly the same task-private update schedule |
| Virtual response steps | Rolled back and charged as overhead | Baselines do not receive hidden persistent updates; overhead is reported explicitly |
| Data | Persistent training samples matched; virtual/meta samples reported as additional estimator data | Same persistent samples; optional equal-extra-data control included |
| Outcome | Quality per shared update and estimator overhead | Same metrics |

Protocol B isolates update geometry but intentionally does not claim equal compute. A third ablation reuses the same inner batch as meta-batch to measure the cost of disjoint meta-data, but it is not the main result because of adaptation overfitting.

### State semantics

- The default diagnostic estimator is **virtual**: both \(\phi_i\) and \(\omega_i\) are restored after estimating the response.
- Persistent task-private training occurs through an explicit common schedule shared by every method.
- An estimator that commits inner updates is named `persistent-inner` and cannot be compared to a baseline lacking the same private updates.
- ConsMTL, scalarization, and every \(K\)-step CompPareto variant use the same persistent private-update allowance under Protocol B.

### Search freeze

- Each method receives the same number of completed trials and the same maximum search GPU-hours.
- Candidate parameter names, ranges/discrete values, sampler seed, per-trial early-stop rule, and maximum retries are committed before launch.
- Infrastructure failure retries reuse the same configuration and seed; numerical divergence counts as a completed failed trial unless a globally preregistered stability rule applies to every method.
- The saved budget allocation—not an after-the-fact choice—states whether baseline surplus compute becomes native private steps, shared updates, or evaluation.

## Ablations

1. Remove private adaptation.
2. Replace compensated curvature with raw shared Hessian/diagonal.
3. Remove loss-scale normalization.
4. Replace max-min retained gain with weighted sum.
5. Remove block-overlap selectors and treat all parameters as shared.
6. Vary private response depth and compute budget.
7. Compare full shared-layer tuning with LoRA on the same blocks.
8. Replace paired semantic batches with independently sampled tasks.

## Statistics

- Three seeds are an exploratory minimum, not sufficient by default for a confirmatory positive 95% training-seed interval. A variance-only dry run estimates between-seed variance without comparing methods; a preregistered power calculation then selects at least five confirmatory seeds or reports the detectable effect if more are infeasible.
- Use a hierarchical bootstrap: resample training seeds first and evaluation examples within seed second. An example-only bootstrap must not be presented as uncertainty over training randomness.
- Report per-seed results, mean, standard deviation, and 95% confidence intervals.
- For many capability slices, control false discovery rate or label slice analysis exploratory.
- Report effect sizes, not only significance.
- Missing/failed runs remain in the ledger with failure reason and retry rule.

## Result schema

Every formal run note records:

- git revision and dirty-state flag;
- base checkpoint and hash;
- model adapter and active parameter blocks;
- dataset manifests and metric revisions;
- task update counts, tokens/samples, and GPU hours;
- all hyperparameters and search parent ID;
- per-task and per-slice metrics;
- failure, retry, and checkpoint-selection status.
