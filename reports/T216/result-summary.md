# Result Summary — T216

## Status

K=1 sweep **complete**. `runs/alternating-showo2-v1/manifest.json` → `status: pass`;
`run_k1.py` exit 0 on 2026-09-28 (container `H20-FoldUMM`, GPU index 3, torch 2.5.1+cu124,
`elapsed_seconds: 26.9`, `gpu_hours: 0.0075`).

**K=1 is a mandatory gate, not a gate pass.** The T216 gate decision requires K=3.
Nothing in this run establishes that any negotiator "supports the gate".

## What was measured

K=1 P0–P3 alternating-protocol diagnostic on Show-o2-1.5B
(`showlab/show-o2-1.5B`, checkpoint revision `07ec16589d4fc5422a74dddbbc4b2cd11e551039`).

Subspaces (prefixes verified against the real state dict):
- Shared: `showo.model.layers.27` (last Qwen2.5-1.5B decoder layer, 46,797,824 params)
- Understanding-private: `und_trans.layers.7` (15,239,504 params)
- Generation-private: `diffusion_head_a.9` (85,999,744 params)

Protocols: P0 (simultaneous baseline, 24 rows), P1 (private-only control, 2 rows),
P2 (shared-then-private, 24 rows), P3 (virtual private-then-shared commit, 12 rows).
Negotiators: `raw_sum`, `normalized_sum`, `pcgrad`, `mgda` (P3 instrumented for
`raw_sum` and `mgda` only, by design). Step-scale grid: `[5e-6, 5e-5, 5e-4]`.
Master seed 216, fixed batches throughout, no `create_graph`, no mid-run RNG reseed,
no persistent joint training, no full-backbone unroll.

## Measured results

Row inventory (`runs/alternating-showo2-v1/raw_rows.json`, 62 rows = 60 mandatory P0/P2/P3 + 2 P1 control):

| Counter | Value |
|---------|-------|
| `numerical.total_rows` | 60 |
| `numerical.nonfinite_mandatory_rows` | 0 |
| `attribution.mandatory_rows_missing_control` | 0 |
| `snapshot_restore.failed` | 0 |
| `persistent_updates` | 0 |
| Attribution identity violations (recomputed: `delta_total != delta_private + delta_controlled`) | 0 / 60 |
| `protocols.*_measured` | all four `true` |

P1 control (private-only descent is the reference arm):

| Task | loss_before | loss_after | delta_private |
|------|-------------|-----------|---------------|
| und | 7.59375 | 7.53125 | -0.0625 |
| gen | 0.0869140625 | 0.08642578125 | -0.00048828125 |

`delta_controlled` (the effect attributable to the shared-subspace displacement,
measured against the P1 control arm) per protocol/task:

| Protocol | Task | rows | `delta_controlled == -0.03125` | `delta_controlled == 0.0` |
|----------|------|------|-------------------------------|---------------------------|
| P0 | und | 12 | 0 | 12 |
| P0 | gen | 12 | 0 | 12 |
| P2 | und | 12 | 8 | 4 |
| P2 | gen | 12 | 0 | 12 |
| P3 | und | 6 | 3 | 3 |
| P3 | gen | 6 | 0 | 6 |

Findings that follow directly from these numbers:

1. **P0 (simultaneous) shows no measurable shared effect anywhere.** All 24 P0 rows have
   `delta_controlled == 0.0`, for every negotiator and every step scale, on both tasks.
   P0's `delta_private` equals the P1 control `delta_private` exactly in every row.
2. **P2 and P3 produce a response only on the understanding task, and only at ±1 bf16 ULP.**
   On `und`, 8/12 P2 rows and 3/6 P3 rows move by -0.03125; the rest are 0.0.
   On `gen`, all 30 P0/P2/P3 rows are bit-identical (`loss_after == 0.08642578125`
   in every one of the 62 rows): the generation task shows no measurable response to
   any treatment in this sweep.
3. **P2 and P3 are near-duplicates at K=1.** Of the 12 directly comparable cells
   (P3's `raw_sum` and `mgda`), 11 agree and 1 disagrees: `(mgda, eta=5e-6, und)`
   is -0.03125 under P2 and 0.0 under P3. Since P2 and P3 apply the shared
   displacement from the same snapshot and then the same K=1 private step, and their
   code paths differ only in which directions are instrumented, this single-cell
   disagreement is at the readout resolution limit rather than evidence of an
   ordering effect.
4. **All measured deltas are at bf16 readout resolution.** The recorded losses are
   bf16: `nextafter` spacing is 0.03125 at 7.59375 (und) and 0.000488281 at
   0.0869140625 (gen). So the und responses (-0.03125) are exactly 1 ULP,
   `delta_private` on und (-0.0625) is 2 ULP, and every gen delta is 0 or 1 ULP.
   No observed effect exceeds 2 ULP of the quantity being read.

## Precision caveat (why the P0 zeros should not be read as "shared gradient has no effect")

Directions are unit-normalized over the shared subspace before stepping
(`unit_normalize` in `protocols.py`, then `p.add_(delta, alpha=eta)`), so the
per-parameter displacement is `eta / sqrt(46,797,824)`:

| `eta` | per-parameter RMS displacement | params that can change at all in bf16 (`|w| < delta / 2^-9`) |
|-------|-------------------------------|-------------------------------------------------------------|
| 5e-6 | 7.31e-10 | `|w| < 3.7e-7` |
| 5e-5 | 7.31e-9 | `|w| < 3.7e-6` |
| 5e-4 | 7.31e-8 | `|w| < 3.7e-5` |

bf16 has 8 explicit mantissa bits, i.e. relative spacing `2^-8`; a `bfloat16`
accumulator therefore leaves `w` unchanged when `|delta| < |w| * 2^-9`. Verified
directly on CPU: `bfloat16(0.1) + 7.31e-8 == bfloat16(0.1)` (unchanged), while
`bfloat16(1e-5) + 7.31e-8` does change. So across the whole step-scale grid the
shared displacement is rounded away for all but the smallest-magnitude shared
weights — the treatment is largely numerically inert rather than scientifically null.

This is stated as the mechanism most consistent with the observed zeros; it was
**not** directly instrumented on the container weights (the checkpoint lives on the
container's `/dockerdata` SSD, not accessible from the reporting host), so it should be
confirmed before any K=3 run — e.g. by counting how many shared elements actually
change under `add_`, or by re-running with fp32 master weights or a larger `eta`.

## What K=1 does and does not establish

Establishes:
- All four negotiators produce finite directions and finite losses on both tasks at K=1
  (0/60 non-finite mandatory rows).
- The attribution identity `delta_total = delta_private + delta_controlled` holds exactly
  on all 60 mandatory rows.
- The snapshot/RNG restore discipline leaves no residue: `persistent_updates == 0`,
  `snapshot_restore.failed == 0`, byte-exact restore asserted with `atol=rtol=0`.
- No mandatory row is missing its P1 control baseline.

Does **not** establish:
- That SP or PS ordering helps or hurts. There is no seed replication (K=1 means one
  shared-step budget and one batch pair, no variance estimate), and the effects measured
  are 0–2 bf16 ULP.
- That any negotiator is preferable. All four are measured neutrally; at K=1 their
  `delta_controlled` values differ only within the 0/-1-ULP band.
- Gate passage. Per the task contract, the gate decision is a K=3 result.
- Anything about convergence, or about T215's failure modes beyond "the three specific
  mechanisms named in `first-report.md` were not used in this run".

## Recommended precondition before K=3

The K=1 sweep is a valid plumbing-and-validity gate (it passed), but the step-scale grid
chosen makes the shared treatment smaller than bf16 parameter resolution. Spending the
remaining budget on K=3 with the same grid would reproduce K=1's ULP-level noise. Before
K=3, either scale `eta` up (to where a meaningful fraction of shared elements actually
change), or keep fp32 master copies of the shared subspace for the displacement/restore
cycle, and record the number of changed elements as a run metric.
