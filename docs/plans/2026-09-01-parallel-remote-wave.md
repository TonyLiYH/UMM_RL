# Parallel remote execution wave — 2026-09-01

## Objective

Keep the remote executor productively occupied for at least 24 execution hours
without overlapping write scopes or bypassing research Gates.

## Authorized independent tracks

| Track | Task | Branch | Default budget | Expected effort | Output |
|---|---|---|---:|---:|---|
| A | T215 Show-o2 finite-response feasibility | `agent/T215-showo2-finite-response-feasibility` | ≤8 H20 GPU-hours | 8–12 h | Reversible state/gradient/unroll feasibility |
| B | T220 UniDDT admission | `agent/T220-uniddt-admission` | ≤10 H20 GPU-hours | 8–12 h | Deep-sharing dual-path admission |
| C | T230 SenseNova-U1 admission | `agent/T230-sensenova-u1-admission` | ≤12 H20 GPU-hours | 10–14 h | Native-pixel/MoT admission and overlap caveats |
| D | T240 UniAR admission | `agent/T240-uniar-admission` | ≤8 H20 GPU-hours | 6–10 h | Homogeneous-objective boundary-control admission |

The nominal aggregate is 32–48 executor-hours and up to 38 H20-equivalent
GPU-hours. GPU hours are upper bounds, not targets. Source/license audits,
environment preparation, hashing, block-map work, reporting, and validation
are included in the expected effort.

Existing T110, T120, and T130 CPU tasks remain independently executable and
increase the available work queue further.

## Scheduling

1. Start all four first reports in parallel; no large download or GPU work
   precedes their committed reports and storage preflights.
2. Prioritize T215 GPU execution because it is on the D0 critical path.
3. Run model downloads and smoke jobs with no more than two large-model GPU
   tasks concurrently; use the remaining time for source, license, block-map,
   and artifact work.
4. T230 stops at an official-release blocker; it must not recreate missing
   U1.5 or unreleased pipelines.
5. Every branch runs its machine acceptance contract before `awaiting_review`.

## Non-overlap guarantee

The four branches have disjoint task, config, run, report, adapter, and test
directories. They may share external read-only model caches, but each task
records its own resolved revision, source URI, execution URI, hashes, and
resource accounting.

## Common first-report requirements

Every first report records:

- official source and revision;
- license and missing components;
- checkpoint/component sizes and hash plan;
- local SSD path, filesystem, and free capacity;
- environment and entry points;
- trainable shared/private/frozen block proposal;
- exact smoke commands;
- resource estimate and stop conditions.

## Stop conditions

- storage preflight fails;
- official code/checkpoint or mandatory license is unavailable;
- proposed work exceeds the declared resource envelope;
- a required path needs an unofficial fork or unreleased component;
- the task would start persistent joint training.
