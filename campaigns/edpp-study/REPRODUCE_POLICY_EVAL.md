# Policy evaluation — how to reproduce the numbers (protocol v2.1, 2026-07-28)

Everything the paper's policy-evaluation section needs is produced by five scripts in
this directory. Read this file before running anything: the previous protocol
(`repro_var_dominance_goodput.sh`, `repro_hetero_ratio_sweep_gp.sh`,
`repro_topology_matrix_gp.sh`) is **superseded and its numbers should not be used** —
see "What was wrong before" at the bottom.

## Quick start

```bash
go build -o blis main.go

# 1. Measure per-cell capacity knees and zero-contention baselines.
#    Writes the protocol constants consumed by everything else.
bash campaigns/edpp-study/repro_knee_sweep.sh          # ~15 min
#    -> then hand-write specs/grid_v2/cells.txt from its output (already committed)

# 2. Main comparison: goodput per policy per regime, at 0.7x and 1.0x the knee.
bash campaigns/edpp-study/repro_grid_v2.sh             # ~60 min, 5 seeds
python3 campaigns/edpp-study/analyze/grid_v2_report.py # table + worst-case regret

# 3. Capacity curves (goodput/throughput/latency vs offered rate, all policies)
#    plus the aggregated never@3M reference.
bash campaigns/edpp-study/repro_policy_curves.sh       # ~90 min, 3 seeds
#    -> out/policy_curves/curves.csv (one row per run)

# 4. Mechanism: SLO-miss decomposition by dimension and serving instance.
bash campaigns/edpp-study/repro_decomposition.sh       # ~20 min, seed 42 + traces

# 5. Structural ablations.
bash campaigns/edpp-study/repro_ratio_sweep_v2.sh      # heterogeneity ratio N
bash campaigns/edpp-study/repro_topology_v2.sh         # 1P3M / 2P2M / 3P1M
```

Scripts are safe to run concurrently (each writes its own spec and metrics files).
`repro_policy_curves.sh` skips runs whose metrics JSON already exists, so it resumes.
The others overwrite.

## The policy set (final — do not silently add arms)

| arm | what it is | decode instance chosen by |
|---|---|---|
| `never` | never disaggregate (prefill where you decode) | load-balancing scorer (`queue-depth:1`) |
| `always` | always disaggregate through the prefill pool | load-balancing scorer |
| `kairos` | reproduction of Kairos prefill deflection; per-seed best over beta in {0.25,0.5,1.0} | its own deflection search |
| `lt-joint` | minimize the deciding request's predicted TTFT over the **full joint action set** | joint (hardware-aware) |
| `dpvar` | **the paper's rule**: congestion + SLO deficits + goodput penalty R(a) = VaR(a) - good_r(a) | joint |
| `never@3M` | *reference row*, curves only: 3 mixed engines, no PD machinery at all | load-balancing scorer |

Deliberately **not** in the set, and why:

- **least-TTFT (non-joint)** — its result is a property of whichever scorer you pair it
  with, and this paper makes no argument about scorers. Everything is joint except the
  static corners and Kairos, which have to pick an instance somehow and use the
  load-balancing scorer.
- **prefix-threshold** — numerically identical to `always` in every cell of these
  workloads (no prefix reuse, every prompt over the threshold). Carries no information.
- **drift-plus-penalty** — not a policy the paper claims.
- **the output-length ORACLE arm** (`--edpp-oracle-output-len`) — reads a realized
  output length, violating INV-9. Removed from every harness so an oracle number
  cannot reach a table by accident. The flag still exists for diagnostics; do not use
  it for anything reported.

**Fairness**: every arm above is deployable. The only oracle read in the decider is
`reqNHatOut`, gated behind `--edpp-oracle-output-len`, which no arm sets.
`--edpp-var-deployable` (set on `dpvar`) makes the co-resident remaining-step estimate
a censored per-class running mean; that is the only place co-resident output lengths
are read anywhere, and the non-VaR arms never read them at all.

## The protocol (why the numbers are what they are)

1. **Rates are knee-relative.** Each cell's capacity knee is measured
   (`repro_knee_sweep.sh`: rate sweep, achieved/offered >= 0.95), and the grid runs at
   0.7x knee (operational) and 1.0x knee (stress). Rates are *not* hand-picked.
2. **SLOs are derived, with no floor.** A zero-contention probe at rate 0.2 gives
   TTFT/ITL/E2E p99; targets are 5x / 4x / 3x those (Method A slowdown multiples,
   Splitwise/Sarathi style). The 4x ITL multiple matters: at 5x the ITL dimension never
   binds anywhere and the rule's ITL machinery goes untested.
3. **Horizons are steady-state.** `num_requests = rate x max(120 s, 10 x E2E SLO)`,
   verified horizon-stable (n vs 2n moves goodput <= 0.006). Short bursts produce
   rankings that flip with horizon length.
4. **One scorer config everywhere**: `--decode-routing-scorers queue-depth:1`,
   `--max-num-running-reqs 16`, including on the heterogeneous cell.
5. **Seeds**: 5 for the grid and ablations, 3 for the curves. The workload spec is
   reseeded per seed (not just the simulator RNG).

Protocol constants live in `specs/grid_v2/cells.txt` (knee rate + baseline p99s per
cell). Changing a cell means re-measuring its knee.

## Known open items

- **`main-full.tex` section V still describes the old policy set** (it names
  least-TTFT, prefix-threshold and drift-plus-penalty, 17 mentions). The prose must be
  brought in line with the table above.
- **Heterogeneous stress rates.** The heterogeneous cell's knee (rate 6) is an *easy*
  test: the fast instance alone absorbs the load, so pure herding scores 1.000 there.
  The regime that separates policies is rates 10-14, where the fast instance saturates
  and the router must spill selectively. `repro_decomposition.sh` includes those as
  `hetero_s10/s12/s14`; the grid does not. Consider adding a stress row.
- **`never@3M` dominates every 1P2M policy on homogeneous cells** at every rate
  measured. On homogeneous hardware with these workloads the disaggregated fleet shape
  is not worth it, and the paper should say so plainly: the router's job is to use a
  *given* disaggregated fleet well.
- Curves are single-seed per rate for latency percentiles; goodput margins under ~0.02
  should not be read as real without more seeds.

## What was wrong before (do not resurrect the old harnesses)

An audit on 2026-07-27 found every cell of the previous protocol compromised:

- Cells ran **over capacity on short bursts** (the decode cell at ~5x its knee, 240
  requests). Goodput measured how much of a burst finished before the backlog caught
  it; doubling the horizon flipped policy rankings.
- **ITL never bound** anywhere (target 100 ms vs realized ~17-31 ms), so the composite
  good was effectively TTFT-and-E2E only.
- TTFT targets were **floored at 1 s** rather than derived, in 3 of 4 cells; the
  heterogeneous cell and ratio sweep used **asserted** 60 s / 8 s / 500 ms targets. The
  60 s TTFT made W* enormous and the congestion term unable to bind at all.
- At that operating point the heterogeneous cell had **no live signal**: the VaR term
  differed across candidates in 0 of 400 decisions and placements were decided by
  floating-point tie-break residue.

Evidence is preserved under `out/audit_2026-07-27/` (see its README).
