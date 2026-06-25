# EDPP Empirical Study — Findings

## Diagnostic cell #1 — synth (decode-bound), rate 2.0, 2000 reqs, equal 4-node hardware

See `out/diag/SUMMARY.md` for the full table; `out/diag/RUNS.md` for the run registry.
Deciders compared at this point: never@4, edpp@{1P3D,2P2D}, always@2P2D, prefix-threshold@{1P3D,2P2D}.

KEY RESULTS (corrected — supersede any earlier "EDPP improves TTFT" wording, which was WRONG):
- Outcome tracks DECODE-CAPABLE NODE COUNT: never@4 > *@1P3D > *@2P2D (monotonic on ITL/E2E).
  Disaggregation does NOT help this decode-bound workload at this load (equal hardware).
- At 1P3D (adequate decode): edpp (0% disagg) ≈ prefix-threshold (100% disagg) — disaggregation
  is NEUTRAL; prefill too cheap to matter where it runs.
- At 2P2D (starved decode): policies only RELOCATE pain. edpp's PARTIAL 16.6% disagg gives the
  WORST TTFT (p99 161s) via HOL blocking (local prefill queues behind decode on 2 nodes).
  always==prefix-threshold (100% disagg) get fast first-token but only 1192/2000 DECODE
  (throughput collapse) — their "good TTFT" is hollow.
- EDPP's one genuinely-good behavior: at 1P3D it correctly DECLINES to disaggregate.
  prefix-threshold (the production default) blindly disaggregates 100% regardless of topology.
- prefix-threshold disaggregates ~everything on synth (tiny inputs trip threshold-16) ⇒ behaves
  like `always` here.

OPEN — THE REAL TEST OF EDPP (Experiment 5, not yet run): none of the above is FAVORABLE to
disaggregation, so "best decider = disaggregate least-harmfully." To see EDPP's *advantage* we
need a regime where per-request disagg helps SOME requests (mixed prefill sizes + adequate decode
capacity), so adaptivity beats both always and never.

---

Status (full sweep): TEMPLATE (fill once the full sweep + `report.py` complete).
Design: `docs/superpowers/specs/2026-06-25-edpp-empirical-study-design.md`.
Artifacts: `campaigns/edpp-study/out/{summary.csv,regret.csv,*.png}`.

Discipline: every prior memory claim below is a **prediction**, marked
confirmed / refuted / refined against the experiment. When an outcome
contradicts a prediction, audit the instrument first (wrong knob, not steady
state, wrong counter) before concluding anything about the algorithm.

## Setup actually run

- Workloads: RAG summarization (prefill-bound) + synthetic-data-gen (decode-bound).
- Load: `aggregate_rate` ∈ {0.5,1,1.5,2,2.5,3}, analyzed vs offered prefill/decode tok/s.
- Topology: equal 4-instance hardware. `never`@4 homogeneous (all-local) baseline;
  disaggregating arms (`edpp`/`always`/`prefix-threshold`) over P/D splits 1P3D, 2P2D, 3P1D.
- Method: bake-then-replay (all arms see the identical trace per cell). `num_requests`=5000.
- Steady state: `N̂_out`-convergence head trim + 5% drain-tail trim.

## Prior claims tested (confirmed / refuted / refined)

- [ ] **Saturation at `aggregate_rate 3.0` for 70B** → knee table (Task 5 Step 3 / `never`@4 p99 vs rate).
- [ ] **EDPP "rarely disaggregates unforced"** → `disagg_frac` column in `summary.csv`.
      Preview (synth/1.0): edpp `disagg_frac`=0 across all splits — consistent so far.
- [ ] **Under-disagg bias direction** → `regret.csv` dominant-term tallies on EDPP SLO misses.

## #1 Outcome — does EDPP beat the baselines?

Per workload × split: EDPP vs `always` / `prefix-threshold`, with `never`@4 overlaid.
A good adaptive decider sits at/above the better of {never, always} at each load point.
- Preview (synth/1.0): edpp ttft_p99 125ms < always 142ms @1P3D — edpp correctly declines
  to disaggregate decode-bound load. Fill the rest from `summary.csv` + plots.

## #2 Mechanism — is the machinery behaving as designed?

- `N̂_out` convergence time (window report; preview: ~index 204 on synth/1.0).
- Backlog conservation: PARTIAL coverage here (no kernel-side ground-truth waiting-token
  counter was added — deliberate scope cut). If the indirect signal is insufficient, add
  the counter as a follow-up.
- Whether prediction error correlates with bad decisions (via the regret join).

## #3 Detective — WHY

Regret events (`regret.csv`) decomposed by dominant rule term — the mechanistic story
behind any EDPP loss, citing trace evidence (which term dominated, disaggregate y/n).

## Anomalies & instrument audits

- **`BLIS_STEP_CSV` not found in code** (calibration memory claims it exists) — flagged,
  calibration-side, not resolved here.
- **`--record-itl` is observe-only**, but the simulator records per-request ITL
  unconditionally into `itl_mean_us` on replay — so ITL outcomes are available without
  the flag. (Initial worry that ITL was unavailable was a misread of the instrument.)
- **`--num-instances 4` alone is fatal** for PD disaggregation (`pool.go` requires role'd
  instances) — the plan's original flag was wrong; corrected to explicit P/D splits.
- **regret_join windowing**: uses the full (un-windowed) request set while `summary.csv`
  uses the steady-state slice — regret tallies may include warmup. Re-window if it matters.
- Record any case where a result contradicted a memory claim and what the instrument audit found.
