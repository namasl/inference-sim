# EDPP Empirical Study — Findings

## Diagnostic cell #1 — synth (decode-bound), rate 2.0, 2000 reqs, equal 4-node hardware

See `out/diag/SUMMARY.md` for the full table; `out/diag/RUNS.md` for the run registry.
Deciders compared at this point: never@4, edpp@{1P3D,2P2D}, always@2P2D, prefix-threshold@{1P3D,2P2D}.

> **CORRECTED 2026-06-26 (instrument audit — the "wrong knob" case the discipline note below warns
> about).** The original KEY RESULTS were produced with BLIS's default `--routing-policy round-robin`
> and NO per-pool scorer flags, so the decode pool fell back to round-robin and pinned ALL decode to one
> node — the "1192/2000 collapse" and "161s TTFT" were artifacts of that misconfiguration. llm-d's
> shipped PD profile is weighted `prefix-cache:2 + queue:1`; BLIS now defaults to it in PD mode. The
> results below are RE-MEASURED under that default (full table + mechanism in
> `out/diag/SESSION_LOG.md` → "CORRECTED SLO/TTFT/E2E TABLE 2026-06-26").

KEY RESULTS (corrected, under the llm-d weighted default; rate 2.0, 2000 reqs, equal 4-node HW):
- **No collapse, ZERO preemptions anywhere** — all arms complete 2000/2000 (the round-robin pin's
  ~10k preemptions + 1192/2000 were the artifact).
- **never@4 wins** (goodput 1.68 rps, E2E p99 275s). Disaggregation does NOT help this decode-bound
  workload at equal hardware. Outcome tracks DECODE-CAPABLE NODE COUNT monotonically across ALL
  topologies: never@4(4) > *@1P3D(3) > *@2P2D(2) > *@3P1D(1) on goodput AND completion.
- **3P1D (1 decode node) is the only REAL collapse** (no routing artifact): nothing to balance →
  KV saturation → always/prefix-thr complete just 1192/2000 with 10,157 preemptions. Those are the
  EXACT numbers the original round-robin 2P2D "collapse" produced — confirming round-robin had pinned
  2P2D to one active decode node (2→1 ≡ 3P1D). edpp@3P1D: worst SLO 0.46 / goodput 0.48 (TTFT p99 765s).
- **EDPP is the WORST decider here**: TTFT p99 518s, SLO 0.75, lowest goodput, at BOTH 2P2D and 1P3D.
  Mechanism (TTFT split by was_disaggregated): disaggregated reqs get fast TTFT (p99 0.2s, prefill on
  dedicated nodes); the ~48% EDPP keeps LOCAL suffer HOL blocking — their prefill queues behind decode
  on the saturated decode nodes — giving local TTFT p99 547s. always/prefix-threshold disaggregate
  100% → no local reqs → fast TTFT (199ms); the decode-queue wait shows up only in E2E. So EDPP's
  PARTIAL disaggregation actively hurts on this uniform decode-bound workload.
- **EDPP's decisions shifted a lot** vs the round-robin runs: it now disaggregates 1045@2P2D (was 331)
  and 1029@1P3D (was 0 — i.e. 1P3D flipped from "stay 100% local / fine" to "51% disagg / TTFT 518s").
  Sensitive to the routing/load signals; needs decision-trace diagnosis (item below).
- prefix-threshold ≈ always on synth (tiny inputs trip threshold-16 → ~100% disagg).

OPEN — WHY does EDPP disaggregate partially (and so much)? On a saturated decode pool the right call
is to disaggregate ~all (like always) so no request's prefill waits behind decode; EDPP's partial
choice is the thing to explain. Use `--routing-decision-trace` + `--edpp-decision-trace`.

OPEN — THE REAL TEST OF EDPP (Experiment 5, not yet run): uniform synth is not FAVORABLE to
disaggregation (per-request adaptivity has no signal to exploit and is shown strictly harmful here).
To see any EDPP *advantage* needs a heterogeneous regime (mixed prefill sizes + adequate decode), so
adaptivity could beat both always and never.

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
