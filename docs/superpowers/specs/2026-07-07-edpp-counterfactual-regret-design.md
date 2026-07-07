# Per-Decision Counterfactual Regret Harness (Design)

Date: 2026-07-07
Branch: `feat/edpp-estimator-validation`
Related: `docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` (§1 lossy-decomposition
argument, §3.4 action set 𝒜, §5.5 reduced-EDPP); `campaigns/edpp-study/FINDINGS.md`.

## 1. Goal & scope

Provide an **exact, policy-free, hindsight** diagnostic of routing-decision quality: for a policy's run on
a recorded trace, quantify how much better each individual `(decode-instance, prefill-location)` decision
*could* have been, by brute-force re-simulation. This answers the concrete open question "does the shipped
(reduced) EDPP make sense, or does its pool-average-decide-then-pick-a-specific-prefill-node structure
(§5.5) cost goodput?" — with a number and the specific mishandled requests. It is **not** an optimal
policy and **not** the offline optimum; it is a per-request one-step-deviation regret measure. The global
MILP optimum (§6 of the formulation) and the full-joint decider (§5.3) are separate, later efforts; this
harness is a prerequisite-free stepping stone toward both and the yardstick-lite they are evaluated against.

Scope:
- **Measurement / evaluation only** for the decider logic: no new routing *policy*. The one new BLIS
  component is a **fixed-plan decider** that forces a supplied per-request action plan.
- Evaluates any existing policy (`never`, `always`, `prefix-threshold`, EDPP-reduced).
- Exact: uses the real simulator dynamics for every counterfactual — no surrogate cost model.

## 2. What it computes (definitions)

- **Action** for request `r`: `a_r = (d_r, p_r)`, `d_r ∈ ℳ` (decode instance), `p_r ∈ 𝒫 ∪ {local}` (§3.4).
- **Plan**: a full assignment `{request_id → a_r}` for every request in the trace.
- **Baseline plan** `π₀`: the plan a policy actually produced on the trace (captured from its run).
- **Goodput** `G(π)`: number (or fraction) of requests meeting their SLO under plan `π` — realized
  `TTFT ≤ deadline` **and** `mean ITL ≤ τ_itl` — read from the sim's computed attainment.
- **One-step deviation**: `π₀` with exactly one request `r`'s action replaced by a candidate `a ∈ 𝒜`,
  every other request pinned to its `π₀` action.
- **Regret** of the policy's decision for `r`:
  `regret(r) = max_{a ∈ 𝒜} G(π₀ with r→a) − G(π₀)`, with `hindsight_best(r) = argmax_a`.
  `regret(r) > 0` ⇒ the decision was locally improvable; `= 0` ⇒ locally best-response (no single-request
  change helps).

**Regret is measured on TOTAL trace goodput, not `r`'s own outcome.** Changing `r`'s action changes the
state later requests face, so each deviation is scored over the whole trace. A deviation that helps `r` but
hurts neighbors nets ≤ 0 regret. This is what makes the measure honest about the decisions' coupling (the
reason a per-request "optimal decision" is not well-defined in isolation, and the reason this is a
*local* diagnostic, not the global optimum).

## 3. Components

### 3.1 Fixed-plan decider (Go, `sim/`)
A `DisaggregationDecider` (interface at `sim/disaggregation.go:52`) that, instead of computing a decision,
looks up `req.ID` in a supplied plan and forces that action:
- `p = local` → return `DisaggregationDecision{Disaggregate: false, DecodePodOverride: d}`.
- `p ∈ 𝒫` → return `{Disaggregate: true, DecodePodOverride: d, PrefillPodHint: p}`.

Requires: (a) the decode override path (already honored at `cluster.go:2284`); (b) **`PrefillPodHint` must
be honored** by `PrefillRoutingEvent.Execute` (`sim/cluster/pd_events.go:28-40`), which today ignores it
and re-runs its own scorer — the one real wiring gap to close. Plan loaded from a CSV via a new flag
(e.g. `--pd-plan <csv>`), columns `request_id, decode_instance, prefill_instance` (`prefill_instance`
empty/`local` for the aggregated action). A request absent from the plan is a fatal config error (the plan
must be total) — no silent fallback (R1).

INV-9: the decider reads only the plan (instance IDs), never `Request.OutputTokens`. INV-6: deterministic
given the plan. INV-13: run/replay parity of the forced routing.

### 3.2 Counterfactual driver (Python, `campaigns/edpp-study/`)
1. **Capture the baseline plan.** Run the target policy once with `--pd-outcome-trace` — that trace already
   emits `prefill_instance` / `decode_instance` per request (Stage A), i.e. `π₀`. Record baseline goodput
   `G0` from the run's metrics (`slo_attainment` / `per_class`).
2. **Self-consistency check (validation of the decider itself).** Feed `π₀` back through the fixed-plan
   decider (`--pd-plan π₀.csv`). The forced run MUST reproduce `G0` and the per-request outcomes
   byte-identically (INV-6). If it does not, the decider does not faithfully replay a plan — stop.
3. **Counterfactual sweep.** For each request `r` in a sampled set `S`, for each `a ∈ 𝒜 \ {π₀(r)}`: write
   `π₀` with `r → a`, run BLIS `--pd-plan`, read `G`. Compute `regret(r)`, `hindsight_best(r)`.
4. **Report (JSON + table).** Per-request: chosen action, hindsight-best action, regret. Aggregates:
   fraction of sampled decisions with positive regret, mean/median/total regret, and a breakdown by
   whether the policy disaggregated the request (does reduced-EDPP's regret concentrate on kept-local or
   disaggregated decisions?). Optional plot.

## 4. Defaults (chosen; flag to revisit)

- **Realistic trace, sampled requests.** Use a genuinely loaded trace (enough requests to saturate — a
  trivial trace makes every hindsight-best `local` and shows nothing). Compute regret over a **sampled**
  subset `S` of `K` requests (`K` configurable, e.g. 50–200), so cost is `K·(|𝒜|−1)` sim runs, bounded and
  parallelizable — not all `N`. Regret is per-request, so a sample gives the distribution.
- **Reuse `--pd-outcome-trace`** to capture `π₀` rather than build new plan-capture instrumentation.
- **Topology / workload:** start on the study's existing synth (decode-bound) at a saturating rate and a
  small multi-instance PD topology (e.g. 1P2D) so `𝒜` is non-trivial (`|𝒜| = |ℳ|·(|𝒫|+1)`).

## 5. Validation ladder

1. **Self-consistency:** fixed-plan(`π₀`) reproduces the policy run's `G0` and per-request outcomes exactly.
2. **Hand cases:** (i) idle cluster, one cheap request → hindsight-best is `local`, regret 0 for a
   `never`-policy baseline and positive for an `always` baseline; (ii) one saturated M + idle P, expensive
   prompt → hindsight-best disaggregates (positive regret for a `never` baseline). These pin the harness's
   correctness on known answers.
3. **Diagnosis run:** reduced-EDPP on the loaded trace → report the regret distribution and where it
   concentrates. This is the empirical verdict on reduced-EDPP's coherence.

## 6. Deliverables & staging

Each stage is an independently testable deliverable (its own plan task set):
1. **Fixed-plan decider** (Go): decider + `--pd-plan` loading + `PrefillPodHint` wiring in the prefill
   route + unit tests + the self-consistency behavior. INV-6/9/13.
2. **Counterfactual driver** (Python, `campaigns/edpp-study/analyze/counterfactual_regret.py` + a
   `repro_counterfactual.sh`): capture `π₀`, self-consistency gate, sweep, regret report; hand-case
   self-tests on tiny traces.
3. **Diagnosis + findings:** run on reduced-EDPP / `never` / `always`, record the regret result in
   FINDINGS ("Counterfactual regret") + README pointer.

## 7. Relationship to the later efforts (context, not scope)

- The fixed-plan decider is **shared infrastructure**: full-joint [C] reuses the same `DecodePodOverride` /
  `PrefillPodHint` override wiring, and the global MILP yardstick reuses the fixed-plan decider to
  cross-validate its optimal plan against realized sim goodput.
- This harness gives a **local** (one-step-deviation) bound, not the **global** optimum; the two agree only
  when no single-request change helps. The MILP (later) supplies the global optimality gap; this supplies
  the interpretable per-decision diagnosis and motivates whether [C] is worth building.

## 8. Testing & invariants

- INV-6 determinism: fixed-plan runs byte-identical across invocations; the self-consistency check is the
  primary guard.
- INV-9: fixed-plan decider never reads `OutputTokens` (routing from the plan only).
- INV-13: forced routing has run/replay parity (the plan is applied identically on both paths).
- Python driver: a self-check on a synthetic tiny trace with a known hindsight-best (mirrors the other
  `analyze/` self-tests).
- No silent fallback: a request missing from the plan is fatal (R1).
