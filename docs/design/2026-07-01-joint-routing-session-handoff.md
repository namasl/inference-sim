# Joint P/D Routing — Session Handoff (2026-07-01)

Checkpoint for the next session, which will **experimentally verify the design in BLIS**. Written
after a long formulation session; the design itself is committed on branch `design/pd-joint-routing`.

## Where things live

- **Design doc (canonical):** `inference-sim/docs/design/2026-06-30-pd-joint-routing-problem-formulation.md`
  (tracked; ~15 commits on `design/pd-joint-routing`, branched off `feat/edpp-pd-disagg`).
- **MASCOTS paper (user's prior queueing model):** `edpp-fresh/MASCOTS2026_paper_96.pdf` — the
  α/β/γ birth–death batch-occupancy model. Relevant as a Layer-2 ingredient (see below).
- **Shipped EDPP + study harness:** `inference-sim/sim/edpp.go`, `sim/edpp_coeffs.go`;
  `campaigns/edpp-study/` (make_specs.py, sweep.sh, analyze/). Frozen coeffs:
  `scripts/calibration/coeffs-llama70b-h100-tp4.json`.
- **Prior worklog (older, related thread):** memory `edpp-pd-routing-reservation-gap` — Q1/Q2
  framing, responsive-z_ttft fix already shipped, synth fully characterized.

## What the design decided (all in the design doc)

- **Realizable system model:** two capability classes — prefill-only `P`, mixed `M` (no "decode-only";
  a decode engine can prefill). Roles fixed per horizon (control-plane). Heterogeneous within pools:
  each instance carries its own coeff vector `θ_i = (α^D, α^P, C0, C1, C_pf, C_attn)`.
- **Action** `a=(d,p)`, `d∈M`, `p∈P∪{local=d}`. Action set `𝒜 = M×(P∪{local})`. Local always feasible.
- **Work model (corrected):** `W_d = C0·o_r + C1·o_r·(a_r+o_r/2)` (trajectory integral; replaces the
  deprecated fixed-`NomDecodeCtx` form). `W_p = C_pf·a_p + C_attn·a_p·(a_r−a_p/2)` (restores the
  cached-prefix cross term the old `(C_attn/2)a_p²` dropped). Per-instance coeffs ⇒ work is
  request×instance.
- **Per-iteration identity:** `T_iter = α_i + Σ_{r∈𝓑_i} δ(s_r)`; work ≠ wall-clock service
  (departure by stage count, not residual work).
- **Admission delay `T_adm`:** obtained by observation (ground truth), by deterministic roll-forward,
  or by Little's law — all **arrival-process-free** (matters because disaggregated decode arrivals
  are non-Poisson prefill departures).
- **Two-fidelity stance:** Layer 1 (observable, runs the policy; arrival-free); Layer 2 (analytical,
  closed-form on the tractable core; validated against Layer 1). Not yet written = Layer 2.
- **Queues:** per-instance congestion `Q_i` (stability only; one shared server), per-class TTFT-deficit
  `Z^T_c`, per-instance ITL-deficit `Z^I_i`; co-residency interference is a **cost**, not a queue.
- **Decision rule (derived, not posited):** drift-plus-penalty →
  `a* = argmin_a [ Σ_i Q_i·Δwork_i(a) + Z^T_c·T̂(a) + Z^I_d·(m_dec + 1{local}·m_pf) + V·c_xfer·1{disagg} ]`.
  `m_dec = δ(r's decode step on d)`, `m_pf = δ(r's prefill chunk on d)`.
- **Soundness:** exact drift bound + virtual-queue↔SLO equivalence ⇒ throughput-optimal + `O(1/V)`
  cost gap, `O(V)` backlog, **under general (non-Poisson) arrivals**; only soft dependency is the
  forward-estimator bias (`T̂`, `m`). Stated as an informal Proposition + proof sketch (§5.4).
- **No drain rates `μ` in the rule** (`b_i` is action-independent; `μ` re-enters only inside `T̂`).
- **Normalization:** relative-form deficit queues (`ttft/τ−1`), work reference `W* ≈ μ_nom·τ_ref`
  (class-agnostic, because `Q_i` is class-agnostic), transfer term carries `τ_ref/τ^T_c` factor.
- **EDPP = reduced case:** same drift rule over a scorer-chosen slice `{(d*,p)}`; full-joint is the
  same method over all `𝒜`. Reduction is a deployment choice, not a necessity; cache affinity is
  subsumed by `J` via per-(request,node) `a_p`.

## Remaining TODOs (priority order)

1. **[NEXT SESSION] Experimental verification in BLIS** — see plan below.
2. **Layer-2 analytical section** — closed-form steady state on the tractable core (prefill / local,
   external-Poisson OK); reconcile with MASCOTS (decode work matches; prefill differs via
   chunk-coupling); use the MASCOTS per-state `Tp(i)/TITL(i)` evaluated at *observed* occupancy as a
   candidate `R_batch`/`T̂`; tighten the §5.4 Proposition toward a full theorem (state Slater/bounded-
   moment conditions precisely).
3. **MILP optimality yardstick (§6)** — write decision variables + constraints (formalizes §3+§3.6);
   offline/clairvoyant optimum on a fixed trace for the optimality gap.
4. **Forward-estimator specifics (§8.4 / §9)** — the roll-forward `R_batch`, `m_dec/m_pf`, and
   `N̂_out` (the deferred "predictor" work). Only soft dependency of the guarantee.
5. **Modeling loose end (§9):** ITL SLO per-request vs per-stream, and how ITL violations are scored
   over a request's decode horizon (interacts with the per-instance `Z^I_i` on a mixed-class batch).
6. **Voice/consistency:** older sections (§1–§2, §6–§8) still have a few declaratives; a light pass.

## Experimental verification plan (to flesh out next session)

Goal: show the derived joint rule (and its corrected pieces) *works* — i.e. its forward estimates
track reality and it makes better P/D decisions than baselines, especially on a **heterogeneous**
workload (the Q2 regime the design targets).

Suggested staging:

- **A. Estimator validation (Layer 1 correctness) — do this first, it underpins everything.**
  Instrument BLIS to log, per request, the realized `T_adm = schedule − enqueue`, realized TTFT/ITL,
  and the forward estimates (`T̂`, roll-forward `R_batch`, `W_d/W_p`). Check predicted-vs-realized
  (the §3.8 validation target). This tests the *one soft dependency* of the guarantee. Use
  `--edpp-decision-trace` / `--routing-decision-trace` (already shipped) as the logging substrate;
  extend if needed.
- **B. Corrected work model.** Replace the deprecated `NomDecodeCtx` decode work with
  `W_d = C0·o + C1·o·(a_r+o/2)` and the cached-prefix prefill cross term; confirm no coeff refit
  needed (same C0,C1,C_pf,C_attn). Regression-check on synth (no-cache: should match old); then RAG /
  shared-prefix (where the cross term bites).
- **C. Occupancy-aware `T̂`.** Implement the admission-delay roll-forward (or Little's-law estimate)
  and confirm it fixes the waiting-only under-prediction documented in the older worklog
  (predicted 14.7s vs realized 542s on saturated decode).
- **D. Full-joint vs reduced EDPP vs baselines.** Build a **heterogeneous** P/M workload (mixed
  prompt sizes / classes) at a decode-adequate topology. Compare: full-joint (choose `d` and `p` by
  `J`), reduced-EDPP (scorer fixes `d`), `always`, `never`, `prefix-threshold`. Report goodput +
  SLO attainment + completions (read completions, not just TTFT-of-completed — see prior worklog
  lesson). Include the offline MILP optimum (once §6 exists) for the optimality gap.
- **E. Both GPU-matched and non-GPU-matched.** The design's real payoff (independent scaling) is only
  visible non-GPU-matched (big untested caveat flagged in the prior worklog). Test both.

Knobs (from memory `edpp-fresh-is-lyapunov-rebuild`): `--edpp-tau-ttft`, `--edpp-tau-itl`,
`--edpp-tau-ref`, `--edpp-v`, `--edpp-c-xfer`, `--edpp-coeffs <json>` (required with `--pd-decider edpp`).
`blis --rate` is ignored with `--workload-spec` (edit `aggregate_rate` in the spec).

## Open theoretical caveats to keep in mind during experiments

- The guarantee holds for the *idealized* rule; the deployed rule inherits it up to forward-estimator
  bias — so estimator validation (A) is not optional, it's the crux.
- `m_pf` in shipped EDPP is the compute-only `C_pf·chunk`; the design's `m_pf = δ(prefill chunk)`
  includes the attention part. Decide whether to upgrade before or after the main experiments.
- ITL fix (`z_itl` responsiveness) from the older worklog is still open there; the new formulation's
  per-instance `Z^I_i` with realized-ITL updates supersedes it — confirm the implementation aligns.
