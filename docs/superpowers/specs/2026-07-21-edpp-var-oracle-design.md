# Value-at-Risk drift oracle for EDPP (Design)

Date: 2026-07-21
Branch: `feat/edpp-estimator-validation`
Related:
- `campaigns/edpp-study/STUDY_REPORT.md` §7 ("The one live idea: change what the drift term measures";
  "The honest next experiment — an oracle upper bound").
- `docs/superpowers/specs/2026-07-15-edpp-least-ttft-baseline-design.md` (the `least-ttft` baseline this
  experiment must beat).
- `docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` (EDPP's drift-plus-penalty rule).
- Existing diagnostic oracle `--edpp-oracle-output-len` (`sim/edpp.go:85`, `reqNHatOut` at `sim/edpp.go:1212`),
  which this composes with.

## 1. Purpose and the bar

Every EDPP experiment in this study converges on one sentence: *every term prices the deciding request's own
experience; the one term that prices the externality (backlog drift) does it in **work** (µs), not **value**
(goodput).* Four failures — the overload collapse (F6), the per-class backfire (F9), the Type-A/B workload
result (E6), and "the shipped rule wins 0/12 cells" (F5) — are all that same missing externality.

This experiment tests the single remaining idea that could rescue the design: keep Neely's drift structure but
**re-price the drift term in value-at-risk** — the marginal goodput destroyed at a decode instance by adding a
request's work there — instead of in microseconds of work.

It is built as a **diagnostic oracle upper bound**, not a deployable policy. The pass/fail bar:

> **Oracle VaR-EDPP must clearly beat the one-line `least-ttft` rule** on the archetypes where `least-ttft`
> currently ties or wins (the moderate-load prefill regime and the decode/balanced archetypes), measured on the
> same `slo_attainment` goodput metric, robust across the 3 seeds.

If a **perfect oracle** cannot beat the one-liner, the method is dead: the characterization paper (why a
principled drift-plus-penalty policy is dominated by its own ablated subsets) is the finding, and we stop
cleanly. If it clears the bar, a deployable approximation becomes worth building (deferred; see §7).

`least-ttft` prices only the deciding request's **own** predicted TTFT. VaR supplies exactly the thing
`least-ttft` is blind to — the **externality** the placement imposes on the co-residents already running on the
instance. That asymmetry is why VaR *could* beat it; that it is the only structural difference is why the
comparison is clean.

## 2. The value-at-risk quantity

Fix a decision on request `R` (class `c`, input length `a_r`, uncached prompt `a_p` on a candidate decode
instance `i`). `least-ttft` and reduced-EDPP already select `i` via the scorer; the joint path enumerates `i`.

**Goodput currency.** The study's goodput is `per_class[...]['slo_attainment']`, computed by
`cluster.SLOAttainmentMultiDim` (`cmd/goodput.go:166`) as the fraction of requests meeting **TTFT ∧ ITL ∧ E2E
jointly**. VaR's per-request "good" indicator `g(·)` uses this **same composite**, so the objective the drift
term optimizes and the metric the bar is judged on are the same currency. Using any other currency would make
the ceiling incomparable to the bar.

**Projected completion.** For each decode co-resident `j` currently on `i`, its projected E2E completion is

```
C_j = now + TrueRemaining_j · t_iter(i)
```

where `TrueRemaining_j` is `j`'s true remaining decode steps (the oracle input — see §4) and `t_iter(i)` is the
trained-physics per-iteration time the rule already computes (`thetaD.tIterDecode(B, kv, sPf)`, `sim/edpp.go`
`tBminus1`). Its ITL is the per-token time; its TTFT is already realized (a decoding co-resident has produced
its first token) and therefore fixed under this decision.

**The two placements and their externality.**

- **Local** (`Disaggregate = false`): R does prefill-on-decode on `i`. During R's `nChunks` prefill
  iterations the batch carries an extra prefill chunk, inflating each co-resident's per-iteration time by the
  `δ_pf-chunk = C_pf·chunk` term the rule already models (`sim/edpp.go:639`); R then joins the decode batch
  (`B+1`). Each co-resident `j` is delayed to `C_j^local = C_j + Δ_j^local`.
- **Disagg** (`Disaggregate = true`): R's prefill runs on the prefill pool `p`; instance `i` is undisturbed
  during the remote prefill + KV transfer window, and R joins `i`'s decode batch only afterward. Fewer of each
  co-resident's steps are slowed, so `Δ_j^disagg < Δ_j^local`. In exchange, R's prefill delays the prefill-pool
  co-residents `k` (pushing their TTFT/first-token completion).

This asymmetry `Δ_j^disagg < Δ_j^local` is the entire mechanism: it is the externality `least-ttft` cannot see.

**The completion-delay model** reuses the existing trained-physics, not a new latency law. A co-resident `j`
with `rem_j = TrueRemaining_j` steps left experiences, under local placement, its first `min(nChunks, rem_j)`
steps at the inflated per-iter time `t_iter(i) + δ_pf-chunk` (R's prefill co-scheduled) and the remainder at
the `B+1` decode per-iter time; under disagg, its steps run at `t_iter(i)` until R arrives post-transfer, then
at the `B+1` time for the tail. `Δ_j` is the difference of these sums against `C_j`. Prefill-pool co-residents
`k` are delayed by R's prefill via the same prefill-side estimators the rule already uses (`tAdmP`,
`tIterPrefill`). *(Exact step-indexing is a micro-plan concern; the design fixes the structure and the
physics source.)*

**Value-at-risk.**

```
VaR_local  = Σ_j [ g(C_j)            − g(C_j^local)  ]
VaR_disagg = Σ_j [ g(C_j)            − g(C_j^disagg) ]  +  Σ_k [ g(k before) − g(k after R's prefill) ]
```

Because TTFT of decoding co-residents is fixed, the marginal decode-side flips come from **ITL** (inflated
during overlap) and **E2E** (delayed by `Δ_j`); prefill-side flips come from **TTFT/E2E** of the `k`.

**Three scoring kernels for `g`** (all run — A is the headline ceiling, B makes the predicted trap
measurable, C is the smoothed deployable-target companion):

- **A — binary flip count.** `g(C) = 1 if (TTFT_met ∧ ITL_met(C) ∧ E2E: C ≤ deadline) else 0`. VaR is the count
  of co-residents whose composite-good flips true→false. This **is** goodput (count of SLO-meeting requests),
  so it gives the cleanest, hyperparameter-free ceiling number. A step function → non-smooth drift signal.
- **B — saturating utility sum.** `g(C) = σ((deadline − C)/scale)`, a smooth utility of slack that saturates to
  ~1 comfortably early and →0 past the deadline. `VaR = Σ [g(before) − g(after)]`. §7's predicted trap: `g ≈ 0`
  and flat for doomed requests → no marginal signal → their placement falls to the other terms, which grab the
  cheapest (contended) resource → reproduces E7's neglect. We **measure** this, not assert it.
- **C — deadline-slack hazard.** `VaR = Σ h(slack_j) · Δ_j`, where the hazard kernel `h` peaks for co-residents
  inside a band around their deadline and **decays gently** on both sides (doomed requests keep a small nonzero
  weight rather than a hard zero, avoiding B's neglect by construction). Carries a band-width scale set from
  the ITL/TTFT SLO. Closest to "marginal goodput destroyed" while remaining an actionable, smooth signal — the
  shape a deployable policy would eventually target.

## 3. Rule wiring — VaR replaces the balance term, nothing else

Today the reduced rule (`sim/edpp.go:633-642`) is `Disaggregate ⟺ lhs > rhs`, where

```
lhs = balanceTermD − balanceTermP           (work-currency externality: Q_i · work-rate)
rhs = transferTerm + ttftTerm + itlTerm     (the deciding request's OWN experience, z-weighted)
```

The VaR rule replaces **only the left-hand side**:

```
lhs_var = VaR_local − VaR_disagg            (value-currency externality)
Disaggregate ⟺ lhs_var > rhs                (rhs unchanged)
```

This is the minimal, most-interpretable change: it swaps the one work-currency externality term for a
value-currency one and keeps the transfer penalty and the z-weighted TTFT/ITL self terms. VaR-EDPP is therefore
`(a least-ttft-like self term) + (the externality least-ttft lacks)` — precisely the shape that should beat the
baseline. Ties (`lhs_var == rhs`) → local, matching the existing strict `>` tie handling.

**Joint path.** `decideJoint` (`sim/edpp.go` ~line 760) enumerates `(d, p)` candidates and picks the
drift-plus-penalty argmin. Under the VaR rule, each candidate's cost uses that candidate's VaR (goodput
destroyed among *that* `d`'s decode co-residents, plus — for disagg candidates — *that* `p`'s prefill
co-residents) in place of the work-currency balance contribution; the transfer/TTFT/ITL terms are unchanged.
This lets VaR select the instance that destroys the least goodput, which is where externality-aware routing
should matter most.

**Selection.** A new decision-rule value `--edpp-rule var` (alongside `""`/`dpp`/`least-ttft`) plus
`--edpp-var-metric flip|util|hazard` (kernels A/B/C). Validated to those values at the CLI boundary and in
`EDPPConfig.validate` (unknown → fatal/panic, R3 style).

## 4. Oracle / INV-9 boundary

VaR needs three inputs per co-resident; they fall on **different** sides of the INV-9 line:

| Input | Source | Status |
|---|---|---|
| `TrueRemaining_j` (co-resident true remaining decode steps) → `C_j`, `Δ_j` | `req.OutputTokens` at snapshot build | **Oracle** (INV-9 violation) |
| R's own true output length `o_r` | `len(req.OutputTokens)` | **Oracle** (existing `--edpp-oracle-output-len`) |
| `deadline_j` (`SLOTargetUs`, arrival, class), ITL target | input-derived (`SLOTargetUs = GatewayEnqueueTime + target`) | **Deployable** |

- **The oracle part is exactly "un-censor co-resident remaining"** that §7 names. The infrastructure already
  exists: `RunningReqState.TrueRemaining` is threaded through the rollforward estimator
  (`sim/admission_estimator.go:15`, used at `sim/admission_estimator.go:103`) and is populated in oracle mode by
  `RunningDecodeState()` (`sim/simulator.go:205`); the control path merely nulls it via `censorOracleRemaining`
  (`sim/edpp.go:1087`). The VaR path reads it **un-censored**, gated behind the same `admissionDetailOracle`
  switch and the VaR-oracle flag.
- **This is a deliberate, gated INV-9 violation.** It emits the same loud `logrus.Warnf` as
  `--edpp-oracle-output-len` ("DIAGNOSTIC oracle … results are an UPPER BOUND, not an achievable policy"), is
  never a deployable policy, and composes with `--edpp-oracle-output-len` (R gets true `o_r` too, for a fully
  clean ceiling).
- **The deadline is deployable** — it derives from input-side quantities, so it does not deepen the oracle. It
  is new state that must be carried on `RunningReqState` and populated at the `RunningDecodeState()` build site
  (which already holds the live `req`, so `req.Deadline`/`SLOTargetUs`/`SLOClass`/`ArrivalTime` are in hand).

## 5. Architecture & components

Isolate the VaR machinery so `sim/edpp.go` (already ~64 KB) does not absorb it:

- **`sim/edpp_var.go` (new).** The VaR unit: the projected-completion model, the three scoring kernels
  (A/B/C behind a small strategy selection), and `VaR_local` / `VaR_disagg` (reduced) and the per-candidate VaR
  (joint). Pure functions of `(R, candidate snapshot(s), co-resident state, kernel, coeffs, SLO targets)` — no
  hidden state, INV-6-clean, unit-testable in isolation. **What it does:** given a placement and the
  co-resident set, return the goodput destroyed. **How you use it:** the decider calls it to form `lhs_var`.
  **What it depends on:** the trained-physics `EDPPCoeffs` (`tIterDecode`, `δ_pf-chunk`, prefill estimators)
  and the (possibly un-censored) `RunningReqState` slices already on the snapshots.
- **`sim/admission_estimator.go`.** Enrich `RunningReqState` with the deployable per-co-resident deadline
  fields (E2E deadline µs, ITL target µs, class). Existing estimators ignore the new fields → byte-identical.
- **`sim/simulator.go`.** `RunningDecodeState()` populates the new deadline fields (deployable) and keeps the
  existing oracle population of `TrueRemaining`.
- **`sim/edpp.go`.** In reduced `Decide`, branch on `rule == "var"` at the decision site to use `lhs_var` in
  place of `balanceTermD − balanceTermP`; in `decideJoint`, use the per-candidate VaR in the candidate cost.
  Add the kernel/oracle config on `EDPPConfig`/`EDPPDecider`. Everything above the decision site (all TTFT/ITL
  estimation) is unchanged and shared, so `rhs` is identical to reduced-EDPP.
- **`cmd/root.go` + `cmd/replay.go`.** `--edpp-rule var` + `--edpp-var-metric`, wired into `EDPPConfig` on
  **both** the run and replay paths (INV-13; the least-ttft baseline shipped an INV-13 miss on exactly this —
  the reject was missing on the replay path — so replay wiring is explicit here). Oracle warning on both.
- **`campaigns/edpp-study/repro_var_oracle.sh` + analysis.** Mirrors `repro_spectrum.sh`: adds `var:flip`,
  `var:util`, `var:hazard` arms next to `least-ttft` / `always` / `never` / `edpp(dpp)`, on the same archetype ×
  utilization × 3-seed grid, same per-archetype auto-derived SLOs, same decode-balancing scorer. Reduced first;
  a joint variant (`JOINT=1`) reuses the joint sweep harness.

## 6. Testing (TDD)

- **VaR kernel units (`sim/edpp_var_test.go`).** A hand-built co-resident set with known `TrueRemaining` and
  deadlines, and a known R, yields the expected value under each kernel: A → the exact flip count; B → a
  slack-graded utility drop that goes ~0 for a doomed co-resident (locks the trap); C → hazard-weighted delay
  concentrated in the near-deadline band with nonzero doomed weight.
- **Externality-asymmetry law.** For any co-resident set, `Δ_j^disagg ≤ Δ_j^local` and hence
  `VaR_local ≥ VaR_disagg` when the prefill pool is idle — the structural law the mechanism rests on
  (a conservation/monotonicity test, not a golden).
- **Rule-swap byte-identity (INV-6).** `rule ∈ {"", "dpp", "least-ttft"}` is byte-identical to today
  (`go test ./sim/... ./cmd/...` green; existing reduced/joint goldens unchanged). VaR is reached only under
  `rule == "var"`.
- **Oracle gate (INV-9).** With the oracle off, co-resident `TrueRemaining` is censored (`-1`) and the VaR path
  must not read a true remaining; with it on, the un-censored values flow. A test that reverts the un-censoring
  turns red (non-vacuous guard).
- **Machinery-bypassed contrast.** Under `rule == "var"`, large `z_ttft`/`z_itl` still move the decision (the
  `rhs` self terms are retained by design) but the **work** balance term does not enter — assert that swapping
  the work backlog `Q_i` leaves the VaR decision unchanged while it would change the `dpp` decision.
- **Joint reduction / self-consistency.** The joint VaR argmin over a candidate set that includes the reduced
  slice agrees with the reduced VaR decision on that slice (mirrors the existing joint self-consistency gate).
- **INV-13 run/replay parity.** A `--edpp-rule var` trace exported and replayed with identical flags produces
  identical per-request metrics.

## 7. Scope, non-goals, and honest limits

- **In scope:** the oracle VaR mechanism (reduced + joint), the three kernels, the deadline plumbing, the
  experiment harness, and the FINDINGS/STUDY_REPORT write-up (a new E-number).
- **Non-goals (deferred):** a **deployable** VaR (co-resident remaining from the censored `N̂_out` estimate
  rather than the oracle) — built only if the oracle clears the bar; and any admission/scheduling lever.
- **Stated up front, not discovered late:**
  - Oracle results are a **ceiling**, never an achievable policy.
  - Kernel B is *expected* to reproduce E7's neglect; that is a measured result, not a failure of the build.
  - **Saturation gives neglect, not triage.** A saturating utility drives doomed requests to `g ≈ 0`, i.e. no
    signal. True triage requires the doomed to *yield* capacity — an admission/scheduling lever EDPP does not
    hold. This is a real boundary on the whole value-currency approach and is part of the finding regardless of
    which way the bar goes.

## 8. Deliverables

1. `sim/edpp_var.go`: completion model + kernels A/B/C + `VaR_local`/`VaR_disagg` + per-candidate joint VaR.
2. Deployable deadline fields on `RunningReqState` (`sim/admission_estimator.go`), populated in
   `RunningDecodeState()` (`sim/simulator.go`).
3. `EDPPConfig`/`EDPPDecider` VaR rule + kernel + oracle gate; reduced and joint decision branches
   (`sim/edpp.go`).
4. `--edpp-rule var` + `--edpp-var-metric` flags wired on run **and** replay, with the oracle warning
   (`cmd/root.go`, `cmd/replay.go`).
5. Unit + invariant tests (§6); full `go test ./...` green; lint clean.
6. `campaigns/edpp-study/repro_var_oracle.sh` + analysis; FINDINGS + STUDY_REPORT §7 update recording the
   verdict against the bar.

## 9. Risks

- **Completion-model fidelity.** The per-co-resident `Δ_j` reuses the rule's trained-physics but adds a
  step-indexing model (overlap window of R's prefill). If it mis-estimates the overlap, VaR is miscalibrated.
  Mitigation: the externality-asymmetry law test pins the qualitative direction; the ceiling claim is robust to
  modest `Δ_j` error because A only cares about deadline crossings.
- **Interpretation, not code.** If oracle VaR beats `least-ttft` on some archetypes but not others, the verdict
  is nuanced — record the full A/B/C × archetype × seed table, not a single operating point (the same discipline
  the least-ttft spec called for).
- **INV-13 on replay** — the least-ttft baseline shipped this exact miss. Replay wiring is a first-class
  deliverable (§5, §8.4), tested (§6).
