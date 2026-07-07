# Full-Joint P/D Routing — Sub-Project 1: Homogeneous Joint Mechanism (`--edpp-joint`) (Design)

Date: 2026-07-07
Branch: `feat/edpp-estimator-validation`
Related: `docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` (§3.4 action set, §5.3 decision
rule, §5.5 reduction & "cache affinity subsumed by J"); `campaigns/edpp-study/FINDINGS.md` "Counterfactual
regret" (the measured target this sub-project chases).

## 1. Goal & scope

Implement the **joint** drift-plus-penalty rule as a mode on the EDPP decider (`--edpp-joint`): enumerate
every action `(d,p) ∈ ℳ×(𝒫∪{local})` and select the **argmin** of the §5.3 objective `J(d,p)` — choosing
the decode instance `d` itself (via `DecodePodOverride`) and the prefill location `p` (via `PrefillPodHint`),
instead of delegating `d` to the external scorer. **Homogeneous P/D pools** (single coeff set), per-class
scalar `z`, but **per-candidate-node cache-aware** `a_p` (see §5). The reduced path (flag off) stays
byte-identical.

**"Homogeneous" here means homogeneous _hardware_ — the decode pool is still heterogeneous in cost.**
The simulator serves instance hardware per role pool-wide (`resolveConfigForRole`), so it cannot cleanly
serve two *same-role* decode instances at different *speeds* — heterogeneous `θ_i` is not testable without a
simulator-side change (deferred, §4). **But even with identical hardware, two decode instances hold different
KV-cache contents**, so for a given request they have different uncached prompt `a_p` → different `W_p`/`T̂` →
**different per-request cost**. That is a genuine per-instance heterogeneity, and one the simulator serves
*faithfully* (per-instance KV cache is real and evolves during the run). So this sub-project already exercises
**two real per-instance levers** for the decode-node choice — **occupancy `Q_i`** (load) and **cache-induced
cost** (per-node `a_p`) — and only the *hardware-coefficient* lever (`θ_i`) is deferred. Not degenerate: it is
the joint rule operating on cost + load heterogeneity that the simulator actually produces. Concretely it has
a measured target (**recover the ~0.06 counterfactual regret**, occupancy-driven) *and* a cost-lever target
(the cache-asymmetric case, §6).

**Success criterion (descriptive, not a predicted win).** Sub-project 1 succeeds when (a) the mechanism is
*correct* — the §5 invariants hold (per-node cache-aware `a_p`, deterministic tie-break, byte-identical
reduced path, INV-9) — and (b) it produces an honest **characterization** of what the joint rule does vs the
scorer across a topology × workload sweep: where it diverges, in which direction, and the observed effect on
regret/goodput. We do **not** pre-commit to "joint wins." The open questions the sweep answers include: does
joint recover the measured ~0.06 counterfactual regret on cache-uniform synth? what does it do when cache is
asymmetric across nodes? — but the answers are whatever the runs show.

## 2. What's new (infra)

1. **Per-instance congestion `Q_i`** (formulation gap 1). The decider maintains
   `qByInstance map[InstanceID]edppPoolWork` (prefill-side and decode-side waiting work per instance),
   incremented at `OnRoute` for the **chosen** `d` (decode work `W_d`) and prefill location (`W_p` on `p`, or
   on `d` when local), drained per-instance at `OnAdmit`/`OnComplete`/`Forget`. Requires threading the chosen
   `(d,p)` instance IDs into `OnRoute`. **The existing pool-level `qpWork`/`qdWork` are left untouched** and
   still drive the reduced path (their sum equals the per-instance totals, so no divergence); `qByInstance`
   is populated always (cheap) but **read only by the joint path**, guaranteeing byte-identical reduced runs.
2. **Candidate enumeration + argmin** (gap 4). When joint, `Decide` loops `d ∈ state.Snapshots` (decode
   candidates) × `p ∈ prefillSnapshots() ∪ {local}`, computes `J(d,p)` (§3), takes the argmin (deterministic
   tie-break, §5), and emits `DecodePodOverride = d*`, `PrefillPodHint = p*` (empty when `p*=local`).
3. **Per-candidate `T̂`** (gap 6). Assemble the `AdmissionContext` per candidate decode snapshot (and per
   candidate prefill snapshot when disaggregating) and evaluate the occupancy-aware estimator
   (`rollforward`, per the driver default) for each — not just the scorer-selected snapshot.
4. **Per-candidate cache-aware `a_p`** (§5.5's "cache affinity subsumed by J"). For each candidate node,
   `a_p(node) = len(InputTokens) − cachedTokens(d.cacheQuery[node](InputTokens))` (block-aligned, mirroring
   the reduced path's cache adjustment). `d.cacheQuery` is already an instance-keyed map (`sim/edpp.go:245`),
   so this is a query per candidate. `W_p` and `T̂` for that candidate use its own `a_p`.
5. **Scorer-vs-joint divergence instrumentation** (the side study). Per decision, log
   `scorer_d` (`state.SelectedInstance` — the scorer's decode pick), `joint_d`/`joint_p` (the argmin),
   `scorer_p` (a **shadow-run** of the prefill scorer over the prefill snapshots at decision time —
   compute-only, logged, not acted on; populated on disaggregate decisions), an `agree` flag, and
   `J_scorer` (J evaluated at the scorer's `(d, shadow-p)`) vs `J_joint`. A column set on the
   `--edpp-decision-trace` (or a small companion). Keep the scorer running in joint mode so the comparison
   exists (a future efficiency pass could skip it).

## 3. The objective, per candidate

Reduced to homogeneous coefficients with per-class scalar `z` (§5.3):

```
J(d,p) =  q_d·W_d(a_r, ô)                            (decode congestion, always on d)
        + (p=local ? q_d : q_p)·W_p(a_p(loc), a_r)   (prefill congestion; a_p from the prefill LOCATION's cache)
        + z_ttft·(T̂_disagg(d,p) or T̂_local(d))       (TTFT term, per-candidate occupancy-aware T̂)
        + z_itl·(m_dec(d) + 1{p=local}·m_pf(d))       (ITL term, per-class z, per-candidate marginal)
        + V·c_xfer·1{p≠local}                         (transfer penalty)
```
all normalized by `W*` as in the reduced rule (same normalizers, homogeneous). `a_p(loc)` is the prefill
location's cache-adjusted uncached tokens (§2.4); for `p=local` it is `d`'s cache, for `p∈𝒫` it is `p`'s
cache. `ô = N̂_out` (INV-9-safe). argmin over `(d,p)`.

## 4. Deferred to sub-project 2 (gated on simulator work)

- **Heterogeneous per-instance `θ_i`** (gap 3) — requires the simulator to serve same-role instances at
  different hardware (intra-pool heterogeneity), which `resolveConfigForRole` does not support today. That
  simulator-side change + `θ_i` loading/indexing + the heterogeneous decode-node "cost" lever is sub-project 2.
- **Per-instance ITL-deficit `Z^I_i`** (gap 2) — keep per-class scalar `z_itl` here.

## 5. Correctness requirements + methodology (we do NOT pre-judge behavior)

**Correctness (engineering invariants the mechanism must satisfy regardless of outcome):**
- **Per-candidate cache-aware `a_p` for BOTH pools.** Prefill-only nodes also hold prefixes, so the prefill
  pool is cache-heterogeneous just like the decode pool. `a_p` is computed from *each candidate location's*
  cache — the decode node's cache for `p=local`, and *each prefill node's* cache for `p∈𝒫` (so the `p`-choice
  is a real cost decision, not just balance). Both via the instance-keyed `d.cacheQuery`.
- **Deterministic tie-break** (lowest instance index) so identical candidates never introduce nondeterminism
  (INV-6); the divergence study does not count a tie as an "improvement".
- **Occupancy-aware `T̂` per candidate** (`rollforward` driver) — the validated estimator, evaluated per
  candidate snapshot.
- **Reduced path byte-identical** when `--edpp-joint` is off; **INV-9** (per-candidate `a_p`/`T̂` read only
  input-side quantities + cache + `N̂_out`, never `OutputTokens`).

**Methodology — run and observe, don't theorize the outcome.** We make **no a-priori claim** about when joint
beats, ties, or loses to the scorer. `J` trades occupancy, cache-warmth (both pools), TTFT/ITL deficit, and
transfer cost — and how those compose depends on the topology and workload in ways not worth predicting on
paper. The validation (§6) is therefore an **exploratory sweep**: set up a topology, run a workload, observe
what joint does vs the scorer; change the topology or workload, observe again. The counterfactual-regret
harness and the scorer-vs-joint divergence log are the *instruments*; the *findings* are whatever the sweep
shows. The dimensions worth varying (because they change which term dominates `J`) are: cache warmth /
asymmetry across nodes, per-instance occupancy, SLO pressure (`z` zero vs positive), prompt size, and transfer
cost — but these are knobs to sweep, not verdicts to confirm.

## 6. Validation — an exploratory topology × workload sweep

Two things are *pass/fail* (the mechanism must be correct); the rest is *observe and report* (run the sweep,
see what happens):

**Correctness gates (pass/fail, no behavior claim):**
- **§5.5 reduction (unit test):** joint `J` restricted to the scorer's `d` reproduces the reduced decider's
  local-vs-disagg decision on that `d`.
- **Reduced path byte-identical** when `--edpp-joint` is off; **INV-6** determinism (deterministic tie-break);
  **INV-9** (per-candidate `a_p`/`T̂` read only input-side + cache + `N̂_out`, never `OutputTokens`).
- **Instrument sanity:** the scorer-vs-joint divergence log is populated (`scorer_d/p`, `joint_d/p`, `J`s);
  the self-consistency gate of the regret harness still passes on the joint plan.

**Exploratory sweep (observe, don't predict).** Run joint vs reduced (+ the divergence log + counterfactual
regret) across a small matrix, and *report what happens* — no pre-committed winner:
- **Topologies:** at least 1P2D and one other (e.g. 2P2D or 1P3D) — varying the number of decode candidates
  and prefill candidates the argmin ranges over.
- **Workloads:** at least (i) cache-uniform synth (shared prefix — the setup where the existing 0.06 regret
  was measured; does joint recover it?), and (ii) cache-asymmetric unique-large-prompt (one node warm, one
  cold — what does joint's per-node `a_p` do?). Add others as questions arise.
- **Per cell, report:** counterfactual regret (joint vs reduced), baseline goodput, scorer-vs-joint divergence
  rate (`d` and `p`), and on divergent decisions the observed direction (occupancy? `a_p`? `J`?). These are
  *observations*, not confirmations.
- **The dimensions to vary** (they change which `J` term dominates): cache warmth/asymmetry, occupancy, SLO
  pressure (`z`=0 vs >0), prompt size, transfer cost. Change one, re-run, see the effect. Expect surprises —
  record them as findings (the step-function and estimator-driver findings this session came exactly this way).
- **Hand cases (tiny, for intuition/debug, not claims):** idle cluster → local; two decode nodes, one loaded;
  large prompt + one warm node. Use them to sanity-check the mechanism, not to assert general behavior.

## 7. Deliverables & staging (for the plan)

1. **Per-instance `Q_i`** — `qByInstance` bookkeeping + thread chosen `(d,p)` into `OnRoute`; pool-level
   untouched; unit tests (per-instance sums == pool-level; drain correctness).
2. **Joint `Decide`** — enumeration + per-candidate `a_p`(cache)/`T̂`/`J` + argmin + deterministic tie-break;
   `--edpp-joint` flag (run+replay); emit `DecodePodOverride`/`PrefillPodHint`; scorer-vs-joint divergence
   logging (scorer_d/p shadow, joint_d/p, J's). Reduced path byte-identical. Unit tests incl. the §5.5
   reduction and the cache-vs-occupancy hand case.
3. **Validation + findings** — counterfactual regret (cache-uniform + cache-asymmetric), the divergence-study
   analysis + `repro_joint.sh`, FINDINGS "Joint mechanism" + README pointer.

## 8. Non-goals

Heterogeneous `θ_i`, simulator intra-pool hardware, per-instance `Z^I_i`, the global MILP optimum (all later).
This sub-project delivers the joint *mechanism* and answers whether it recovers the measured homogeneous
decode-placement regret — with the cache edge case handled correctly and tested, not assumed.
