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

**Success criterion (deliberately not "joint beats scorer" — see §5).** On the workloads tested, joint's
divergences from the scorer are explained and net-beneficial (lower counterfactual regret than reduced-EDPP
on cache-uniform synth), AND we characterize where joint does *not* win (the adversarial cache-asymmetric
case). Honest and falsifiable.

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

## 5. Rigor: edge cases where "joint > scorer" can fail (test, don't assume)

The claim is **not** "joint always picks the idle node." `J` trades congestion, cache-warmth, TTFT, and
transfer. Enumerated failure modes and how this design addresses each:

1. **Cache asymmetry (the headline edge case).** A large *unique* prompt on an idle but **cache-cold** node
   pays full prefill (large `a_p` → large `W_p`/`T̂`), while a busier node holding the prefix has small `a_p`.
   A cache-*blind* joint rule would mispick the idle-cold node and **lose to the cache-aware
   `precise-prefix-cache` scorer**. → Addressed by **per-candidate `a_p`** (§2.4): `J` sees each node's cache
   warmth and can prefer warm-but-busy over idle-cold. **Must be tested** on an adversarial cache-asymmetric
   workload (§6), not assumed.
2. **`z=0` regime (no SLO pressure).** The TTFT/ITL terms vanish; `J` reduces to congestion + transfer. Cache
   still enters via `W_p` (congestion), so per-candidate `a_p` keeps this correct; but note the decode-node
   choice is then occupancy+cache only.
3. **True ties.** Identical candidates → arbitrary argmin "divergence" that isn't a real win. → **Deterministic
   tie-break** (lowest instance index, INV-6), and the divergence study excludes ties from "improvement".
4. **Per-candidate estimator error.** `T̂` is an estimate; under saturation a candidate's `T̂` can be biased,
   skewing the argmin. → The occupancy-aware estimator is the validated one; the divergence study logs
   `J_scorer` vs `J_joint` so a bad argmin is visible.

## 6. Validation

- **Counterfactual regret, cache-uniform (the win):** run `--edpp-joint --edpp-tadm-estimator rollforward` on
  1P2D synth (shared prefix ⇒ cache ~uniform across decode nodes), through `repro_counterfactual.sh`. Expect
  regret **below** reduced-EDPP's 0.06 (ideally ~0 — joint recovers the `instance_1` decode placement),
  baseline goodput ≥ 0.99. This is the primary target.
- **Counterfactual regret, cache-asymmetric (the cost-lever demonstration AND honesty test):** a workload of
  **unique large prompts** where one decode node is cache-warm and another idle-cold — a *real* per-instance
  cost heterogeneity the simulator serves faithfully (no hardware change needed). This is the positive test of
  the **cache-cost lever**: joint's per-candidate `a_p` should make it prefer the warm node (or accept a busier
  warm node over an idle-cold one) — a decision the occupancy-only reduced balance term cannot express but the
  `precise-prefix-cache` scorer can, so it is also the fair head-to-head. Report joint vs reduced regret and
  the scorer-divergence direction. If joint mispicks (cache-blind failure resurfacing), that is a recorded
  finding, not hidden.
- **Scorer-vs-joint divergence study:** from the §2.5 log, report divergence rate (`d` and `p`), and on
  divergent decisions the direction (does joint pick lower-occupancy / lower-`a_p` / lower-`J`?), and whether
  divergence correlates with recovered regret. This is the mechanism figure.
- **§5.5 reduction (refactor-safety unit test):** joint `J` restricted to the scorer's `d` reproduces the
  reduced decider's local-vs-disagg decision on that `d`.
- **Hand cases:** two decode nodes, one loaded → joint picks the emptier for a kept-local request; idle
  cluster → local; large-prompt + one warm node → joint picks warm (cache case in miniature).
- **Invariants:** INV-6 determinism (deterministic tie-break); **reduced path byte-identical when flag off**;
  INV-9 (per-candidate `a_p`/`T̂` read only input-side + cache + `N̂_out`, never `OutputTokens`).

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
