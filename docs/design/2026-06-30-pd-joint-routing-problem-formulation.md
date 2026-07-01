# Joint Routing + Disaggregation Decision: Problem Formulation and System Model

**Date:** 2026-06-30
**Status:** Draft — problem/system model only. No algorithm or implementation committed yet.
**Branch:** `design/pd-joint-routing`

> In this document we fix the **system model** and the **problem statement** for prefill/decode
> (P/D) routing in a disaggregated LLM-serving deployment. We adopt one stance throughout: we
> describe the system at **two fidelities** — an *exact, observable* layer that the deployed policy
> evaluates from measured state (§3.7–§3.8), and a *tractable, analytical* layer we use for
> closed-form reasoning and then validate against the first. We deliberately stop short of
> committing a specific estimator for each latency term; those choices are deferred (§9).

---

## 1. Motivation: routing and deciding are one problem, currently split

Today the system makes two decisions with two separate mechanisms, in sequence:

1. **Routing** (the scorer / endpoint-picker): given the decode pool, pick a decode instance;
   given the prefill pool, pick a prefill instance. This is intra-pool load balancing.
2. **Deciding** (the P/D decider, e.g. `prefix-threshold` or EDPP): given the decode instance
   is already fixed, choose whether to prefill locally or disaggregate.

This is a **hierarchical decomposition** of a single underlying assignment problem. It is not
lossless, because the two decisions are coupled:

- The *value* of disaggregating a request depends on **which** decode instance it lands on:
  offloading prefill helps a congested decode instance far more than an idle one.
- The *best* decode instance depends on **whether** the request will be disaggregated: if it
  will, choose the instance on decode capacity alone (it never prefills); if it won't, choose
  on combined prefill+decode capacity. These can be different instances.

A decomposed pipeline picks the decode instance *blind to the split*, then picks the split
*blind to the decode alternatives it could have chosen*. A **joint** formulation treats
`(instance, split)` as one action and can express choices the pipeline cannot.

In this document we formalize that joint problem.

---

## 2. Scope decisions (fixed for this formulation)

| # | Decision | Choice fixed here | Rationale / deferred alternative |
|---|----------|-------------------|----------------------------------|
| S1 | Instance roles | **Fixed per instance for the horizon** (control-plane decision) | Per-request role flipping is not realizable in llm-d / Dynamo data planes (see §7). Runtime role *re*allocation = **deferred extension** (§8). |
| S2 | Capability classes | **Two classes: prefill-only (P) and mixed (M)** | "Decode-only" is not a real class: a vLLM decode worker is a full engine and can prefill locally. Matches llm-d's `prefill` vs `prefill-decode`/`decode` operational reality. |
| S3 | Prefill target set when disaggregating | **(i) prefill-only (P) nodes exclusively** | A mixed node never prefills *another* request. Option (ii) "any prefill-capable node" = **deferred extension** (§8). |
| S4 | Instance homogeneity | **Heterogeneous within both pools** | P instances may differ from one another; M instances may differ from one another (mixed GPUs, TP degree, KV capacity, service rate). This is what makes intra-pool instance choice cost-relevant, not just balance-relevant — and what makes the joint policy non-trivial. |
| S5 | Predictor accuracy | **Out of scope** | How latency terms are *estimated* is a separate work item, addressed after the model is fixed. |

---

## 3. System model

### 3.1 Instances

A fixed set of instances $\mathcal{I}$, each carrying a capability label fixed for the horizon:

$$c(i) \in \{\mathsf{P}\ (\text{prefill-only}),\ \mathsf{M}\ (\text{mixed: prefill and decode})\}$$

Derived sets:

- decode-capable instances: $\mathcal{M} = \{i : c(i)=\mathsf{M}\}$
- prefill-capable instances: $\mathcal{P}^{+} = \{i : c(i)\in\{\mathsf{P},\mathsf{M}\}\}$
- prefill-pool (per S3, the disaggregation prefill targets): $\mathcal{P} = \{i : c(i)=\mathsf{P}\}$

**Heterogeneity (S4).** Each instance $i$ carries its own hardware/serving parameters: GPU type,
tensor-parallel degree, KV-block capacity $K_i$, interconnect bandwidth, and — crucially — its own
**latency-coefficient vector**

$$\theta_i = (\alpha^D_i,\ \alpha^P_i,\ C0_i,\ C1_i,\ C_{\!pf,i},\ C_{\!attn,i})$$

A $\mathsf{P}$ instance needs only the prefill subset $(\alpha^P_i, C_{\!pf,i}, C_{\!attn,i})$; an
$\mathsf{M}$ instance carries the full vector (it does both phases). No two instances are assumed
identical, even within the same class.

**Consequence — work is request×instance.** Because the coefficients are per-instance, the work a
request lands (§3.6) depends on *which* instance serves it: $W_d$ is evaluated with the decode
node's $(C0,C1)$, $W_p$ with the prefill location's $(C_{\!pf},C_{\!attn})$. The *same* request has
different cost on different instances — this is exactly why instance choice changes cost (not just
load balance) and why the joint policy (§5.3) is non-trivial under heterogeneity.

**Calibration note (practical).** Coefficients are fit per *hardware/TP profile* (model × GPU × TP),
not per physical instance: instances sharing a profile share $\theta$. So heterogeneity = a small
set of profile coefficient-files plus an instance→profile assignment, not $|\mathcal{I}|$
independent fits.

Topologies are simply initial label vectors:

- `never@4` = four $\mathsf{M}$ instances (no dedicated prefill).
- `2P2D` = two $\mathsf{P}$ + two $\mathsf{M}$ instances.

### 3.2 Requests

Each request $r$ is characterized by:

- arrival time $t_r$,
- prompt (prefill) length $a_r$ (known at arrival),
- output (decode) length $o_r$ (a random variable, revealed only as decode proceeds),
- an SLO class with targets $(\tau^{\text{ttft}}_r, \tau^{\text{itl}}_r)$ — time-to-first-token
  and inter-token latency.

### 3.3 Phases and costs

A request proceeds through up to three cost stages:

1. **Prefill** — process $a_r$ prompt tokens at the chosen prefill location $p_r$; produces a KV
   cache of size $\propto a_r$. Cost depends on $a_r$ **and** on the heterogeneous parameters of
   $p_r$.
2. **KV transfer** — *only if disaggregated* ($p_r \neq d_r$): move the KV cache from $p_r$ to
   $d_r$; cost $\propto a_r$ and the interconnect. Adds to TTFT.
3. **Decode** — run $o_r$ autoregressive steps on $d_r$; the KV cache must be resident on $d_r$.
   Per-step cost depends on the running batch / context and on $d_r$'s parameters.

### 3.4 Decision / action

For each request the policy emits a single action — a (decode instance, prefill location) pair:

$$a_r = (d_r,\; p_r), \qquad d_r \in \mathcal{M},\quad p_r \in \mathcal{P} \cup \{\text{local}=d_r\}$$

- $p_r = \text{local}$ ⇒ **aggregated**: $d_r$ prefills and decodes the request itself (no transfer).
- $p_r \in \mathcal{P}$ ⇒ **disaggregated**: prefill on a dedicated prefill-only node, KV transfer,
  decode on $d_r$.

The single set-union $\mathcal{P} \cup \{\text{local}\}$ folds the **"disaggregate?"** question
into the **"which instance?"** question: route and decide are coordinates of one action.

**Feasibility.** Because every decode-capable instance is itself prefill-capable
($\mathcal{M}\subseteq\mathcal{P}^{+}$), the **local option is always feasible**. No request is
ever forced to disaggregate by capability alone. (This is the clean consequence of dropping the
decode-only class in S2.)

### 3.5 Resources and coupling

- **KV capacity.** Each instance $i$ has finite KV-block capacity $K_i$, which caps the number of
  requests it can concurrently hold in its decode batch.
- **Compute.** Prefill is compute/throughput-bound; decode is memory-bandwidth- and KV-capacity-bound.
- **Co-residency interference on M.** On a mixed node, a local prefill competes with that node's
  ongoing decode steps for the same iterations (head-of-line interference). This is the intrinsic
  cost of `local` that disaggregation trades against the transfer cost.

### 3.6 Work model (per-request demand)

Each request contributes a quantum of **work** (in µs) to the instance(s) it touches. Work is the
forward/demand quantity that feeds the congestion queue and the forward latency estimate (it is
*not* the SLO-deficit queue, which reads realized latency — see §5.2). Both phases follow one
principle.

**Unifying principle.** Work in each phase splits into:

- **compute** = coeff × (number of *new* tokens computed) — linear in tokens;
- **attention** = coeff × (number of new tokens) × (average context they attend over), where the
  average context is `start_context + new_tokens/2` (context grows linearly across the phase, so
  its mean is the midpoint).

| phase | new tokens | start context | avg context | attention work |
|-------|-----------|---------------|-------------|----------------|
| prefill | uncached `a_p` | cached prefix `p_cached` | `p_cached + a_p/2` | `CAttn·a_p·(p_cached + a_p/2)` |
| decode | output `o` | full prompt `input` | `input + o/2` | `C1·o·(input + o/2)` |

**Decode work** (on decode node `d`, using `d`'s coefficients):

```
W_d(d) = C0_d · o   +   C1_d · o · (input + o/2)
```

- `C0_d` = decode per-step compute overhead (context-independent), `C1_d` = KV-read cost per resident
  token (the `decode_compute_coeff` / `decode_memory_coeff` in plain terms) — **per-instance** (§3.1).
- `input` is the **full** prompt length, **not** the uncached `a_p`: prefix-cache hits reduce
  prefill *compute*, but the full prompt resides in KV and is re-read every decode step.
- `o` is the estimated output length `N̂_out` (running per-class mean, or a demand model). INV-9-safe:
  `input` is known at routing; `o` is estimated, never read from actual `OutputTokens`.

**Prefill work** (on prefill location `q`, using `q`'s coefficients; `q = p` for disaggregated,
`q = d` for local). With full input `N`, uncached `a_p`, cached prefix `p_cached = N − a_p`:

```
W_p(q) = CPf_q · a_p   +   CAttn_q · a_p · (p_cached + a_p/2)
       = CPf_q · a_p   +   (CAttn_q/2) · a_p²   +   CAttn_q · a_p · p_cached
```

- `CPf_q` = exposed prefill compute per token, `CAttn_q` = prefill attention coefficient (µs/token²)
  — **per-instance** (§3.1). For the `local` option these are the mixed node `d`'s own prefill
  coefficients.
- Only the uncached tokens are computed as queries, but each attends over the **full** context up
  to its position (cached prefix included) — hence `a_p · (p_cached + a_p/2)`, not `a_p²` and not
  `N²`.

**Relationship to the prior `Wp`/`NomDecodeCtx` forms (corrections recorded here):**

- The decode-work form `w_d = N̂_out · δ̄_decode(NomDecodeCtx) = N̂_out·(C0 + C1·NomDecodeCtx)` is
  **deprecated** in this formulation. `NomDecodeCtx` is a single fixed assumed context (e.g. 2048)
  plugged in for every request and every step; it ignores the request's actual prompt length and
  the growth of context during decode, mis-estimating the (dominant) memory term in both directions.
  `W_d` above replaces it with the trajectory integral — same calibrated `C0`, `C1`, no refitting.
- The prior prefill work `Wp = CPf·a_p + (CAttn/2)·a_p²` is correct **only in the no-cache limit**
  (`p_cached = 0`, `a_p = N`). With prefix caching it drops the cross term `CAttn·a_p·p_cached` —
  the cost of uncached tokens attending back over the cached prefix. `W_p` above restores it.
  In no-cache traffic (e.g. tiny-prompt synth) the cross term ≈ 0 and the two agree; it matters on
  shared-prefix / RAG workloads.

The work in $W_p$ and $W_d$ is a *demand*, not a latency. The next two subsections connect it to
wall-clock time, and in doing so expose a subtlety we must respect: a request's work is drained
from a queue the moment it enters the running batch, so a backlog that counts only *waiting* work
omits the *currently-running* batch entirely — the very thing that decides how long a new arrival
waits. We therefore develop the time model with that residual occupancy in view.

### 3.7 From work to time: the per-iteration identity

We connect work to time through the mechanics of continuous batching, in a form we can evaluate
from observed state.

At any instant, instance $i$ runs a *batch* $B_i$ — the set of requests active on it. Each active
request $r$ sits at a definite *stage* $s_r$: either a specific prefill chunk, or decode step $k$
(its $k$-th output token). We write $\delta(s_r)$ for the **marginal work** request $r$ contributes
in the current iteration given its stage. It is the single-iteration slice of the §3.6 work, read
off the stage:

- prefill chunk of `chunk` tokens: $\delta = C_{\!pf}\cdot\text{chunk} + C_{\!attn}\cdot\text{chunk}\cdot(\text{context cached so far})$;
- decode step $k$: $\delta = C0 + C1\cdot(\text{input}+k)$.

Every quantity here is observed from $r$'s current stage and the instance coefficients $\theta_i$;
nothing is estimated except, for stages not yet reached, the output length $o$.

One iteration then takes
$$T^{\text{iter}}_i \;=\; \alpha_i \;+\; \sum_{r\in B_i}\delta(s_r),$$
where $\alpha_i$ is the per-iteration baseline from $\theta_i$ (kernel launch, weight load) and the
sum is the batch's total marginal work. This is the state-resolved form of the linear iteration-time
law $T=\alpha+N\delta$; we keep a per-request $\delta(s_r)$ rather than one shared mean $\delta$,
precisely because we can observe each stage.

**Work is not wall-clock service.** Under continuous batching a request advances exactly one stage
per iteration, so its remaining lifetime is a fixed number of iterations — its remaining *stage
count* $(c_{\text{rem}}+o_{\text{rem}})$ — while each of those iterations lasts $T^{\text{iter}}_i$,
which depends on the *whole* co-resident batch. Two distinct quantities therefore attach to a
running request $r$, and we will need both:

- its **residual work** $W'_r=\sum_{\text{remaining stages}}\delta(s)$ — $r$'s future contribution
  to every co-resident request's iteration time;
- its **departure**, set by its remaining stage count (it frees a slot and its KV only when it
  completes), *not* by $W'_r$.

We obtain both by bookkeeping: we observe how far $r$ has progressed (completed chunks and $k$
tokens), read off the remaining stages, and take $W'_r$ as the tail of the same §3.6 integral —
exact in the known quantities, depending only on the estimated $o$. This decoupling (own work
spread across shared iterations; departure governed by stage count) is why a request's service time
is not its work, and it is the fact §3.8 uses to obtain the admission delay.

### 3.8 Admission delay, and how we obtain it

The part of TTFT that queueing governs is the **admission delay** $T^{\text{adm}}_r$: the time from
when request $r$ becomes eligible (enqueued after routing) until it enters the running batch and
begins its own prefill. We obtain it in two complementary ways, neither of which assumes anything
about the arrival process.

**By observation (ground truth).** The simulator — and a real server, through its metrics —
records $r$'s enqueue and schedule instants, so
$$T^{\text{adm}}_r \;=\; t^{\text{schedule}}_r - t^{\text{enqueue}}_r$$
is measured exactly. We use it both as the target any forward estimate must reproduce and to
calibrate the coefficients.

**By forward prediction (what the policy needs).** A router commits to an action before it can
observe the outcome, and it can never measure the alternative it did not take, so it needs an
estimate of $T^{\text{adm}}$ under a candidate placement. We form one from observed state, without a
stochastic model:

- If the target instance has a free batch slot and enough free KV for the prompt *now*, the request
  is admitted on the next iteration and $T^{\text{adm}}\approx 0$.
- Otherwise we roll the observed batch forward deterministically: each iteration costs
  $T^{\text{iter}}_i$ (§3.7), running requests depart after their remaining stage counts, and we
  accumulate elapsed time until a slot and the required KV free. This is a short, deterministic
  look-ahead over observed state and the work formulas — not a solved queue.
- Where a single aggregate suffices, Little's law gives $T^{\text{adm}}\approx \bar L^{\text{q}}_i/\lambda^{\text{adm}}_i$,
  with the mean waiting count $\bar L^{\text{q}}_i$ and the admission rate $\lambda^{\text{adm}}_i$
  both measured. Little's law holds for any ergodic system, so this estimate is valid however
  bursty or correlated the arrivals are.

**Why the arrival process does not enter.** For a disaggregated request, the decode instance sees
as its arrivals the prefill instances' *departures* delayed by KV transfer — a correlated,
non-Poisson stream. Both estimates above avoid this: the roll-forward conditions on the *actual
current batch*, and Little's law needs no arrival assumption at all. A memoryless (Poisson)
description remains reasonable for the *external* request stream seen by the prefill instances and
by the aggregated (`local`) path; we use it only in the analytical layer where it is needed
(steady state, and the offline yardstick of §6), and we validate that layer against the observed
$T^{\text{adm}}$ above. This is the observable layer promised in the opening: the deployed policy's
forward estimates rest on it, and the analytical layer must reproduce it.

---

## 4. Objective

Maximize **goodput** (equivalently, minimize the time-average rate of SLO violation across TTFT
and ITL targets), subject to:

- per-instance KV-capacity and compute constraints, and
- queue stability (no instance's backlog grows unboundedly).

---

## 5. Policy family and baselines

Every routing policy in this study is a **restriction of the same action set** in §3.4, which
keeps comparisons clean:

| Policy | Restriction on $a_r$ |
|--------|----------------------|
| `never` (aggregate) | force $p_r = \text{local}$ |
| `always` | force $p_r \neq \text{local}$ (always remote prefill) |
| `prefix-threshold` | $p_r = \text{local}$ iff non-cached prompt tokens $\le$ threshold; decode instance from scorer |
| EDPP (current) | decode instance from scorer; split from the Lyapunov rule, using **pool-level** virtual queues |
| **Joint (target)** | choose $(d_r, p_r)$ jointly — see §6 |

### 5.1 The joint policy direction (not yet specified in detail)

The joint objective chooses the full action in one optimization, e.g. a drift-plus-penalty form
with **per-instance** virtual queues $Q_i$:

$$a_r^{\*} \;=\; \arg\min_{a}\; \Big[\, V\cdot \text{penalty}(a) \;+\; \sum_{i\in\mathcal{I}} Q_i \cdot \Delta\text{work}_i(a) \,\Big]$$

where `penalty(a)` is the transfer/KV-movement cost (zero for `local`) plus any soft SLO cost, and
$\Delta\text{work}_i(a)$ is the work action $a$ adds to instance $i$. In this single argmin,
load balancing emerges from the $\sum_i Q_i\,\Delta\text{work}_i$ term (work flows to lighter
instances) and the P/D split emerges from the penalty-vs-queue trade-off — there is no separate
scorer.

### 5.2 Queue model (LOCKED)

The formulation uses **two families of state**, deliberately kept distinct:

1. **Congestion queues — one scalar work-backlog queue $Q_i$ per instance.** This is the only
   role of the congestion family: **stability** (is instance $i$ being given more work than its
   throughput can sustain?). Total committed work — prefill *and* decode — pools into a single
   $Q_i$, because a mixed node is **one server, not two**: a vLLM engine interleaves prefill chunks
   and decode steps in the same iterations, contending for the same compute and KV. Two independent
   per-phase congestion queues would model an M node as two parallel servers, which it is not.
   Heterogeneity (S4) enters as each instance's own service rate.

2. **SLO-deficit queues — split by latency type.** A **TTFT-deficit** queue and an **ITL-deficit**
   queue (per class; possibly per instance), accumulating how much each latency target is being
   missed. The prefill/decode asymmetry lives *here*, for free: TTFT is a prefill-side latency, ITL
   a decode-side latency — so splitting the SLO family by latency type captures the phase
   distinction **without** doubling the congestion state.

3. **Co-residency interference is a cost, not a queue.** "A local prefill on an M node inflates its
   decode ITL" is a penalty/cost statement (§3.5); it feeds the ITL-deficit drift, weighed against
   the transfer cost the disaggregated option pays. This is the heart of the P/D trade-off and
   belongs in `penalty(a)` / the drift cost, not in a second congestion queue.

**Why this split.** Stability is a scalar-work property of each box (one congestion queue is
correct and matches the single-server reality); the phase-specific latency effects belong to the
SLO family (TTFT vs ITL) and the cost (interference vs transfer). This keeps the drift algebra
tractable and keeps one capacity constraint per box, which the offline MILP (§6) can express
directly.

**Fallback trigger (escape hatch).** If a concrete decision the policy must make turns out to be
inexpressible with a single congestion queue — e.g. we want the *congestion* term itself, not the
ITL-deficit term, to steer prefill away from decode-busy M nodes — revisit two-queue congestion
(option b). Not expected to be needed.

The exact penalty terms and the SLO-deficit drift are specified in §5.3; the estimators
(the forward quantities) remain deferred (§9).

### 5.3 Penalty and drift — the joint decision rule

**State.** Three queue families (§5.2), all maintained by observation:

- `Q_i` — congestion backlog at instance `i` (work-µs), conservation-bookkept.
- `Z^T_c` — TTFT-deficit virtual queue for SLO class `c`.
- `Z^I_i` — ITL-deficit virtual queue at instance `i`.

**Lyapunov function** (quadratic, standard):

```
L(t) = ½ [ Σ_i Q_i²  +  Σ_c (Z^T_c)²  +  Σ_i (Z^I_i)² ]
```

**Drift-plus-penalty decision.** For each request `r` of class `c`, choose the action
`a = (d, p)` that minimizes the part of the one-step drift bound it controls, plus `V ×` penalty:

```
J(a) = Σ_i Q_i · Δwork_i(a)            (congestion drift)
     + Z^T_c · T̂(a)                    (TTFT-deficit drift)
     + Z^I_d · [ m_dec(d) + 1{p=local}·m_pf(d) ]   (ITL-deficit drift, on the decode node d)
     + V · c_xfer · 1{p ≠ local}        (penalty)
```

with the forward (modeled) quantities:

- `Δwork_i(a)` — work the action lands on instance `i`, from §3.6, **evaluated with each target
  instance's own coefficients** (§3.1):
  - `p = local`: `Δwork_d = W_p(d) + W_d(d)` (both phases on `d`, using `d`'s prefill *and* decode
    coefficients); all other instances 0.
  - `p ∈ 𝒫`: `Δwork_d = W_d(d)`, `Δwork_p = W_p(p)`.
  - Under heterogeneity the same request yields different `Δwork` on different instances, so the
    congestion term `Σ_i Q_i·Δwork_i` ranks instances by cost *and* load jointly.
- `T̂(a)` — forward TTFT estimate for the request under action `a`, obtained as in §3.8 (the
  admission-delay roll-forward or Little's-law estimate, both occupancy-aware):
  `T̂_local(d) = T^adm(d) + own-prefill-on-d`;
  `T̂_disagg(d,p) = T^adm(p) + prefill-on-p + transfer + T^adm(d)`. The `−τ_c` term in the
  constraint is constant within a class and drops out of the argmin.
- `m_dec(d)` — marginal per-step decode pressure the request adds to `d`'s batch (steers the
  choice of `d` toward ITL-healthy decode nodes; identical for local and disagg, so it does **not**
  drive the P/D split).
- `m_pf(d)` — per-step prefill-chunk interference on `d`'s co-resident decodes when prefill is
  **local** (the `c_pf · chunk` term). For disaggregation the prefill lands on a `𝖯`-only node,
  which carries no ITL constraint, so it contributes **no** ITL drift — this asymmetry is the
  interference cost the `local` option pays and the `disaggregate` option escapes.

**Joint argmin.** The policy selects over the full action set in one optimization:

```
a* = argmin over { (d, local) : d ∈ 𝓜 } ∪ { (d, p) : d ∈ 𝓜, p ∈ 𝒫 }  of  J(a)
```

Load balancing (choice of `d`, choice of `p`) and the P/D split fall out of the same argmin; there
is no separate scorer.

**Reduction to the pairwise P/D rule (fixed `d`).** For a fixed decode node `d` (e.g. one chosen by
an external scorer, as today) and the best prefill target
`p* = argmin_{p∈𝒫} [ Q_p·W_p + Z^T_c·T̂_disagg(d,p) ]`, **disaggregate iff** `J(d,p*) < J(d,local)`,
i.e.

```
  (Q_d − Q_{p*})·W_p  +  Z^I_d · m_pf(d)     >     Z^T_c · ( T̂_disagg − T̂_local )  +  V · c_xfer
  └──────────────┬───────────────────────┘         └───────────────────┬─────────────────────┘
   congestion relief (move W_p off d)                TTFT change from disaggregating
   + ITL interference relief on d                    + transfer penalty
```

This is structurally the current EDPP `lhs > rhs` rule, generalized: the congestion term uses the
corrected `W_p` (§3.6), the ITL interference relief `Z^I_d·m_pf(d)` is now an explicit observed-queue
term on the LHS, and `T̂` is occupancy-aware.

**Queue updates (all observed).**

```
Q_i      : conservation bookkeeping — add Δwork at route/admit, drain at service (§5.2)
Z^T_c    : at each request's first token,  Z^T_c ← max( Z^T_c + (ttft_realized − τ^T_c), 0 )
Z^I_i    : per decode step (or time slot) on i,  Z^I_i ← max( Z^I_i + (itl_realized_i − τ^I), 0 )
```

The TTFT update is per-request at first-token (one-shot); the ITL update is per-instance, integrated
over the instance's decode steps (sustained) — the asymmetry locked in §5.2.

**Normalization.** Each term is divided by its natural scale (the `τ`'s and work normalizers `w*`)
so the terms are dimensionless and `V` is a pure penalty knob; this is the dimensionless-invariance
property already anchored in the current implementation. The standard `[V ↔ 1/V]` trade-off applies:
larger `V` lowers transfer cost at the price of larger SLO-deficit backlog.

**Observed vs. modeled (cross-ref §5.2).** The queue *states* `Q_i, Z^T_c, Z^I_i` are observed; the
*forward* quantities `T̂(a), m_dec, m_pf, Δwork` (and `N̂_out` inside `W_d`) are modeled — they price
an action not yet taken. The reactive interference handling is automatic (a busy decode node already
has high `Z^I_d`); `m_pf(d)` is the optional **anticipatory** layer (§5.1 / the deferred predictor
work owes its estimate).

---

## 6. Optimality yardstick (for the paper)

To report "how close to optimal" any policy is, we will compute an **offline / clairvoyant**
optimum over a fixed recorded arrival trace:

- Formulate placement + timing as a **MILP** (binary assignment variables for prefill and decode
  instance per request; KV-capacity and compute constraints; SLO-deficit objective), or an LP
  relaxation for a lower bound on cost.
- Run every policy and the offline optimum on the **same** trace; report the optimality gap.

Caveats to state explicitly in the paper:

- The offline optimum has hindsight the online policy never has, so the gap is a **pessimistic**
  bound (a policy can be far from offline-optimal yet optimal among online policies).
- The exact MILP is solvable only at small/medium scale; show the gap there and argue it carries,
  or fall back to a cheaper goodput upper bound (LP relaxation / flow / bin-packing) at scale.

The MILP's constraints *are* the §3 system model written formally, so building it is also a check
on the formulation.

---

## 7. Why the realizable model looks the way it does (production grounding)

Confirmed against upstream sources (2026-06-30):

- **llm-d** assigns roles via the static Kubernetes label `llm-d.ai/role` with values `prefill`,
  `decode`, `prefill-decode`. All endpoints live in one `InferencePool`; `prefill-profile` and
  `decode-profile` *filter by label* at scheduling time. A "decode" pod is a full engine that can
  prefill locally — hence our two-class P/M model. The label is set at deploy time; the router
  never relabels a node per request.
  (`llm-d/llm-d` `docs/architecture/advanced/disaggregation/README.md`.)
- **NVIDIA Dynamo** deploys prefill and decode as **separate worker types in distinct pools**
  (`VllmPrefillWorker`, `VllmDecodeWorker`); each pool's planner has a `mode` ∈
  {`disagg`,`prefill`,`decode`}. The (global) planner scales replica/GPU counts — it does **not**
  convert a worker between roles per request. (`ai-dynamo/dynamo` `examples/global_planner`.)

**Conclusion:** in shipping data planes an instance's capability is fixed by the control plane on a
slow timescale; the per-request router only chooses *among* instances given fixed capabilities.
"This request needs only prefill; this one can do both" is **not** expressible per-request — it is a
node property. Our model respects this (S1, S2).

---

## 8. Out of scope / deferred extensions

1. **Runtime role-allocation loop** — letting the control plane convert instances between $\mathsf{P}$
   and $\mathsf{M}$ on a slow timescale to track load (Dynamo's planner, llm-d's autoscaler). This is a
   *separate, slower* control loop with a different actuator; modeling it requires an explicit
   two-timescale treatment.
2. **Fully-flexible per-request roles** — $p_r$ ranging over *all* instances including per-request
   role flips. Not realizable in current data planes; would be a distinct research contribution.
3. **Prefill on mixed nodes (option ii)** — allowing $p_r \in \mathcal{P}^{+}$ (a lightly-loaded $\mathsf{M}$
   node takes a neighbor's prefill). More expressive; blurs the P/M separation; rarely deployed.
4. **Predictor / estimator accuracy** — how the latency and backlog terms feeding any policy are
   estimated. Addressed after the model is fixed.

---

## 9. Open questions (to resolve next)

- The observable layer for `T̂` is now specified (§3.7–§3.8: per-iteration identity, admission-delay
  roll-forward / Little's law). Remaining forward-estimator specifics: the marginal ITL terms
  `m_dec`, `m_pf`, and the output-length estimate `N̂_out` (the deferred predictor work — §8 item 4).
- The analytical (Layer-2) reduction — closed-form steady-state on the tractable core and the
  provable policy guarantees — is not yet written here; §6 sketches its yardstick role.
- Heterogeneity parameterization: which per-instance parameters the model carries explicitly
  (KV capacity, service-rate coefficients, interconnect bandwidth) and how they enter the costs.
- MILP decision variables and constraints (§6) — the formal write-up of §3.
- Whether SLO classes are modeled per-request or per-stream, and how ITL violations are scored
  over a request's decode horizon.
