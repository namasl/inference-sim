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

We view this as a **hierarchical decomposition** of a single underlying assignment problem, and it
is not lossless, because the two decisions are coupled:

- The *value* of disaggregating a request depends on **which** decode instance it lands on:
  offloading prefill helps a congested decode instance far more than an idle one.
- The *best* decode instance depends on **whether** the request will be disaggregated: if it
  will, choose the instance on decode capacity alone (it never prefills); if it won't, choose
  on combined prefill+decode capacity. These can be different instances.

A decomposed pipeline picks the decode instance *blind to the split*, then picks the split
*blind to the decode alternatives it could have chosen*. A **joint** formulation treats
`(instance, split)` as one action and can express choices the pipeline cannot.

In this document we formalize that joint problem.

### 1.1 Why a model, and not just measured timing

A reader may reasonably object: the simulator and a real server both expose per-request timestamps
— arrival, schedule, first token — so why build a work model at all, rather than route on measured
timing directly? We do rely on measurement heavily; it calibrates and validates everything in
§3.7–§3.8. But three things a model supplies cannot be measured:

1. **Counterfactuals.** A router commits to an action before it can observe the outcome, and it
   never sees the alternative it did not take. Choosing between "route to $A$ or $B$" or "local or
   disaggregate" requires an estimate of *each* candidate's latency, whereas timestamps are
   retrospective and describe only the action already chosen. The work model exists to score the
   actions not yet taken.
2. **Extrapolation.** Measurement covers only operating points already visited. A parameterized
   model predicts ones that have not been — a new topology, a different hardware mix, a load not yet
   reached, or the effect of adding a replica — which is exactly the generalization a provisioning
   or autoscaling decision needs.
3. **Provable guarantees.** Stability and near-optimality are properties of a model, not of a log.
   We establish them through Lyapunov drift (§5.3), which — importantly — does not assume memoryless
   arrivals and so survives the correlated, non-Poisson stream a decode instance sees under
   disaggregation (§3.8).

Measurement and model are therefore not competitors. Timestamps are the ground truth against which
we calibrate the coefficients and validate the predictions; the model is what turns that observable
history into the forward, counterfactual, and provable statements a policy — and a paper — require.

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

**Unifying principle.** Work in each phase splits into a *compute* part, linear in the number of new
tokens computed, and an *attention* part — the number of new tokens times the average context they
attend over. Because context grows linearly across a phase, that average is the midpoint,
$(\text{start context})+(\text{new tokens})/2$.

| phase | new tokens | start context | avg. context | attention work |
|-------|-----------|---------------|--------------|----------------|
| prefill | uncached $a_p$ | cached prefix $a_r-a_p$ | $a_r-a_p/2$ | $C_{\!attn}\,a_p\,(a_r-a_p/2)$ |
| decode | output $o_r$ | full prompt $a_r$ | $a_r+o_r/2$ | $C1\,o_r\,(a_r+o_r/2)$ |

Here $a_r$ is the full prompt length and $o_r$ the output length (both from §3.2), and $a_p\le a_r$ is
the *uncached* portion of the prompt — the tokens not already resident in the target's KV cache.

**Decode work** (on decode node $d$, with $d$'s coefficients):
$$W_d(d) \;=\; C0_d\,o_r \;+\; C1_d\,o_r\,(a_r + o_r/2).$$
$C0_d$ is the decode per-step compute overhead (context-independent) and $C1_d$ the KV-read cost per
resident token (§3.1, per-instance). The decode context uses the **full** prompt $a_r$, not the
uncached $a_p$: prefix-cache hits reduce prefill *compute*, but the full prompt resides in KV and is
re-read every decode step. The output length is the estimate $o_r=\hat N_{\text{out}}$ (a running
per-class mean, or a demand model), keeping the term INV-9-safe — $a_r$ is known at routing, $o_r$ is
estimated, never read from the realized output.

**Prefill work** (on prefill location $q$, with $q$'s coefficients; $q=p$ when disaggregated,
$q=d$ when local):
$$W_p(q) \;=\; C_{\!pf,q}\,a_p \;+\; C_{\!attn,q}\,a_p\,(a_r-a_p/2)\;=\;C_{\!pf,q}\,a_p+\tfrac12 C_{\!attn,q}\,a_p^2+C_{\!attn,q}\,a_p\,(a_r-a_p).$$
$C_{\!pf,q}$ is the exposed prefill compute per token and $C_{\!attn,q}$ the prefill attention
coefficient (µs/token², §3.1); for the `local` option these are the mixed node $d$'s own prefill
coefficients. Only the uncached tokens are computed as queries, but each attends over the full
context up to its position (cached prefix included) — hence $a_p\,(a_r-a_p/2)$, neither $a_p^2$ nor
$a_r^2$.

**Relationship to the earlier forms (corrections recorded here).** The previous decode form
$\hat N_{\text{out}}\cdot\bar\delta_{\text{dec}}(\text{NomDecodeCtx})=\hat N_{\text{out}}\,(C0+C1\cdot\text{NomDecodeCtx})$
is **deprecated**: its fixed nominal context (e.g. 2048), applied to every request and every step,
ignores the actual prompt length and the growth of context during decode, mis-estimating the
(dominant) memory term in both directions. $W_d$ above is the trajectory integral of the same
$C0, C1$ — no refitting. The previous prefill form $C_{\!pf}\,a_p+\tfrac12 C_{\!attn}\,a_p^2$ is the
no-cache limit ($a_p=a_r$) of $W_p$; with a cached prefix it drops the cross term
$C_{\!attn}\,a_p\,(a_r-a_p)$ — the cost of uncached tokens attending back over the cached prefix. The
two agree on no-cache traffic (tiny-prompt synth) and diverge on shared-prefix / RAG workloads.

The work in $W_p$ and $W_d$ is a *demand*, not a latency. The next two subsections connect it to
wall-clock time, and in doing so expose a subtlety we must respect: a request's work is drained
from a queue the moment it enters the running batch, so a backlog that counts only *waiting* work
omits the *currently-running* batch entirely — the very thing that decides how long a new arrival
waits. We therefore develop the time model with that residual occupancy in view.

### 3.7 From work to time: the per-iteration identity

We connect work to time through the mechanics of continuous batching, in a form we can evaluate
from observed state.

At any instant, instance $i$ runs a *batch* $\mathcal{B}_i$ — the set of requests active on it. Each
active request $r$ sits at a definite *stage* $s_r$: either a specific prefill chunk, or decode step
$k$ (its $k$-th output token). We write $\delta(s_r)$ for the **marginal work** request $r$
contributes in the current iteration given its stage. It is the single-iteration slice of the §3.6
work, read off the stage:

- prefill chunk of $\text{chunk}$ tokens: $\delta = C_{\!pf}\,\text{chunk} + C_{\!attn}\,\text{chunk}\cdot(\text{context processed so far})$;
- decode step $k$: $\delta = C0 + C1\,(a_r+k)$.

Every quantity here is observed from $r$'s current stage and the instance coefficients $\theta_i$;
nothing is estimated except, for stages not yet reached, the output length $o_r$.

One iteration then takes
$$T^{\text{iter}}_i \;=\; \alpha_i \;+\; \sum_{r\in \mathcal{B}_i}\delta(s_r),$$
where $\alpha_i$ is the per-iteration baseline from $\theta_i$ ($\alpha^P_i$ or $\alpha^D_i$ of §3.1,
per the iteration's phase; kernel launch, weight load) and the sum is the batch's total marginal work. This is the state-resolved form of the linear iteration-time
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

We express every baseline as a **restriction of the same action set** (§3.4), which keeps the
comparisons clean:

| Policy | Restriction on $a_r$ |
|--------|----------------------|
| `never` (aggregate) | force $p_r = \text{local}$ |
| `always` | force $p_r \neq \text{local}$ (always remote prefill) |
| `prefix-threshold` | $p_r = \text{local}$ iff non-cached prompt tokens $\le$ threshold; decode instance from scorer |
| EDPP — reduced ($d$ external) | the §5.3 drift rule minimized over the slice $\{(d^\star,p): p\in\mathcal{P}\cup\{\text{local}\}\}$, with $d^\star$ from a scorer (today's shipped decider; it also uses pool-level rather than per-instance queues) |
| **EDPP — full joint (target)** | the *same* §5.3 drift rule minimized over all of $\mathcal{A}=\mathcal{M}\times(\mathcal{P}\cup\{\text{local}\})$; no scorer |

The last two rows are the **same method** — the drift-plus-penalty decider of §5.3 — on two different
action sets; §5.5 makes this precise.

### 5.1 The problem, as constrained stochastic optimization

Before naming any rule, we state what it must achieve. Write $g(t)$ for the operating cost we would
rather avoid — here the transfer / KV-movement cost incurred by disaggregation in epoch $t$. Our
objective (§4) is to minimize its time average subject to stability and the SLO constraints:

$$\min\ \ \limsup_{T\to\infty}\frac1T\sum_{t=0}^{T-1}\mathbb{E}\big[g(t)\big]
\qquad\text{s.t.}\qquad
\begin{aligned}
&\text{every congestion backlog } Q_i \text{ is mean-rate stable},\\
&\overline{\text{ttft}}_c \le \tau^T_c \quad \forall\,\text{class } c,\\
&\overline{\text{itl}}_i \le \tau^I \quad \forall\,\text{instance } i,
\end{aligned}$$

where $\overline{(\cdot)}$ is a time average. The two SLO rows are *time-average* constraints, and
stability of the work backlogs is what keeps throughput sustainable. The rest of §5 turns these
constraints into a per-request rule; the decision formula appears only in §5.3, as the minimizer of
a drift bound — not as a starting assumption.

### 5.2 Queue model

We keep **two families of state**, deliberately distinct:

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
   belongs in the drift cost, not in a second congestion queue.

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

### 5.3 Deriving the decision rule

**Queue dynamics.** Each queue of §5.2 evolves by a standard update driven by *observed* signals.
The congestion backlog at instance $i$ gains the work routed onto it and drains at its service rate:
$$Q_i(t{+}1)=\max\{Q_i(t)-b_i(t),\,0\}+A_i(t),$$
with $A_i(t)$ the work routed onto $i$ this epoch (exogenous arrivals) and $b_i(t)$ the work $i$
serves (set by the server, not by the routing action). The deficit
queues integrate realized violations:
$$Z^T_c(t{+}1)=\max\{Z^T_c(t)+(\text{ttft}_r-\tau^T_c),\,0\}\ \text{ at each class-}c\text{ first token},$$
$$Z^I_i(t{+}1)=\max\{Z^I_i(t)+(\text{itl}_i-\tau^I),\,0\}\ \text{ per decode step on }i.$$
The TTFT update fires once per request at its observed first token; the ITL update accrues per
instance over its decode steps — the asymmetry of §5.2. By the standard virtual-queue argument,
mean-rate stability of $Z^T_c$ (resp. $Z^I_i$) is equivalent to the time-average TTFT (resp. ITL)
constraint of §5.1. So it suffices to keep every queue stable while minimizing $g$.

**Lyapunov function.** Take the quadratic backlog measure
$$L(t)=\tfrac12\Big[\textstyle\sum_i Q_i(t)^2+\sum_c \big(Z^T_c(t)\big)^2+\sum_i \big(Z^I_i(t)\big)^2\Big],$$
and let $\Delta(t)=\mathbb{E}\big[L(t{+}1)-L(t)\mid \text{state}(t)\big]$ be its one-epoch conditional
drift.

**Drift bound.** Applying $(\max\{x-b,0\}+a)^2\le x^2+a^2+b^2+2x(a-b)$ to the congestion queues and
$(\max\{x+y,0\})^2\le x^2+y^2+2xy$ to the deficit queues, then collecting the bounded second-moment
terms into a constant $B$,
$$\Delta(t)\ \le\ B\ +\ \sum_i Q_i\,\mathbb{E}[A_i-b_i]\ +\ \sum_c Z^T_c\,\mathbb{E}[\text{ttft}-\tau^T_c]\ +\ \sum_i Z^I_i\,\mathbb{E}[\text{itl}_i-\tau^I].$$
$B$ is finite because per-request work and per-step latencies are bounded (finite prompt/output
lengths and batch size).

**Drift-plus-penalty.** The Lyapunov-optimization principle minimizes, each epoch, a bound on
$\Delta(t)+V\,\mathbb{E}[g(t)\mid\text{state}]$, where $V>0$ tunes how hard cost is pushed down
against backlog. Consider the single request $r$ (class $c$) being routed, with action $a=(d,p)$.
Only $r$'s own placement is controllable this epoch; the service terms $b_i$ and the targets $\tau$
do not depend on $a$. Reading off the action-dependent part of the bound term by term:

- **Congestion.** Write $\Delta\text{work}_i(a)$ for the §3.6 work the action places on instance $i$,
  each piece evaluated with that instance's own coefficients (§3.1): if $p=\text{local}$ then
  $\Delta\text{work}_d = W_p(d)+W_d(d)$ (both phases on $d$) and zero elsewhere; if $p\in\mathcal{P}$
  then $\Delta\text{work}_d = W_d(d)$ and $\Delta\text{work}_p = W_p(p)$. This epoch routes the single
  request $r$, so the only exogenous arrival to any backlog is $r$ itself and $\mathbb{E}[A_i]=\Delta\text{work}_i(a)$.
  Splitting the congestion term of the bound accordingly,
  $$\sum_i Q_i\,\mathbb{E}[A_i-b_i]\;=\;\underbrace{\sum_i Q_i\,\Delta\text{work}_i(a)}_{\text{depends on }a}\;-\;\underbrace{\sum_i Q_i\,\mathbb{E}[b_i]}_{\text{independent of }a},$$
  and the second sum — the server's own drain — is constant across the actions we compare, so it
  drops from the argmin. The action-dependent congestion contribution is thus $\sum_i Q_i\,\Delta\text{work}_i(a)$.
- **TTFT.** Only class $c$'s deficit sees this request — the other classes' queues are untouched by
  $a$ — and the target $\tau^T_c$ is fixed, so
  $$\sum_{c'} Z^T_{c'}\,\mathbb{E}[\text{ttft}-\tau^T_{c'}]\;=\;\underbrace{Z^T_c\,\mathbb{E}[\text{ttft}_r\mid a]}_{\text{depends on }a}\;-\;\underbrace{Z^T_c\,\tau^T_c}_{\text{independent of }a}.$$
  We estimate the conditional expectation by the observable forward value $\hat T(a)$ of §3.8. The
  action-dependent TTFT term is thus $Z^T_c\,\hat T(a)$.
- **ITL.** Routing $r$ perturbs the per-step latency of the *one* decode node $d$ it lands on; every
  other instance's ITL is untouched by $a$, a $\mathsf P$-only prefill target does no decode (so has
  no ITL constraint), and $\tau^I$ is fixed. Hence
  $$\sum_i Z^I_i\,\mathbb{E}[\text{itl}_i-\tau^I]\;=\;\underbrace{Z^I_d\big(m_{dec}(d)+\mathbf{1}\{p=\text{local}\}\,m_{pf}(d)\big)}_{\text{depends on }a}\;-\;\underbrace{\textstyle\sum_i Z^I_i\,\tau^I}_{\text{independent of }a}.$$
  Here the marginal per-step latency $r$ adds to $d$ is, by the iteration-time identity of §3.7, its
  own marginal work $\delta$: since $T^{\text{iter}}_d=\alpha_d+\sum_{r'\in \mathcal{B}_d}\delta(s_{r'})$,
  admitting $r$ raises every co-resident request's per-step time by $r$'s $\delta$. So we **define**
  $$m_{dec}(d)=\delta(\text{$r$'s decode step on }d),\qquad m_{pf}(d)=\delta(\text{$r$'s prefill chunk on }d),$$
  the first incurred whenever $r$ decodes on $d$ (i.e. always), the second only when prefill is local
  and thus co-schedules on $d$. (Today's EDPP approximates $m_{pf}$ by its compute part
  $C_{\!pf,d}\cdot\text{chunk}$, dropping the attention part of $\delta$.) This is a first-order,
  *instantaneous* marginal; its persistence over $r$'s lifetime is not in this one-shot term but is
  recovered reactively through $Z^I_d$ (see §5.4).
- **Penalty.** With $c_{\text{xfer}}$ the per-request transfer / KV-movement cost,
  $g(a)=c_{\text{xfer}}\,\mathbf{1}\{p\ne\text{local}\}$, so $V\,\mathbb{E}[g(a)]=V\,c_{\text{xfer}}\,\mathbf{1}\{p\ne\text{local}\}$.

Collecting the four and discarding the action-independent constant $B$ and the $-\tau$ shifts, the
epoch bound is minimized by

$$a_r^{*}=\arg\min_{a}\ \Big[\ \underbrace{\textstyle\sum_i Q_i\,\Delta\text{work}_i(a)}_{\text{congestion}}\;+\;\underbrace{Z^T_c\,\hat T(a)}_{\text{TTFT}}\;+\;\underbrace{Z^I_d\big[m_{dec}(d)+\mathbf{1}\{p=\text{local}\}\,m_{pf}(d)\big]}_{\text{ITL}}\;+\;\underbrace{V\,c_{\text{xfer}}\,\mathbf{1}\{p\ne\text{local}\}}_{\text{penalty}}\ \Big].$$

This is the decision rule — obtained as the minimizer of the drift-plus-penalty bound, not posited.
Every quantity in it is either an observed queue state ($Q_i,\,Z^T_c,\,Z^I_d$) or a forward estimate
from §3.6–§3.8.

**No drain rates.** Notably the server service rates $\mu_i$ (equivalently the $b_i$) do **not**
appear in the rule: $b_i$ dropped as action-independent, so $\mu$ is needed neither to rank actions
nor to update $Q_i$ (drained by *observed* completions, §5.2). Service timing re-enters only inside
$\hat T(a)$, through the iteration-time roll-forward $T^{\text{iter}}=\alpha+\sum\delta$ (§3.7), where
it comes from the coefficients $\theta$ — not from a separately-fit $\mu$. (This removes the explicit
$q/\mu$ balance term the earlier EDPP carried.)

We now give the forward quantities explicitly:

- $\Delta\text{work}_i(a)$ — defined at the congestion step above. Under heterogeneity the same
  request yields a *different* $\Delta\text{work}$ on different instances, so the congestion term
  $\sum_i Q_i\,\Delta\text{work}_i$ ranks candidates by cost *and* load jointly, not by load alone.
- $\hat T(a)$ — forward TTFT estimate for the request under action $a$, obtained as in §3.8 (the
  admission-delay roll-forward or Little's-law estimate, both occupancy-aware):
  $\hat T_{\text{local}}(d)=T^{\text{adm}}(d)+(\text{$r$'s own prefill on }d)$;
  $\hat T_{\text{disagg}}(d,p)=T^{\text{adm}}(p)+(\text{prefill on }p)+(\text{transfer})+T^{\text{adm}}(d)$.
- $m_{dec}(d),\ m_{pf}(d)$ — defined at the ITL step above. $m_{dec}(d)$ is incurred whether or not we
  disaggregate (decode is on $d$ either way), so it steers *which* decode node but **not** the
  local-vs-disaggregate split; $m_{pf}(d)$ is incurred **only** locally, so it is precisely the
  interference cost the `local` option pays and disaggregation escapes.

**Joint argmin.** Writing $J(a)$ for the bracketed objective above, the policy selects over the full
action set in one optimization:
$$a_r^{*}=\arg\min_{a\in\mathcal{A}} J(a),\qquad \mathcal{A}=\{(d,\text{local}):d\in\mathcal{M}\}\cup\{(d,p):d\in\mathcal{M},\,p\in\mathcal{P}\}.$$
Load balancing (the choice of $d$ and of $p$) and the P/D split fall out of the same argmin; there is
no separate scorer. When the decode node is instead fixed by an external scorer, the rule collapses
to the current EDPP pairwise decider — we show that reduction in §5.5.

**Units and normalization.** In raw form the three drift terms share units — writing work in
time-units (work is a service time), $Q_i\,\Delta\text{work}_i$, $Z^T_c\,\hat T$, and $Z^I_d\,m$ are
each a $(\text{time})^2$, and the penalty $V\,c_{\text{xfer}}$ matches only if $V$ carries units of
time. But their *magnitudes* diverge badly: a work backlog $Q_i$ can be seconds of accumulated work,
while the ITL deficit is measured against $\tau^I\!\approx\!50$ ms and the TTFT deficit against
$\tau^T\!\approx\!500$ ms. Summed raw, the congestion term swamps the SLO terms and ITL all but
vanishes. We therefore measure each violation against its own target. The cleanest route is to
define the deficit queues in *relative* form,
$$Z^T_c\!\leftarrow\!\max\{Z^T_c+(\text{ttft}_r/\tau^T_c-1),\,0\},\qquad Z^I_i\!\leftarrow\!\max\{Z^I_i+(\text{itl}_i/\tau^I-1),\,0\},$$
which enforces the identical constraint ($\text{ttft}\le\tau^T\Leftrightarrow\text{ttft}/\tau^T\le1$)
but makes each deficit a dimensionless *fraction of target* — so a 10% TTFT miss and a 10% ITL miss
count equally — and to normalize the congestion pair by a reference work $W^{\star}$:
$q_i=Q_i/W^{\star}$, $\Delta w_i=\Delta\text{work}_i/W^{\star}$. Every term of the rule is then
dimensionless and $V$ is a pure, dimensionless knob.

**Choosing $W^{\star}$ — class-agnostic, tied to $\tau_{\text{ref}}$.** The congestion queue $Q_i$
carries *no class*: a mixed instance pools requests of every class, and stability is a property of
that one shared GPU, not of any class (a per-class congestion queue would model per-class servers
sharing no resource — the same error as splitting a mixed node into two servers). So $W^{\star}$ must
be class-agnostic as well. We set it against a class-agnostic reference,
$$W^{\star}\;\approx\;\mu_{\text{nom}}\cdot \tau_{\text{ref}},$$
the work whose backlog induces one $\tau_{\text{ref}}$-worth of delay; then $q_i=Q_i/W^{\star}$ reads
as "backlog measured in $\tau_{\text{ref}}$-delays," directly commensurate with the fraction-of-target
SLO deficits. The per-class sensitivity is deliberately *not* placed here — it is already carried by
the per-class TTFT term $Z^T_c\hat T$ (weighted by $\tau^T_c$), so a per-class $W^{\star}_c$ would
double-count it. Two constants enter:

- $\tau_{\text{ref}}$ — a **fixed reference TTFT**: one system-wide constant, independent of any
  operating class target $\tau^T_c$. It is the single class-agnostic time-scale the formulation uses
  for cross-cutting normalization — the same constant that appears in the transfer-penalty factor
  below. (A natural pick is a representative interactive TTFT, e.g. the tightest class target or a
  round number near it; its role is only to fix a common scale, and the argmin is invariant to it up
  to the compensating factors.)
- $\mu_{\text{nom}}$ — a **nominal service rate**: the work served per unit time by a representative
  instance. It is used only as this one-time calibration constant and does **not** reintroduce $\mu$
  into the per-request rule.

A pleasant consequence: with a class-agnostic $W^{\star}$ but per-class $\tau^T_c$ on the SLO and
transfer terms, those terms scale as $1/(\tau^T_c)^2$ while congestion scales as
$1/(\tau_{\text{ref}})^2$. So a **tight-SLO** class (small $\tau^T_c$) yields latency-dominated
decisions and a **loose** class yields balance-dominated ones — the behavior we want, for free. A
per-class $W^{\star}_c$ would flatten this, making the stability-vs-SLO weight identical across
classes.

Two further consequences: (i) the argmin is invariant to an overall rescaling of $J$, so only the
*relative* term weights — set by the normalizers — matter; and (ii) the transfer penalty must carry
the fixed $\tau_{\text{ref}}/\tau^T_c$ factor so it scales as $1/(\tau^T_c)^2$ like the SLO terms,
otherwise a loose $\tau^T_c$ shrinks the balance/SLO terms faster than the penalty and spuriously
suppresses disaggregation. The standard $[V\!\leftrightarrow\!1/V]$ trade-off then holds: larger $V$
lowers transfer cost at the price of larger (dimensionless) SLO-deficit backlog.

**Observed vs. modeled.** The queue *states* $Q_i, Z^T_c, Z^I_i$ are observed; the *forward*
quantities $\hat T(a), m_{dec}, m_{pf}, \Delta\text{work}$ (and $\hat N_{\text{out}}$ inside $W_d$)
are modeled — they price an action not yet taken. Reactive interference handling is automatic (a
busy decode node already carries a high $Z^I_d$); the $m$ terms are the optional **anticipatory**
correction whose approximation §5.4 examines.

### 5.4 Is the rule sound? What is exact, and what is approximate

The derivation is a standard Lyapunov drift-plus-penalty argument, so it inherits the usual
guarantees — but only to the extent its inputs are exact. We separate the two.

**Exact / rigorous.**

- The drift bound and the quadratic Lyapunov step hold verbatim, needing only bounded per-request
  work and per-step latency (finite prompt/output lengths and batch) for the constant $B$ to be finite.
- The congestion term $\sum_i Q_i\,\Delta\text{work}_i(a)$ is exact — it is the arrival's contribution
  to each backlog, with the action-independent service $b_i$ correctly dropped.
- The virtual-queue construction is exact: mean-rate stability of $Z^T_c, Z^I_i$ is equivalent to the
  §5.1 time-average SLO constraints. The standard result then applies — under the idealized rule the
  policy is **throughput-optimal** (stable whenever any policy could be) and its time-average cost is
  within $O(1/V)$ of the constrained optimum, with worst-case backlog $O(V)$: the usual
  $[V\!\leftrightarrow\!1/V]$ trade-off.
- Crucially, none of this assumes memoryless arrivals. The drift argument holds for general (bursty,
  correlated) arrivals with bounded moments, so the guarantee **survives the non-Poisson decode
  stream** of §3.8.

Stated precisely:

> **Proposition (stability and near-optimality; informal).** Suppose per-request work and per-step
> latency are bounded, and the forward estimates are unbiased — $\mathbb{E}[\hat T(a)]=\mathbb{E}[\text{ttft}\mid a]$,
> and likewise for the ITL marginals. If *some* policy meets all §5.1 constraints with time-average
> cost $g^\star$ and slack $\epsilon>0$ (i.e. it keeps each SLO strictly below target on average),
> then the rule of §5.3 (i) keeps every queue mean-rate stable — so all SLO constraints hold on time
> average — and (ii) attains time-average cost $\bar g \le g^\star + B/V$, with mean total backlog
> $O(V/\epsilon)$.

*Proof sketch.* The rule minimizes the drift-plus-penalty bound over the action space, so its bound
is no larger than that of **any** other policy — in particular a stationary, queue-blind policy that
draws actions from the constraint-feasible distribution achieving $g^\star$ with slack $\epsilon$.
Substituting that comparison policy on the right-hand side, feasibility drives each queue-weighted
expectation to $\le -\epsilon$ and the penalty term to $Vg^\star$, giving
$$\Delta(t)+V\,\mathbb{E}[g(t)]\ \le\ B + Vg^\star - \epsilon\textstyle\sum_q \text{queue}_q(t).$$
Take expectations, sum over $t=0,\dots,T-1$, and telescope $L$: dropping the non-negative backlog
term yields $\bar g \le g^\star + B/V$, and dropping the penalty term bounds the mean backlog by
$O\big((B+Vg^\star)/\epsilon\big)=O(V/\epsilon)$, hence mean-rate stability. Bounded work/latency
keep $B$ finite. Neither the comparison policy nor the telescoping invokes the arrival law, so the
result holds for **general (non-Poisson) arrivals**. Finally, unbiasedness is what lets the
*implemented* rule (which uses $\hat T$ and the $m$ terms in place of the true expectations) occupy
the same inequality; any bias enters additively and is bounded by the estimator error — the quantity
§3.8 makes observable and validates. $\qquad\blacksquare$

**Approximate (and where the honesty lies).**

- **TTFT term.** The bound contains $\mathbb{E}[\text{ttft}_r\mid a]$; we substitute the observable
  forward estimate $\hat T(a)$ (§3.8). The guarantee is exact when $\hat T$ is unbiased and degrades
  gracefully with its bias — which is exactly why §3.8 makes $\hat T$ measurable and validates it
  against the realized $T^{\text{adm}}$. The *theory* attaches to the rule with true
  $\mathbb{E}[\text{ttft}]$; the *implementation* substitutes $\hat T$, and the gap is the estimator error.
- **ITL term.** $m_{dec}, m_{pf}$ approximate the horizon-integrated ITL impact of $r$ by its
  *instantaneous* per-step marginal work on $d$. The persistence over $r$'s co-residence is not in
  this one-shot term; it is captured *reactively*, since $Z^I_d$ stays elevated over that horizon as
  realized ITL misses accrue. The $m$ terms are thus an anticipatory correction on top of reactive
  backpressure, not a claim of exactness.
- **Per-arrival epochs.** We apply the per-slot principle once per arrival, treating each routing as
  one decision epoch — standard when an action affects only its own request's placement.

**Two points the derivation settles.**

- *No double counting.* The congestion term (weighted by $Q_i$) and the TTFT term (weighted by
  $Z^T_c$) both correlate with load, yet they enforce *different* constraints — stability versus the
  latency SLO — and are driven by different queues. There is no redundancy.
- *Why keep congestion queues at all,* when $\hat T$ already sees occupancy? Because SLO deficits
  alone do not bound backlog: under a loose SLO a policy could hold deficits near zero while work
  piles up. The congestion queues deliver the stability (throughput-optimality) half of the guarantee.

**Net.** "This joint rule is stable and within $O(1/V)$ of optimal, under general arrivals" is sound
for the idealized rule; the deployed rule inherits it up to the bias of the forward estimates
$\hat T, m_{dec}, m_{pf}$, which we therefore hold to the observable validation of §3.8.

### 5.5 Reduction to the pairwise rule (recovering EDPP)

Recall the joint action set $\mathcal{A}$ (§5.3), of size $|\mathcal{A}|=|\mathcal{M}|\,(|\mathcal{P}|+1)$;
the joint rule minimizes $J$ over all of it. The current EDPP decider instead
restricts to the slice $\{(d^{\star},p): p\in\mathcal{P}\cup\{\text{local}\}\}$ for a single decode
node $d^{\star}$ chosen by an **external scorer** (prefix-cache / queue scores, not $J$), and
minimizes only over $p$. It therefore optimizes the *same* objective on a one-dimensional slice
whose location is set by a non-drift mechanism. To see the reduction, fix $d=d^{\star}$ and let the
best prefill target be
$p^{\star}=\arg\min_{p\in\mathcal{P}}\big[\,Q_p\,W_p+Z^T_c\,\hat T_{\text{disagg}}(d,p)\,\big]$. The
joint rule then disaggregates iff $J(d,p^{\star})<J(d,\text{local})$, i.e.

$$\underbrace{(Q_d-Q_{p^{\star}})\,W_p + Z^I_d\,m_{pf}(d)}_{\text{congestion relief on }d\ +\ \text{ITL interference relief}}\ >\ \underbrace{Z^T_c\big(\hat T_{\text{disagg}}-\hat T_{\text{local}}\big) + V\,c_{\text{xfer}}}_{\text{TTFT change from disaggregating}\ +\ \text{transfer penalty}}.$$

This is structurally today's EDPP `lhs > rhs` rule, generalized in three ways: the congestion term
uses the corrected work $W_p$ (§3.6); the ITL interference relief $Z^I_d\,m_{pf}(d)$ is now an
explicit observed-queue term on the left; and $\hat T$ is occupancy-aware (§3.8). Fixing $d$
externally is exactly the loss the joint formulation removes (§1) — the pairwise rule cannot see
that a *different* decode node would have made disaggregation unnecessary.

**The two are one method, not two.** Nothing mathematical forces the reduction: the full joint rule
*is* EDPP — the same drift-plus-penalty decider of §5.3 — evaluated over the whole action set
$\mathcal{A}$ rather than a scorer-chosen slice. The scorer is not even required for cache locality.
The cached prefix $p_{\text{cached}}$ in the work model (§3.6) is per-(request, node), so a node that
already holds the prompt's KV has a smaller uncached $a_p$, hence smaller $W_p$ and $\hat T$, and the
joint argmin prefers it on its own — cache affinity is subsumed by $J$, not delegated to a separate
scorer. What the full version costs is compute — $|\mathcal{A}|=|\mathcal{M}|(|\mathcal{P}|+1)$
candidate scorings per request instead of $(|\mathcal{P}|+1)$ — and llm-d scorer parity; what it buys
is never sitting on a suboptimal slice. Optimizing over a superset, it is provably no worse than the
reduced case. The reduction is therefore a deployment choice (integration on llm-d's existing
scorer), not a mathematical necessity.

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

We confirmed the following against upstream sources (2026-06-30):

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

**We conclude:** in shipping data planes an instance's capability is fixed by the control plane on a
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

- The observable layer for $\hat T$ is now specified (§3.7–§3.8: per-iteration identity,
  admission-delay roll-forward / Little's law). Remaining forward-estimator specifics: the marginal
  ITL terms $m_{dec}, m_{pf}$ and the output-length estimate $\hat N_{\text{out}}$ (the deferred
  predictor work — §8 item 4).
- The analytical (Layer-2) reduction — closed-form steady-state on the tractable core and the
  provable policy guarantees — is not yet written here; §6 sketches its yardstick role.
- Heterogeneity parameterization: which per-instance parameters the model carries explicitly
  (KV capacity, service-rate coefficients, interconnect bandwidth) and how they enter the costs.
- MILP decision variables and constraints (§6) — the formal write-up of §3.
- Whether SLO classes are modeled per-request or per-stream, and how ITL violations are scored
  over a request's decode horizon.
