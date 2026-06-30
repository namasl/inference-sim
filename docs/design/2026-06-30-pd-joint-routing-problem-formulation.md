# Joint Routing + Disaggregation Decision: Problem Formulation and System Model

**Date:** 2026-06-30
**Status:** Draft — problem/system model only. No algorithm or implementation committed yet.
**Branch:** `design/pd-joint-routing`

> This document fixes the **system model** and the **problem statement** for prefill/decode
> (P/D) routing in a disaggregated LLM-serving deployment. It deliberately stops short of
> proposing a specific policy implementation. Predictor-accuracy questions (how individual
> latency terms are estimated) are explicitly **out of scope here** and deferred.

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

This document formalizes the joint problem.

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

**Heterogeneity (S4).** Each instance $i$ carries its own hardware/serving parameters:
GPU type, tensor-parallel degree, KV-block capacity $K_i$, and per-phase service-rate
coefficients. No two instances are assumed identical, even within the same class.

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
scorer. The exact queue definitions, penalty terms, and estimators are deferred to a follow-up
design.

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

- Exact form of the per-instance virtual queues and penalty in §5.1 (the joint policy design).
- Heterogeneity parameterization: which per-instance parameters the model carries explicitly
  (KV capacity, service-rate coefficients, interconnect bandwidth) and how they enter the costs.
- MILP decision variables and constraints (§6) — the formal write-up of §3.
- Whether SLO classes are modeled per-request or per-stream, and how ITL violations are scored
  over a request's decode horizon.
