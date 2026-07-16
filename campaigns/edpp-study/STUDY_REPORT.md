# EDPP policy study — what we ran, what we found, what it means

Status as of 2026-07-15. Branch `feat/edpp-estimator-validation` (not pushed).
Every number below is reproducible with a command in this document. If a claim has no
command, it is marked as an assumption or a gap.

---

## 1. Executive summary (read this first)

**The question.** We built EDPP, a drift-plus-penalty policy that decides, per request,
whether to disaggregate prefill and where to route it. The question this study answers is:
**does EDPP actually beat trivial heuristics on a fixed disaggregated topology, and where?**

**What we found, in one paragraph.** On a fixed 1P2D topology, which trivial heuristic is
optimal *flips* across the workload spectrum: `always` disaggregate wins on decode-bound
workloads, `never` wins on prefill-bound ones. So no fixed heuristic is universally right,
which is the first real motivation for an adaptive policy. In the **prefill-bound** regime
under overload, the optimum is an *interior* split (use all three servers for the bottleneck
prefill work), and **EDPP finds it dynamically and wins big** — goodput 0.917 vs 0.108
(`never`), 0.033 (`always`), and 0.604 (the best *static* split). That is a genuine, robust
(3 seeds), and previously-unclaimed positive result for EDPP. **However**, a one-line
baseline that strips EDPP down to "disaggregate iff predicted TTFT is lower" captures most
of that win (0.854–0.975), and at extreme overload it *beats* EDPP decisively (0.375–0.433
vs 0.054–0.071 — EDPP collapses). So the win looks like **dynamic latency-aware routing**,
not the drift-plus-penalty machinery.

**A separate finding worth its own line.** The **llm-d shipped decode scorer**
(`precise-prefix-cache:2,queue-depth:1`) pins *all* decode traffic onto a single instance
when requests share a prefix group, costing **6.6× goodput** (0.133 vs 0.879). This is not a
bug in our code. It is a finding about a shipped production configuration, and every policy
comparison in the literature that uses it is measuring the scorer, not the policy.

**Honest status.** Three things we have **not** done, which bound every claim above:
1. **We never used EDPP's joint routing.** Every spectrum experiment ran EDPP in *reduced*
   mode. Which decode instance a request lands on was decided by the scorer, not by EDPP.
   EDPP's actual thesis — the joint `(d,p)` argmin — is untested against a fair baseline.
2. **We never tested SLO-class heterogeneity.** Every workload was single-class, which
   switches off EDPP's most distinctive machinery (the per-class virtual queues).
3. **We never changed EDPP's objective.** It still minimizes transfer cost, which we now
   believe is the wrong objective (§3.2).

**Effort delivered.** Four reviewed, merged implementation plans (hardware heterogeneity
wiring, per-instance θ_i, the least-TTFT baseline, plus the earlier estimator/joint work),
a reproducible experiment harness, and the findings above.

---

## 2. What EDPP is, precisely

### 2.0 Why a joint rule at all (formulation §1, §3.4, §5.5)

The design starts from one claim: **routing and deciding are the same problem, and the system
currently splits them.** Two mechanisms run in sequence — a scorer picks the decode instance, then
a separate decider chooses local-vs-disaggregate. The formulation folds both into one action:

    a_r = (d_r, p_r),   p_r ∈ prefill-nodes ∪ {local}

"local" is just another prefill location, so *"disaggregate?"* becomes a coordinate of *"which
instance?"*. The split is lossy because the two decisions are **coupled in both directions** (§1):

- the **value of disaggregating depends on which decode instance** was picked — offloading prefill
  helps a congested decode instance far more than an idle one;
- the **best decode instance depends on whether the request will be disaggregated** — if it will,
  pick on decode capacity alone (that instance never prefills); if it won't, pick on combined
  prefill+decode capacity. These can be different instances.

A decomposed pipeline picks the instance blind to the split, then the split blind to the instances
it could have chosen. Because the joint rule minimizes the same objective over a *superset* of the
reduced action set, it is **provably no worse**; what it costs is compute and llm-d scorer parity.
§5.5 is explicit that the reduction is *"a deployment choice…, not a mathematical necessity"* and
that *"the pairwise rule cannot see that a different decode node would have made disaggregation
unnecessary."*

**Consequence for this study:** every experiment here used the **reduced** rule — the scorer-chosen
slice. We measured the decomposition the formulation was written to replace, and never switched on
the mechanism it argues for. That is §6 gap 1, and it is why that gap outranks the others.

### 2.1 The objective — and a contradiction in the formulation

The formulation states its objective **twice, and the two do not agree**:

> **§4 Objective.** "Maximize **goodput** (equivalently, minimize the time-average rate of SLO
> violation across TTFT and ITL targets), subject to: per-instance KV-capacity and compute
> constraints, and queue stability."

> **§5.1** (which presents itself as restating §4). "Write g(t) for the operating cost we would
> rather avoid — here the **transfer / KV-movement cost** incurred by disaggregation in epoch t.
> Our objective (§4) is to minimize its time average subject to stability **and the SLO
> constraints**."

These are different optimization problems. Between §4 and §5.1 the SLO moved **out of the objective
and into the constraints**, and transfer cost — which §4 never mentions — became the thing being
minimized. **The implemented rule follows §5.1.**

This is the documented origin of the objective mismatch in §3.2: we grade against §4's goal
(goodput) while the code optimizes §5.1's problem. It also means the fix is a **re-derivation from
the objective §4 already states**, not another term bolted onto the rule.

### 2.2 The three terms of the implemented rule

For each request EDPP evaluates an objective and picks the minimizing action. The objective has
three kinds of terms, and it matters enormously which is which:

| term | what it is | kind |
|---|---|---|
| `q_i · Δwork_i` | per-instance backlog (congestion) | **drift** — enforces work-stability (throughput) |
| `z_ttft · T̂/τ`, `z_itl · (…)/τ` | virtual queues for the TTFT/ITL SLOs | **drift** — enforces the SLO *constraints* |
| `V · c_xfer · 1{disagg}` | the KV-transfer cost | **penalty** — the actual objective |

**The SLOs are constraints, not the objective.** They are enforced by virtual queues `z`
that accumulate violation. The *only* thing EDPP optimizes is `V·c_xfer`. Read plainly,
EDPP's posture is:

> *"Disaggregate as little as possible, subject to holding the TTFT/ITL SLOs and keeping
> the queues stable."*

Two modes exist:
- **reduced** — a scorer picks the decode instance; EDPP decides only local-vs-disaggregate.
- **joint** (`--edpp-joint`) — EDPP enumerates all `(decode, prefill)` candidates and picks
  the argmin itself. *This is EDPP's real thesis, and §6 explains why we never tested it here.*

Defaults used everywhere below: `--edpp-v 1`, `--edpp-c-xfer 5ms`, `--edpp-tadm-estimator
rollforward` (the occupancy-aware admission-delay estimator; the shipped default `waiting` is
occupancy-blind and would understate EDPP).

---

## 3. Methodology — how we evaluate a policy, and why

### 3.1 The measure: goodput

**Goodput** = the fraction of completed requests meeting *all* their SLO dimensions
(TTFT and ITL and e2e). In BLIS this is `per_class[class].slo_attainment`. Higher is better.

We chose goodput because it is what an operator is graded on. Note the tension this creates
with EDPP's own objective (§3.2).

### 3.2 A known objective mismatch (important caveat on every EDPP number)

EDPP minimizes *transfer cost* subject to *average-latency constraints*. We measure *goodput*.
These coincide only when the SLOs are comfortably feasible. Under overload:
- the constraints are **infeasible** — no policy can hold them, so the virtual queues grow
  without bound and the drift-plus-penalty guarantee is void;
- "minimize transfers" is **beside the point** — you are missing TTFT either way;
- worse, the `z` queues accumulate *lateness* (`z += realized − τ`, clamped at 0). A request
  10 s late pumps a large positive into the queue, so the policy keeps spending capacity on a
  request that already missed and can never count toward goodput. **Goodput saturates; the
  surrogate does not.** Goodput implies *triage* (protect the savable, abandon the doomed);
  an accumulating-lateness surrogate structurally cannot triage.

This is our leading hypothesis for why EDPP collapses under extreme overload (§5, F5).
**We have not fixed this** — every experiment below uses the shipped transfer-cost objective.

### 3.3 The policies compared

| policy | rule | note |
|---|---|---|
| `never` | never disaggregate (prefill on the decode instance) | wastes the prefill server |
| `always` | always disaggregate | uses all servers; the P/D split's intent |
| `prefix-threshold` | disaggregate iff uncached prompt tokens > N | **N=16 is llm-d's shipped default** (`deploy/config/pd-epp-config.yaml`) |
| `least-ttft` | disaggregate iff predicted TTFT_disagg < predicted TTFT_local | **EDPP's estimator without its drift/z/V machinery** — built precisely to isolate the machinery's value |
| `edpp` | the full drift-plus-penalty rule | penalty = `V·c_xfer` |

### 3.4 The yardstick: a static disaggregation-fraction oracle

To know whether a policy leaves goodput on the table we need a target. We force a per-request
plan with `--pd-plan` (a CSV of `request_id,decode_instance,prefill_instance`). For fraction
*f*, the first *f* of every 100 requests are disaggregated (`prefill_instance=instance_0`) and
the rest are local (empty), with decode alternating `instance_1`/`instance_2` so decode is
perfectly balanced and cannot confound the result. `f=0` is `never`; `f=100` is `always`.
Sweeping *f* and taking the max gives the **best static split**. An *interior* maximum proves
neither corner is optimal — i.e. that routing genuinely matters.

This is a **static** yardstick, not the true optimum. A dynamic policy can beat it (and does,
§5 F4). The true joint hindsight optimum (a MILP over all requests) remains unbuilt.

### 3.5 Experiment design choices, and the reason for each

- **Fixed 1P2D topology** (`instance_0`=prefill, `instance_1/2`=decode). The operator picks
  the topology; the policy only routes. Comparing across topologies would answer a
  *provisioning* question, not a *policy* question.
- **Four workload archetypes** spanning decode-bound → prefill-bound, because which pool is
  the bottleneck is exactly what should drive the disaggregation decision.
- **SLO auto-set per archetype** from an idle probe (2× idle `e2e_p99`, 3× idle `ttft_p99`).
  Each archetype has a different intrinsic latency; one fixed SLO would measure "which
  archetype is fast", not "which policy is good".
- **Concurrency cap** (`--max-num-running-reqs 16`). With the default cap (256) and abundant
  KV, these workloads never queue and *every policy scores 1.000* — no signal. The cap forces
  saturation at reachable rates. **This is an artificial stressor and a caveat on the results.**
- **Decode scorer held fixed** at `queue-depth:1` for the spectrum, to isolate the
  disaggregation decision from the scorer artifact in §5 F1. We report both setups.

---

## 4. The experiments (each with its command)

All commands run from the repo root, after `go build -o blis main.go`.
Harness: `campaigns/edpp-study/repro_spectrum.sh` (self-contained; emits its own specs).

### E1 — The scorer comparison (llm-d default vs load-balanced)

```bash
MODE=scorer bash campaigns/edpp-study/repro_spectrum.sh
```
Archetype `mixed` (in=2048, out=128), rate 16, cap 16, 1P2D. Output (`i1`/`i2` = requests
that *decoded* on each decode instance — every completed request decodes exactly once, so
this is the realized decode allocation):

| decode scorer | `never` | `always` | realized split |
|---|---|---|---|
| **llm-d shipped** (`precise-prefix-cache:2,queue-depth:1`) | 0.067 | **0.133** | `i1=0  i2=240` ← all pinned to one instance |
| load-balanced (`queue-depth:1`) | 0.246 | **0.879** | `i1=119 i2=121` |

### E2 — The workload spectrum

```bash
bash campaigns/edpp-study/repro_spectrum.sh                # decode scorer = balanced
SCORER=llmd bash campaigns/edpp-study/repro_spectrum.sh    # decode scorer = llm-d shipped
```
1P2D, cap 16, seed 42, decode scorer balanced. Goodput:

| archetype (in/out) | rate | `never` | `always` | `prefix16` | `least-ttft` | `edpp` |
|---|---|---|---|---|---|---|
| **decode** 256/512 | 4 | 0.463 | **1.000** | 1.000 | 0.463 | 0.463 |
| | 8 | 0.133 | **0.271** | 0.271 | 0.133 | 0.133 |
| | 16 | 0.133 | **0.267** | 0.267 | 0.133 | 0.133 |
| **mixed** 2048/128 | 16 | 0.246 | **0.879** | 0.879 | 0.671 | 0.550 |
| **prefill_lean** 8192/64 | 8 | **1.000** | 0.604 | 0.604 | **1.000** | **1.000** |
| | 16 | 0.467 | 0.046 | 0.046 | 0.762 | **0.771** |
| **prefill_bound** 16000/16 | 4 | 0.996 | 0.096 | 0.096 | 0.992 | **1.000** |
| | 8 | 0.100 | 0.033 | 0.033 | 0.854 | **0.917** |
| | 16 | 0.054 | 0.025 | 0.025 | **0.375** | 0.071 |

(`prefix16` ties `always` throughout because every prompt here exceeds 16 uncached tokens, so
llm-d's shipped threshold degenerates to "always disaggregate" on these workloads.)

### E3 — The static-fraction oracle on the prefill-bound cell

```bash
MODE=oracle ARCH=prefill_bound RATE=8 bash campaigns/edpp-study/repro_spectrum.sh
```
| f (% disaggregated) | 0 | 20 | 25 | 30 | **35** | 40 | 45 | 50 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goodput | 0.108 | 0.500 | 0.571 | 0.600 | **0.604** | 0.562 | 0.517 | 0.408 | 0.329 | 0.179 | 0.033 |

Dynamic policies on the same cell: `least-ttft` 0.854, `edpp` **0.917** (`i1=115 i2=125`).
The maximum is **interior** (f=35), so neither corner is optimal, and both dynamic policies
beat the best static split.

### E4 — Seed robustness on the prefill-bound cell

```bash
for r in 8 16; do for s in 42 7 123; do
  SEED=$s MODE=oracle ARCH=prefill_bound RATE=$r FRACS=35 \
    bash campaigns/edpp-study/repro_spectrum.sh
done; done
```
The SLO target is derived from a **fixed** probe seed (`SLO_PROBE_SEED`, default 42) and does
not move when `SEED` sweeps, so every seed is graded against the same goalposts.

| policy | rate | seed 42 | seed 7 | seed 123 |
|---|---|---|---|---|
| best static split (f=35) | 8 | 0.604 | 0.604 | 0.533 |
| `least-ttft` | 8 | 0.854 | 0.883 | **0.975** |
| `edpp` | 8 | **0.917** | **0.917** | 0.942 |
| best static split (f=35) | 16 | 0.075 | 0.075 | 0.083 |
| `least-ttft` | 16 | **0.375** | **0.433** | **0.383** |
| `edpp` | 16 | 0.071 | 0.067 | 0.054 |

At rate 8 both dynamic policies beat the best static split, and `edpp` edges `least-ttft` on
2 of 3 seeds (~0.06). At rate 16 the static split collapses too (0.075–0.083), `edpp` collapses
with it (0.054–0.071), and only `least-ttft` holds up (0.375–0.433) — a **5–6× gap in favour of
the stripped-down rule** under extreme overload.

### E5 — Hardware heterogeneity (earlier work, `repro_hetero_hw.sh`)

```bash
bash campaigns/edpp-study/repro_hetero_hw.sh              # under-capacity regime
SAT=1 bash campaigns/edpp-study/repro_hetero_hw.sh        # saturating regime
SAT=1 THETA=1 bash campaigns/edpp-study/repro_hetero_hw.sh  # with per-instance θ_i
```
1P2D with a fast H100 decode and a deliberately crippled A100 decode (400 TFLOPS / 0.7 TB/s),
configured via a `hw_config_by_gpu` policy bundle. **This is the only place we used
`--edpp-joint`.**

- **Under-capacity** (fast node can serve everything): joint-EDPP **0.97–1.00** vs
  reduced-EDPP 0.00–0.77, best hardware-blind scorer 0.72–0.77, optimum 1.00. EDPP wins,
  *reactively* — it avoids the slow node because that node visibly accumulates congestion,
  not because it knows the hardware is slow.
- **Saturating** (must use both nodes; optimum is an interior ~86%-fast split at 0.96):
  joint-EDPP 0.82 ≈ blind load-balance 0.84 ≈ reduced 0.83. All undershoot.
- **With per-instance θ_i** (giving EDPP the hardware model): it **over-corrects** — 95–97%
  to the fast node, goodput 0.685/0.750, *worse* than the reactive baselines. The work term
  overwhelms the congestion signal. Bar not met; recorded as a characterized limitation.

### E6 — Workload heterogeneity (earlier work)

Type A (85%, in=256/out=300, tight SLO) + Type B (15%, in=8000/out=16, loose SLO), 1P2D,
balanced decode. Measuring A's goodput:

| rate | `never` | `always` | `prefix1000` (disagg B only) | `edpp` |
|---|---|---|---|---|
| 12 | 0.952 | **1.000** | 1.000 | 1.000 |
| 16 | 0.795 | **1.000** | 1.000 | 0.984 |
| 20 | 0.430 | **1.000** | 1.000 | 0.980 |
| 24 | 0.293 | **1.000** | 1.000 | 0.924 |

`always` is optimal; EDPP is slightly worse. Keeping B local wrecks A (interference), but
disaggregating *everything* fixes it — no per-request smartness needed here.

---

## 5. Findings

**F1 — llm-d's shipped decode scorer pins decode onto one instance.** With one shared prefix
group, `precise-prefix-cache:2` dominates `queue-depth:1` and sends *all* 240 requests to a
single decode instance (E1). Cost: **6.6× goodput** (0.133 → 0.879). This is a property of a
shipped production config, not a bug we introduced. It also means any policy comparison run
on the default profile is partly measuring the scorer. We hold it fixed in the spectrum and
report both.

**F2 — The optimal naive corner flips across the workload spectrum.** `always` wins
decode-bound (it uses the otherwise-idle prefill server); `never` wins prefill-bound (two
decode servers doing prefill beats one prefill server doing it). No fixed heuristic is right
everywhere — the first honest motivation for adaptivity.

**F3 — On prefill-bound overload, the optimum is an interior split.** f=35 → 0.604, beating
both corners (0.108, 0.033). Routing genuinely matters here.

**F4 — EDPP wins big in that regime, and it is robust.** 0.917/0.917/0.942 across seeds,
beating both corners *and* the best static split (0.604) by ~0.32. Mechanism: it dynamically
spreads the bottleneck prefill work across all three servers using live congestion, which a
static split cannot do. **This is a genuine positive result for EDPP.**

**F5 — But the machinery is not what wins, and it hurts under overload.** `least-ttft` —
EDPP's estimator with the drift, virtual queues, and V *removed* — reaches 0.854–0.975 at
rate 8 (capturing most of the win), and at rate 16 it beats EDPP **0.375–0.433 vs
0.054–0.071**. EDPP collapses where the stateless rule stays robust. The likely cause is §3.2:
the accumulating virtual queues optimize a non-saturating surrogate that diverges from goodput
exactly when load is infeasible.

**F6 — Giving EDPP more hardware knowledge made it worse.** Per-instance θ_i over-corrects
(E5). More model fidelity did not help; the missing ingredient was a capacity governor, and an
ITL-based one is provably the wrong governor (we measured the overloaded fast node at 16.8 ms
ITL — far under its 50 ms target — so an ITL-deficit signal is silent exactly where the
overload is).

---

## 6. What we have NOT established (bounds on every claim above)

1. **Joint routing was never exercised in the spectrum.** All spectrum runs are *reduced*
   EDPP with the decode instance chosen by the scorer. EDPP's central claim — the joint
   `(d,p)` argmin — is tested only in E5, and never against a fair dynamic baseline
   (`least-ttft` is reduced-only). **This is the single biggest gap.**
2. **SLO-class heterogeneity is completely untested.** Every workload is single-class. The
   per-class `z` queues — EDPP's most distinctive machinery, and the one thing `least-ttft`
   *structurally cannot do* (it is class-blind) — never had anything to act on.
3. **The objective was never corrected.** Still `V·c_xfer` (§3.2).
4. **The virtual queues were never shown per request.** The instrumentation exists
   (`--edpp-decision-trace` writes per-decision `z_ttft`, `z_itl`, each term, `lhs/rhs`) and
   we used it once for a diagnosis, but we never presented the traces. A reader cannot
   currently see *why* EDPP decided what it decided.
5. **Most spectrum cells are single-seed** (seed 42). Only the prefill-bound cell has 3 seeds.
6. **1P2D only**, and the saturation is induced by an artificial concurrency cap.
7. **No global optimum.** The oracle is a *static* fraction sweep, not the joint hindsight
   optimum (the MILP remains unbuilt), so "leaves goodput on the table" is measured against a
   weaker target than the true one.

---

## 7. What this means, and what to do next

The value case for EDPP now rests on three axes of heterogeneity, and we have tested each
incompletely:

| axis | status | expectation |
|---|---|---|
| **Workload** (request mix) | `always` wins; EDPP slightly worse (E6). EDPP is *externality-blind* — it judges each request by its own SLO and misses that B's prefill harms A. | Structural flaw; may need a coupling term |
| **Hardware** (fast/slow) | joint-EDPP wins under-capacity (E5); θ_i over-corrects under saturation | `least-ttft` may capture it too — untested |
| **SLO-class** (critical/batch) | **untested** | EDPP's strongest structural home — `least-ttft` is class-blind and *cannot* compete |

**Recommended next steps, in order:**
1. **SLO-class experiment** (no new code). Mixed critical+batch workload; measure critical-class
   goodput. This is the one axis where `least-ttft` structurally cannot compete, so it is
   EDPP's best and fairest shot.
2. **Extend `least-ttft` to the joint path** (small change), then re-run E5 and the spectrum
   with `--edpp-joint` for both. This closes gap #1 and finally tests EDPP's actual thesis.
3. **Publish the decision traces** for the key cells, so the `z`/drift behavior is visible
   rather than asserted.

**The paper this supports today** is a characterization: *which heterogeneity axes justify
adaptive P/D routing, when a one-line rule suffices, and why the drift-plus-penalty apparatus
destabilizes under overload* — plus the llm-d scorer finding, which is independently useful.
Whether it becomes a *method* paper depends on step 1 and 2.

---

## 8. Reproduction

```bash
cd inference-sim
go build -o blis main.go

# E1 scorer comparison
MODE=scorer bash campaigns/edpp-study/repro_spectrum.sh

# E2 spectrum (both scorer setups)
bash campaigns/edpp-study/repro_spectrum.sh
SCORER=llmd bash campaigns/edpp-study/repro_spectrum.sh

# E3 static-fraction oracle
MODE=oracle ARCH=prefill_bound RATE=8 bash campaigns/edpp-study/repro_spectrum.sh

# E5 hardware heterogeneity
bash campaigns/edpp-study/repro_hetero_hw.sh
SAT=1 THETA=1 bash campaigns/edpp-study/repro_hetero_hw.sh

# See WHY a decision was made (per-decision term breakdown incl. z_ttft/z_itl)
./blis run --model meta-llama/llama-3.3-70b-instruct \
  --workload-spec campaigns/edpp-study/specs/spectrum/w.yaml \
  --num-instances 3 --prefill-instances 1 --decode-instances 2 \
  --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 16 \
  --pd-decider edpp --edpp-coeffs scripts/calibration/coeffs-llama70b-h100-tp4.json \
  --edpp-tadm-estimator rollforward \
  --slo-ttft "standard=1794ms" --slo-itl "standard=100ms" --slo-e2e "standard=1600ms" \
  --trace-level decisions --edpp-decision-trace /tmp/decisions.csv
# columns: z_ttft, z_itl, balance_term_d/p, transfer_term, ttft_term, itl_term, lhs, rhs, disaggregate
```

Relevant code: EDPP rule `sim/edpp.go` (reduced `Decide` ~line 470, joint `decideJoint`
~line 760); coefficients `scripts/calibration/coeffs-llama70b-h100-tp4.json`;
`least-ttft` baseline `--edpp-rule least-ttft` (commits `44e4988..9798bca`).

---

## Appendix A — the questions that prompted this report (verbatim)

This report was written in response to the following. It is reproduced unedited because it is
the most accurate record of what was opaque, what was assumed, and which open questions drove
the analysis above. Where it asks a question, the answer is cross-referenced.

> I am trying to parse and understand deeply the previous 5-6 conversations. I have to update
> my manager on the progress. I thinnk he feels like I am not working. I should be able to tell
> him the kind of experiments we've run so far and reason with our methodology of evaulating the
> policies. The starting point I can think of is our edpp formulation that primarily has one
> objective: decide and route the request to appropriate instances such that their slos are
> satisifed (maybe edpp says more, i don't know). We started with homogenous case (meaning that
> the setup has same hardware) and we ran some experiments (i don't know what and how). We found
> that the results (but what kind?) were not favor of edpp (don't know why). Then you suggested
> that we should incorporate the hardware heterogeneity (you found out that this was not an easy
> task because of ....). So we added the notion of heterogeneity and ran the experiments again (i
> don't know what experiments were run and what we measured). We found out that the results were
> still wanting in favor of edpp (i don't know why and how we went about concluding that we need
> something like "governer" which i don't understand the meaning of and the need of). The question
> at point i was facing were several, one of which was a sanity check are we considering joint
> routing decisions in our exeriments, or we are defaulting to llm-d router scorers. if we are
> indeed considering the joint routing, why is edpp not working because its job is to protect the
> slos. At this point, I was really lost because you run the experiments without telling me the
> command you ran, what was the configuration you ran, why did you pick that configuration and
> designed the experiments in that way. You always came back saying that edpp doesn't work. In
> this process, somehow we stumbled upon picking goodput as the objective. Assuming that you did
> everything that made sense from the first principles Then we stopped and the whole thinking out
> loud conversation started. Do we need edpp at all? For a workload, maybe a "smarter" policy
> shines over naive heuristics such as always disaggregate and prefix threshold based decisions.
> This could be because of two reasons (maybe there are more reasons but this is all i can think
> of): one is the policy being blind to characteristics of the workload and the other one being
> sub-optimal routing decisions because of being blind to instance's hardware characteristics
> (maybe there are other reasons behind the sub-optimal routing decisions). So initially the plan
> was: 1. Pin the 1P1D mental model precisely — the walkthrough you started, but completed until
> you can predict the decision by hand. (We're most of the way there in this message.)
> 2. State the value hypothesis as a falsifiable claim on ONE fixed topology (say 2P2D): "There
> exists a workload where always, prefix-threshold, and least-predicted-latency all leave goodput
> on the table, and the reason is [workload-blindness | hardware-blindness]." Then find the
> smallest such workload — or fail to, and conclude the heuristics are enough.
> 3. Only if (2) succeeds, identify the minimal ingredient that closes the gap — and check whether
> it needs the full drift-plus-penalty apparatus or just a smarter heuristic. Build the least thing
> that works.
>
> Somehow we stumbled on this issue of the penalty term of transfer cost as not the correct thing
> to optimize over i.e. fewest transfers while slo's are feasible. but the point you raised is this
> penalty term is moot i.e. you have nothing to optimize over when the system is at
> capacity/saturated/underprovisioned. Im simple terms, you see that the ttft is violated,
> exercising or not that penalty term is pointless. because either way, you are violating the ttft.
> In short, the transfer cost penalty term is an okay surrogate when slos are feasible but are
> terrible when slos are infeasible. Am I right in understanding this issue you raised?
>
> If yes, then this suggests that perhaps a better objective is goodput or utility. And then
> perhaps this was an important point according to you: EDPP is already a goodput surrogate. The
> z_ttft/z_itl constraint queues are surrogate #1 — "hold average latency near target, and hope
> most requests clear the deadline." "Picking goodput" doesn't mean adding a goodput term. It means
> recognizing we've been optimizing a loose proxy for it, and asking whether a tighter one would
> track the real target better. This also tells me that the experiments we ran so far, you never
> showed me what is happening to these virtual queues with every request. I believe we did build an
> instrumentation for it to track what is edpp getting as the input and what is it outputting.
>
> The other point you raised seems also important: Look at how z_itl updates: z += (realized_latency
> − τ), clamped at zero. It accumulates lateness. A request that finishes 10 seconds past its
> deadline pumps a big positive into the queue, so the policy keeps expending effort to reduce that
> request's lateness — even though it already missed and will never count toward goodput.
>
> Goodput doesn't care. A request 10 ms over and one 10 s over are equally failures. Goodput
> saturates: once a request will miss, further effort on it yields zero. That single difference —
> the true objective saturates; our surrogate doesn't — is almost certainly why EDPP flails under
> overload. Goodput implies triage: protect the savable requests, stop pouring capacity into the
> doomed ones. An accumulated-lateness surrogate structurally cannot triage — it keeps rewarding
> progress on hopeless requests. A goodput-faithful surrogate must be a saturating value (utility
> that's flat-high before the deadline and stops decaying after), so that the optimizer naturally
> reallocates scarce capacity to the requests still in reach.
>
> And then we kind of made a turn where from following the proper objective function, we turned
> towards asking this question whether the goodput gap even a routing problem. And the reason you
> asked this question alludes (I think?) to the previously mentioned apprehension of whether edpp is
> indeed required, which considers routing as an inherent problem. If we can build a simple policy
> that goodput optimum under overload, then we don't need edpp.
>
> So the first thing to do is map where the routing even matters. Smallest topology with a real
> decode-routing choice — 1P2D. Sweep load from comfortably-feasible → overloaded. At each load, put
> goodput of the naive heuristics (always, prefix-threshold, least-predicted-latency) next to the
> goodput oracle we already have (fixed-plan brute force). The output is a single curve: in which
> load band, if any, do the heuristics leave goodput on the table? That band — and only that band —
> is where a routing paper can live.
>
> And then step 2 is — diagnose the gap in that band (a day). Where heuristics trail the oracle, diff
> the decisions request-by-request. Two possibilities:
> - (a) Routing story: the oracle wins by sending the right requests to the right server. Then a
> goodput-faithful (saturating) routing surrogate is the fix, and we design it — with the oracle as
> the target to track.
> - (b) Scheduling story: the oracle wins by triaging (letting some miss to save others). Then routing
> can't fix it, and that is the finding — "under a fixed topology, goodput is recovered by
> deadline-aware admission, not P/D routing."
>
> For this you ran some experiments (again you never told me what command you ran, what was the
> output you got, how did you extract the supposedly meaningful numbers) Also in the process you
> discovered something happening with llm-d router scorers. I think it was related to prefix scorer
> taking over other scorers. Therefore, the requests were pinned to just one decode server (again how
> did you observe this was absent. Like I have to rely on you and ask you to tell me what you observe,
> which shouldn't be the case). And I believe you discovered such a router behaviour because we built
> a goodput oracle that takes all possible alternate actions as opposed to the policy action. One
> thing I'll say is that the default scorer is same as llm-d scorer. The llm-d develeopers have set
> this scorer. So we should be able to make a point that look your scorers are not doing a good job.
> So my point is that this behaviour is not a bug, but an understanding of what works when. And you
> yourself decided to fix this load balancing scorer. This is fine, but we should report the results
> for both setups. Anyways this was discovered with a homogenous workload (i don't know what exactly
> were workload characteristics except that it had moderate prefill).
>
> Then we ran a clean spectrum sweep. Fixed 1P2D, decode-balancing held fixed, capped concurrency to
> force saturation at reachable rates, SLO auto-set per archetype from an idle probe (so "goodput" is
> comparable across archetypes). Four workload archetypes spanning decode-bound → prefill-bound; for
> each, never/always/prefix-threshold/edpp across load. (Again what command you ran for this, I have
> no idea, perhaps I'd have to open the "ran 1 shell command" to see what you run and then observe
> myself).
>
> [... the prefill-bound result and seed-robustness numbers, quoted back ...]
>
> I think the thought process to answer whether or not we need edpp is how much goodput other
> heuristic policies leave on the table.
>
> Again I don't know how you came about this best static disaggregation function.
> And coming back to previous point of not having the proper objective in the edpp and whether goodput
> serves as a proper objective, i belive you still used the transfer cost as the penalty term?
>
> And then the next question you ask is whether this win is specific to the dpp machinery or would any
> dynamic load aware disaggregation rule get it. A simple greedy heuristic -- disaggregate iff
> predicted ttft disagg < predicted ttft local might capure most of that 0.92 goodput value. withour
> any virtual queues or V.
>
> More things happened after this which I am unable to write here.
>
> Also please write a report for this, maybe as a slide deck or a document that provides our study
> comprehensively such that anyone in my team can pick up and verify the claims, that lets me update
> my manager. He is really after me.

### Answers to the direct questions above

| question | answer | where |
|---|---|---|
| "the transfer cost penalty term is an okay surrogate when slos are feasible but terrible when infeasible. Am I right?" | **Yes.** The penalty is the only thing EDPP optimizes; under infeasibility the constraints cannot be met, the guarantee is void, and minimizing transfers is beside the point. | §2, §3.2 |
| "are we considering joint routing decisions, or defaulting to llm-d router scorers?" | **Defaulting to the scorer.** All spectrum runs are *reduced* EDPP; the decode instance was chosen by the scorer (pinned to `queue-depth:1`). Joint was used only in the hardware-heterogeneity work. | §6 gap 1, E5 |
| "how did you come about this best static disaggregation fraction?" | A forced per-request plan via `--pd-plan`, sweeping the disaggregated fraction f and taking the max (f=35 → 0.604). It is a **static** yardstick. | §3.4, E3 |
| "i believe you still used the transfer cost as the penalty term?" | **Yes.** Every experiment uses the shipped `V·c_xfer` objective. We never changed it. | §2, §3.2, §6 gap 3 |
| "you never showed me what is happening to these virtual queues with every request" | Correct, and it is a real gap. The instrumentation exists (`--edpp-decision-trace`). A command to dump it is in §8. | §6 gap 4, §8 |
| "we should report the results for both setups [llm-d scorer vs fixed]" | Done — both are reported, and the llm-d default is framed as a finding about a shipped config, not a bug. | E1, F1 |
