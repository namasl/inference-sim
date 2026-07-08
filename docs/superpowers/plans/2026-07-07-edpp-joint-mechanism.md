# Full-Joint P/D — Sub-Project 1: Homogeneous Joint Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `--edpp-joint` mode to the EDPP decider that enumerates all `(d,p)` actions and selects the drift-plus-penalty argmin — choosing the decode instance itself (not the scorer) — with per-instance congestion `Q_i` and per-candidate cache-aware `a_p`, plus a scorer-vs-joint divergence log; then run an exploratory topology×workload sweep.

**Architecture:** Reuse the existing `EDPPDecider` machinery. Add per-instance waiting-work bookkeeping (`qByInstance`) alongside the untouched pool-level scalars (reduced path stays byte-identical). In joint mode, `Decide` loops over decode snapshots × prefill snapshots ∪ {local}, computes `J(d,p)` with per-candidate `a_p` (via the instance-keyed `cacheQuery`) and per-candidate occupancy-aware `T̂`, argmins with a deterministic tie-break, and emits `DecodePodOverride`/`PrefillPodHint` (wiring already exists). A per-decision log records the scorer's picks vs the joint picks.

**Tech Stack:** Go 1.22 (`sim/`, `sim/cluster/`, `cmd/`, `sim/trace/`), Python3 + pandas, bash.

## Global Constraints

- Branch: `feat/edpp-estimator-validation`. Spec: `docs/superpowers/specs/2026-07-07-edpp-joint-mechanism-design.md`.
- **Homogeneous hardware**: single coeff set (`d.coeffs`), per-class scalar `z`. Per-instance `θ_i`, per-instance `Z^I_i`, and simulator intra-pool hardware heterogeneity are OUT of scope (sub-project 2).
- **Reduced path byte-identical** when `--edpp-joint` is off: leave the existing `qpWork`/`qdWork` scalars and the existing `Decide` local-vs-disagg path untouched; joint code is gated behind the flag.
- **Per-candidate cache-aware `a_p` for BOTH pools**: `a_p(node) = len(InputTokens) − cachedTokens(d.cacheQuery[node](InputTokens))`, block-aligned exactly as the reduced path computes it for `SelectedInstance` (`sim/edpp.go` ~line 400). Decode node's cache for `p=local`; each prefill node's cache for `p∈𝒫`.
- **Deterministic tie-break**: on equal `J`, pick the lowest instance index (then local before disagg) — INV-6.
- **Occupancy-aware `T̂`**: joint mode uses the estimator selected by `--edpp-tadm-estimator` (the harness passes `rollforward`); evaluated per candidate snapshot.
- **INV-9**: `Decide` reads only `len(req.InputTokens)`, cache, `N̂_out`, snapshots — never `req.OutputTokens`. **INV-13**: `--edpp-joint` on run AND replay.
- The objective `J(d,p)` is the **normalized** form in spec §3/§3.1 — every term carries its divisor:
  `q_d·W_d + (p=local?q_d:q_p)·W_p(a_p(loc),a_r) + z_ttft·T̂(a)/τ_ttft + z_itl·(m_dec+1{local}m_pf)/τ_itl + V·c_xfer·(τ_ref/τ_ttft)·1{disagg}`, with `q_i=Q_i/W*`. Use the **absolute** per-candidate `T̂(a)` (NOT the reduced rule's `ttft_p−ttft_d` difference); extract the reduced rule's normalized term helpers (`/τ`, `W*`), not its differenced decision expression. Two documented homogeneous-cut deviations from §5.3: per-class `z_itl` (not per-instance `Z^I_d`) and single `θ` (not `θ_i`).
- Go tests: `go test ./sim/... -run <name>`; build `go build -o blis main.go`; gofmt before commit.

---

### Task 1: Per-instance congestion queues `Q_i`

**Files:**
- Modify: `sim/edpp.go` (add `qByInstance`; populate in `OnRoute`; drain in `OnAdmit`/`OnComplete`/`Forget`)
- Modify: the `OnRoute` call site(s) in `sim/cluster/` that must pass the chosen decode/prefill instance IDs
- Test: `sim/edpp_qbyinstance_test.go`

**Interfaces:**
- Consumes: existing `edppPendingWork` (`sim/edpp.go:264`), `OnRoute(req, key, toPrefill, apTokens)`.
- Produces:
  - `EDPPDecider.qByInstance map[InstanceID]*edppInstWork` where `type edppInstWork struct{ wp, wd float64 }` (waiting work at that instance).
  - Extended `OnRoute(req *Request, key string, toPrefill bool, apTokens int, decodeInst, prefillInst string)` — the chosen decode instance, and prefill instance (`""`/`==decodeInst` for local). (Confirm the call sites and update them; keep the pool-level scalars updated exactly as today.)
  - `EDPPDecider.QByInstance() map[string]struct{Wp, Wd float64}` accessor (for tests + the joint reader).

- [ ] **Step 1: Write the failing test**

```go
// sim/edpp_qbyinstance_test.go
package sim

import "testing"

func TestQByInstance_SumsMatchPoolLevel(t *testing.T) {
	d := newTestEDPPDecider(t) // helper that builds a decider with known coeffs; mirror existing edpp_test.go setup
	// disagg route: prefill work → prefill inst "P0", decode work → decode inst "M1"
	d.OnRoute(&Request{ID: "r1", InputTokens: make([]int, 1000), SLOClass: "batch"}, "r1", true, 1000, "M1", "P0")
	// local route: prefill+decode both → "M0"
	d.OnRoute(&Request{ID: "r2", InputTokens: make([]int, 500), SLOClass: "batch"}, "r2", false, 500, "M0", "")

	q := d.QByInstance()
	// P0 holds r1's prefill work; M1 holds r1's decode work; M0 holds r2's (wp+wd).
	if q["P0"].Wp <= 0 || q["P0"].Wd != 0 {
		t.Fatalf("P0 = %+v, want Wp>0 Wd==0", q["P0"])
	}
	if q["M1"].Wd <= 0 || q["M1"].Wp != 0 {
		t.Fatalf("M1 = %+v, want Wd>0 Wp==0", q["M1"])
	}
	if q["M0"].Wd <= 0 {
		t.Fatalf("M0 = %+v, want Wd>0 (local wp+wd)", q["M0"])
	}
	// INVARIANT: per-instance sums equal the pool-level scalars (byte-identical reduced bookkeeping).
	var sumWp, sumWd float64
	for _, v := range q {
		sumWp += v.Wp
		sumWd += v.Wd
	}
	if !floatEq(sumWp, d.qpWork) || !floatEq(sumWd, d.qdWork) {
		t.Fatalf("per-instance sums (wp=%v wd=%v) != pool-level (qp=%v qd=%v)", sumWp, sumWd, d.qpWork, d.qdWork)
	}
}
```
(Reuse the existing test helpers/`floatEq` in `sim/edpp_test.go`; if none, add a tiny `floatEq(a,b)=math.Abs(a-b)<1e-9`.)

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestQByInstance -v` → FAIL (`QByInstance`/extended `OnRoute` undefined).

- [ ] **Step 3: Add `qByInstance` + extend `OnRoute`**

In `sim/edpp.go`: add field `qByInstance map[InstanceID]*edppInstWork` (init in `NewEDPPDecider`), type `edppInstWork struct{ wp, wd float64 }`, and record the routed instances on `edppPendingWork` (`decodeInst, prefillInst string`) so drains can find them. Extend `OnRoute` to accept `decodeInst, prefillInst string`; keep the existing pool-scalar updates verbatim and mirror them per-instance:
```go
func (d *EDPPDecider) OnRoute(req *Request, key string, toPrefill bool, apTokens int, decodeInst, prefillInst string) {
	d.awaitingFirstToken[key] = &edppAwaiting{startUs: req.ArrivalTime, class: req.SLOClass}
	if apTokens <= 0 {
		return
	}
	wp := d.coeffs.Wp(apTokens, len(req.InputTokens))
	wd := d.coeffs.Wd(len(req.InputTokens), d.nHatFor(req.SLOClass).mean())
	pw := edppPendingWork{toPrefill: toPrefill, decodeInst: decodeInst, prefillInst: prefillInst}
	if toPrefill {
		pw.wp, pw.wd = wp, wd
		d.qpWork += wp
		d.qdWork += wd
		d.instWork(prefillInst).wp += wp
		d.instWork(decodeInst).wd += wd
	} else {
		pw.wd = wp + wd
		d.qdWork += wp + wd
		d.instWork(decodeInst).wd += wp + wd
	}
	d.pending[key] = pw
}
// instWork returns (creating if needed) the per-instance work accumulator; "" is ignored (nil-safe).
func (d *EDPPDecider) instWork(id string) *edppInstWork { ... }
```
In `OnAdmit`/`OnComplete`/`Forget`: wherever the pool-level scalar is decremented for a `pending[key]`, also decrement the same amount from `qByInstance[pw.decodeInst]`/`[pw.prefillInst]` (guard against going negative like the pool path does). Add the `QByInstance()` accessor. **Confirm the exact drain arithmetic in situ** and mirror it per-instance.

- [ ] **Step 4: Update the `OnRoute` call site(s)**

Find where the cluster calls `decider.OnRoute(...)` (grep `OnRoute(` in `sim/cluster/`), and pass the chosen decode instance (the parent's `DecodeInstanceID` or the local decode target) and prefill instance (`parent.PrefillInstanceID` when disagg, else `""`). Confirm the chosen instances are known at that call site; if the prefill instance isn't chosen until later, pass `""` there and attribute prefill work at prefill-routing time (note this in situ). Keep behavior identical for the pool-level path.

- [ ] **Step 5: Run the test + full sim suite**

Run: `go test ./sim/ -run TestQByInstance -v && go test ./sim/...`
Expected: PASS; the sum-equals-pool invariant holds; no other sim test regresses (reduced path unchanged). gofmt.

- [ ] **Step 6: Commit**

```bash
git add sim/edpp.go sim/edpp_qbyinstance_test.go sim/cluster/
git commit -m "feat(edpp): per-instance congestion queues qByInstance (sums == pool-level)"
```

---

### Task 2: Joint `Decide` — enumeration + per-candidate cache-aware `a_p` + argmin + `--edpp-joint`

**Files:**
- Modify: `sim/edpp.go` (joint branch in `Decide`; `EDPPConfig.Joint bool`), `cmd/root.go` (flag), the EDPP construction site `sim/cluster/cluster.go:~448`
- Test: `sim/edpp_joint_test.go`

**Interfaces:**
- Consumes: `qByInstance`/`QByInstance()` (Task 1), `d.cacheQuery` (instance-keyed), `d.coeffs.Wp/Wd`, the admission estimator, `state.Snapshots` (decode candidates), `d.prefillSnapshots()` (prefill candidates).
- Produces: joint `Decide` sets `DisaggregationDecision{Disaggregate, DecodePodOverride, PrefillPodHint}` from the argmin; a helper `jointCandidateCost(req, class, d, p, state) (J float64, ...)` used by both Decide and the divergence log (Task 3).

- [ ] **Step 1: Write the failing tests**

```go
// sim/edpp_joint_test.go — behavior tests on constructed snapshots.
func TestJoint_PicksLowerOccupancyDecode(t *testing.T) {
	// Two decode candidates, identical except M1 has much less queued work than M0.
	// Homogeneous cache (both cold). A kept-local request must go to M1.
	d := newJointTestDecider(t) // Joint:true, known coeffs, cacheQuery returns 0 (cold) for all
	state := twoDecodeState(t, /*M0 busy*/ 500.0, /*M1 idle*/ 0.0) // helper builds RouterState w/ 2 decode snapshots + SelectedInstance="M0"
	dec := d.Decide(reqBatch("r1", 200), state)
	if dec.DecodePodOverride != "M1" {
		t.Fatalf("joint decode pick = %q, want M1 (lower occupancy)", dec.DecodePodOverride)
	}
}

func TestJoint_PrefersCacheWarmOverIdleCold(t *testing.T) {
	// Large prompt. M0 is busier but cache-WARM (a_p small); M1 idle but COLD (a_p = full).
	// The cache-cost term should make joint pick M0 despite higher occupancy.
	d := newJointTestDecider(t)
	state := twoDecodeState(t, 50.0 /*M0 mild load*/, 0.0 /*M1 idle*/)
	d.cacheQuery = map[string]func([]int) int{
		"M0": func(toks []int) int { return len(toks) },        // fully cached → a_p≈0
		"M1": func(toks []int) int { return 0 },                // cold → a_p = full
	}
	dec := d.Decide(reqBatch("r2", 8000), state) // large prompt
	if dec.DecodePodOverride != "M0" {
		t.Fatalf("joint pick = %q, want M0 (cache-warm beats idle-cold for a large prompt)", dec.DecodePodOverride)
	}
}

func TestJoint_DeterministicTieBreak(t *testing.T) {
	// Fully identical candidates → lowest index, and stable across runs.
	d := newJointTestDecider(t)
	state := twoDecodeState(t, 0.0, 0.0)
	a := d.Decide(reqBatch("r3", 100), state).DecodePodOverride
	b := d.Decide(reqBatch("r3", 100), state).DecodePodOverride
	if a != b || a != "M0" {
		t.Fatalf("tie-break not deterministic/lowest-index: %q then %q", a, b)
	}
}

func TestJoint_ReducesToScorerSliceMatchesReduced(t *testing.T) {
	// §5.5 reduction: joint J restricted to the scorer's d reproduces the reduced local-vs-disagg call.
	// Build one decider in reduced mode, one in joint mode, force the same single decode candidate,
	// and assert the disaggregate decision matches.
	... // construct identical state with a single decode snapshot == SelectedInstance
}
```
(These need small helpers `newJointTestDecider`, `twoDecodeState`, `reqBatch` — add them in the test file, mirroring the snapshot/`RouterState` construction in `sim/edpp_test.go`.)

- [ ] **Step 2: Run to verify they fail**

Run: `go test ./sim/ -run TestJoint -v` → FAIL (joint branch/`EDPPConfig.Joint` absent).

- [ ] **Step 3: Implement the joint branch + `a_p` helper**

Add `EDPPConfig.Joint bool` and `EDPPDecider.joint bool`. Add a cache-aware `a_p` helper (extract the reduced path's block-aligned computation at `sim/edpp.go` ~line 400 into a reusable `apForInstance(req, instID) int`). In `Decide`, at the top: `if d.joint { return d.decideJoint(req, state) }` (leaving the reduced body untouched). Implement `decideJoint`:
```go
func (d *EDPPDecider) decideJoint(req *Request, state *RouterState) DisaggregationDecision {
	class := req.SLOClass
	zt, zi := d.zFor(class) // per-class scalar z (reuse existing accessor)
	decodeSnaps := state.Snapshots
	prefillSnaps := d.prefillSnapshots() // may be empty
	type cand struct{ d, p string; local bool; J float64 }
	var best *cand
	consider := func(c cand) {
		// deterministic tie-break: strictly-less replaces; equal keeps the earlier (lower index / local-first) candidate
		if best == nil || c.J < best.J-1e-12 {
			cc := c; best = &cc
		}
	}
	for _, ds := range decodeSnaps {              // d ∈ ℳ (sorted by instance id for determinism)
		dID := ds.InstanceID
		qd := d.instWorkNorm(dID, /*decode*/ true)  // q_d = qByInstance[dID].wd / W*
		// local: prefill+decode on dID
		apLoc := d.apForInstance(req, dID)
		Jlocal := qd*d.coeffs.Wd(len(req.InputTokens), d.nHatFor(class).mean()) +
			qd*d.coeffs.Wp(apLoc, len(req.InputTokens)) +
			zt*d.tHatLocalTerms(req, ds) + zi*d.itlTerms(req, ds, /*local*/ true)
		consider(cand{d: dID, local: true, J: Jlocal})
		// disagg: decode on dID, prefill on each p
		for _, ps := range prefillSnaps {
			pID := ps.InstanceID
			apP := d.apForInstance(req, pID)
			qp := d.instWorkNorm(pID, /*decode*/ false)
			Jdis := qd*d.coeffs.Wd(len(req.InputTokens), d.nHatFor(class).mean()) +
				qp*d.coeffs.Wp(apP, len(req.InputTokens)) +
				zt*d.tHatDisaggTerms(req, ds, ps) + zi*d.itlTerms(req, ds, false) +
				d.transferPenalty(class)
			consider(cand{d: dID, p: pID, J: Jdis})
		}
	}
	if best == nil { return DisaggregationDecision{} } // no candidates → reduced fallback / never
	if best.local {
		return DisaggregationDecision{Disaggregate: false, DecodePodOverride: best.d}
	}
	return DisaggregationDecision{Disaggregate: true, DecodePodOverride: best.d, PrefillPodHint: best.p}
}
```
Reuse the reduced rule's existing term computations for `tHatLocalTerms`/`tHatDisaggTerms` (the `ttft_p`/`ttft_d` construction, per-candidate snapshot instead of the selected one), `itlTerms` (the collapsed ITL term), `transferPenalty` (`V·c_xfer·τ_ref/τ_ttft` factor), and normalization by `W*` — extract them from the current `Decide` body so both paths share them (DRY). Ensure `decodeSnaps`/`prefillSnaps` are iterated in a deterministic (sorted-by-ID) order.

- [ ] **Step 4: Wire `--edpp-joint`**

Register `--edpp-joint` (bool) in `registerSimConfigFlags` (run+replay), carry to `EDPPConfig.Joint` at the construction site (`cluster.go:~448`). Default false → reduced path.

- [ ] **Step 5: Run the joint tests + full suite**

Run: `go test ./sim/ -run TestJoint -v && go build -o blis main.go && go test ./sim/...`
Expected: the 4 joint tests PASS; reduced-path tests unchanged; build green. gofmt.

- [ ] **Step 6: Commit**

```bash
git add sim/edpp.go cmd/root.go cmd/replay.go sim/cluster/cluster.go sim/edpp_joint_test.go
git commit -m "feat(edpp): --edpp-joint mode — (d,p) argmin with per-candidate cache-aware a_p + per-instance Q_i"
```

---

### Task 3: Scorer-vs-joint divergence log

**Files:**
- Modify: `sim/edpp.go` (populate a divergence record in joint `Decide`), `sim/trace/` (a small CSV writer or columns on the edpp decision trace), `sim/cluster/` (wire the trace flag)
- Test: `sim/edpp_joint_test.go` (assert the record fields)

**Interfaces:**
- Consumes: joint `Decide`, `state.SelectedInstance` (scorer's decode pick), `d.prefillSnapshots()` + the prefill scorer (shadow-run for `scorer_p`), `jointCandidateCost`.
- Produces: per-decision fields `scorer_d, joint_d, scorer_p, joint_p, agree_d, agree_p, J_scorer, J_joint`, emitted when a divergence-trace flag is set.

- [ ] **Step 1: Write the failing test** — assert that after a joint decision where the scorer's `SelectedInstance` differs from the argmin, the captured record has `scorer_d == SelectedInstance`, `joint_d == override`, `agree_d == false`, and `J_joint <= J_scorer` (the argmin is no worse than the scorer's slice by construction). Include a `scorer_p` shadow assertion on a disagg decision.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement.** In `decideJoint`, when the divergence trace is enabled, compute `scorer_d = state.SelectedInstance`, shadow-run the prefill scorer over `prefillSnaps` for `scorer_p` (compute-only, not acted on), evaluate `J` at the scorer's `(scorer_d, scorer_p-or-local)` slice, and record `{scorer_d, joint_d, scorer_p, joint_p, agree, J_scorer, J_joint}`. Emit via a new `--edpp-joint-trace <csv>` (mirror `--edpp-decision-trace` plumbing) OR as extra columns on the existing decision trace — pick one and keep it gated/zero-cost-off. `J_joint <= J_scorer` always (argmin over a superset of the scorer's slice) — assert it as an internal invariant.

- [ ] **Step 4: Run tests + suite.** gofmt.

- [ ] **Step 5: Commit** `feat(edpp): scorer-vs-joint divergence trace (scorer_d/p vs joint_d/p, J's)`.

---

### Task 4: Exploratory sweep + findings

**Files:**
- Create: `campaigns/edpp-study/repro_joint.sh`, `campaigns/edpp-study/analyze/joint_divergence.py` (+ its self-test)
- Modify: `campaigns/edpp-study/FINDINGS.md`, `campaigns/edpp-study/README.md`

**Interfaces:**
- Consumes: `blis --edpp-joint --edpp-tadm-estimator rollforward --edpp-joint-trace`, `repro_counterfactual.sh` (reuse for regret), the divergence trace.

- [ ] **Step 1: Correctness gates first (pass/fail).** In `repro_joint.sh`: (a) a run with `--edpp-joint` OFF must be byte-identical to the current reduced run (diff metrics/decision trace); (b) the counterfactual self-consistency gate passes on the joint plan. Fail loudly otherwise.

- [ ] **Step 2: Write `repro_joint.sh` — the sweep.** For each cell in {topology ∈ (1P2D, 2P2D)} × {workload ∈ (synth_cf cache-uniform, a new cache-asymmetric unique-large-prompt spec)}: run reduced-EDPP and `--edpp-joint` (both with `--edpp-tadm-estimator rollforward`), emit `--pd-outcome-trace` + `--edpp-joint-trace` + `--metrics-path`; run `counterfactual_regret.py` (reuse) for regret and `joint_divergence.py` for the divergence summary. Add the cache-asymmetric spec via `make_specs.py` (unique prompts, no shared prefix, large input distribution). Echo a per-cell summary.

- [ ] **Step 3: Write `joint_divergence.py` + self-test.** Reads the `--edpp-joint-trace` CSV; reports `d`-divergence rate, `p`-divergence rate, and on divergent rows the direction (share where joint picked lower-`J` / lower-occupancy / lower-`a_p`). Self-test on a synthetic trace with known divergences (plain-assert, like the other analyzers).

- [ ] **Step 4: Run the sweep; RECORD WHAT HAPPENS.** Run `bash campaigns/edpp-study/repro_joint.sh` (small K). Report the observed numbers honestly per cell — regret (joint vs reduced), goodput, divergence rate + direction — WITHOUT forcing a "joint wins" narrative. If joint loses on the cache-asymmetric cell (or anywhere), that is a finding.

- [ ] **Step 5: Write FINDINGS + README.** FINDINGS "Joint mechanism (sub-project 1)": the correctness gates (reduction, byte-identical-off), then the sweep table (per cell: regret joint/reduced, divergence rate/direction) and an honest reading — where joint diverges, the direction, and the observed effect; explicitly note this is homogeneous-hardware (cache+occupancy levers, θ_i deferred) and a LOCAL diagnostic. README pointer. Reproduce: `bash campaigns/edpp-study/repro_joint.sh`.

- [ ] **Step 6: Commit** `docs(edpp): joint-mechanism sweep + divergence findings`.

---

## Notes for the implementer (confirm-in-situ)

- **`OnRoute` threading (Task 1 Step 4):** the exact cluster call site of `decider.OnRoute` and whether the chosen prefill instance is known there vs at prefill-routing time — confirm and thread accordingly; if prefill is chosen later, attribute prefill work then (note it), keeping pool-level behavior identical.
- **Extracting shared term helpers (Task 2 Step 3):** the reduced `Decide` computes `ttft_p`/`ttft_d`, the ITL term, transfer penalty, and `W*` normalization inline. Extract these into helpers callable per-candidate WITHOUT changing the reduced path's numbers (verify a reduced-mode test stays byte-identical). This is the load-bearing refactor; do it carefully.
- **Trace form (Task 3):** choose new `--edpp-joint-trace` vs columns on `--edpp-decision-trace`; either is fine if gated and zero-cost-off.
- **Cache-asymmetric spec (Task 4):** unique prompts (no `prefix_group`/large per-request variance) so the two decode nodes genuinely diverge in cache warmth as the run proceeds; confirm via the divergence log that `a_p` differs across candidates.
