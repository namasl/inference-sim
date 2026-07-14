package sim

import "testing"

// newJointTestDecider builds a decider in joint mode with the known test coeffs and a
// default-cold cacheQuery (every instance returns 0 cached blocks ⇒ a_p = full prompt).
func newJointTestDecider(t *testing.T) *EDPPDecider {
	t.Helper()
	cfg := defaultTestEDPPConfig()
	cfg.Joint = true
	// Prefill pool provides a single candidate "P0" so disagg is enumerable.
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "P0"}} }
	d := NewEDPPDecider(cfg, newTestAffineModel(), coldCacheQuery("M0", "M1", "P0"), prefill)
	return d
}

// coldCacheQuery returns a cacheQuery map where every listed instance reports 0 cached
// blocks (fully cold ⇒ a_p = full prompt length).
func coldCacheQuery(ids ...string) map[string]func([]int) int {
	m := make(map[string]func([]int) int, len(ids))
	for _, id := range ids {
		m[id] = func([]int) int { return 0 }
	}
	return m
}

// twoDecodeState builds a RouterState with two decode snapshots M0 and M1, whose queued
// decode work is expressed via ResidentPrefillTokens/BatchSize proxies. Here we drive
// occupancy through the per-instance congestion queue directly (see caller), so the
// snapshots only need distinct IDs; m0Load/m1Load set BatchSize as a coarse occupancy
// proxy consumed by the admission-delay predictor.
func twoDecodeState(t *testing.T, m0Load, m1Load float64) *RouterState {
	t.Helper()
	return &RouterState{
		SelectedInstance: "M0",
		Snapshots: []RoutingSnapshot{
			{ID: "M0", BatchSize: int(m0Load)},
			{ID: "M1", BatchSize: int(m1Load)},
		},
	}
}

// reqBatch builds a "batch"-class Request with nInput input tokens.
func reqBatch(id string, nInput int) *Request {
	return &Request{ID: id, InputTokens: make([]int, nInput), SLOClass: "batch"}
}

func TestJoint_PicksLowerOccupancyDecode(t *testing.T) {
	// Two decode candidates, identical except M0 carries much more queued decode work
	// than M1. A kept-local request must go to M1 (lower congestion q_d).
	d := newJointTestDecider(t)
	state := twoDecodeState(t, 500.0, 0.0)
	// Load the per-instance congestion queue: heavy decode backlog on M0, none on M1.
	d.OnRoute(reqBatch("bg", 20000), "bg", false, 20000, "M0", "")
	dec := d.Decide(reqBatch("r1", 200), state)
	if dec.DecodePodOverride != "M1" {
		t.Fatalf("joint decode pick = %q, want M1 (lower occupancy)", dec.DecodePodOverride)
	}
}

func TestJoint_PrefersCacheWarmOverIdleCold(t *testing.T) {
	// Large prompt. M0 is mildly loaded but cache-WARM (a_p≈0); M1 is idle but COLD
	// (a_p = full). The cache-cost term must make joint keep local on M0 despite M1 idle.
	d := newJointTestDecider(t)
	state := twoDecodeState(t, 50.0, 0.0)
	d.cacheQuery = map[string]func([]int) int{
		"M0": func(toks []int) int { return len(toks) / d.cfg.BlockSize }, // fully cached → a_p≈0
		"M1": func(toks []int) int { return 0 },                           // cold → a_p = full
		"P0": func(toks []int) int { return 0 },
	}
	dec := d.Decide(reqBatch("r2", 8000), state)
	if dec.DecodePodOverride != "M0" {
		t.Fatalf("joint pick = %q, want M0 (cache-warm beats idle-cold for a large prompt)", dec.DecodePodOverride)
	}
}

func TestJoint_DisaggToWarmPrefillNode(t *testing.T) {
	// Locks per-node a_p sourcing for disagg candidates: apForInstance must read the
	// PREFILL node's cache, not the decode node's. Large prompt; decode nodes M0/M1 are
	// COLD (local prefill on d = full a_p, expensive) while the disagg PREFILL node P0 is
	// cache-WARM (a_p≈0 → cheap prefill + tiny transfer). Joint must disaggregate to P0.
	d := newJointTestDecider(t)
	state := twoDecodeState(t, 0.0, 0.0)
	d.cacheQuery = map[string]func([]int) int{
		"M0": func(toks []int) int { return 0 },                           // cold decode
		"M1": func(toks []int) int { return 0 },                           // cold decode
		"P0": func(toks []int) int { return len(toks) / d.cfg.BlockSize }, // warm prefill → a_p≈0
	}
	// z_ttft pressure so the (expensive local prefill) TTFT term dominates over the small
	// transfer penalty, favoring the warm prefill node.
	d.ensureZ("batch").zTTFT = 1e7
	dec := d.Decide(reqBatch("r4", 8000), state)
	if !dec.Disaggregate {
		t.Fatalf("joint kept local; want disagg to warm prefill node (large cold-local prefill)")
	}
	if dec.PrefillPodHint != "P0" {
		t.Fatalf("joint disagg prefill hint = %q, want P0 (warm prefill node)", dec.PrefillPodHint)
	}
}

func TestJoint_DeterministicTieBreak(t *testing.T) {
	// Fully identical candidates → lowest index, stable across repeated calls.
	d := newJointTestDecider(t)
	state := twoDecodeState(t, 0.0, 0.0)
	a := d.Decide(reqBatch("r3", 100), state).DecodePodOverride
	b := d.Decide(reqBatch("r3", 100), state).DecodePodOverride
	if a != b || a != "M0" {
		t.Fatalf("tie-break not deterministic/lowest-index: %q then %q", a, b)
	}
}

// lowestIDPrefillScorer mimics a prefill routing policy that always picks the
// lowest-ID prefill snapshot. Used as an injected shadow scorer in divergence-trace tests.
func lowestIDPrefillScorer(_ *Request, snaps []RoutingSnapshot) string {
	best := ""
	for _, s := range snaps {
		if best == "" || s.ID < best {
			best = s.ID
		}
	}
	return best
}

func TestJoint_DivergenceTrace_DecodeOverride(t *testing.T) {
	// The decode scorer pre-selects M0, but M0 carries heavy queued decode work, so the
	// joint argmin overrides to M1. The divergence trace must record scorer_d == M0,
	// joint_d == M1 (the override), agree_d == false, and J_joint <= J_scorer.
	d := newJointTestDecider(t)
	d.cfg.JointTraceEnabled = true
	d.SetPrefillScorer(lowestIDPrefillScorer)
	state := twoDecodeState(t, 500.0, 0.0) // SelectedInstance = M0
	d.OnRoute(reqBatch("bg", 20000), "bg", false, 20000, "M0", "")

	dec := d.Decide(reqBatch("r1", 200), state)
	tr := dec.EDPPJointTrace
	if tr == nil {
		t.Fatal("expected non-nil EDPPJointTrace when JointTraceEnabled")
	}
	if tr.ScorerD != "M0" {
		t.Errorf("ScorerD = %q, want M0 (state.SelectedInstance)", tr.ScorerD)
	}
	if tr.JointD != dec.DecodePodOverride || tr.JointD != "M1" {
		t.Errorf("JointD = %q, want M1 (== DecodePodOverride %q)", tr.JointD, dec.DecodePodOverride)
	}
	if tr.AgreeD {
		t.Errorf("AgreeD = true, want false (scorer M0 != joint M1)")
	}
	if tr.AgreeD != (tr.ScorerD == tr.JointD) {
		t.Errorf("AgreeD (%v) inconsistent with ScorerD==JointD (%v==%v)", tr.AgreeD, tr.ScorerD, tr.JointD)
	}
	if tr.JJoint > tr.JScorer+1e-9 {
		t.Errorf("internal invariant violated: JJoint (%g) must be <= JScorer (%g)", tr.JJoint, tr.JScorer)
	}
	// This is a kept-local decision, so no prefill node is involved.
	if dec.Disaggregate {
		t.Fatalf("expected local decision for this state")
	}
	if tr.ScorerP != "" || tr.JointP != "" {
		t.Errorf("local decision must leave ScorerP/JointP empty, got %q/%q", tr.ScorerP, tr.JointP)
	}
}

func TestJoint_DivergenceTrace_ScorerPOnDisagg(t *testing.T) {
	// A disaggregating decision (large cold-local prompt, warm prefill node) must populate
	// scorer_p via the shadow prefill scorer, and the J_joint <= J_scorer invariant holds.
	d := newJointTestDecider(t)
	d.cfg.JointTraceEnabled = true
	d.SetPrefillScorer(lowestIDPrefillScorer)
	state := twoDecodeState(t, 0.0, 0.0)
	d.cacheQuery = map[string]func([]int) int{
		"M0": func(toks []int) int { return 0 },
		"M1": func(toks []int) int { return 0 },
		"P0": func(toks []int) int { return len(toks) / d.cfg.BlockSize }, // warm prefill
	}
	d.ensureZ("batch").zTTFT = 1e7

	dec := d.Decide(reqBatch("r4", 8000), state)
	if !dec.Disaggregate {
		t.Fatalf("expected disagg decision")
	}
	tr := dec.EDPPJointTrace
	if tr == nil {
		t.Fatal("expected non-nil EDPPJointTrace")
	}
	if tr.ScorerP != "P0" {
		t.Errorf("ScorerP = %q, want P0 (shadow prefill scorer pick on disagg)", tr.ScorerP)
	}
	if tr.JointP != dec.PrefillPodHint {
		t.Errorf("JointP = %q, want %q (== PrefillPodHint)", tr.JointP, dec.PrefillPodHint)
	}
	if tr.AgreeP != (tr.ScorerP == tr.JointP) {
		t.Errorf("AgreeP (%v) inconsistent with ScorerP==JointP", tr.AgreeP)
	}
	if tr.JJoint > tr.JScorer+1e-9 {
		t.Errorf("internal invariant violated: JJoint (%g) must be <= JScorer (%g)", tr.JJoint, tr.JScorer)
	}
}

func TestJoint_DivergenceTrace_NilWhenDisabled(t *testing.T) {
	d := newJointTestDecider(t) // JointTraceEnabled defaults false
	dec := d.Decide(reqBatch("r1", 200), twoDecodeState(t, 0.0, 0.0))
	if dec.EDPPJointTrace != nil {
		t.Errorf("EDPPJointTrace must be nil when JointTraceEnabled is off, got %+v", dec.EDPPJointTrace)
	}
}

func TestDecideJoint_HomogeneousCoeffsByGPU_ByteIdentical(t *testing.T) {
	// INV-6 byte-identity: a joint decider whose CoeffsByGPU maps every GPU type back to
	// the global coeffs must produce the SAME decision as one with no CoeffsByGPU, because
	// coeffsFor returns d.coeffs for every candidate and the per-candidate wd/mDec recompute
	// identical float values.
	base := defaultTestEDPPConfig().Coeffs
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "P0"}} }
	cache := coldCacheQuery("M0", "M1", "P0")

	cfgPlain := defaultTestEDPPConfig()
	cfgPlain.Joint = true
	cfgDup := cfgPlain
	cfgDup.CoeffsByGPU = map[string]EDPPCoeffs{"H100": base, "A100": base}

	newState := func() *RouterState {
		return &RouterState{
			SelectedInstance: "M0",
			Snapshots: []RoutingSnapshot{
				{ID: "M0", GPUType: "H100", BatchSize: 3, KvTokensInUse: 512},
				{ID: "M1", GPUType: "A100", BatchSize: 1, KvTokensInUse: 128},
			},
		}
	}

	dPlain := NewEDPPDecider(cfgPlain, newTestAffineModel(), cache, prefill)
	dDup := NewEDPPDecider(cfgDup, newTestAffineModel(), cache, prefill)
	// Identical virtual-queue pressure on both so a real (non-degenerate) argmin runs.
	for _, dd := range []*EDPPDecider{dPlain, dDup} {
		z := dd.ensureZ("batch")
		z.zTTFT = 1e6
		z.zITL = 1e6
	}

	got := dDup.Decide(reqBatch("r", 256), newState())
	want := dPlain.Decide(reqBatch("r", 256), newState())
	if got.Disaggregate != want.Disaggregate ||
		got.DecodePodOverride != want.DecodePodOverride ||
		got.PrefillPodHint != want.PrefillPodHint {
		t.Fatalf("duplicate-θ decision {%v %q %q} != plain {%v %q %q} (byte-identity broken)",
			got.Disaggregate, got.DecodePodOverride, got.PrefillPodHint,
			want.Disaggregate, want.DecodePodOverride, want.PrefillPodHint)
	}
}

func TestDecideJoint_PerInstanceTheta_PrefersFastGPU(t *testing.T) {
	// θ_i is consumed: two decode candidates with IDENTICAL live state differ only in
	// GPUType. The fast H100 sits at the HIGHER instance ID, so a tie-break would pick the
	// slow A100 — asserting the fast node wins proves θ_i (not ordering) drove the argmin.
	base := defaultTestEDPPConfig().Coeffs
	slow := base
	slow.AlphaD *= 4
	slow.AlphaP *= 4 // keep AlphaD≈AlphaP (coeffs validator rejects >10% divergence)
	slow.C0 *= 4
	slow.C1 *= 4
	slow.CPf *= 4

	cfg := defaultTestEDPPConfig()
	cfg.Joint = true
	cfg.CoeffsByGPU = map[string]EDPPCoeffs{"H100": base, "A100": slow}
	// No prefill pool: only local decode candidates are enumerated, so the decode-side
	// physics (θ_i) is the sole differentiator between the two candidates.
	d := NewEDPPDecider(cfg, newTestAffineModel(), coldCacheQuery("instance_1", "instance_2"), nil)
	d.ensureZ("batch").zTTFT = 1e7 // TTFT pressure so the θ-dependent T̂_local dominates J

	state := &RouterState{
		SelectedInstance: "instance_1",
		Snapshots: []RoutingSnapshot{
			{ID: "instance_1", GPUType: "A100", BatchSize: 1},
			{ID: "instance_2", GPUType: "H100", BatchSize: 1},
		},
	}
	dec := d.Decide(reqBatch("r", 256), state)
	if dec.DecodePodOverride != "instance_2" {
		t.Fatalf("joint picked %q, want instance_2 (fast H100); θ_i not driving decode selection", dec.DecodePodOverride)
	}
}

func TestJoint_ReducesToScorerSliceMatchesReduced(t *testing.T) {
	// §5.5 reduction: joint J restricted to the scorer's single d reproduces the reduced
	// local-vs-disagg decision. Build one reduced decider and one joint decider sharing an
	// identical single-decode / single-prefill state and identical virtual-queue state, and
	// assert the Disaggregate decision matches across a sweep of z_ttft pressures.
	cold := coldCacheQuery("d0", "p0")
	prefill := func() []RoutingSnapshot {
		return []RoutingSnapshot{{ID: "p0", ResidentPrefillTokens: 64, BatchSize: 1, MaxBatchSize: 8}}
	}
	newState := func() *RouterState {
		return &RouterState{
			SelectedInstance: "d0",
			Snapshots: []RoutingSnapshot{
				{ID: "d0", BatchSize: 2, MaxBatchSize: 8, KvTokensInUse: 1024, ResidentPrefillTokens: 0},
			},
		}
	}

	for _, zt := range []float64{0, 1e5, 1e6, 1e7, 1e8} {
		cfgR := defaultTestEDPPConfig()
		reduced := NewEDPPDecider(cfgR, newTestAffineModel(), cold, prefill)
		cfgJ := defaultTestEDPPConfig()
		cfgJ.Joint = true
		joint := NewEDPPDecider(cfgJ, newTestAffineModel(), cold, prefill)

		// Identical backlog on the single decode/prefill instance (per-instance == pool).
		for _, dd := range []*EDPPDecider{reduced, joint} {
			dd.OnRoute(reqBatch("bg", 4000), "bg", true, 4000, "d0", "p0")
			dd.ensureZ("batch").zTTFT = zt
		}

		r := reduced.Decide(reqBatch("q", 2000), newState())
		j := joint.Decide(reqBatch("q", 2000), newState())
		if r.Disaggregate != j.Disaggregate {
			t.Fatalf("zt=%g: reduced Disaggregate=%v but joint=%v (reduction §5.5 broken)", zt, r.Disaggregate, j.Disaggregate)
		}
		// When joint disaggregates it must name the single prefill node; the decode node
		// is always the sole candidate d0.
		if j.DecodePodOverride != "d0" {
			t.Fatalf("zt=%g: joint decode override = %q, want d0", zt, j.DecodePodOverride)
		}
		if j.Disaggregate && j.PrefillPodHint != "p0" {
			t.Fatalf("zt=%g: joint disagg prefill hint = %q, want p0", zt, j.PrefillPodHint)
		}
	}
}
