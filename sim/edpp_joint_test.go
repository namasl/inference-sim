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
