package sim

import (
	"math"
	"testing"
)

// goodSelf is the reward half of the goodput-objective diagnostic: the arriving request R's own
// smoothed goodput under a candidate placement. These tests pin the laws it must satisfy, not its
// exact numbers — a rewrite that preserves the behavior must still pass.

// A request comfortably inside every target scores full good (1 under flip, ~1 under util); a
// request that blows its TTFT budget scores 0 under flip and near-0 under util.
func TestGoodSelf_FlipComposite(t *testing.T) {
	slo := varSLO{tauTTFTUs: 100_000, tauITLUs: 50_000, tauE2EUs: 5_000_000}
	// tHat well under 100ms, ITL (tIterAfter) under 50ms, e2e = 20ms + 100·40ms = 4.02s < 5s.
	good := goodSelf(slo, 20_000, 40_000, 100, varKernelFlip)
	if good != 1 {
		t.Fatalf("in-budget request: good = %v, want 1", good)
	}
	// Push TTFT past its 100ms target: the composite flips to 0.
	miss := goodSelf(slo, 120_000, 40_000, 100, varKernelFlip)
	if miss != 0 {
		t.Fatalf("TTFT-missing request: good = %v, want 0", miss)
	}
	// Push ITL past its 50ms target with TTFT still fine: also 0.
	itlMiss := goodSelf(slo, 20_000, 60_000, 100, varKernelFlip)
	if itlMiss != 0 {
		t.Fatalf("ITL-missing request: good = %v, want 0", itlMiss)
	}
	// A huge output length blows the E2E deadline even with fine TTFT/ITL: 0.
	e2eMiss := goodSelf(slo, 20_000, 40_000, 1_000_000, varKernelFlip)
	if e2eMiss != 0 {
		t.Fatalf("E2E-missing request: good = %v, want 0", e2eMiss)
	}
}

// A zero threshold disables that dimension's conjunct (matches the VaR kernels and
// cluster.SLOAttainmentMultiDim). With every target zero, good is trivially 1 under flip.
func TestGoodSelf_ZeroThresholdDisablesConjunct(t *testing.T) {
	slo := varSLO{tauTTFTUs: 0, tauITLUs: 0, tauE2EUs: 0}
	if good := goodSelf(slo, 999_999, 999_999, 999, varKernelFlip); good != 1 {
		t.Fatalf("all-thresholds-zero: good = %v, want 1 (no dimension priced)", good)
	}
	// Only E2E priced: a fast first token but an enormous output should still miss E2E.
	e2eOnly := varSLO{tauE2EUs: 1_000_000}
	if good := goodSelf(e2eOnly, 1_000, 40_000, 10_000, varKernelFlip); good != 0 {
		t.Fatalf("E2E-only, huge output: good = %v, want 0", good)
	}
}

// Under the util kernel good is a slack-utility product in (0,1) and is non-increasing as the
// projected TTFT grows (a later first token is never better for the request's own goodput).
func TestGoodSelf_UtilMonotoneInTTFT(t *testing.T) {
	slo := varSLO{tauTTFTUs: 100_000, tauITLUs: 50_000, tauE2EUs: 5_000_000}
	prev := 2.0 // above the (0,1] range, so the first sample always passes the check
	for _, tHat := range []float64{0, 25_000, 50_000, 100_000, 200_000, 400_000} {
		g := goodSelf(slo, tHat, 40_000, 100, varKernelUtil)
		if g < 0 || g > 1 {
			t.Fatalf("util good out of [0,1]: %v at tHat=%v", g, tHat)
		}
		if g > prev+1e-12 {
			t.Fatalf("util good rose with larger tHat (%v): %v > %v", tHat, g, prev)
		}
		prev = g
	}
}

// Under the util kernel good is non-increasing as the output length grows (a longer decode only
// pushes the end-to-end completion out).
func TestGoodSelf_UtilMonotoneInOutputLen(t *testing.T) {
	slo := varSLO{tauTTFTUs: 100_000, tauE2EUs: 5_000_000}
	prev := 2.0
	for _, nOut := range []float64{1, 10, 50, 100, 200, 500} {
		g := goodSelf(slo, 20_000, 40_000, nOut, varKernelUtil)
		if g > prev+1e-12 {
			t.Fatalf("util good rose with larger nOut (%v): %v > %v", nOut, g, prev)
		}
		prev = g
	}
}

// Pure function (INV-6): identical inputs yield byte-identical output.
func TestGoodSelf_Deterministic(t *testing.T) {
	slo := varSLO{tauTTFTUs: 100_000, tauITLUs: 50_000, tauE2EUs: 5_000_000}
	for _, k := range []varKernel{varKernelFlip, varKernelUtil, varKernelHazard} {
		a := goodSelf(slo, 37_000, 41_000, 137, k)
		b := goodSelf(slo, 37_000, 41_000, 137, k)
		if a != b {
			t.Fatalf("goodSelf non-deterministic for kernel %v: %v != %v", k, a, b)
		}
	}
}

// --- Wiring: the goodput objective on the normalized dpVaR path (jointVaRComponents) ---

// goodputVarDecider builds a decider on the headline auto-normalized drift-plus-VaR path,
// with the goodput reframing on or off. The prefill pool offers one candidate so disagg is
// enumerable; caches are cold.
func goodputVarDecider(t *testing.T, goodput bool, tauTTFTUs int64) *EDPPDecider {
	t.Helper()
	cfg := defaultTestEDPPConfig()
	cfg.Joint = true
	cfg.Rule = "var"
	cfg.VarMetric = "flip"
	cfg.VarKeepCongestion = true
	cfg.VarNormalize = true
	cfg.VarGoodputObjective = goodput
	if tauTTFTUs != 0 {
		cfg.TauTTFTUs = tauTTFTUs
	}
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "P0"}} }
	return NewEDPPDecider(cfg, newTestAffineModel(), coldCacheQuery("M0", "P0"), prefill)
}

// jointEvalCtxFor builds the per-decision context the joint cost functions consume, mirroring
// decideJoint's construction (zTTFT/zITL left at 0 — no virtual-queue pressure).
func jointEvalCtxFor(d *EDPPDecider, req *Request, nHatOut float64) *jointEvalCtx {
	return &jointEvalCtx{
		req: req, n: d.normFor(req.SLOClass),
		reqKVNeed: d.reqKVNeed(req), nHatOut: nHatOut, nowUs: 0,
	}
}

// The normalization spread floor ε₀ (varNormFloor) is one arriving request's work on the nominal
// decode instance in reference units, scaled by VarNormalizeFloorScale. Three laws pin the wiring
// independent of the exact work numbers: it is strictly positive so it guards the min-max division,
// it is linear in the scale so the sensitivity sweep moves it predictably, and an absent scale (0)
// resolves to the unit default. It stays positive even with no decode snapshots.
func TestVarNormFloor_PositiveLinearAndDefaulted(t *testing.T) {
	mk := func(scale float64) *EDPPDecider {
		cfg := defaultTestEDPPConfig()
		cfg.Joint = true
		cfg.Rule = "var"
		cfg.VarMetric = "flip"
		cfg.VarKeepCongestion = true
		cfg.VarNormalize = true
		cfg.VarNormalizeFloorScale = scale
		prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "P0"}} }
		return NewEDPPDecider(cfg, newTestAffineModel(), coldCacheQuery("M0", "P0"), prefill)
	}
	req := reqBatch("r1", 200)
	snaps := []RoutingSnapshot{{ID: "M0"}}
	floorFor := func(scale float64) float64 {
		d := mk(scale)
		return d.varNormFloor(jointEvalCtxFor(d, req, 128), snaps)
	}
	f1 := floorFor(1)
	if !(f1 > 0) {
		t.Fatalf("floor must be positive (it guards the division), got %v", f1)
	}
	if f2 := floorFor(2); math.Abs(f2-2*f1) > 1e-9*math.Max(1, f1) {
		t.Fatalf("floor must be linear in scale: f(1)=%v f(2)=%v", f1, f2)
	}
	if f0 := floorFor(0); f0 != f1 {
		t.Fatalf("absent scale (0) must resolve to the unit default: f(0)=%v f(1)=%v", f0, f1)
	}
	d := mk(1)
	if g := d.varNormFloor(jointEvalCtxFor(d, req, 128), nil); !(g > 0) {
		t.Fatalf("floor with no decode snapshots must stay positive, got %v", g)
	}
}

// The goodput flag changes ONLY the value term and (on disagg) the self term; it never touches
// the congestion drift. This is the core wiring contract of the reframing.
func TestGoodputWiring_CongestionUnchanged(t *testing.T) {
	off, on := goodputVarDecider(t, false, 0), goodputVarDecider(t, true, 0)
	req := reqBatch("r1", 200)
	ds := RoutingSnapshot{ID: "M0"}
	ps := RoutingSnapshot{ID: "P0"}
	for _, tc := range []struct {
		name string
		ps   *RoutingSnapshot
	}{{"local", nil}, {"disagg", &ps}} {
		congOff, _, _ := off.jointVaRComponents(jointEvalCtxFor(off, req, 128), ds, tc.ps)
		congOn, _, _ := on.jointVaRComponents(jointEvalCtxFor(on, req, 128), ds, tc.ps)
		if congOff != congOn {
			t.Fatalf("%s: goodput flag changed congestion drift: off=%v on=%v", tc.name, congOff, congOn)
		}
	}
}

// On a kept-local candidate the goodput flag subtracts the request's own good (in [0,1]) from the
// value term and leaves the self term (z-TTFT/ITL) untouched — there is no transfer penalty locally.
func TestGoodputWiring_LocalSubtractsGood(t *testing.T) {
	off, on := goodputVarDecider(t, false, 0), goodputVarDecider(t, true, 0)
	req := reqBatch("r1", 200)
	ds := RoutingSnapshot{ID: "M0"}

	_, vvOff, selfOff := off.jointVaRComponents(jointEvalCtxFor(off, req, 128), ds, nil)
	_, vvOn, selfOn := on.jointVaRComponents(jointEvalCtxFor(on, req, 128), ds, nil)

	if selfOff != selfOn {
		t.Fatalf("local: goodput flag must not touch the self term: off=%v on=%v", selfOff, selfOn)
	}
	good := vvOff - vvOn
	if good <= 0 || good > 1+1e-9 {
		t.Fatalf("local: value term must drop by a good_r in (0,1]; got Δ=%v (off=%v on=%v)", good, vvOff, vvOn)
	}
}

// On a disagg candidate the flag (a) subtracts the request's own good from the value term and
// (b) drops the standalone transfer penalty from the self term (it flows through the request's own
// projected TTFT instead). The self-term drop must equal exactly the transfer penalty.
func TestGoodputWiring_DisaggDropsTransferPenalty(t *testing.T) {
	off, on := goodputVarDecider(t, false, 0), goodputVarDecider(t, true, 0)
	req := reqBatch("r1", 200)
	ds := RoutingSnapshot{ID: "M0"}
	ps := RoutingSnapshot{ID: "P0"}

	_, vvOff, selfOff := off.jointVaRComponents(jointEvalCtxFor(off, req, 128), ds, &ps)
	_, vvOn, selfOn := on.jointVaRComponents(jointEvalCtxFor(on, req, 128), ds, &ps)

	wantXfer := off.transferPenalty(off.normFor(req.SLOClass), off.cXferUsFor(req))
	if wantXfer <= 0 {
		t.Fatalf("test precondition: transfer penalty must be positive, got %v", wantXfer)
	}
	if gotDrop := selfOff - selfOn; !floatNear(gotDrop, wantXfer, 1e-6) {
		t.Fatalf("disagg: self-term drop = %v, want the transfer penalty %v", gotDrop, wantXfer)
	}
	good := vvOff - vvOn
	if good <= 0 || good > 1+1e-9 {
		t.Fatalf("disagg: value term must drop by a good_r in (0,1]; got Δ=%v", good)
	}
}

// When the request cannot make its SLO (flip good = 0), the goodput flag has NO effect on the value
// term — the reward half is zero, so VaR − good_r == VaR. This ties the value-term delta to good_r
// semantics rather than to an unconditional constant.
func TestGoodputWiring_NoGoodNoValueChange(t *testing.T) {
	// τ_ttft = 1µs forces the projected TTFT past its target ⇒ flip good = 0.
	off, on := goodputVarDecider(t, false, 1), goodputVarDecider(t, true, 1)
	req := reqBatch("r1", 200)
	ds := RoutingSnapshot{ID: "M0"}

	_, vvOff, _ := off.jointVaRComponents(jointEvalCtxFor(off, req, 128), ds, nil)
	_, vvOn, _ := on.jointVaRComponents(jointEvalCtxFor(on, req, 128), ds, nil)
	if !floatNear(vvOff, vvOn, 1e-9) {
		t.Fatalf("SLO-missing request: value term changed despite good_r=0: off=%v on=%v", vvOff, vvOn)
	}
}

func floatNear(a, b, tol float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}
