package sim

import (
	"math"
	"testing"
)

// edppAffineModel is a test LatencyModel whose StepTime is exactly α + Σδ(r),
// matching the structural form the EDPP finite-difference extraction assumes:
//
//	prefill probe (ProgressIndex < len(InputTokens)): δ = kp · NumNewTokens
//	decode probe  (otherwise, len(OutputTokens) > 0):  δ = c0 + c1 · ProgressIndex
//
// This lets the unit tests assert exact α/δ recovery and exact normalizer values.
type edppAffineModel struct {
	alpha int64 // per-step fixed cost
	kp    int64 // prefill marginal per chunk token
	c0    int64 // decode per-request overhead
	c1    int64 // decode marginal per context token
}

func (m *edppAffineModel) StepTime(batch []*Request) int64 {
	if len(batch) == 0 {
		return 1 // matches the LatencyModel contract: StepTime([]) >= 1, never α
	}
	s := m.alpha
	for _, r := range batch {
		if r.ProgressIndex < int64(len(r.InputTokens)) {
			s += m.kp * int64(r.NumNewTokens) // prefill
		} else {
			s += m.c0 + m.c1*r.ProgressIndex // decode
		}
	}
	return s
}

func (m *edppAffineModel) QueueingTime(*Request) int64      { return 0 }
func (m *edppAffineModel) OutputTokenProcessingTime() int64 { return 0 }
func (m *edppAffineModel) PostDecodeFixedOverhead() int64   { return 0 }

func newTestAffineModel() *edppAffineModel {
	return &edppAffineModel{alpha: 1000, kp: 10, c0: 100, c1: 1}
}

func defaultTestEDPPConfig() EDPPConfig {
	return EDPPConfig{
		TauTTFTUs:        100_000, // 100 ms
		TauITLUs:         50_000,  // 50 ms
		V:                1.0,
		CXferUs:          5_000, // 5 ms
		NomPrefillTokens: 512,
		NomDecodeCtx:     2048,
		BlockSize:        16,
	}
}

// --- Coefficient extraction (Task 1) ---

func TestEDPP_ExtractAlpha_CancelsDelta(t *testing.T) {
	m := newTestAffineModel()
	// Prefill probe of 300 tokens: StepTime([p]) = 1000 + 10·300, [p,p] = 1000 + 20·300.
	// α = 2·(1000+3000) − (1000+6000) = 8000 − 7000 = 1000.
	if got := edppExtractAlpha(m, edppPrefillProbe(300)); got != 1000 {
		t.Errorf("prefill α = %d, want 1000", got)
	}
	// Decode probe at ctx 2048: [d] = 1000+100+2048, [d,d] = 1000+2·(100+2048).
	// α = 2·3148 − 5296 = 6296 − 5296 = 1000.
	if got := edppExtractAlpha(m, edppDecodeProbe(2048)); got != 1000 {
		t.Errorf("decode α = %d, want 1000", got)
	}
}

func TestEDPP_MarginalDelta_RecoversPerCopyWork(t *testing.T) {
	m := newTestAffineModel()
	// prefill δ for 300 tokens = kp·300 = 3000.
	if got := edppMarginalDelta(m, edppPrefillProbe(300)); got != 3000 {
		t.Errorf("prefill δ = %d, want 3000", got)
	}
	// decode δ at ctx 2048 = c0 + c1·2048 = 100 + 2048 = 2148.
	if got := edppMarginalDelta(m, edppDecodeProbe(2048)); got != 2148 {
		t.Errorf("decode δ = %d, want 2148", got)
	}
}

func TestEDPP_AlphaZero_YieldsConservingRate(t *testing.T) {
	// §11 conservation anchor: with α → 0, μ^nom → 1 (work-conserving server).
	m := &edppAffineModel{alpha: 0, kp: 10, c0: 100, c1: 1}
	d := NewEDPPDecider(defaultTestEDPPConfig(), m, nil, nil)
	n := d.normFor("") // default class
	if math.Abs(n.muDNom-1.0) > 1e-9 {
		t.Errorf("α=0 ⇒ μ_d^nom = %v, want 1.0", n.muDNom)
	}
	if math.Abs(n.muPNom-1.0) > 1e-9 {
		t.Errorf("α=0 ⇒ μ_p^nom = %v, want 1.0", n.muPNom)
	}
}

// --- Constructor / normalizers (Task 2) ---

func TestEDPP_Constructor_PrecomputesNormalizers(t *testing.T) {
	m := newTestAffineModel()
	cfg := defaultTestEDPPConfig()
	d := NewEDPPDecider(cfg, m, nil, nil)
	n := d.normFor("") // default class

	// μ_d^nom = 1 − α_d/τ_itl = 1 − 1000/50000 = 0.98
	if math.Abs(n.muDNom-0.98) > 1e-9 {
		t.Errorf("μ_d^nom = %v, want 0.98", n.muDNom)
	}
	// T_iter_p^nom = StepTime([prefillProbe(512)]) = 1000 + 10·512 = 6120
	// μ_p^nom = 1 − 1000/6120
	wantMuP := 1 - 1000.0/6120.0
	if math.Abs(n.muPNom-wantMuP) > 1e-9 {
		t.Errorf("μ_p^nom = %v, want %v", n.muPNom, wantMuP)
	}
	// W*_d = μ_d^nom · τ_ttft = 0.98 · 100000 = 98000
	if math.Abs(n.wStarD-98000) > 1e-6 {
		t.Errorf("W*_d = %v, want 98000", n.wStarD)
	}
	// δ̄_d = decode δ at ctx 2048 = 2148 ; δ̄_p = prefill δ at 512 = 5120
	if d.deltaBarD != 2148 {
		t.Errorf("δ̄_d = %d, want 2148", d.deltaBarD)
	}
	if d.deltaBarP != 5120 {
		t.Errorf("δ̄_p = %d, want 5120", d.deltaBarP)
	}
}

func TestEDPP_Constructor_RejectsInvalidConfig(t *testing.T) {
	m := newTestAffineModel()
	bad := []EDPPConfig{
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.TauTTFTUs = 0; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.TauITLUs = 0; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.V = -1; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.CXferUs = -1; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.NomPrefillTokens = 0; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.NomDecodeCtx = 0; return c }(),
		func() EDPPConfig { c := defaultTestEDPPConfig(); c.BlockSize = 0; return c }(),
	}
	for i, c := range bad {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("config[%d] expected panic, got none", i)
				}
			}()
			_ = NewEDPPDecider(c, m, nil, nil)
		}()
	}
}

// --- Decide rule + §11 behavioral anchors (Tasks 3 & 5) ---

func decodeState(selected string, queue, batch int, itlUs float64) *RouterState {
	return &RouterState{
		SelectedInstance: selected,
		Snapshots: []RoutingSnapshot{
			{ID: selected, QueueDepth: queue, BatchSize: batch, ITL: itlUs},
		},
	}
}

func TestEDPP_SignalDirection_QdNonDecreasingInBatch(t *testing.T) {
	// §11 signal-direction anchor: holding backlog (QueueDepth) fixed while raising
	// running batch B, normalized q_d must NOT decrease (guards the live-μ-in-normalizer bug).
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)

	lowB := []RoutingSnapshot{{ID: "d0", QueueDepth: 5, BatchSize: 2}}
	highB := []RoutingSnapshot{{ID: "d0", QueueDepth: 5, BatchSize: 40}}

	n := d.normFor("")
	qdLow, _ := d.normalizedBacklogs(lowB, nil, n)
	qdHigh, _ := d.normalizedBacklogs(highB, nil, n)
	if qdHigh < qdLow {
		t.Errorf("q_d decreased as batch grew: low=%v high=%v", qdLow, qdHigh)
	}
}

func TestEDPP_NOutIndependence(t *testing.T) {
	// §11 N_out-independence anchor: the decision must not read Request.OutputTokens (INV-9).
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	state := decodeState("d0", 10, 8, 60_000)

	reqA := &Request{ID: "a", InputTokens: make([]int, 800)}
	reqB := &Request{ID: "b", InputTokens: make([]int, 800), OutputTokens: make([]int, 4096)}

	if d.Decide(reqA, state).Disaggregate != d.Decide(reqB, state).Disaggregate {
		t.Errorf("decision depended on OutputTokens (INV-9 violation)")
	}
}

func TestEDPP_DisaggregationPayoffSign(t *testing.T) {
	// §11 disaggregation-payoff-sign anchor: under an ITL-SLO breach (z_itl large) with
	// idle prefill capacity (no prefill backlog), the rule must shift toward P.
	cfg := defaultTestEDPPConfig()
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

	req := &Request{ID: "r", InputTokens: make([]int, 800)}
	state := decodeState("d0", 10, 8, 60_000) // decode ITL above τ_itl

	// Drive z_itl large via realized ITL misses (E8), prefill stays idle.
	for i := 0; i < 50; i++ {
		d.OnComplete("", 0, 200_000) // realized ITL = 200ms >> τ_itl, default class
	}
	if !d.Decide(req, state).Disaggregate {
		t.Errorf("expected P (disaggregate) under z_itl breach with idle prefill")
	}
}

func TestEDPP_OnComplete_UpdatesVirtualQueues(t *testing.T) {
	// E8: Z_ttft and Z_itl accumulate positive violations and floor at 0.
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	d.OnComplete("", 150_000, 80_000) // TTFT 150ms (τ=100ms), ITL 80ms (τ=50ms)
	z := d.zByClass[""]
	if z == nil || z.zTTFT != 50_000 {
		t.Errorf("Z_ttft = %v, want 50000", z)
	}
	if z.zITL != 30_000 {
		t.Errorf("Z_itl = %v, want 30000", z.zITL)
	}
	// A large under-target completion must floor each queue at 0, not go negative.
	d.OnComplete("", 0, 0)
	if z.zTTFT != 0 || z.zITL != 0 {
		t.Errorf("virtual queues not floored at 0: zTTFT=%v zITL=%v", z.zTTFT, z.zITL)
	}
}

func TestEDPP_PerClass_TargetsResolve(t *testing.T) {
	// A class with an explicit override uses it; an unlisted class falls back to defaults.
	cfg := defaultTestEDPPConfig()
	cfg.TauTTFTByClassUs = map[string]int64{"critical": 30_000}
	cfg.TauITLByClassUs = map[string]int64{"critical": 10_000}
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)

	tT, tI := d.targetsFor("critical")
	if tT != 30_000 || tI != 10_000 {
		t.Errorf("critical targets = (%d,%d), want (30000,10000)", tT, tI)
	}
	tT, tI = d.targetsFor("batch") // unlisted → defaults
	if tT != cfg.TauTTFTUs || tI != cfg.TauITLUs {
		t.Errorf("batch targets = (%d,%d), want defaults (%d,%d)", tT, tI, cfg.TauTTFTUs, cfg.TauITLUs)
	}
}

func TestEDPP_PerClass_IndependentVirtualQueues(t *testing.T) {
	// An ITL breach reported for one class must not bleed into another class's
	// virtual queue: the SLO-pressure feedback is per-class.
	cfg := defaultTestEDPPConfig()
	cfg.TauTTFTByClassUs = map[string]int64{"critical": 30_000}
	cfg.TauITLByClassUs = map[string]int64{"critical": 10_000}
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot { return nil })

	for i := 0; i < 50; i++ {
		d.OnComplete("critical", 0, 200_000) // breach only the critical class
	}
	if d.zByClass["critical"] == nil || d.zByClass["critical"].zITL == 0 {
		t.Fatalf("critical z_itl did not accumulate")
	}
	if z := d.zByClass["batch"]; z != nil && z.zITL != 0 {
		t.Errorf("batch z_itl leaked from critical breach: %v", z.zITL)
	}

	state := decodeState("d0", 10, 8, 60_000)
	// Critical (breached) shifts toward P; batch (no breach, loose target) stays D.
	if !d.Decide(&Request{ID: "c", InputTokens: make([]int, 800), SLOClass: "critical"}, state).Disaggregate {
		t.Errorf("critical request should disaggregate under its own ITL breach")
	}
	if d.Decide(&Request{ID: "b", InputTokens: make([]int, 800), SLOClass: "batch"}, state).Disaggregate {
		t.Errorf("batch request should not disaggregate (no breach on its class)")
	}
}

func TestEDPP_DeltaPfChunk_UsesChunkBudgetNotFullPrompt(t *testing.T) {
	// δ_pf-chunk is the marginal work of ONE prefill chunk = min(a_p, ChunkTokens)
	// tokens, NOT the whole prefill demand W_p. With the affine model δ_pf(s) = kp·s
	// (kp = 10), the inflation must track the chunk budget, not the prompt length.
	const ap = 8000

	// Capped well below a_p ⇒ inflation reflects the chunk (256·10), not 8000·10.
	capped := NewEDPPDecider(func() EDPPConfig { c := defaultTestEDPPConfig(); c.ChunkTokens = 256; return c }(),
		newTestAffineModel(), nil, nil)
	if got := capped.chunkInflation(ap); got != 2560 {
		t.Errorf("capped δ_pf-chunk = %v, want 2560 (kp·256)", got)
	}

	// No cap (0) ⇒ the whole prefill counts as one chunk: 8000·10.
	uncapped := NewEDPPDecider(func() EDPPConfig { c := defaultTestEDPPConfig(); c.ChunkTokens = 0; return c }(),
		newTestAffineModel(), nil, nil)
	if got := uncapped.chunkInflation(ap); got != 80000 {
		t.Errorf("uncapped δ_pf-chunk = %v, want 80000 (kp·8000)", got)
	}

	// When a_p <= ChunkTokens the whole (small) prefill is one chunk: 100·10.
	if got := capped.chunkInflation(100); got != 1000 {
		t.Errorf("small-prompt δ_pf-chunk = %v, want 1000 (kp·100)", got)
	}
}

func TestEDPP_EmptyPrompt_DoesNotDisaggregate(t *testing.T) {
	d := NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, nil)
	if d.Decide(&Request{ID: "r"}, decodeState("d0", 1, 1, 60_000)).Disaggregate {
		t.Errorf("empty prompt must not disaggregate")
	}
}

// Compile-time interface compliance.
var (
	_ DisaggregationDecider = (*EDPPDecider)(nil)
	_ SLOFeedbackDecider    = (*EDPPDecider)(nil)
)
