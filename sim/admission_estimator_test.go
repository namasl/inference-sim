package sim

import "testing"

func TestWaitingEstimator_ReproducesFormula(t *testing.T) {
	e, err := NewAdmissionEstimator("waiting")
	if err != nil {
		t.Fatal(err)
	}
	// waiting: QWork/Mu (the current admission term).
	got := e.EstimateTAdm(AdmissionContext{QWork: 5000, Mu: 0.5})
	if got != 10000 {
		t.Fatalf("waiting = %v, want 10000", got)
	}
}

func TestLittleEstimator(t *testing.T) {
	e, _ := NewAdmissionEstimator("little")
	// T_adm ≈ L̄_q / λ_adm : QueueDepth=8 waiting, AdmissionRate=0.002 req/µs → 4000µs.
	got := e.EstimateTAdm(AdmissionContext{QueueDepth: 8, AdmissionRate: 0.002})
	if got != 4000 {
		t.Fatalf("little = %v, want 4000", got)
	}
	// Zero admission rate → 0 (avoid div by zero; no signal).
	if e.EstimateTAdm(AdmissionContext{QueueDepth: 8, AdmissionRate: 0}) != 0 {
		t.Fatal("little with zero rate must be 0")
	}
}

func TestFluidEstimator(t *testing.T) {
	e, _ := NewAdmissionEstimator("fluid")
	// Slot + KV already free → ~0.
	free := AdmissionContext{BatchSize: 2, MaxBatchSize: 4, FreeKVBlocks: 100, ReqKVNeed: 10, TIter: 1000, RemainingStepsEst: 20}
	if got := e.EstimateTAdm(free); got != 0 {
		t.Fatalf("free slot must give 0, got %v", got)
	}
	// Full batch, zero waiting work: waiting would give 0; fluid must give a large T_adm.
	// Wave mean-field: QueueDepth=0 → ⌈(0+1)/4⌉=1 wave → 1·20·1000 = 20000µs.
	full := AdmissionContext{BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0, ReqKVNeed: 10, TIter: 1000, RemainingStepsEst: 20}
	got := e.EstimateTAdm(full)
	if got < 19999 || got > 20001 {
		t.Fatalf("full-batch fluid = %v, want ~20000", got)
	}
	// Contrast: waiting on the same full/zero-waiting state gives 0 (the bug).
	w, _ := NewAdmissionEstimator("waiting")
	if w.EstimateTAdm(full) != 0 {
		t.Fatal("waiting must give 0 here (documents the bug fluid fixes)")
	}
}

func TestFluidEstimator_WaveMeanField(t *testing.T) {
	e, _ := NewAdmissionEstimator("fluid")
	// Free slot + KV → 0.
	if got := e.EstimateTAdm(AdmissionContext{BatchSize: 2, MaxBatchSize: 4, FreeKVBlocks: 100, ReqKVNeed: 10, QueueDepth: 0, RemainingStepsEst: 20, TIter: 1000}); got != 0 {
		t.Fatalf("free slot → 0, got %v", got)
	}
	// Full batch, short queue (QueueDepth+1 <= BatchSize → 1 wave): ⌈(0+1)/4⌉·20·1000 = 20000.
	full1 := AdmissionContext{BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0, ReqKVNeed: 10, QueueDepth: 0, RemainingStepsEst: 20, TIter: 1000}
	if got := e.EstimateTAdm(full1); got < 19999 || got > 20001 {
		t.Fatalf("short-queue 1 wave → ~20000, got %v", got)
	}
	// Deep queue: QueueDepth=9, BatchSize=4 → ⌈10/4⌉=3 waves → 3·20·1000 = 60000.
	deep := AdmissionContext{BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0, ReqKVNeed: 10, QueueDepth: 9, RemainingStepsEst: 20, TIter: 1000}
	if got := e.EstimateTAdm(deep); got < 59999 || got > 60001 {
		t.Fatalf("deep-queue 3 waves → ~60000, got %v", got)
	}
}

func TestRollforwardEstimator(t *testing.T) {
	e, _ := NewAdmissionEstimator("rollforward")
	// Batch full (2/2), no free KV. Two running reqs with remaining steps 3 and 5,
	// holding 10 and 10 KV blocks. Request needs 8 blocks.
	// Roll forward at T_iter=1000µs: after 3 iters (3000µs) req A departs → frees 10 blocks
	// AND a slot. 10 ≥ 8 and slot free → admit. T_adm = 3·1000 = 3000µs.
	ctx := AdmissionContext{
		BatchSize: 2, MaxBatchSize: 2, FreeKVBlocks: 0, ReqKVNeed: 8, TIter: 1000,
		Running:           []RunningReqState{{TrueRemaining: 3, KVBlocks: 10}, {TrueRemaining: 5, KVBlocks: 10}},
		RemainingStepsEst: 4,
	}
	got := e.EstimateTAdm(ctx)
	if got < 2999 || got > 3001 {
		t.Fatalf("rollforward = %v, want ~3000", got)
	}
}

func TestRollforwardEstimator_UsesEstimateWhenNoOracle(t *testing.T) {
	e, _ := NewAdmissionEstimator("rollforward")
	// TrueRemaining=-1 (no oracle) → use RemainingStepsEst for each running req.
	ctx := AdmissionContext{
		BatchSize: 1, MaxBatchSize: 1, FreeKVBlocks: 0, ReqKVNeed: 5, TIter: 1000,
		Running: []RunningReqState{{TrueRemaining: -1, KVBlocks: 10}}, RemainingStepsEst: 4,
	}
	// Single req departs after est 4 steps → frees slot+KV → T_adm = 4000µs.
	got := e.EstimateTAdm(ctx)
	if got < 3999 || got > 4001 {
		t.Fatalf("rollforward(est) = %v, want ~4000", got)
	}
}

func TestNewAdmissionEstimator_UnknownIsError(t *testing.T) {
	if _, err := NewAdmissionEstimator("nope"); err == nil {
		t.Fatal("expected error for unknown estimator")
	}
}

func TestOracleVariants(t *testing.T) {
	// Oracle variants exist and use TrueRemaining even when an estimate is also present.
	e, err := NewAdmissionEstimator("rollforward_oracle")
	if err != nil {
		t.Fatal(err)
	}
	ctx := AdmissionContext{
		BatchSize: 1, MaxBatchSize: 1, FreeKVBlocks: 0, ReqKVNeed: 5, TIter: 1000,
		Running: []RunningReqState{{TrueRemaining: 2, KVBlocks: 10}}, RemainingStepsEst: 99,
	}
	// Oracle uses TrueRemaining=2 (not est 99) → 2000µs.
	if got := e.EstimateTAdm(ctx); got < 1999 || got > 2001 {
		t.Fatalf("rollforward_oracle = %v, want ~2000 (uses TrueRemaining)", got)
	}
}

func TestDeployableGuard(t *testing.T) {
	if IsDeployableEstimator("rollforward") != true {
		t.Fatal("rollforward is deployable")
	}
	if IsDeployableEstimator("rollforward_oracle") != false {
		t.Fatal("oracle is NOT deployable")
	}
}
