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
	// N_ahead=1 slot needed; X̂_dep = B/(R̄·T_iter) = 4/(20·1000)=2e-4 dep/µs → T_adm=1/2e-4=5000µs.
	full := AdmissionContext{BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0, ReqKVNeed: 10, TIter: 1000, RemainingStepsEst: 20}
	got := e.EstimateTAdm(full)
	if got < 4999 || got > 5001 {
		t.Fatalf("full-batch fluid = %v, want ~5000", got)
	}
	// Contrast: waiting on the same full/zero-waiting state gives 0 (the bug).
	w, _ := NewAdmissionEstimator("waiting")
	if w.EstimateTAdm(full) != 0 {
		t.Fatal("waiting must give 0 here (documents the bug fluid fixes)")
	}
}

func TestNewAdmissionEstimator_UnknownIsError(t *testing.T) {
	if _, err := NewAdmissionEstimator("nope"); err == nil {
		t.Fatal("expected error for unknown estimator")
	}
}
