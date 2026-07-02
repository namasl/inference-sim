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

func TestNewAdmissionEstimator_UnknownIsError(t *testing.T) {
	if _, err := NewAdmissionEstimator("nope"); err == nil {
		t.Fatal("expected error for unknown estimator")
	}
}
