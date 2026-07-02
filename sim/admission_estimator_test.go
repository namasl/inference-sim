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

func TestNewAdmissionEstimator_UnknownIsError(t *testing.T) {
	if _, err := NewAdmissionEstimator("nope"); err == nil {
		t.Fatal("expected error for unknown estimator")
	}
}
