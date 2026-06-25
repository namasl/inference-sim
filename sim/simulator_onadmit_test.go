package sim

import "testing"

// TestSimulator_OnAdmit_FiresOnFirstAdmission verifies that the OnAdmit callback
// is fired exactly once when a request first enters the running batch.
func TestSimulator_OnAdmit_FiresOnFirstAdmission(t *testing.T) {
	sim := mustNewSimulator(t, newTestSimConfig())
	var admitted []string
	sim.OnAdmit = func(req *Request, tick int64) { admitted = append(admitted, req.ID) }

	req := &Request{
		ID:           "r1",
		ArrivalTime:  0,
		InputTokens:  make([]int, 3),
		OutputTokens: make([]int, 4),
		State:        StateQueued,
	}
	sim.InjectArrival(req)
	sim.Run()

	if len(admitted) == 0 {
		t.Fatalf("OnAdmit not fired for r1; got %v", admitted)
	}
	if admitted[0] != "r1" {
		t.Fatalf("OnAdmit fired with wrong ID; got %v, want r1", admitted)
	}

	count := 0
	for _, id := range admitted {
		if id == "r1" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("OnAdmit fired %d times for r1, want 1", count)
	}
}
