package cmd

import (
	"math"
	"os"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
)

// TestResolveEDPPCoeffs_NonEDPPDecider verifies that non-EDPP deciders return the
// zero value regardless of the coeffs path argument.
func TestResolveEDPPCoeffs_NonEDPPDecider(t *testing.T) {
	got := resolveEDPPCoeffs("never", "anything")
	if got != (sim.EDPPCoeffs{}) {
		t.Errorf("expected zero EDPPCoeffs for non-edpp decider, got %+v", got)
	}
}

// TestResolveEDPPCoeffs_FrozenLlama70b verifies that a valid frozen coefficients
// file is loaded and returns coefficients with the expected AlphaD value.
func TestResolveEDPPCoeffs_FrozenLlama70b(t *testing.T) {
	path := "../scripts/calibration/coeffs-llama70b-h100-tp4.json"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("frozen coeffs file absent (%s), skipping", path)
	}

	got := resolveEDPPCoeffs("edpp", path)
	const wantAlphaD = 16613.537554540144
	const eps = 1e-6
	if math.Abs(got.AlphaD-wantAlphaD) > eps {
		t.Errorf("AlphaD: want %v (±%v), got %v", wantAlphaD, eps, got.AlphaD)
	}
}

// TestResolveEDPPCoeffs_OtherDeciders checks that non-"edpp" decider values
// (e.g. "always", "prefix-threshold") also return the zero value.
func TestResolveEDPPCoeffs_OtherDeciders(t *testing.T) {
	for _, decider := range []string{"always", "prefix-threshold"} {
		got := resolveEDPPCoeffs(decider, "irrelevant-path")
		if got != (sim.EDPPCoeffs{}) {
			t.Errorf("decider %q: expected zero EDPPCoeffs, got %+v", decider, got)
		}
	}
}
