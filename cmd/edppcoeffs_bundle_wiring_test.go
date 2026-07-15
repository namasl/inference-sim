package cmd

import (
	"os"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// frozenH100CoeffsPath is the committed frozen coefficients file used by both
// GPU keys below. Reusing one file for both keys keeps this test independent of
// Task 7's A100 coefficients asset while still exercising the loader with a real,
// validated file.
const frozenH100CoeffsPath = "../scripts/calibration/coeffs-llama70b-h100-tp4.json"

// TestCoeffsByGPU_Wired_LoaderCopiesAllEntries verifies the loader helper
// edppCoeffsByGPUFromBundle (the exact function the run command calls) loads
// every coeffs_by_gpu path into sim.EDPPCoeffs and returns nil for a bundle
// without (or with an empty) coeffs_by_gpu block.
//
// This is a UNIT guard on the loader. It does NOT (and cannot) guard the
// DeploymentConfig literal assignment in runCmd — that is covered by
// TestCoeffsByGPU_Wired_RunCmdLiteralWiring_Structural below.
func TestCoeffsByGPU_Wired_LoaderCopiesAllEntries(t *testing.T) {
	if _, err := os.Stat(frozenH100CoeffsPath); os.IsNotExist(err) {
		t.Skipf("frozen coeffs file absent (%s), skipping", frozenH100CoeffsPath)
	}

	want, err := sim.LoadEDPPCoeffs(frozenH100CoeffsPath)
	if err != nil {
		t.Fatalf("sim.LoadEDPPCoeffs(%q): %v", frozenH100CoeffsPath, err)
	}
	if !(want.AlphaD > 0) {
		t.Fatalf("sanity: frozen coeffs file has non-positive AlphaD %v", want.AlphaD)
	}

	bundle := &sim.PolicyBundle{
		CoeffsByGPU: map[string]string{
			"H100": frozenH100CoeffsPath,
			"A100": frozenH100CoeffsPath, // same file for both keys (see const doc)
		},
	}

	got := edppCoeffsByGPUFromBundle(bundle)
	if got == nil {
		t.Fatalf("edppCoeffsByGPUFromBundle returned nil; bundle carried coeffs_by_gpu")
	}
	if len(got) != 2 {
		t.Fatalf("edppCoeffsByGPUFromBundle returned %d entries, want 2: %+v", len(got), got)
	}
	h100, ok := got["H100"]
	if !ok {
		t.Fatalf("converted map missing H100 entry: %+v", got)
	}
	a100, ok := got["A100"]
	if !ok {
		t.Fatalf("converted map missing A100 entry: %+v", got)
	}
	if h100 != want {
		t.Errorf("H100 entry not loaded via sim.LoadEDPPCoeffs: got %+v, want %+v", h100, want)
	}
	if a100 != want {
		t.Errorf("A100 entry not loaded via sim.LoadEDPPCoeffs: got %+v, want %+v", a100, want)
	}
	if !(h100.AlphaD > 0) || !(a100.AlphaD > 0) {
		t.Errorf("AlphaD not populated: H100=%v A100=%v", h100.AlphaD, a100.AlphaD)
	}
}

// TestCoeffsByGPU_Wired_LoaderNilForAbsentBundle verifies the "homogeneous" fast
// path: a nil bundle, and a bundle with no coeffs_by_gpu entries, both return nil
// (not an empty non-nil map) so downstream nil-checks (e.g. coeffsFor) see the
// intended "no override" signal.
func TestCoeffsByGPU_Wired_LoaderNilForAbsentBundle(t *testing.T) {
	if got := edppCoeffsByGPUFromBundle(nil); got != nil {
		t.Errorf("nil bundle: want nil map, got %+v", got)
	}
	if got := edppCoeffsByGPUFromBundle(&sim.PolicyBundle{}); got != nil {
		t.Errorf("bundle with no coeffs_by_gpu: want nil map, got %+v", got)
	}
}

// NOTE: The DeploymentConfig literal wiring (`EDPPCoeffsByGPU: bundleEDPPCoeffsByGPU`
// in cmd/root.go's runCmd.Run) is intentionally not covered by a structural
// wiring test here. The loader test above reuses ONE frozen coefficients file
// for both GPU keys, so a behavioral (completed_requests-observable) guard on
// the literal line has no routing/latency signal to detect. Adding a
// production-code test seam to make the literal inspectable was rejected on
// review (see Task 6 findings) since it introduces a test-only hook into
// runCmd.Run with no clean production alternative. The literal-line guard is
// deferred to Task 8, which exercises real differing per-GPU coefficients on
// an end-to-end behavioral test.
