package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

// edppCoeffsJSONMarker prefixes the one stdout line the child process prints:
// the JSON encoding of config.EDPPCoeffsByGPU, captured via the test-only
// edppCoeffsByGPUWiringHook seam (cmd/root.go) immediately after the
// DeploymentConfig literal is built.
const edppCoeffsJSONMarker = "EDPP_COEFFS_BY_GPU_JSON:"

// TestCoeffsByGPU_Wired_RunCmdLiteralWiring_Structural is the regression guard on
// the DeploymentConfig literal in runCmd (cmd/root.go):
//
//	EDPPCoeffsByGPU: bundleEDPPCoeffsByGPU,
//
// Unlike the analogous hw_config_by_gpu wiring guard (cmd/hwconfig_bundle_wiring_test.go),
// this cannot be a behavioral (completed_requests) test: the loader test above
// intentionally reuses ONE frozen coefficients file for both GPU keys (so it does
// not depend on Task 7's A100 asset), which means the two keys carry IDENTICAL
// EDPPCoeffs — there is no routing/latency difference to observe downstream
// whether or not the literal line is present. So this test instead inspects the
// actual DeploymentConfig struct built inside runCmd.Run, via a test-only seam
// (edppCoeffsByGPUWiringHook, cmd/root.go) that fires with config.EDPPCoeffsByGPU
// immediately after the literal is constructed. This exercises the REAL
// construction code path (bundle -> edppCoeffsByGPUFromBundle ->
// bundleEDPPCoeffsByGPU -> the literal field), not a reimplementation of it.
//
// Like TestHWConfigByGPU_RunCmdLiteralWiring_PerPoolSpeedObservable, the actual
// run happens in a child process (the hook sets logrus's level and other
// process-global state that must not leak into sibling tests in this package).
// The hook marshals the captured map to JSON, prints it prefixed with
// edppCoeffsJSONMarker, and os.Exit(0)s before the simulation runs.
//
// If the `EDPPCoeffsByGPU: bundleEDPPCoeffsByGPU,` literal line is removed,
// config.EDPPCoeffsByGPU is the zero value (nil) regardless of the bundle, the
// hook prints "null", and this test goes RED.
func TestCoeffsByGPU_Wired_RunCmdLiteralWiring_Structural(t *testing.T) {
	if os.Getenv("BLIS_EDPP_COEFFS_WIRING_CHILD") == "1" {
		runEDPPCoeffsWiringChild(t)
		return
	}
	if _, err := os.Stat(frozenH100CoeffsPath); os.IsNotExist(err) {
		t.Skipf("frozen coeffs file absent (%s), skipping", frozenH100CoeffsPath)
	}

	captured := runEDPPCoeffsWiringChildProcess(t)
	if captured == nil {
		t.Fatalf("EDPPCoeffsByGPU literal not wired: config.EDPPCoeffsByGPU is nil despite bundle carrying coeffs_by_gpu for H100 and A100")
	}
	h100, ok := captured["H100"]
	if !ok {
		t.Fatalf("EDPPCoeffsByGPU missing H100 key: %+v", captured)
	}
	if !(h100.AlphaD > 0) {
		t.Errorf("EDPPCoeffsByGPU[\"H100\"].AlphaD = %v, want > 0", h100.AlphaD)
	}
	if _, ok := captured["A100"]; !ok {
		t.Errorf("EDPPCoeffsByGPU missing A100 key: %+v", captured)
	}
}

// runEDPPCoeffsWiringChildProcess re-execs this test in a child process and
// returns the EDPPCoeffsByGPU map it reported (nil if the wiring is broken).
func runEDPPCoeffsWiringChildProcess(t *testing.T) map[string]sim.EDPPCoeffs {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestCoeffsByGPU_Wired_RunCmdLiteralWiring_Structural")
	cmd.Env = append(os.Environ(), "BLIS_EDPP_COEFFS_WIRING_CHILD=1")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("child run failed: %v\nstdout:\n%s", err, string(out))
	}
	var jsonLine string
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, edppCoeffsJSONMarker) {
			jsonLine = strings.TrimPrefix(line, edppCoeffsJSONMarker)
			break
		}
	}
	if jsonLine == "" {
		t.Fatalf("child run produced no %s line in stdout:\n%s", edppCoeffsJSONMarker, string(out))
	}
	var captured map[string]sim.EDPPCoeffs
	if err := json.Unmarshal([]byte(jsonLine), &captured); err != nil {
		t.Fatalf("unmarshal child JSON %q: %v", jsonLine, err)
	}
	return captured
}

// runEDPPCoeffsWiringChild is the child-process body: it configures and
// executes the real runCmd against a two-GPU coeffs_by_gpu bundle, installs
// edppCoeffsByGPUWiringHook to print config.EDPPCoeffsByGPU as JSON, and exits
// before the simulation runs.
func runEDPPCoeffsWiringChild(t *testing.T) {
	dir := t.TempDir()
	bundlePath := filepath.Join(dir, "bundle.yaml")
	bundleYAML := fmt.Sprintf(`coeffs_by_gpu:
  H100: %s
  A100: %s
`, frozenH100CoeffsPath, frozenH100CoeffsPath)
	if err := os.WriteFile(bundlePath, []byte(bundleYAML), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write bundle: %v\n", err)
		os.Exit(2)
	}

	mcDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir model config: %v\n", err)
		os.Exit(2)
	}
	configJSON := `{
  "architectures": ["LlamaForCausalLM"],
  "num_attention_heads": 4,
  "num_hidden_layers": 2,
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_key_value_heads": 4,
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`
	if err := os.WriteFile(filepath.Join(mcDir, "config.json"), []byte(configJSON), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write config.json: %v\n", err)
		os.Exit(2)
	}

	// Explicit --hardware-config (mirrors writeHWWiringFixtures in
	// hwconfig_bundle_wiring_test.go) so this run never falls back to
	// network auto-fetch of hardware calibration.
	hwPath := filepath.Join(dir, "hw.json")
	hwJSON := `{
  "H100": {
    "MemoryGiB": 80.0,
    "TFlopsPeak": 312.0,
    "BwPeakTBs": 3.35,
    "mfuPrefill": 0.5,
    "mfuDecode": 0.5
  }
}`
	if err := os.WriteFile(hwPath, []byte(hwJSON), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write hw config: %v\n", err)
		os.Exit(2)
	}

	args := []string{
		"--model", "test-model",
		"--latency-model", "roofline",
		"--hardware", "H100",
		"--tp", "1",
		"--model-config-folder", mcDir,
		"--hardware-config", hwPath,
		"--policy-config", bundlePath,
		"--total-kv-blocks", "4000",
		"--block-size-in-tokens", "16",
		"--num-instances", "1",
		"--seed", "42",
		"--workload", "distribution",
		"--rate", "80",
		"--num-requests", "10",
		"--prompt-tokens", "64",
		"--output-tokens", "32",
		"--horizon", "60000000",
		"--defaults-filepath", "../defaults.yaml",
		"--log", "error",
	}
	if err := runCmd.ParseFlags(args); err != nil {
		fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
		os.Exit(2)
	}

	edppCoeffsByGPUWiringHook = func(m map[string]sim.EDPPCoeffs) {
		b, err := json.Marshal(m)
		if err != nil {
			fmt.Fprintf(os.Stderr, "marshal EDPPCoeffsByGPU: %v\n", err)
			os.Exit(2)
		}
		fmt.Println(edppCoeffsJSONMarker + string(b))
		os.Exit(0)
	}
	runCmd.Run(runCmd, nil)
	// runCmd.Run should never return here: the hook above always os.Exit()s
	// before the simulation runs. If it does return, the hook was never
	// invoked (runCmd.Run did not reach the DeploymentConfig literal).
	fmt.Fprintln(os.Stderr, "runCmd.Run returned without invoking edppCoeffsByGPUWiringHook")
	os.Exit(2)
}
