package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// frozenH100CoeffsPath is the committed frozen coefficients file used by both
// GPU keys below. Reusing one file for both keys keeps this test independent of
// Task 7's A100 coefficients asset while still exercising the loader with a real,
// validated file.
const frozenH100CoeffsPath = "../scripts/calibration/coeffs-llama70b-h100-tp4.json"

// slowA100CoeffsPath is the committed slow-device θ_i file (fit from the sim's own
// trained-physics at 400 TFLOPS / 0.7 TB/s — Task 7). Its decode coefficients are
// ~4.8× the H100 file's (c0 8.87 vs 5.35, c1 0.228 vs 0.048), so a joint decider
// that consults it scores the A100 decode candidate as far costlier and shifts
// decode load toward the fast H100 candidate. That behavioral shift is what the
// literal-wiring guard below observes.
const slowA100CoeffsPath = "../scripts/calibration/coeffs-llama70b-a100crippled-tp4.json"

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

// instance1CompletedRE extracts the completed_requests reported for instance_1 (the
// fast H100 decode instance) from the run command's per-instance stdout metrics. The
// completed_requests field is printed on the line immediately after instance_id, so an
// anchored two-line match unambiguously reads instance_1's own count.
var instance1CompletedRE = regexp.MustCompile(`"instance_id":\s*"instance_1",\s*"completed_requests":\s*(\d+)`)

// TestCoeffsByGPU_RunCmdLiteralWiring_DecodeSplitObservable is the durable
// behavioral guard on the DeploymentConfig literal in runCmd (cmd/root.go):
//
//	EDPPCoeffsByGPU: bundleEDPPCoeffsByGPU,
//
// It drives the ACTUAL `blis run` command end-to-end, in a child process, on a
// heterogeneous 1P2D joint-EDPP scenario where two decode instances run on
// different hardware (fast H100 vs slow A100 via hw_config_by_gpu), once WITH a
// coeffs_by_gpu bundle (H100 fast + A100 slow θ_i files) and once WITHOUT it. The
// node_pools, hw_config_by_gpu, workload, seed, and global --edpp-coeffs are
// IDENTICAL in both runs, so the ONLY thing that changes is whether per-instance
// θ_i reaches the joint decider via the literal line.
//
// With the literal present, the joint decider scores the slow-A100 decode
// candidate with its ~4.8×-costlier θ_i and proactively shifts decode load onto
// the fast H100 instance, so instance_1 completes strictly MORE requests than in
// the homogeneous (no-coeffs_by_gpu) run. If the literal line is removed,
// config.EDPPCoeffsByGPU is nil in both runs, both use the single global
// --edpp-coeffs for every candidate, the realized decode splits match, and this
// test goes RED.
//
// Child processes are used (matching TestHWConfigByGPU_RunCmdLiteralWiring_*)
// because runCmd.Run calls logrus.Fatalf on misconfiguration, which would
// os.Exit the whole test binary if run in-process.
func TestCoeffsByGPU_RunCmdLiteralWiring_DecodeSplitObservable(t *testing.T) {
	if os.Getenv("BLIS_COEFFS_WIRING_CHILD") == "1" {
		runCoeffsWiringChild(t)
		return
	}
	for _, p := range []string{frozenH100CoeffsPath, slowA100CoeffsPath} {
		if _, err := os.Stat(p); os.IsNotExist(err) {
			t.Skipf("coeffs asset absent (%s), skipping behavioral guard", p)
		}
	}

	withTheta := runCoeffsWiringChildProcess(t, "theta")
	homogeneous := runCoeffsWiringChildProcess(t, "plain")

	t.Logf("instance_1 (fast) completed_requests: with-coeffs_by_gpu=%d homogeneous=%d", withTheta, homogeneous)

	// The homogeneous run must genuinely split decode across BOTH instances,
	// otherwise there is no baseline against which the θ_i shift is observable.
	if homogeneous <= 0 {
		t.Fatalf("homogeneous run put %d decode on the fast instance; fixture failed to reach the joint decider", homogeneous)
	}
	// With per-instance θ_i wired, the joint decider shifts strictly more decode
	// onto the fast H100 instance. A margin (not just !=) rules out incidental
	// one-request jitter and pins the DIRECTION the literal produces.
	const margin = 10
	if withTheta < homogeneous+margin {
		t.Errorf("EDPPCoeffsByGPU literal not wired: with-coeffs_by_gpu completed %d on the fast "+
			"instance, homogeneous completed %d (expected at least %d more). The per-instance θ_i "+
			"must reach the joint decider via the DeploymentConfig literal "+
			"`EDPPCoeffsByGPU: bundleEDPPCoeffsByGPU` in runCmd (cmd/root.go).",
			withTheta, homogeneous, margin)
	}
}

// runCoeffsWiringChildProcess re-execs this test in a child process for the given
// mode ("theta" | "plain") and returns instance_1's completed_requests.
func runCoeffsWiringChildProcess(t *testing.T, mode string) int {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestCoeffsByGPU_RunCmdLiteralWiring_DecodeSplitObservable")
	cmd.Env = append(os.Environ(), "BLIS_COEFFS_WIRING_CHILD=1", "BLIS_COEFFS_WIRING_MODE="+mode)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("child run (mode=%s) failed: %v\nstdout:\n%s", mode, err, string(out))
	}
	m := instance1CompletedRE.FindSubmatch(out)
	if m == nil {
		t.Fatalf("child run (mode=%s) produced no instance_1 completed_requests in stdout:\n%s", mode, string(out))
	}
	n, err := strconv.Atoi(string(m[1]))
	if err != nil {
		t.Fatalf("child run (mode=%s) unparsable completed_requests %q: %v", mode, m[1], err)
	}
	return n
}

// runCoeffsWiringChild is the child-process body: it writes a synthetic model
// config and a heterogeneous 1P2D bundle (with or without coeffs_by_gpu per
// BLIS_COEFFS_WIRING_MODE), then executes the real runCmd under the joint EDPP
// decider, printing per-instance metrics (including instance_1's completed_requests)
// to stdout.
func runCoeffsWiringChild(t *testing.T) {
	withCoeffs := os.Getenv("BLIS_COEFFS_WIRING_MODE") == "theta"

	dir := t.TempDir()
	mcDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir model config: %v\n", err)
		os.Exit(2)
	}
	// A mid-size Llama-shaped config so trained-physics charges enough per-request
	// decode work that the fast H100 pool saturates under the offered load and the
	// joint decider is forced to spill decode onto the slow A100 pool — the regime
	// where per-instance θ_i changes the split.
	configJSON := `{
  "architectures": ["LlamaForCausalLM"],
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_key_value_heads": 8,
  "torch_dtype": "float16",
  "max_position_embeddings": 8192
}`
	if err := os.WriteFile(filepath.Join(mcDir, "config.json"), []byte(configJSON), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write config.json: %v\n", err)
		os.Exit(2)
	}

	// node_pools + hw_config_by_gpu are IDENTICAL in both modes (fast H100 pool
	// absorbs prefill + decode instance_1; slow A100 pool takes decode instance_2).
	// Only coeffs_by_gpu is added in "theta" mode.
	bundle := `node_pools:
  - {name: fast, gpu_type: H100, gpus_per_node: 8, gpu_memory_gib: 80.0, initial_nodes: 1, max_nodes: 1}
  - {name: slow, gpu_type: A100, gpus_per_node: 4, gpu_memory_gib: 80.0, initial_nodes: 1, max_nodes: 1}
hw_config_by_gpu:
  H100: {tflops_peak: 1979.0, bw_peak_tbs: 3.35, mfu_prefill: 0.5, mfu_decode: 0.5}
  A100: {tflops_peak: 400.0,  bw_peak_tbs: 0.7,  mfu_prefill: 0.5, mfu_decode: 0.5}
`
	if withCoeffs {
		bundle += `coeffs_by_gpu:
  H100: ` + frozenH100CoeffsPath + `
  A100: ` + slowA100CoeffsPath + `
`
	}
	bundlePath := filepath.Join(dir, "bundle.yaml")
	if err := os.WriteFile(bundlePath, []byte(bundle), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write bundle: %v\n", err)
		os.Exit(2)
	}

	args := []string{
		"--model", "test-model",
		"--latency-model", "trained-physics",
		"--hardware", "H100",
		"--tp", "4",
		"--model-config-folder", mcDir,
		"--policy-config", bundlePath,
		"--total-kv-blocks", "20000",
		"--block-size-in-tokens", "16",
		"--num-instances", "3",
		"--prefill-instances", "1",
		"--decode-instances", "2",
		"--seed", "42",
		"--workload", "distribution",
		"--rate", "60",
		"--num-requests", "240",
		"--prompt-tokens", "256",
		"--output-tokens", "128",
		"--max-num-running-reqs", "4",
		"--horizon", "600000000",
		"--defaults-filepath", "../defaults.yaml",
		"--log", "error",
		"--pd-decider", "edpp",
		"--edpp-coeffs", frozenH100CoeffsPath,
		"--edpp-tau-ttft", "10s",
		"--edpp-tau-itl", "50ms",
		"--edpp-tadm-estimator", "rollforward",
		"--edpp-joint",
		"--slo-ttft", "batch=60s",
		"--slo-e2e", "batch=8s",
		"--slo-itl", "batch=500ms",
	}
	if err := runCmd.ParseFlags(args); err != nil {
		fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
		os.Exit(2)
	}
	runCmd.Run(runCmd, nil)
	os.Exit(0)
}
