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

// twoPoolBundleYAML builds a policy-config carrying two node pools (H100 + A100).
// When withHWConfig is true it also emits a hw_config_by_gpu block whose A100 entry
// is calibrated catastrophically slow; when false the block is omitted entirely so
// both pools fall back to the single CLI --hardware calibration.
func twoPoolBundleYAML(withHWConfig bool) string {
	pools := `node_pools:
  - name: h100-pool
    gpu_type: H100
    gpus_per_node: 1
    initial_nodes: 1
    max_nodes: 1
    gpu_memory_gib: 80
  - name: a100-pool
    gpu_type: A100
    gpus_per_node: 1
    initial_nodes: 1
    max_nodes: 1
    gpu_memory_gib: 80
`
	if !withHWConfig {
		return pools
	}
	// H100 fast (mirrors the CLI --hardware calibration), A100 crippled: a
	// ~1e-6 TFlops peak inflates every roofline step past any finite horizon,
	// so the A100-pool instance completes essentially nothing.
	return pools + `hw_config_by_gpu:
  H100:
    tflops_peak: 312.0
    bw_peak_tbs: 3.35
    mfu_prefill: 0.5
    mfu_decode: 0.5
  A100:
    tflops_peak: 0.000001
    bw_peak_tbs: 0.00000001
    mfu_prefill: 0.5
    mfu_decode: 0.5
`
}

// writeHWWiringFixtures writes a minimal HF config.json and a hardware config whose
// H100 entry is fast, so the ONLY thing that can slow the A100-pool instance is the
// per-GPU hw_config_by_gpu override reaching it via DeploymentConfig.HWConfigByGPU.
func writeHWWiringFixtures(t *testing.T) (mcFolder, hwPath string) {
	t.Helper()
	dir := t.TempDir()
	mcDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(mcDir, 0o755); err != nil {
		t.Fatalf("mkdir model config: %v", err)
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
		t.Fatalf("write config.json: %v", err)
	}
	hwFile := filepath.Join(dir, "hw.json")
	hwJSON := `{
  "H100": {
    "MemoryGiB": 80.0,
    "TFlopsPeak": 312.0,
    "BwPeakTBs": 3.35,
    "mfuPrefill": 0.5,
    "mfuDecode": 0.5
  }
}`
	if err := os.WriteFile(hwFile, []byte(hwJSON), 0o644); err != nil {
		t.Fatalf("write hw config: %v", err)
	}
	return mcDir, hwFile
}

// TestHWConfigByGPU_ConverterCopiesAllFields verifies the field-copy helper
// hwConfigByGPUFromBundle (the exact function the run command calls) reproduces
// every hw_config_by_gpu field, not just the two guarded by bundle.Validate().
//
// This is a UNIT guard on the converter. It does NOT (and cannot) guard the
// DeploymentConfig literal assignment in runCmd — that is covered by
// TestHWConfigByGPU_RunCmdLiteralWiring_PerPoolSpeedObservable below.
func TestHWConfigByGPU_ConverterCopiesAllFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "policy.yaml")
	yaml := `node_pools:
  - name: h100-pool
    gpu_type: H100
    gpus_per_node: 1
    initial_nodes: 1
    max_nodes: 1
    gpu_memory_gib: 80
  - name: a100-pool
    gpu_type: A100
    gpus_per_node: 1
    initial_nodes: 1
    max_nodes: 1
    gpu_memory_gib: 80
hw_config_by_gpu:
  H100:
    tflops_peak: 1979.0
    bw_peak_tbs: 3.35
    mfu_prefill: 0.5
    mfu_decode: 0.5
  A100:
    tflops_peak: 1248.0
    bw_peak_tbs: 2.0
    mfu_prefill: 0.3
    mfu_decode: 0.3
`
	if err := os.WriteFile(path, []byte(yaml), 0o644); err != nil {
		t.Fatalf("write policy YAML: %v", err)
	}
	bundle, err := sim.LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if err := bundle.Validate(); err != nil {
		t.Fatalf("bundle.Validate: %v", err)
	}

	hwMap := hwConfigByGPUFromBundle(bundle)
	if hwMap == nil {
		t.Fatalf("hwConfigByGPUFromBundle returned nil; bundle carried hw_config_by_gpu")
	}
	h100, ok := hwMap["H100"]
	if !ok {
		t.Fatalf("converted map missing H100 entry")
	}
	a100, ok := hwMap["A100"]
	if !ok {
		t.Fatalf("converted map missing A100 entry")
	}
	if h100.TFlopsPeak != 1979.0 || a100.TFlopsPeak != 1248.0 {
		t.Errorf("per-GPU TFlopsPeak not wired through: got H100=%v A100=%v, want 1979.0 and 1248.0",
			h100.TFlopsPeak, a100.TFlopsPeak)
	}
	if h100.BwPeakTBs != 3.35 || h100.MfuPrefill != 0.5 || h100.MfuDecode != 0.5 {
		t.Errorf("H100 fields not copied verbatim: %+v", h100)
	}
	if a100.BwPeakTBs != 2.0 || a100.MfuPrefill != 0.3 || a100.MfuDecode != 0.3 {
		t.Errorf("A100 fields not copied verbatim: %+v", a100)
	}
}

// completedRequestsRE extracts "completed_requests": N from the run command's
// stdout metrics JSON.
var completedRequestsRE = regexp.MustCompile(`"completed_requests":\s*(\d+)`)

// TestHWConfigByGPU_RunCmdLiteralWiring_PerPoolSpeedObservable is the real
// regression guard on the DeploymentConfig literal in runCmd (cmd/root.go):
//
//	HWConfigByGPU: bundleHWConfigByGPU,
//
// It drives the ACTUAL `blis run` command (runCmd.Run) end-to-end, in a child
// process, against a two-pool (H100 + A100) policy bundle, once WITH a
// hw_config_by_gpu block that cripples the A100 pool and once WITHOUT it. The
// H100 hardware config and CLI --hardware are identical and fast in both runs,
// so the ONLY path by which the A100-pool instance can slow down is the per-GPU
// override travelling bundle -> hwConfigByGPUFromBundle -> DeploymentConfig
// literal -> per-instance construction.
//
// With the literal line present, the crippled-A100 run completes strictly FEWER
// requests than the no-override run. If the literal line is removed,
// config.HWConfigByGPU is nil in both runs, both pools use the fast CLI
// calibration, the completion counts match, and this test goes RED.
//
// Child processes are used (matching TestReplayCmd_NodePoolsBundleFatal) because
// runCmd.Run calls logrus.Fatalf on any misconfiguration, which would os.Exit the
// whole test binary if run in-process.
func TestHWConfigByGPU_RunCmdLiteralWiring_PerPoolSpeedObservable(t *testing.T) {
	if os.Getenv("BLIS_HW_WIRING_CHILD") == "1" {
		runHWWiringChild(t)
		return
	}

	crippled := runHWWiringChildProcess(t, "crippled")
	baseline := runHWWiringChildProcess(t, "baseline")

	t.Logf("completed_requests: crippled-A100=%d baseline(no hw_config_by_gpu)=%d", crippled, baseline)
	if baseline <= 0 {
		t.Fatalf("baseline run completed %d requests; fixture/workload too weak to observe a difference", baseline)
	}
	if crippled >= baseline {
		t.Errorf("HWConfigByGPU literal not wired: crippled-A100 run completed %d requests, "+
			"no-override run completed %d; expected strictly fewer. The A100 pool's slow "+
			"hw_config_by_gpu calibration must reach its instance via the DeploymentConfig "+
			"literal `HWConfigByGPU: bundleHWConfigByGPU` in runCmd (cmd/root.go).",
			crippled, baseline)
	}
}

// runHWWiringChildProcess re-execs this test in a child process for the given mode
// ("crippled" | "baseline") and returns the completed_requests it reported.
func runHWWiringChildProcess(t *testing.T, mode string) int {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestHWConfigByGPU_RunCmdLiteralWiring_PerPoolSpeedObservable")
	cmd.Env = append(os.Environ(), "BLIS_HW_WIRING_CHILD=1", "BLIS_HW_WIRING_MODE="+mode)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("child run (mode=%s) failed: %v\nstdout:\n%s", mode, err, string(out))
	}
	// runCmd prints one metrics block per instance followed by the aggregated
	// cluster block last. The aggregate is the cluster-wide total we compare on,
	// so take the LAST completed_requests match.
	all := completedRequestsRE.FindAllSubmatch(out, -1)
	if all == nil {
		t.Fatalf("child run (mode=%s) produced no completed_requests in stdout:\n%s", mode, string(out))
	}
	m := all[len(all)-1]
	n, err := strconv.Atoi(string(m[1]))
	if err != nil {
		t.Fatalf("child run (mode=%s) unparsable completed_requests %q: %v", mode, m[1], err)
	}
	return n
}

// runHWWiringChild is the child-process body: it configures and executes the real
// runCmd against the two-pool bundle selected by BLIS_HW_WIRING_MODE, printing the
// metrics JSON (including completed_requests) to stdout.
func runHWWiringChild(t *testing.T) {
	mode := os.Getenv("BLIS_HW_WIRING_MODE")
	withHWConfig := mode == "crippled"

	dir := t.TempDir()
	bundlePath := filepath.Join(dir, "bundle.yaml")
	if err := os.WriteFile(bundlePath, []byte(twoPoolBundleYAML(withHWConfig)), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write bundle: %v\n", err)
		os.Exit(2)
	}
	mcFolder, hwPath := writeHWWiringFixtures(t)

	args := []string{
		"--model", "test-model",
		"--latency-model", "roofline",
		"--hardware", "H100",
		"--tp", "1",
		"--model-config-folder", mcFolder,
		"--hardware-config", hwPath,
		"--policy-config", bundlePath,
		"--total-kv-blocks", "4000",
		"--block-size-in-tokens", "16",
		"--num-instances", "2",
		"--seed", "42",
		"--workload", "distribution",
		"--rate", "80",
		"--num-requests", "40",
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
	// runCmd.Run prints "=== Simulation Metrics ===" + metrics JSON to stdout.
	runCmd.Run(runCmd, nil)
	os.Exit(0)
}
