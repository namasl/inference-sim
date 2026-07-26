package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"testing"
)

// edppRuleCompletedRE matches the presence of at least one completed_requests
// count in the run command's per-instance stdout metrics, proving the run
// produced real metrics (not just "didn't crash").
var edppRuleCompletedRE = regexp.MustCompile(`"completed_requests":\s*\d+`)

// TestEDPPRule_LeastTTFT_AcceptedReducedAndJoint is the durable behavioral
// guard on the --edpp-rule CLI wiring (cmd/root.go): --edpp-rule least-ttft must
// be accepted and thread through to a completed run both (1) on the reduced EDPP
// path and (2) combined with --edpp-joint, which selects the least-TTFT-joint arm
// (hardware-aware least-TTFT over the full action set). The combination was once
// rejected because the joint path silently ignored the rule; now it is honored.
//
// Both scenarios re-exec `blis run` as a child process (BLIS_EDPPRULE_CHILD=1)
// because a config error would call logrus.Fatalf / os.Exit, which would kill the
// whole test binary if invoked in-process (matching
// TestCoeffsByGPU_RunCmdLiteralWiring_* in edppcoeffs_bundle_wiring_test.go).
func TestEDPPRule_LeastTTFT_AcceptedReducedAndJoint(t *testing.T) {
	if os.Getenv("BLIS_EDPPRULE_CHILD") == "1" {
		runEDPPRuleChild(t)
		return
	}
	if _, err := os.Stat(frozenH100CoeffsPath); os.IsNotExist(err) {
		t.Skipf("frozen coeffs file absent (%s), skipping", frozenH100CoeffsPath)
	}

	for _, scenario := range []string{"reduced", "joint"} {
		out, err := runEDPPRuleChildProcess(t, scenario)
		if err != nil {
			t.Fatalf("--edpp-rule least-ttft (%s) run failed (expected success): %v\noutput:\n%s", scenario, err, out)
		}
		if !edppRuleCompletedRE.Match(out) {
			t.Fatalf("%s: run produced no completed_requests metrics in stdout:\n%s", scenario, out)
		}
	}
}

// runEDPPRuleChildProcess re-execs this test in a child process for the given
// scenario ("reduced" | "joint") and returns the child's combined output and
// exit error (nil on success).
func runEDPPRuleChildProcess(t *testing.T, scenario string) ([]byte, error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestEDPPRule_LeastTTFT_AcceptedReducedAndJoint")
	cmd.Env = append(os.Environ(), "BLIS_EDPPRULE_CHILD=1", "BLIS_EDPPRULE_SCENARIO="+scenario)
	return cmd.CombinedOutput()
}

// runEDPPRuleChild is the child-process body: it builds a tiny 1P2D EDPP
// config with --edpp-rule least-ttft (plus --edpp-joint in the "joint"
// scenario) and executes the real runCmd.
func runEDPPRuleChild(t *testing.T) {
	scenario := os.Getenv("BLIS_EDPPRULE_SCENARIO")

	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)

	args := []string{
		"--model", "test-model",
		"--latency-model", "trained-physics",
		"--hardware", "H100",
		"--tp", "1",
		"--model-config-folder", mcFolder,
		"--hardware-config", hwPath,
		"--total-kv-blocks", "2000",
		"--block-size-in-tokens", "16",
		"--num-instances", "3",
		"--prefill-instances", "1",
		"--decode-instances", "2",
		"--seed", "42",
		"--workload", "distribution",
		"--rate", "20",
		"--num-requests", "40",
		"--prompt-tokens", "64",
		"--output-tokens", "32",
		"--max-num-running-reqs", "4",
		"--horizon", "60000000",
		"--defaults-filepath", "../defaults.yaml",
		"--log", "error",
		"--pd-decider", "edpp",
		"--edpp-coeffs", frozenH100CoeffsPath,
		"--edpp-tau-ttft", "10s",
		"--edpp-tau-itl", "50ms",
		"--edpp-rule", "least-ttft",
	}
	if scenario == "joint" {
		args = append(args, "--edpp-joint")
	}

	if err := runCmd.ParseFlags(args); err != nil {
		fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
		os.Exit(2)
	}
	runCmd.Run(runCmd, nil)
	os.Exit(0)
}
