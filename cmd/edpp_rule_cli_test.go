package cmd

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"testing"
)

// edppRuleCompletedRE matches the presence of at least one completed_requests
// count in the run command's per-instance stdout metrics, proving the run
// produced real metrics (not just "didn't crash").
var edppRuleCompletedRE = regexp.MustCompile(`"completed_requests":\s*\d+`)

// TestEDPPRule_LeastTTFT_AcceptedAndJointRejected is the durable behavioral
// guard on the --edpp-rule CLI wiring (cmd/root.go): the flag must (1) be
// accepted and thread through to a completed run under the reduced EDPP path,
// and (2) be rejected at the CLI boundary (R3) when combined with
// --edpp-joint, since least-ttft is a reduced-path baseline that bypasses the
// joint argmin machinery entirely.
//
// Both scenarios re-exec `blis run` as a child process
// (BLIS_EDPPRULE_CHILD=1) because logrus.Fatalf calls os.Exit, which would
// kill the whole test binary if invoked in-process (matching
// TestCoeffsByGPU_RunCmdLiteralWiring_* in edppcoeffs_bundle_wiring_test.go).
func TestEDPPRule_LeastTTFT_AcceptedAndJointRejected(t *testing.T) {
	if os.Getenv("BLIS_EDPPRULE_CHILD") == "1" {
		runEDPPRuleChild(t)
		return
	}
	if _, err := os.Stat(frozenH100CoeffsPath); os.IsNotExist(err) {
		t.Skipf("frozen coeffs file absent (%s), skipping", frozenH100CoeffsPath)
	}

	// Case 1: --edpp-rule least-ttft alone completes and produces metrics.
	out, err := runEDPPRuleChildProcess(t, "accept")
	if err != nil {
		t.Fatalf("--pd-decider edpp --edpp-rule least-ttft run failed (expected success): %v\noutput:\n%s", err, out)
	}
	if !edppRuleCompletedRE.Match(out) {
		t.Fatalf("run produced no completed_requests metrics in stdout:\n%s", out)
	}

	// Case 2: --edpp-rule least-ttft + --edpp-joint together must fail (R3: CLI -> Fatalf).
	out, err = runEDPPRuleChildProcess(t, "reject")
	if err == nil {
		t.Fatalf("expected non-zero exit for --edpp-rule least-ttft --edpp-joint, got exit 0; output:\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("unexpected error type for reject scenario: %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("reject scenario: expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	const wantMsg = "cannot be combined with --edpp-joint"
	if !strings.Contains(string(out), wantMsg) {
		t.Errorf("reject scenario: fatal message should contain %q, got:\n%s", wantMsg, out)
	}
}

// runEDPPRuleChildProcess re-execs this test in a child process for the given
// scenario ("accept" | "reject") and returns the child's combined output and
// exit error (nil on success).
func runEDPPRuleChildProcess(t *testing.T, scenario string) ([]byte, error) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestEDPPRule_LeastTTFT_AcceptedAndJointRejected")
	cmd.Env = append(os.Environ(), "BLIS_EDPPRULE_CHILD=1", "BLIS_EDPPRULE_SCENARIO="+scenario)
	return cmd.CombinedOutput()
}

// runEDPPRuleChild is the child-process body: it builds a tiny 1P2D EDPP
// config with --edpp-rule least-ttft (plus --edpp-joint in the "reject"
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
	if scenario == "reject" {
		args = append(args, "--edpp-joint")
	}

	if err := runCmd.ParseFlags(args); err != nil {
		fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
		os.Exit(2)
	}
	runCmd.Run(runCmd, nil)
	os.Exit(0)
}
