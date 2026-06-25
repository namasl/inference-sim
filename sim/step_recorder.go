package sim

import (
	"encoding/csv"
	"os"
	"strconv"

	"github.com/sirupsen/logrus"
)

// StepRecorderEnvVar names the environment variable that, when set to a file
// path, enables per-step latency-law recording. Off by default: when the
// variable is unset newStepRecorderFromEnv returns nil and the simulator does
// no extra work, so deterministic stdout (INV-6) is unaffected.
const StepRecorderEnvVar = "BLIS_STEP_CSV"

// stepRecorderHeader lists the per-step CSV columns in write order. These are
// exactly the regressors of the E3 latency law T_iter = α + c0·B_dec + c1·KV +
// c_pf·S_pf (plus the context-weighted prefill term pf_ctx for the c_attn
// basis), captured once per executed engine step.
var stepRecorderHeader = []string{
	"step_idx", "t_iter_us", "b_dec", "kv", "s_pf", "pf_ctx", "batch_size",
}

// stepRecorder streams one CSV row per executed engine step to a file. It is a
// calibration data tap: the rows feed offline OLS fitting of the E3
// coefficients. It writes to a file (never stdout) so it cannot perturb the
// deterministic result channel.
type stepRecorder struct {
	file *os.File
	cw   *csv.Writer
}

// newStepRecorderFromEnv returns a stepRecorder writing to the path in
// BLIS_STEP_CSV, or nil if the variable is unset. A non-empty path that cannot
// be opened is a startup misconfiguration, so we fail loud rather than silently
// dropping calibration data.
func newStepRecorderFromEnv() *stepRecorder {
	path := os.Getenv(StepRecorderEnvVar)
	if path == "" {
		return nil
	}
	f, err := os.Create(path)
	if err != nil {
		logrus.Fatalf("step recorder: cannot create %s=%q: %v", StepRecorderEnvVar, path, err)
	}
	cw := csv.NewWriter(f)
	if err := cw.Write(stepRecorderHeader); err != nil {
		logrus.Fatalf("step recorder: cannot write header to %q: %v", path, err)
	}
	return &stepRecorder{file: f, cw: cw}
}

// record appends one per-step row. bDec is the decode-request count, kv the
// summed resident decode context (Σ ProgressIndex), sPf the prefill tokens this
// step, and pfCtx the context-weighted prefill regressor Σ tᵢ·(sᵢ+tᵢ/2) — the
// attention basis the latency model uses, so a fit on it recovers c_attn.
func (r *stepRecorder) record(stepIdx int, tIterUs int64, bDec int, kv, sPf, pfCtx int64, batchSize int) {
	row := []string{
		strconv.Itoa(stepIdx),
		strconv.FormatInt(tIterUs, 10),
		strconv.Itoa(bDec),
		strconv.FormatInt(kv, 10),
		strconv.FormatInt(sPf, 10),
		strconv.FormatInt(pfCtx, 10),
		strconv.Itoa(batchSize),
	}
	if err := r.cw.Write(row); err != nil {
		logrus.Fatalf("step recorder: write failed: %v", err)
	}
}

// close flushes and closes the underlying file. Safe to call on a nil receiver.
func (r *stepRecorder) close() {
	if r == nil {
		return
	}
	r.cw.Flush()
	if err := r.cw.Error(); err != nil {
		logrus.Errorf("step recorder: flush failed: %v", err)
	}
	if err := r.file.Close(); err != nil {
		logrus.Errorf("step recorder: close failed: %v", err)
	}
}
