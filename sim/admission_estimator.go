package sim

import "fmt"

// RunningReqState is one running decode request's state for the roll-forward
// estimator. TrueRemaining is the oracle remaining step count (-1 when the
// oracle is not populated); StepsDone is decode steps completed; KVBlocks held.
type RunningReqState struct {
	StepsDone     int64
	KVBlocks      int64
	TrueRemaining int64
}

// AdmissionContext bundles everything an admission-delay estimator may read for
// one pool. The EDPPDecider assembles it from its backlog/rate state and the
// (possibly enriched) selected snapshot. Times/work in µs.
type AdmissionContext struct {
	QWork             float64 // waiting-backlog work (µs)
	Mu                float64 // occupancy-aware drain rate
	BatchSize         int
	MaxBatchSize      int
	FreeKVBlocks      int64
	ReqKVNeed         int64   // KV blocks this request needs
	TIter             float64 // occupancy-aware per-iteration time (µs)
	QueueDepth        int
	AdmissionRate     float64           // req/µs admitted at this pool (for little)
	RemainingStepsEst float64           // mean estimated remaining decode steps (N̂_out-based, for fluid)
	Running           []RunningReqState // per-running-request state (for rollforward)
}

// AdmissionDelayEstimator predicts the admission delay (µs) a request would incur
// at a pool given its current state. Pure function of ctx (INV-6).
type AdmissionDelayEstimator interface {
	EstimateTAdm(ctx AdmissionContext) float64
	Name() string
}

type waitingEstimator struct{}

func (waitingEstimator) Name() string { return "waiting" }
func (waitingEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.Mu <= 0 {
		return 0
	}
	return ctx.QWork / ctx.Mu
}

type littleEstimator struct{}

func (littleEstimator) Name() string { return "little" }
func (littleEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.AdmissionRate <= 0 {
		return 0
	}
	return float64(ctx.QueueDepth) / ctx.AdmissionRate
}

type fluidEstimator struct{}

func (fluidEstimator) Name() string { return "fluid" }
func (fluidEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	// Admit next iteration if a slot AND enough KV already fit.
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return 0
	}
	// Occupancy-conditioned departure rate X̂_dep = B / (R̄ · T_iter) departures per µs.
	if ctx.RemainingStepsEst <= 0 || ctx.TIter <= 0 || ctx.BatchSize <= 0 {
		return 0
	}
	xDep := float64(ctx.BatchSize) / (ctx.RemainingStepsEst * ctx.TIter)
	if xDep <= 0 {
		return 0
	}
	// N_ahead: at least one departure needed for a slot; add KV-driven departures if KV-bound.
	nAhead := 1.0
	return nAhead / xDep
}

// NewAdmissionEstimator returns the estimator by name. Little/fluid/rollforward
// and the oracle variants are added in later tasks.
func NewAdmissionEstimator(name string) (AdmissionDelayEstimator, error) {
	switch name {
	case "", "waiting":
		return waitingEstimator{}, nil
	case "little":
		return littleEstimator{}, nil
	case "fluid":
		return fluidEstimator{}, nil
	default:
		return nil, fmt.Errorf("unknown admission estimator %q", name)
	}
}
