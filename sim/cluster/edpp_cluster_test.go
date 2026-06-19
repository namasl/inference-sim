package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// newTestEDPPDeploymentConfig mirrors newTestDisaggDeploymentConfig but selects the
// EDPP decider and supplies its knobs. Tight τ_itl makes the run breach the ITL SLO
// so the z-feedback path is exercised.
func newTestEDPPDeploymentConfig(numInstances, prefill, decode int) DeploymentConfig {
	modelCfg := sim.ModelConfig{
		NumLayers:       2,
		NumHeads:        4,
		HiddenDim:       64,
		IntermediateDim: 128,
		BytesPerParam:   2.0,
	}
	hwCfg := sim.HardwareCalib{TFlopsPeak: 1.0, BwPeakTBs: 0.001}
	betas := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0}
	alphas := []float64{100, 1, 100}
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs(betas, alphas),
			ModelHardwareConfig: sim.NewModelHardwareConfig(modelCfg, hwCfg, "test-model", "H100", 1, 1, false, "trained-physics", 0),
		},
		NumInstances:            numInstances,
		PrefillInstances:        prefill,
		DecodeInstances:         decode,
		PDDecider:               "edpp",
		EDPPTauTTFTUs:           100_000,
		EDPPTauITLUs:            5_000,
		EDPPV:                   0.1,
		EDPPCXferUs:             1_000,
		EDPPNomPrefillTokens:    512,
		EDPPNomDecodeCtx:        2048,
		RoutingPolicy:           "round-robin",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
	}
}

func TestEDPP_Cluster_WiringAndFeedback(t *testing.T) {
	config := newTestEDPPDeploymentConfig(4, 2, 2)
	cs := NewClusterSimulator(config, newTestRequests(5), nil)

	// The "edpp" name must resolve to an EDPPDecider...
	if _, ok := cs.disaggregationDecider.(*sim.EDPPDecider); !ok {
		t.Fatalf("disaggregationDecider = %T, want *sim.EDPPDecider", cs.disaggregationDecider)
	}
	// ...and it must be registered as the SLO-feedback sink so completions update Z.
	if cs.sloFeedback == nil {
		t.Fatalf("sloFeedback is nil; EDPP completion feedback not wired")
	}

	// End-to-end run must complete without error and produce decode output (full path).
	mustRun(t, cs)
	if cs.AggregatedMetrics().TotalOutputTokens == 0 {
		t.Error("TotalOutputTokens = 0; EDPP run produced no decode output")
	}
}
