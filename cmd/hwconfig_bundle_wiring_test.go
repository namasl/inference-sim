package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// bundleWithTwoPools writes a policy-config YAML carrying two node pools (H100 + A100)
// and a hw_config_by_gpu block, loads it via the real LoadPolicyBundle + Validate path,
// and returns the parsed bundle. The h100/a100 TFlopsPeak values are the caller's to pick
// so the same helper serves both the field-wiring assertion (realistic values) and the
// behavioral assertion (catastrophically slow A100).
func bundleWithTwoPools(t *testing.T, h100TFlops, h100BW, a100TFlops, a100BW float64) *sim.PolicyBundle {
	t.Helper()
	yaml := fmt.Sprintf(`node_pools:
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
    tflops_peak: %g
    bw_peak_tbs: %g
    mfu_prefill: 0.5
    mfu_decode: 0.5
  A100:
    tflops_peak: %g
    bw_peak_tbs: %g
    mfu_prefill: 0.3
    mfu_decode: 0.3
`, h100TFlops, h100BW, a100TFlops, a100BW)

	path := filepath.Join(t.TempDir(), "policy.yaml")
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
	return bundle
}

// TestHWConfigByGPU_Wired_PerInstanceHardwareDiffers verifies the cmd conversion path:
// bundle.HWConfigByGPU -> DeploymentConfig.HWConfigByGPU. Before this wiring the field was
// never populated, so every node-pool-placed instance fell back to the single CLI --gpu
// calibration and the map below would be nil.
//
// Two assertions:
//  1. Field wiring: hwConfigByGPUFromBundle (the exact function the run command calls)
//     produces a map carrying BOTH pool-specific TFlopsPeak values (1979.0 for H100,
//     1248.0 for A100) with every field copied verbatim.
//  2. Behavioral integration: feeding that map into the same DeploymentConfig the run
//     command builds, a two-pool / two-instance cluster whose A100 pool is calibrated
//     catastrophically slow completes strictly FEWER requests than an identical cluster
//     with no hw_config_by_gpu (all instances on the fast CLI calibration). This proves
//     the per-GPU calibration reaches per-instance construction and diverges by pool —
//     impossible when HWConfigByGPU is nil.
func TestHWConfigByGPU_Wired_PerInstanceHardwareDiffers(t *testing.T) {
	// --- Assertion 1: field-level wiring with realistic pool calibrations. ---
	realistic := bundleWithTwoPools(t, 1979.0, 3.35, 1248.0, 2.0)
	hwMap := hwConfigByGPUFromBundle(realistic)
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
	// All six fields must be copied (not just the two Validate() guards).
	if h100.BwPeakTBs != 3.35 || h100.MfuPrefill != 0.5 || h100.MfuDecode != 0.5 {
		t.Errorf("H100 fields not copied verbatim: %+v", h100)
	}
	if a100.BwPeakTBs != 2.0 || a100.MfuPrefill != 0.3 || a100.MfuDecode != 0.3 {
		t.Errorf("A100 fields not copied verbatim: %+v", a100)
	}

	// --- Assertion 2: the wired map takes effect per instance (pool-authoritative). ---
	// Slow A100 calibration -> roofline step time >> horizon -> its instance completes ~0.
	// Fast H100 -> completes its share. Identical no-map run: both instances fast -> more done.
	slow := bundleWithTwoPools(t, 312.0, 3.35, 1e-6, 1e-8)
	slowMap := hwConfigByGPUFromBundle(slow)

	nodePools := []cluster.NodePoolConfig{
		{Name: "h100-pool", GPUType: "H100", GPUsPerNode: 1, InitialNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80},
		{Name: "a100-pool", GPUType: "A100", GPUsPerNode: 1, InitialNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80},
	}

	completed := func(hwByGPU map[string]sim.HardwareCalib) int {
		cfg := cluster.DeploymentConfig{
			SimConfig:     baseHWWiringSimCfg(),
			NumInstances:  2,
			NodePools:     nodePools,
			HWConfigByGPU: hwByGPU,
		}
		cs := cluster.NewClusterSimulator(cfg, makeHWWiringReqs(), nil)
		if err := cs.Run(); err != nil {
			t.Fatalf("ClusterSimulator.Run: %v", err)
		}
		return cs.AggregatedMetrics().CompletedRequests
	}

	completedSlow := completed(slowMap)
	completedFast := completed(nil) // no hw_config_by_gpu -> all instances on fast CLI calib

	if completedSlow >= completedFast {
		t.Errorf("HWConfigByGPU not applied per instance: slow-A100-pool cluster completed %d, "+
			"no-override cluster completed %d; expected strictly fewer (the A100 pool's slow "+
			"calibration must reach its instance via DeploymentConfig.HWConfigByGPU)",
			completedSlow, completedFast)
	}
}

// baseHWWiringSimCfg mirrors the roofline-mode SimConfig used by the sim/cluster HWConfigByGPU
// tests. The CLI calibration here is a fast H100 so that the *only* way the A100 instance
// slows down is the per-GPU override.
func baseHWWiringSimCfg() sim.SimConfig {
	mc := sim.ModelConfig{
		NumLayers: 4, HiddenDim: 256, NumHeads: 4, NumKVHeads: 4,
		BytesPerParam: 2.0, IntermediateDim: 512, VocabSize: 1000,
	}
	fastH100 := sim.HardwareCalib{TFlopsPeak: 312.0, BwPeakTBs: 3.35, MfuPrefill: 0.5, MfuDecode: 0.5}
	return sim.SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		ModelHardwareConfig: sim.NewModelHardwareConfig(mc, fastH100, "test-model", "H100", 1, 1, false, "roofline", 0),
		KVCacheConfig:       sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(8, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs(nil, []float64{0, 0, 0}),
	}
}

func makeHWWiringReqs() []*sim.Request {
	reqs := make([]*sim.Request, 8)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("req_%d", i),
			ArrivalTime:  int64(i) * 100,
			InputTokens:  make([]int, 50),
			OutputTokens: make([]int, 20),
			State:        sim.StateQueued,
		}
	}
	return reqs
}
