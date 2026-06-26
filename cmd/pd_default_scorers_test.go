package cmd

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestResolvePoolScorerConfigs_PDDefaultsToLLMDProfile verifies that PD runs
// default to llm-d's shipped PD scheduling profile (prefix-cache:2 + queue:1,
// per deploy/config/pd-epp-config.yaml in llm-d-router) when the per-pool scorer
// flags are unset — rather than falling back to the cluster-wide round-robin
// policy, which does not load-balance the decode pool. Both run and replay use
// this helper, so the default is identical across them (INV-13 parity).
func TestResolvePoolScorerConfigs_PDDefaultsToLLMDProfile(t *testing.T) {
	wantLLMD := []sim.ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 2},
		{Name: "queue-depth", Weight: 1},
	}

	tests := []struct {
		name      string
		flagVal   string
		pool      string
		pdEnabled bool
		want      []sim.ScorerConfig
	}{
		{"pd enabled, decode flag empty → llm-d default", "", "decode", true, wantLLMD},
		{"pd enabled, prefill flag empty → llm-d default", "", "prefill", true, wantLLMD},
		{"pd disabled, flag empty → nil (fall back to main policy)", "", "decode", false, nil},
		{
			"explicit flag overrides default (pd enabled)",
			"queue-depth:1", "decode", true,
			[]sim.ScorerConfig{{Name: "queue-depth", Weight: 1}},
		},
		{
			"explicit flag honored even when pd disabled",
			"kv-utilization:3", "decode", false,
			[]sim.ScorerConfig{{Name: "kv-utilization", Weight: 3}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolvePoolScorerConfigs(tt.flagVal, tt.pool, tt.pdEnabled)
			if len(got) != len(tt.want) {
				t.Fatalf("got %d configs %+v, want %d %+v", len(got), got, len(tt.want), tt.want)
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Errorf("config[%d] = %+v, want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}
}
