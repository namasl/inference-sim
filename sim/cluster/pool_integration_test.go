package cluster

import (
	"testing"
)

// TestBackwardCompatibility_PoolsDisabled verifies BC-PD-1 and BC-PD-10:
// When pools are disabled (both counts = 0), simulation behavior is unchanged.
func TestBackwardCompatibility_PoolsDisabled(t *testing.T) {
	config := newTestDeploymentConfig(4)
	config.PrefillInstances = 0
	config.DecodeInstances = 0
	config.PDDecider = "" // default

	requests := newTestRequests(10)
	cs := NewClusterSimulator(config, requests)

	// Verify pools are not configured
	if cs.poolsConfigured() {
		t.Error("poolsConfigured() = true, want false when both counts are 0")
	}

	// Verify disaggregationDecider is nil when pools are disabled
	if cs.disaggregationDecider != nil {
		t.Error("disaggregationDecider should be nil when pools are disabled")
	}

	mustRun(t, cs)

	// Verify no parent requests were created (no disaggregation occurred)
	if len(cs.ParentRequests()) != 0 {
		t.Errorf("ParentRequests() = %d, want 0 (no disaggregation when pools disabled)", len(cs.ParentRequests()))
	}

	// Verify requests were processed normally
	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests == 0 {
		t.Error("CompletedRequests = 0, want > 0 (requests should complete normally)")
	}
}

// TestPoolsEnabled_NeverDisaggregate verifies BC-PD-1, BC-PD-4, BC-PD-5:
// When pools are enabled but decider is "never", DisaggregationDecisionEvent
// is scheduled but always routes to standard path.
func TestPoolsEnabled_NeverDisaggregate(t *testing.T) {
	config := newTestDeploymentConfig(4)
	config.PrefillInstances = 2
	config.DecodeInstances = 2
	config.PDDecider = "never"
	config.PDTransferBandwidthGBps = 25.0
	config.PDTransferBaseLatencyMs = 0.05
	config.PDKVBytesPerToken = 512

	requests := newTestRequests(10)
	cs := NewClusterSimulator(config, requests)

	// Verify pools are configured
	if !cs.poolsConfigured() {
		t.Error("poolsConfigured() = false, want true when both counts > 0")
	}

	// Verify pool membership
	membership := cs.PoolMembership()
	if len(membership) != 4 {
		t.Errorf("PoolMembership() len = %d, want 4", len(membership))
	}

	mustRun(t, cs)

	// Verify no parent requests were created (NeverDisaggregate always returns false)
	if len(cs.ParentRequests()) != 0 {
		t.Errorf("ParentRequests() = %d, want 0 (NeverDisaggregate should not create parent requests)", len(cs.ParentRequests()))
	}

	// Verify requests were processed normally
	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests == 0 {
		t.Error("CompletedRequests = 0, want > 0 (requests should complete normally)")
	}
}

// TestPoolsEnabled_AlwaysDisaggregate verifies BC-PD-6:
// When decider is "always", all requests are disaggregated.
func TestPoolsEnabled_AlwaysDisaggregate(t *testing.T) {
	config := newTestDeploymentConfig(4)
	config.PrefillInstances = 2
	config.DecodeInstances = 2
	config.PDDecider = "always"
	config.PDTransferBandwidthGBps = 25.0
	config.PDTransferBaseLatencyMs = 0.05
	config.PDKVBytesPerToken = 512

	requests := newTestRequests(10)
	cs := NewClusterSimulator(config, requests)

	mustRun(t, cs)

	// Verify all requests were disaggregated (parent requests created)
	parentReqs := cs.ParentRequests()
	if len(parentReqs) != len(requests) {
		t.Errorf("ParentRequests() = %d, want %d (all requests should be disaggregated)", len(parentReqs), len(requests))
	}

	// Verify requests were processed
	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests == 0 {
		t.Error("CompletedRequests = 0, want > 0 (disaggregated requests should complete)")
	}
}

// TestPoolTopologyValidation_CLI verifies BC-PD-11, BC-PD-12, BC-PD-13:
// Invalid pool configurations are rejected at construction time.
func TestPoolTopologyValidation_CLI(t *testing.T) {
	tests := []struct {
		name     string
		prefill  int
		decode   int
		total    int
		wantPanic bool
	}{
		{
			name:      "negative prefill",
			prefill:   -1,
			decode:    2,
			total:     4,
			wantPanic: true,
		},
		{
			name:      "negative decode",
			prefill:   2,
			decode:    -1,
			total:     4,
			wantPanic: true,
		},
		{
			name:      "only prefill set",
			prefill:   2,
			decode:    0,
			total:     4,
			wantPanic: true,
		},
		{
			name:      "sum exceeds total",
			prefill:   3,
			decode:    3,
			total:     4,
			wantPanic: true,
		},
		{
			name:      "valid configuration",
			prefill:   2,
			decode:    2,
			total:     4,
			wantPanic: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if tt.wantPanic && r == nil {
					t.Error("expected panic but none occurred")
				}
				if !tt.wantPanic && r != nil {
					t.Errorf("unexpected panic: %v", r)
				}
			}()

			config := newTestDeploymentConfig(tt.total)
			config.PrefillInstances = tt.prefill
			config.DecodeInstances = tt.decode
			config.PDDecider = "never"
			config.PDTransferBandwidthGBps = 25.0
			config.PDTransferBaseLatencyMs = 0.05
			config.PDKVBytesPerToken = 512

			requests := newTestRequests(5)
			_ = NewClusterSimulator(config, requests)
		})
	}
}
