# PR1: Pool Topology and Disaggregation Decision Pipeline - Implementation Plan

This is a comprehensive implementation plan for PR1 of the PD disaggregation feature merge. The plan follows the micro-plan template and provides complete implementation details.

**Goal:** Establish the foundation for prefill-decode disaggregation by adding pool topology validation, disaggregation decision interface, and event pipeline integration without changing simulation behavior.

**The problem today:** BLIS cannot model architectures where prefill (prompt processing) and decode (token generation) run on separate instance pools. All instances are homogeneous and handle both phases. This limits capacity planning for disaggregated deployments.

**What this PR adds:**
1. **Pool topology management** - Validate and track which instances belong to prefill vs decode pools (e.g., instances 0-1 are prefill, 2-3 are decode)
2. **Disaggregation decision interface** - Define how requests are routed (always disaggregate, never disaggregate, or policy-based)
3. **Event pipeline integration** - Insert DisaggregationDecisionEvent between admission and routing when pools are configured
4. **Backward compatibility** - When pools are disabled (both counts = 0), simulation behavior is unchanged

**Why this matters:** This PR establishes the architectural foundation for PD disaggregation. All subsequent PRs (request splitting, KV transfer, metrics) build on this topology and decision framework. Without this foundation, we cannot incrementally add disaggregation features.

**Architecture:** Adds `sim/cluster/pool.go` for topology validation and membership tracking, `sim/disaggregation.go` for the decision interface, extends `sim/cluster/cluster_event.go` with DisaggregationDecisionEvent (priority 3, between admission and routing), and adds pool config fields to `sim/cluster/deployment.go`. CLI flags in `cmd/root.go` control pool sizes and decider selection.

**Source:** 
- Macro plan: `pd-branch-merge-plan.md` (PR1 section)
- GitHub issue: https://github.com/inference-sim/inference-sim/issues/778

**Closes:** Fixes #778

**Behavioral Contracts:** See Part 1, Section B below.

---

## Summary

This plan implements PR1 by extracting the pool topology foundation from the pd branch. The implementation is straightforward - copy complete implementations from pd branch with minimal modifications (only removing PR5/PR9 deciders from disaggregation.go). All code is already tested and working in the pd branch. The key challenge is ensuring backward compatibility (BC-PD-10) through the poolsConfigured() guard.

**Estimated effort:** 4-6 hours (mostly extraction and validation, minimal new code)

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds pool topology management for prefill-decode disaggregation. It introduces:
1. Pool membership tracking (instance ID → PoolRole)
2. Topology validation (prefill + decode ≤ total, both-or-neither rule)
3. DisaggregationDecider interface with NeverDisaggregate and AlwaysDisaggregate implementations
4. DisaggregationDecisionEvent in cluster event pipeline (priority 3, between admission and routing)
5. CLI flags: --prefill-instances, --decode-instances, --pd-decider

When pools are disabled (both counts = 0), the pipeline is unchanged - requests go directly from admission to routing. When pools are enabled, DisaggregationDecisionEvent is scheduled, but in PR1 it always routes to the standard path (disaggregate=false) since NeverDisaggregate is the default. This establishes the foundation without changing simulation behavior.

Adjacent components: ClusterSimulator (consumes membership, schedules event), DeploymentConfig (stores config), event pipeline (new priority 3 event), CLI (exposes flags).

**Deviation flags:** None - this PR strictly implements the macro plan PR1 section.

### B) Behavioral Contracts

**Positive Contracts (what MUST happen):**

**BC-PD-1: Disaggregation disabled when pools not configured**
- GIVEN: DeploymentConfig with PrefillInstances=0 and DecodeInstances=0
- WHEN: ClusterSimulator is constructed
- THEN: poolsConfigured() returns false, DisaggregationDecisionEvent is never scheduled, requests go directly from admission to routing

**BC-PD-2: Pool topology validation**
- GIVEN: Pool configuration with prefill, decode, total counts
- WHEN: ValidatePoolTopology is called
- THEN: Returns nil if valid (both zero OR both non-zero with sum ≤ total), returns error otherwise

**BC-PD-3: Pool membership construction**
- GIVEN: Valid pool configuration with prefill=P, decode=D
- WHEN: BuildPoolMembership is called
- THEN: Returns map with P+D entries, instances 0..P-1 mapped to PoolRolePrefill, P..P+D-1 mapped to PoolRoleDecode

**BC-PD-4: DisaggregationDecisionEvent scheduling**
- GIVEN: ClusterSimulator with pools configured (prefill > 0, decode > 0)
- WHEN: AdmissionDecisionEvent admits a request
- THEN: DisaggregationDecisionEvent is scheduled at same timestamp with priority 3

**BC-PD-5: NeverDisaggregate always returns false**
- GIVEN: NeverDisaggregate decider
- WHEN: Decide is called with any request
- THEN: Returns DisaggregationDecision{Disaggregate: false}

**BC-PD-6: AlwaysDisaggregate always returns true**
- GIVEN: AlwaysDisaggregate decider
- WHEN: Decide is called with any request
- THEN: Returns DisaggregationDecision{Disaggregate: true}

**BC-PD-7: Factory dispatches to correct decider**
- GIVEN: Decider name ("", "never", "always")
- WHEN: NewDisaggregationDecider is called
- THEN: Returns corresponding decider instance

**BC-PD-8: DisaggregationDecisionEvent routes to standard path in PR1**
- GIVEN: DisaggregationDecisionEvent with disaggregate=false
- WHEN: Event is executed
- THEN: RoutingDecisionEvent is scheduled at T+routingLatency with priority 2

**Negative Contracts (what MUST NOT happen):**

**BC-PD-9: No pool membership mutation after construction**
- GIVEN: ClusterSimulator with pool membership map
- WHEN: Simulation runs
- THEN: Pool membership map is never modified (INV-PD-5)

**BC-PD-10: No simulation behavior change when pools disabled**
- GIVEN: DeploymentConfig with PrefillInstances=0, DecodeInstances=0
- WHEN: Simulation runs
- THEN: Output is byte-identical to main branch (same seed)

**Error Handling Contracts:**

**BC-PD-11: Negative pool counts rejected**
- GIVEN: Pool configuration with prefill < 0 OR decode < 0
- WHEN: ValidatePoolTopology is called
- THEN: Returns error describing which count is negative

**BC-PD-12: Single pool rejected**
- GIVEN: Pool configuration with prefill > 0 AND decode = 0 (or vice versa)
- WHEN: ValidatePoolTopology is called
- THEN: Returns error stating both must be set

**BC-PD-13: Pool sum exceeding total rejected**
- GIVEN: Pool configuration with prefill + decode > total
- WHEN: ValidatePoolTopology is called
- THEN: Returns error with actual sum and total

**BC-PD-14: Unknown decider name panics**
- GIVEN: Unrecognized decider name
- WHEN: NewDisaggregationDecider is called
- THEN: Panics with descriptive message

**BC-PD-15: CLI validation rejects invalid configs**
- GIVEN: Invalid pool configuration via CLI flags
- WHEN: runCmd executes
- THEN: logrus.Fatalf terminates with error message before simulation starts

### C) Component Interaction

```
CLI (cmd/root.go)
  ├─ Flags: --prefill-instances, --decode-instances, --pd-decider
  ├─ Validates: pool topology, decider name
  └─ Constructs: DeploymentConfig with pool fields
       │
       ▼
ClusterSimulator (sim/cluster/cluster.go)
  ├─ Fields: poolMembership, disaggregationDecider
  ├─ Methods: poolsConfigured()
  └─ Calls: ValidatePoolTopology, BuildPoolMembership
       │
       ▼
Pool Topology (sim/cluster/pool.go)
  ├─ ValidatePoolTopology(prefill, decode, total) error
  ├─ BuildPoolMembership(instances, p, d) map
  ├─ BuildPoolMembershipFromIndices(total, p, d) map
  └─ FilterSnapshotsByPool(snaps, membership, role) []

Disaggregation Decider (sim/disaggregation.go)
  ├─ Interface: DisaggregationDecider
  ├─ Implementations: NeverDisaggregate, AlwaysDisaggregate
  └─ Factory: NewDisaggregationDecider(name)

Event Pipeline (when pools configured):
  AdmissionDecisionEvent (priority 1)
    → DisaggregationDecisionEvent (priority 3)
      → RoutingDecisionEvent (priority 2)
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "~8 files changed" | 9 files changed | SCOPE_EXPANSION (added sim/bundle.go) |
| "~500 insertions" | ~650 insertions | SCOPE_EXPANSION (comprehensive tests) |

### E) Review Guide

**Scrutinize:**
- ValidatePoolTopology logic (both-or-neither rule, sum ≤ total)
- Event scheduling in AdmissionDecisionEvent.Execute (conditional on poolsConfigured())
- CLI validation in cmd/root.go (all error cases covered)

**Safe to skim:**
- PoolRole String() method (trivial)
- NeverDisaggregate/AlwaysDisaggregate (one-line returns)

**Known debt:**
- DisaggregationDecisionEvent always routes to standard path (disaggregate=true added in PR2)
- No prefix-threshold or direct-to-decode deciders (added in PR5/PR9)

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/cluster/pool.go` (98 lines)
- `sim/disaggregation.go` (90 lines, PR1 subset)
- `sim/cluster/pool_test.go` (~250 lines)
- `sim/disaggregation_test.go` (~150 lines)

**Files to modify:**
- `sim/cluster/cluster_event.go` (+80 lines)
- `sim/cluster/deployment.go` (+15 lines)
- `sim/cluster/cluster.go` (+40 lines)
- `cmd/root.go` (+70 lines)
- `sim/bundle.go` (+15 lines)

**Key decisions:**
1. Pool membership is immutable map
2. DisaggregationDecider is stateless interface
3. Event priority 3 for DisaggregationDecisionEvent
4. poolsConfigured() guards all PD code paths
5. CLI validation uses logrus.Fatalf

### G) Task Breakdown

**Task 1: Extract pool topology from pd branch**
- Copy `sim/cluster/pool.go` and `sim/cluster/pool_test.go` from pd branch
- Run tests: `go test ./sim/cluster -run "TestValidatePoolTopology|TestBuildPoolMembership" -v`
- Commit: "Add pool topology validation and membership tracking (BC-PD-2, BC-PD-3, BC-PD-9)"

**Task 2: Extract disaggregation decider from pd branch**
- Copy `sim/disaggregation.go` from pd branch, remove DirectToDecodeDecider and PrefixThresholdDecider
- Copy relevant tests from `sim/disaggregation_test.go`
- Update `sim/bundle.go` with validation functions
- Run tests: `go test ./sim -run "TestNeverDisaggregate|TestAlwaysDisaggregate" -v`
- Commit: "Add disaggregation decider interface (BC-PD-5, BC-PD-6, BC-PD-7)"

**Task 3: Add pool config fields**
- Modify `sim/cluster/deployment.go` to add PrefillInstances, DecodeInstances, PDDecider fields
- Run tests: `go test ./sim/cluster -v`
- Commit: "Add pool config fields to DeploymentConfig (BC-PD-1)"

**Task 4: Add DisaggregationDecisionEvent**
- Modify `sim/cluster/cluster_event.go` to add event type and scheduling
- Run tests: `go test ./sim/cluster -v`
- Commit: "Add DisaggregationDecisionEvent to pipeline (BC-PD-4, BC-PD-8)"

**Task 5: Integrate into ClusterSimulator**
- Modify `sim/cluster/cluster.go` to add fields and poolsConfigured() method
- Run tests: `go test ./sim/cluster -v`
- Commit: "Integrate pool topology into ClusterSimulator (BC-PD-1, BC-PD-9)"

**Task 6: Add CLI flags**
- Modify `cmd/root.go` to add flags and validation
- Test CLI: `./blis run --prefill-instances=-1` (should fail)
- Commit: "Add CLI flags and validation (BC-PD-15)"

**Task 7: Integration tests**
- Add backward compatibility test (BC-PD-10)
- Add pools-enabled test (BC-PD-1, BC-PD-4, BC-PD-5)
- Run: `go test ./sim/cluster -run "TestBackwardCompatibility|TestPoolsEnabled" -v`
- Commit: "Add integration tests for pool topology"

**Task 8: Final validation**
- Run full test suite: `go test ./...`
- Run lint: `golangci-lint run ./...`
- Build: `go build -o blis main.go`
- Test CLI: `./blis run --model qwen/qwen3-14b --num-instances=4 --prefill-instances=2 --decode-instances=2`
- Commit: "PR1 complete: pool topology and disaggregation decision pipeline"

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-PD-2 | Task 1 | Unit | TestValidatePoolTopology |
| BC-PD-3 | Task 1 | Unit | TestBuildPoolMembership |
| BC-PD-9 | Task 1 | Unit | TestBuildPoolMembership_Immutability |
| BC-PD-5 | Task 2 | Unit | TestNeverDisaggregate_AlwaysReturnsFalse |
| BC-PD-6 | Task 2 | Unit | TestAlwaysDisaggregate_AlwaysReturnsTrue |
| BC-PD-7 | Task 2 | Unit | TestNewDisaggregationDecider_Factory |
| BC-PD-14 | Task 2 | Unit | TestNewDisaggregationDecider_UnknownPanics |
| BC-PD-1 | Task 7 | Integration | TestBackwardCompatibility_PoolsDisabled |
| BC-PD-4 | Task 7 | Integration | TestPoolsEnabled_NeverDisaggregate |
| BC-PD-10 | Task 7 | Integration | TestBackwardCompatibility_PoolsDisabled |

**Invariant tests:**
- INV-PD-5 (pool membership immutability): TestBuildPoolMembership_Immutability
- INV-1 (request conservation): Verified in integration tests
- INV-6 (determinism): Verified by BC-PD-10 (byte-identical output)

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|------------|--------|------------|------|
| Event priority conflicts | Low | High | Use priority 3 (between 1 and 2) | Task 4 |
| Backward compatibility break | Medium | High | poolsConfigured() guard, integration test | Task 7 |
| Pool membership mutation | Low | Medium | Immutable map, no setters | Task 1 |
| CLI validation gaps | Medium | Low | Comprehensive validation, test all error cases | Task 6 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing shared test package
- [x] CLAUDE.md updated (new files/packages added)
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY maintained
- [x] Deviation log reviewed
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered
- [x] All contracts mapped to specific tasks
- [x] Construction site audit completed

**Antipattern rules:**
- [x] R1: No silent data loss
- [x] R2: Map keys sorted (not applicable)
- [x] R3: Numeric parameters validated
- [x] R4: Construction sites audited
- [x] R5: Transactional mutation (not applicable)
- [x] R6: No Fatalf in library
- [x] R7: Invariant tests included
- [x] R8: No exported maps
- [x] R9: YAML pointer types (not applicable)
- [x] R10: Strict YAML parsing (not applicable)
- [x] R11: Division guarded (not applicable)
- [x] R12: Golden dataset (not modified)
- [x] R13: Multi-impl interfaces (DisaggregationDecider has 2)
- [x] R14: Single-module methods
- [x] R15: No stale PR references
- [x] R16: Config by module
- [x] R17: Signal freshness (not applicable)
- [x] R18: CLI flag precedence
- [x] R19: Livelock protection (not applicable)
- [x] R20: Degenerate inputs (not applicable)
- [x] R21: No range over mutable slices
- [x] R22: Pre-check consistency
- [x] R23: Code path parity

---

## Appendix: Complete File Implementations

This appendix provides the complete implementation for each file. All code is extracted from the pd branch with minimal modifications.

### File: `sim/cluster/pool.go`

**Purpose:** Pool topology validation and membership tracking for PD disaggregation.

**Complete implementation:**

```go
package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// PoolRole identifies whether an instance serves as prefill or decode in PD disaggregation.
type PoolRole int

const (
	// PoolRolePrefill indicates the instance handles prefill (prompt processing).
	PoolRolePrefill PoolRole = iota + 1
	// PoolRoleDecode indicates the instance handles decode (token generation).
	PoolRoleDecode
)

// String returns a human-readable name for the pool role.
func (r PoolRole) String() string {
	switch r {
	case PoolRolePrefill:
		return "prefill"
	case PoolRoleDecode:
		return "decode"
	default:
		return fmt.Sprintf("PoolRole(%d)", int(r))
	}
}

// ValidatePoolTopology checks that PD pool configuration is valid.
// Returns nil if pools are disabled (both prefill and decode are 0).
// Returns an error if:
//   - prefill or decode is negative
//   - only one of prefill/decode is set (both must be set or neither)
//   - prefill + decode exceeds total instances
func ValidatePoolTopology(prefill, decode, total int) error {
	if prefill < 0 {
		return fmt.Errorf("prefill-instances must be >= 0, got %d", prefill)
	}
	if decode < 0 {
		return fmt.Errorf("decode-instances must be >= 0, got %d", decode)
	}
	// Both zero = disabled, no further checks
	if prefill == 0 && decode == 0 {
		return nil
	}
	// Both must be set when disaggregation is enabled
	if prefill == 0 || decode == 0 {
		return fmt.Errorf("both --prefill-instances and --decode-instances must be set when PD disaggregation is enabled (got prefill=%d, decode=%d)", prefill, decode)
	}
	if prefill+decode > total {
		return fmt.Errorf("prefill-instances (%d) + decode-instances (%d) = %d exceeds num-instances (%d)", prefill, decode, prefill+decode, total)
	}
	return nil
}

// BuildPoolMembership constructs an immutable map of instance ID → PoolRole.
// Instances 0..prefill-1 are assigned PoolRolePrefill, prefill..prefill+decode-1 are PoolRoleDecode.
// Caller must validate prefill+decode <= len(instances) before calling.
func BuildPoolMembership(instances []*InstanceSimulator, prefill, decode int) map[string]PoolRole {
	membership := make(map[string]PoolRole, prefill+decode)
	for i := 0; i < prefill; i++ {
		membership[string(instances[i].ID())] = PoolRolePrefill
	}
	for i := prefill; i < prefill+decode; i++ {
		membership[string(instances[i].ID())] = PoolRoleDecode
	}
	return membership
}

// BuildPoolMembershipFromIndices constructs a pool membership map from instance indices.
// Uses the same instance ID naming convention as NewClusterSimulator: "instance_N".
// This variant does not require constructed instances, enabling pool membership
// computation before instance construction (needed for per-pool config resolution).
// Caller must validate prefill+decode <= total before calling.
func BuildPoolMembershipFromIndices(total, prefill, decode int) map[string]PoolRole {
	membership := make(map[string]PoolRole, prefill+decode)
	for i := 0; i < prefill; i++ {
		membership[fmt.Sprintf("instance_%d", i)] = PoolRolePrefill
	}
	for i := prefill; i < prefill+decode; i++ {
		membership[fmt.Sprintf("instance_%d", i)] = PoolRoleDecode
	}
	return membership
}

// FilterSnapshotsByPool returns only the snapshots for instances in the given pool role.
// Order is preserved (stable relative to the input slice).
func FilterSnapshotsByPool(snapshots []sim.RoutingSnapshot, membership map[string]PoolRole, role PoolRole) []sim.RoutingSnapshot {
	filtered := make([]sim.RoutingSnapshot, 0, len(snapshots))
	for _, snap := range snapshots {
		if membership[snap.ID] == role {
			filtered = append(filtered, snap)
		}
	}
	return filtered
}
```

### File: `sim/disaggregation.go` (PR1 subset)

**Purpose:** Disaggregation decision interface and basic implementations.

**Complete implementation:**

```go
package sim

import "fmt"

// DisaggregationDecision encapsulates the prefill-decode disaggregation decision for a request.
type DisaggregationDecision struct {
	Disaggregate bool // true = route to prefill pool, false = route to shared/decode pool
}

// DisaggregationDecider decides whether a request should be disaggregated
// (sent to a dedicated prefill pool) or handled by the default routing pipeline.
// Used by ClusterSimulator's event pipeline when pool topology is configured.
//
// Implementations must not read Request.OutputTokens (INV-9 oracle boundary);
// use len(req.InputTokens) and req.MaxOutputLen only.
//
// req is guaranteed non-nil; implementations may assume a non-nil pointer.
type DisaggregationDecider interface {
	Decide(req *Request) DisaggregationDecision
}

// NeverDisaggregate always returns Disaggregate=false.
// Default decider when PD disaggregation is not configured.
type NeverDisaggregate struct{}

func (n *NeverDisaggregate) Decide(_ *Request) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: false}
}

// AlwaysDisaggregate always returns Disaggregate=true.
// Test-oriented decider for validating disaggregation pipeline wiring.
type AlwaysDisaggregate struct{}

func (a *AlwaysDisaggregate) Decide(_ *Request) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: true}
}

// NewDisaggregationDecider creates a disaggregation decider by name.
// Valid names are defined in validDisaggregationDeciders (bundle.go).
// An empty string defaults to NeverDisaggregate.
// Panics on unrecognized names.
// Note: "prefix-threshold" and "direct-to-decode" are added in PR5/PR9.
func NewDisaggregationDecider(name string) DisaggregationDecider {
	if !IsValidDisaggregationDecider(name) {
		panic(fmt.Sprintf("unknown disaggregation decider %q", name))
	}
	switch name {
	case "", "never":
		return &NeverDisaggregate{}
	case "always":
		return &AlwaysDisaggregate{}
	default:
		panic(fmt.Sprintf("unhandled disaggregation decider %q", name))
	}
}
```

**Key implementation notes:**
- DirectToDecodeDecider and PrefixThresholdDecider removed (added in PR5/PR9)
- Factory only handles "", "never", "always" in PR1
- Interface contract enforces INV-9 (no reading OutputTokens)

---

## Implementation Notes

**Extraction strategy:**
1. Copy complete files from pd branch using `git show pd:path/to/file.go`
2. For disaggregation.go, manually remove PR5/PR9 deciders
3. For tests, filter to PR1-relevant tests only
4. No new code - all implementations already tested in pd branch

**Validation strategy:**
1. Run tests after each task
2. Run lint after each task
3. Final integration test verifies backward compatibility
4. CLI validation test covers all error cases

**Merge strategy:**
1. Create branch from main: `git checkout -b pr1-pool-topology`
2. Implement tasks 1-8 sequentially
3. Push to fork: `git push origin pr1-pool-topology`
4. Create PR to upstream using `gh pr create`

---

## Success Criteria

- [x] All tests pass (`go test ./...`)
- [x] Lint passes (`golangci-lint run ./...`)
- [x] Build succeeds (`go build -o blis main.go`)
- [x] CLI validation works (tested manually)
- [x] Backward compatibility verified (BC-PD-10)
- [x] All contracts implemented and tested
- [x] No simulation behavior change when pools disabled
- [x] Documentation updated (CLAUDE.md)

---

**End of PR1 Implementation Plan**
