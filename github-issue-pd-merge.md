# GitHub Issue: PD Branch Merge Plan

## Issue Title
[Epic] Prefill-Decode Disaggregation Feature - Merge Plan (10 PRs)

## Labels
- `epic`
- `enhancement`
- `disaggregation`
- `architecture`

## Issue Body

---

## Overview

This epic tracks the merge of the `pd` branch into `main` through a series of 10 focused pull requests. The PD branch implements prefill-decode disaggregation simulation capability, allowing BLIS to model architectures where prefill (prompt processing) and decode (token generation) run on separate instance pools.

**Total scope:** ~15,768 insertions, ~34,860 deletions across 210 files  
**Strategy:** Split into 10 focused PRs organized in 2 phases, each adding standalone value

---

## Phase 1: Core Disaggregation Pipeline (PRs 1-5)

### PR1: Pool Topology and Disaggregation Decision Pipeline

**Status:** 🔲 Not Started

**Goal:** Establish the foundation for PD disaggregation without changing simulation behavior.

**What it adds:**
- Pool topology validation and membership tracking (`sim/cluster/pool.go`)
- `DisaggregationDecider` interface with `NeverDisaggregate` and `AlwaysDisaggregate` implementations (`sim/disaggregation.go`)
- `DisaggregationDecisionEvent` in cluster event pipeline (stub that always routes locally)
- CLI flags: `--prefill-instances`, `--decode-instances`, `--pd-decider`
- Pool membership map construction and validation

**Files changed:** ~8 files
- `sim/cluster/pool.go` (new)
- `sim/disaggregation.go` (new)
- `sim/cluster/cluster_event.go` (add DisaggregationDecisionEvent)
- `sim/cluster/deployment.go` (add pool config fields)
- `cmd/root.go` (add CLI flags)
- Tests for pool topology and disaggregation interface

**Why merge this first:** Establishes the architectural foundation. All code paths default to `NeverDisaggregate`, so simulation behavior is unchanged (backward compatible). Enables incremental development of subsequent PRs.

**Validation:** Run existing test suite - all tests pass, output byte-identical to main branch.

**Insertions/Deletions:** ~500 insertions, ~50 deletions

---

### PR2: End-to-End Disaggregated Request Flow

**Status:** 🔲 Not Started  
**Dependencies:** PR1

**Goal:** Implement the complete disaggregated request lifecycle with KV transfer simulation.

**What it adds:**
- Request splitting into prefill and decode sub-requests
- `ParentRequest` tracking structure (`sim/cluster/parent_request.go`)
- Four new event types: `PrefillRoutingEvent`, `KVTransferStartedEvent`, `KVTransferCompletedEvent`, `DecodeRoutingEvent` (`sim/cluster/pd_events.go`)
- Pool-filtered routing (prefill pool → decode pool)
- KV transfer duration simulation based on block count and bandwidth
- Per-pool routing scorer configuration (`--prefill-routing-scorers`, `--decode-routing-scorers`)
- CLI flags: `--pd-transfer-bandwidth`, `--pd-transfer-base-latency`, `--pd-kv-bytes-per-token`
- Decode-only batch formation path (`sim/batch_formation.go`)
- `EnqueueDecodeSubRequest()` method in simulator

**Files changed:** ~15 files
- `sim/cluster/pd_events.go` (new, ~350 lines)
- `sim/cluster/parent_request.go` (new, ~50 lines)
- `sim/cluster/cluster.go` (parent tracking, completion detection)
- `sim/cluster/cluster_event.go` (DisaggregationDecisionEvent bifurcation)
- `sim/cluster/deployment.go` (transfer config, per-pool scorers)
- `sim/batch_formation.go` (decode-only path)
- `sim/simulator.go` (EnqueueDecodeSubRequest method)
- `cmd/root.go` (CLI flags)
- Comprehensive tests for disaggregated flow

**Why merge this second:** This is the core functionality. After PR2, users can run basic disaggregated simulations and observe end-to-end request completion through prefill→transfer→decode. Enables capacity planning experiments.

**Validation:** New tests verify INV-PD-1 through INV-PD-5 (KV completeness, pool exclusivity, transfer conservation, phase causality, pool stability). Existing tests still pass.

**Insertions/Deletions:** ~2,500 insertions, ~500 deletions

---

### PR3: Disaggregation-Aware Metrics

**Status:** 🔲 Not Started  
**Dependencies:** PR2

**Goal:** Add observability for disaggregated simulations through specialized metrics.

**What it adds:**
- `PDMetrics` struct with parent TTFT, transfer duration, per-pool throughput, load imbalance ratio
- `CollectPDMetrics()` function (`sim/cluster/pd_metrics.go`)
- `ParentRequests()` and `PerInstanceMetricsByID()` accessors on `ClusterSimulator`
- `=== PD Metrics ===` output section in CLI (guarded, only prints when disaggregation active)
- Backward compatibility: `RawMetrics.PD` is nil when disaggregation disabled

**Files changed:** ~6 files
- `sim/cluster/pd_metrics.go` (new, ~210 lines)
- `sim/cluster/metrics.go` (extend RawMetrics with PD field)
- `sim/cluster/cluster.go` (add accessor methods)
- `cmd/root.go` (call CollectPDMetrics, print PD section)
- Tests for metrics accuracy and backward compatibility

**Why merge this third:** Closes the observability gap. Users can now see transfer overhead, parent-level TTFT, and per-pool throughput. Essential for understanding disaggregated simulation results.

**Validation:** Tests verify parent TTFT accuracy, transfer duration correctness, per-pool throughput calculation, load imbalance ratio. Non-disaggregated simulations produce nil PD metrics (backward compatible).

**Insertions/Deletions:** ~400 insertions, ~50 deletions

---

### PR4: Disaggregation Trace Instrumentation

**Status:** 🔲 Not Started  
**Dependencies:** PR2

**Goal:** Add decision trace records for disaggregation pipeline events.

**What it adds:**
- Four new trace record types: `DisaggregationRecord`, `PrefillRoutingRecord`, `DecodeRoutingRecord`, `KVTransferRecord` (`sim/trace/record.go`)
- Recording methods in `SimulationTrace`: `RecordDisaggregation()`, `RecordPrefillRouting()`, `RecordDecodeRouting()`, `RecordKVTransfer()`
- Instrumentation calls in event handlers (controlled by existing `--trace-level` flag)
- Counterfactual analysis for pool routing decisions (reuses existing `computeCounterfactual()`)

**Files changed:** ~5 files
- `sim/trace/record.go` (add 4 new record types)
- `sim/trace/trace.go` (add 4 recording methods)
- `sim/cluster/cluster_event.go` (instrument DisaggregationDecisionEvent)
- `sim/cluster/pd_events.go` (instrument routing and transfer events)
- Tests for trace coverage and backward compatibility

**Why merge this fourth:** Pure observability layer. Enables debugging of disaggregation decisions and routing choices. No simulation behavior changes.

**Validation:** Tests verify record coverage for disaggregated simulations, isolation for non-disaggregated simulations, counterfactual analysis for pool routing. Existing trace records unchanged.

**Insertions/Deletions:** ~300 insertions, ~20 deletions

---

### PR5: Prefix-Aware Disaggregation Decider

**Status:** 🔲 Not Started  
**Dependencies:** PR2

**Goal:** Add intelligent disaggregation decision based on prefix cache hit rate.

**What it adds:**
- `PrefixThresholdDecider` implementation using router-side prefix cache
- `DisaggregationObserver` interface for cache-aware decisions
- Global virtual instance for prefix cache tracking
- CLI flag: `--pd-prefix-threshold` (default 512 non-cached tokens)
- Integration with existing `PrefixCacheIndex` from PR18

**Files changed:** ~4 files
- `sim/disaggregation.go` (add PrefixThresholdDecider)
- `sim/cluster/cluster.go` (wire up observer)
- `cmd/root.go` (add CLI flag)
- Tests for prefix-aware decision logic

**Why merge this fifth:** Adds production-ready disaggregation policy. Users can now run realistic simulations where disaggregation decisions consider prefix cache efficiency. Completes Phase 1 core functionality.

**Validation:** Tests verify threshold-based decisions, cache hit rate influence, backward compatibility with other deciders.

**Insertions/Deletions:** ~200 insertions, ~20 deletions

---

## Phase 2: Advanced Features (PRs 6-9)

### PR6: Per-Pool Hardware Configuration

**Status:** 🔲 Not Started  
**Dependencies:** PR5

**Goal:** Enable heterogeneous hardware across prefill and decode pools.

**What it adds:**
- `PoolOverrides` type for per-pool config (`sim/cluster/resolve.go`)
- `ResolvePoolConfig()` function for config resolution
- Per-pool CLI flags: `--prefill-tp`, `--decode-tp`, `--prefill-hardware`, `--decode-hardware`, `--prefill-latency-model`, `--decode-latency-model`, `--prefill-max-model-len`, `--decode-max-model-len`
- Config resolution logic in `DeploymentConfig.ToSimConfig()`
- INV-P2-1: Pool-config consistency invariant

**Files changed:** ~5 files
- `sim/cluster/resolve.go` (new, ~85 lines)
- `sim/cluster/deployment.go` (add override fields, resolution logic)
- `cmd/root.go` (add per-pool CLI flags)
- Tests for config resolution and validation

**Why merge this sixth:** Enables realistic heterogeneous deployments (e.g., H100 for prefill, L40S for decode). Critical for cost optimization experiments.

**Validation:** Tests verify config resolution for all pool roles, default fallback behavior, validation of conflicting configs.

**Insertions/Deletions:** ~600 insertions, ~50 deletions

---

### PR7: KV Transfer Contention Model

**Status:** 🔲 Not Started  
**Dependencies:** PR5

**Goal:** Model bandwidth contention when multiple transfers occur simultaneously.

**What it adds:**
- Fair-share bandwidth contention model (INV-P2-2)
- Active transfer tracking in `ClusterSimulator`
- Dynamic bandwidth allocation: `effective_bandwidth = total_bandwidth / active_transfers`
- CLI flag: `--pd-transfer-contention` (bool, default false)
- Transfer queue depth tracking for metrics

**Files changed:** ~4 files
- `sim/cluster/cluster.go` (transfer tracking, bandwidth allocation)
- `sim/cluster/pd_events.go` (update transfer duration calculation)
- `sim/cluster/pd_metrics.go` (add transfer queue depth metrics)
- `cmd/root.go` (add CLI flag)
- Tests for contention model and conservation

**Why merge this seventh:** Adds realism for high-throughput scenarios. Users can model network bottlenecks and understand when transfer bandwidth becomes a constraint.

**Validation:** Tests verify fair-share allocation, transfer conservation (INV-PD-3), single-transfer full bandwidth usage.

**Insertions/Deletions:** ~300 insertions, ~30 deletions

---

### PR8: Prefill-Decode Interference Model

**Status:** 🔲 Not Started  
**Dependencies:** PR5

**Goal:** Model performance degradation when prefill and decode co-locate on decode instances.

**What it adds:**
- `InterferenceLatencyModel` wrapper applying slowdown multiplier to StepTime (`sim/cluster/interference.go`)
- Phase composition detection (prefill-dominant vs decode-dominant batches)
- CLI flags: `--pd-interference-prefill`, `--pd-interference-decode` (float64, default 0)
- INV-P2-3: Interference monotonicity invariant (multiplier ≥ 1.0)
- INV-P2-4: Decode-targeted routing for non-disaggregated requests

**Files changed:** ~5 files
- `sim/cluster/interference.go` (new, ~160 lines)
- `sim/cluster/cluster.go` (wrap latency model for decode instances)
- `sim/cluster/deployment.go` (add interference config)
- `cmd/root.go` (add CLI flags)
- Tests for interference calculation and invariants

**Why merge this eighth:** Models the key tradeoff of disaggregation - eliminating interference vs adding transfer overhead. Essential for understanding when disaggregation provides net benefit.

**Validation:** Tests verify multiplier calculation, phase detection, monotonicity invariant, decode-targeted routing.

**Insertions/Deletions:** ~500 insertions, ~40 deletions

---

### PR9: Direct-to-Decode Decider

**Status:** 🔲 Not Started  
**Dependencies:** PR5

**Goal:** Add short-prompt bypass for disaggregation decision.

**What it adds:**
- `DirectToDecodeDecider` implementation bypassing disaggregation for short prompts
- CLI flag: `--pd-direct-decode-threshold` (default 256 input tokens)
- Optimization for latency-sensitive short requests
- Integration with decode pool routing (INV-P2-4)

**Files changed:** ~3 files
- `sim/disaggregation.go` (add DirectToDecodeDecider)
- `cmd/root.go` (add CLI flag)
- Tests for threshold-based bypass logic

**Why merge this ninth:** Adds latency optimization for short requests. Users can model hybrid policies where only long prompts are disaggregated.

**Validation:** Tests verify threshold-based decisions, decode pool routing for bypassed requests, backward compatibility.

**Insertions/Deletions:** ~150 insertions, ~20 deletions

---

## Bug Fixes and Cleanup (PR10)

### PR10: Bug Fixes and Edge Case Handling

**Status:** 🔲 Not Started  
**Dependencies:** PR1-9

**Goal:** Address edge cases and bugs discovered during PD development.

**What it includes:**
- Fix: Guard `processCompletions` against ProgressIndex overshoot in PD mode (#687)
- Fix: Guard decode-only batch path against zero-input requests (#628)
- Cleanup: Remove obsolete code paths and documentation
- Update: Documentation and examples for PD disaggregation

**Files changed:** ~10 files
- `sim/simulator.go` (completion guard)
- `sim/batch_formation.go` (zero-input guard)
- `examples/pd-disaggregation-demo.yaml` (new example)
- Documentation updates in `docs/guide/cluster.md`, `docs/reference/configuration.md`
- Cleanup of obsolete test helpers

**Why merge this last:** Consolidates bug fixes and cleanup. Ensures the feature is production-ready with comprehensive documentation.

**Validation:** All tests pass, documentation is up-to-date, example configurations work correctly.

**Insertions/Deletions:** ~300 insertions, ~200 deletions

---

## File Change Summary

| PR | Files Changed | Insertions | Deletions | Key Files |
|----|---------------|------------|-----------|-----------|
| PR1 | ~8 | ~500 | ~50 | pool.go, disaggregation.go, cluster_event.go |
| PR2 | ~15 | ~2,500 | ~500 | pd_events.go, parent_request.go, cluster.go |
| PR3 | ~6 | ~400 | ~50 | pd_metrics.go, metrics.go |
| PR4 | ~5 | ~300 | ~20 | record.go, trace.go |
| PR5 | ~4 | ~200 | ~20 | disaggregation.go |
| PR6 | ~5 | ~600 | ~50 | resolve.go, deployment.go |
| PR7 | ~4 | ~300 | ~30 | cluster.go, pd_events.go |
| PR8 | ~5 | ~500 | ~40 | interference.go, cluster.go |
| PR9 | ~3 | ~150 | ~20 | disaggregation.go |
| PR10 | ~10 | ~300 | ~200 | simulator.go, batch_formation.go, docs |

**Total:** ~65 files directly modified (many files appear in multiple PRs for incremental changes)

---

## Merge Strategy

### Prerequisites
- ✅ All PRs must pass CI (build, lint, test)
- ✅ Each PR must include comprehensive tests
- ✅ Each PR must maintain backward compatibility (non-disaggregated simulations unchanged)
- ✅ Each PR must update relevant documentation

### Review Process
1. **Phase 1 PRs (1-5):** Sequential review and merge. Each PR depends on the previous.
2. **Phase 2 PRs (6-9):** Can be reviewed in parallel after PR5 merges. Each is independent.
3. **PR10:** Final cleanup after all feature PRs merge.

### Rollback Strategy
- Each PR is independently revertible
- Phase 1 PRs form a dependency chain - reverting PR2 requires reverting PR3-5
- Phase 2 PRs are independent - can revert individually without affecting others

### Testing Strategy
- **Unit tests:** Each PR includes focused unit tests for new functionality
- **Integration tests:** PR2, PR3, PR4 include end-to-end disaggregation tests
- **Regression tests:** All PRs must pass existing test suite (backward compatibility)
- **Golden dataset:** PR2 regenerates golden dataset with disaggregation scenarios

---

## Success Criteria

### Phase 1 Complete When:
- ✅ Users can run disaggregated simulations with basic policies
- ✅ Metrics and traces provide full observability
- ✅ Prefix-aware disaggregation policy available

### Phase 2 Complete When:
- ✅ Heterogeneous hardware configurations supported
- ✅ Contention and interference models available
- ✅ Latency optimization policies implemented

### Each PR Must Satisfy:
1. ✅ All CI checks pass (build, lint, test)
2. ✅ Backward compatibility maintained (non-disaggregated simulations unchanged)
3. ✅ Comprehensive test coverage (unit + integration)
4. ✅ Documentation updated (inline comments + user guides)
5. ✅ Standalone value (PR adds usable functionality, not just scaffolding)

---

## Related Documentation

- Macro plans: `pr2-pd-request-flow-plan.md`, `pr3-pd-metrics-plan.md`, `pr4-pd-traces-plan.md`
- Design guidelines: `docs/contributing/templates/design-guidelines.md`
- Invariants: `docs/contributing/standards/invariants.md` (INV-PD-1 through INV-PD-5, INV-P2-1 through INV-P2-4)

---

## Notes

- The PD branch contains ~34,860 deletions, many from cleanup of obsolete code paths and documentation. These deletions are distributed across PRs where relevant.
- Some files (e.g., `cluster.go`, `deployment.go`, `cmd/root.go`) appear in multiple PRs for incremental additions. Each PR adds a focused set of changes to these files.
- The macro plans (`pr2-pd-request-flow-plan.md`, `pr3-pd-metrics-plan.md`, `pr4-pd-traces-plan.md`) provide detailed implementation guidance for Phase 1 PRs.
- Phase 2 PRs build on the foundation but are independently valuable - users can adopt them selectively based on their simulation needs.

---

## Tracking Checklist

- [ ] PR1: Pool Topology and Disaggregation Decision Pipeline
- [ ] PR2: End-to-End Disaggregated Request Flow
- [ ] PR3: Disaggregation-Aware Metrics
- [ ] PR4: Disaggregation Trace Instrumentation
- [ ] PR5: Prefix-Aware Disaggregation Decider
- [ ] PR6: Per-Pool Hardware Configuration
- [ ] PR7: KV Transfer Contention Model
- [ ] PR8: Prefill-Decode Interference Model
- [ ] PR9: Direct-to-Decode Decider
- [ ] PR10: Bug Fixes and Edge Case Handling