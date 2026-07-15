# `least-ttft` Disaggregation Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `least-ttft` decision-rule mode to the EDPP decider — disaggregate iff `ttftP < ttftD` — reusing EDPP's exact predicted TTFTs but bypassing the drift-plus-penalty machinery, to isolate whether EDPP's prefill-bound win is drift-plus-penalty-specific or generic dynamic least-TTFT routing.

**Architecture:** A single decision branch on the EDPP decider. A new `EDPPConfig.Rule` field (default `"dpp"`; new value `"least-ttft"`) selects, at the reduced decision site (`sim/edpp.go:623`), between the existing `lhs > rhs` and `ttftP < ttftD`. All estimation above line 623 (rollforward admission `T̂`, work model, chunk terms, scorer-selected decode instance) is shared and unchanged, so the ONLY difference from reduced-EDPP is the final comparison. A `--edpp-rule` CLI flag threads it through `DeploymentConfig.EDPPRule`, mirroring `--edpp-joint`.

**Tech Stack:** Go 1.22, cobra CLI.

## Global Constraints

- Branch: `feat/edpp-estimator-validation`. Spec: `docs/superpowers/specs/2026-07-15-edpp-least-ttft-baseline-design.md`.
- **Reduced path only.** The joint (`--edpp-joint`) path is untouched. `--edpp-rule least-ttft` combined with `--edpp-joint` is a hard `logrus.Fatalf` at the CLI boundary (unsupported combination).
- **Backward-compatible / INV-6 byte-identity:** the default rule is `"dpp"`, and an empty `Rule` (`""`, how existing tests construct `EDPPConfig`) behaves as `"dpp"`. With `Rule ∈ {"", "dpp"}` every run is byte-identical to today. The decision branch is the only behavioral change and it is inert unless `Rule == "least-ttft"`.
- **`least-ttft` uses ONLY `ttftP`/`ttftD`** (the values already computed at `sim/edpp.go:602-603`). It does not read the backlog/balance drift terms (`balanceTermD/P`), the SLO virtual queues (`z_ttft`/`z_itl`), or the standalone transfer penalty. `ttftP` already includes the transfer latency `c_xfer`, so transfer stays counted as latency.
- **Ties → local:** `Disaggregate = (ttftP < ttftD)` (strict), matching reduced-EDPP's `Disaggregate = lhs > rhs` (ties → local).
- **Trace consistency:** the `EDPPDecisionTrace.Disaggregate` field (`sim/edpp.go:634`) must reflect the rule actually used, not always `lhs > rhs`.
- Go: `go build ./... && go test ./sim/... ./cmd/...`; gofmt before every commit. Commit ONLY the files each task names (never `git add -A`; ~12 untracked scratch files/dirs must stay untracked).

**Existing config-flow pattern to mirror (from `--edpp-joint`):** `cmd/root.go` var `edppJoint` (line 166) + flag registration (line 1348) + set on the `DeploymentConfig` literal (near line 1974, beside `EDPPTAdmEstimator: edppTAdmEstimator`) → `DeploymentConfig.EDPPJoint` (`sim/cluster/deployment.go:90`) → `EDPPConfig{Joint: config.EDPPJoint}` (`sim/cluster/cluster.go:460`).

---

### Task 1: `Rule` field + decision branch on the EDPP decider (`sim/edpp.go`)

**Files:**
- Modify: `sim/edpp.go` (`EDPPConfig` struct ~line 82; `EDPPConfig.validate()` line 155; `EDPPDecider` struct ~line 293; `NewEDPPDecider` ~line 353; reduced decision site lines 623 + 634)
- Test: `sim/edpp_test.go`

**Interfaces:**
- Produces: `EDPPConfig.Rule string` (default meaning `"dpp"`; also accepts `"least-ttft"`); `EDPPDecider.rule string`. Task 2 (cmd/cluster) sets `EDPPConfig.Rule` from `DeploymentConfig.EDPPRule`.

- [ ] **Step 1: Write the failing tests**

Add to `sim/edpp_test.go`. Reuse the existing reduced-path test helpers — grep for how current reduced tests build the decider and state (`NewEDPPDecider(defaultTestEDPPConfig(), newTestAffineModel(), nil, <prefill closure>)`, and the `decodeState(...)` / inline `&RouterState{SelectedInstance:..., Snapshots:[]RoutingSnapshot{...}}` pattern). The tests below name helper stand-ins; replace with the real ones in situ.

```go
// least-ttft disaggregates when local decode is congested (ttftD high) and stays
// local when the prefill pool is congested (ttftP high) — decided purely on predicted TTFT.
func TestDecideReduced_LeastTTFT_DecidesOnPredictedTTFT(t *testing.T) {
	// Prefill pool EMPTY/idle -> ttftP low; decode instance heavily loaded -> ttftD high => disaggregate.
	cfg := defaultTestEDPPConfig()
	cfg.Rule = "least-ttft"
	d := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot {
		return []RoutingSnapshot{{ID: "p0", BatchSize: 0, ResidentPrefillTokens: 0}}
	})
	reqBusy := newTestRequestInput(600)                       // uncached prompt so a_p > 0
	stateBusyDecode := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{
		{ID: "d0", BatchSize: 64, KvTokensInUse: 60000, QueueDepth: 40}, // congested decode
	}}
	if !d.Decide(reqBusy, stateBusyDecode).Disaggregate {
		t.Fatal("least-ttft: expected Disaggregate=true when local decode is congested (ttftD > ttftP)")
	}

	// Now flip it: idle decode, congested prefill pool -> ttftP high => stay local.
	d2 := NewEDPPDecider(cfg, newTestAffineModel(), nil, func() []RoutingSnapshot {
		return []RoutingSnapshot{{ID: "p0", BatchSize: 64, ResidentPrefillTokens: 120000, QueueDepth: 40}}
	})
	stateIdleDecode := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{
		{ID: "d0", BatchSize: 0, KvTokensInUse: 0, QueueDepth: 0},
	}}
	if d2.Decide(reqBusy, stateIdleDecode).Disaggregate {
		t.Fatal("least-ttft: expected Disaggregate=false when prefill pool is congested (ttftP > ttftD)")
	}
}

// The KEY guard: least-ttft ignores the SLO virtual queues. Blowing up z must NOT change
// its decision, whereas under dpp the same z DOES change the decision.
func TestDecideReduced_LeastTTFT_IgnoresVirtualQueues(t *testing.T) {
	req := newTestRequestInput(600)
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{
		{ID: "d0", BatchSize: 8, KvTokensInUse: 4000, QueueDepth: 2},
	}}
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "p0", BatchSize: 4, ResidentPrefillTokens: 2000}} }

	cfgLT := defaultTestEDPPConfig()
	cfgLT.Rule = "least-ttft"
	base := NewEDPPDecider(cfgLT, newTestAffineModel(), nil, prefill)
	baseDec := base.Decide(req, state).Disaggregate

	withZ := NewEDPPDecider(cfgLT, newTestAffineModel(), nil, prefill)
	withZ.zByClass[req.SLOClass] = &edppClassState{zTTFT: 1e12, zITL: 1e12} // huge SLO deficit
	if withZ.Decide(req, state).Disaggregate != baseDec {
		t.Fatal("least-ttft decision changed when z virtual queues were inflated (machinery leaked in)")
	}

	// Contrast: under dpp the same huge z SHOULD move the decision (proves the guard is meaningful).
	cfgDPP := defaultTestEDPPConfig() // Rule "" == dpp
	dppNoZ := NewEDPPDecider(cfgDPP, newTestAffineModel(), nil, prefill)
	dppZ := NewEDPPDecider(cfgDPP, newTestAffineModel(), nil, prefill)
	dppZ.zByClass[req.SLOClass] = &edppClassState{zTTFT: 1e12, zITL: 1e12}
	if dppNoZ.Decide(req, state).Disaggregate == dppZ.Decide(req, state).Disaggregate {
		t.Fatal("dpp decision did NOT change under huge z — the contrast guard is vacuous; retune the state")
	}
}

// Unknown rule is rejected at construction (R3 panic style).
func TestNewEDPPDecider_RejectsUnknownRule(t *testing.T) {
	cfg := defaultTestEDPPConfig()
	cfg.Rule = "bogus"
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for unknown EDPPConfig.Rule")
		}
	}()
	_ = NewEDPPDecider(cfg, newTestAffineModel(), nil, nil)
}

// Default (Rule "") is byte-identical to explicit "dpp".
func TestDecideReduced_EmptyRuleEqualsDPP(t *testing.T) {
	req := newTestRequestInput(600)
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{{ID: "d0", BatchSize: 8, KvTokensInUse: 4000}}}
	prefill := func() []RoutingSnapshot { return []RoutingSnapshot{{ID: "p0", ResidentPrefillTokens: 2000}} }
	e := defaultTestEDPPConfig()               // Rule ""
	dfl := defaultTestEDPPConfig(); dfl.Rule = "dpp"
	if NewEDPPDecider(e, newTestAffineModel(), nil, prefill).Decide(req, state) !=
		NewEDPPDecider(dfl, newTestAffineModel(), nil, prefill).Decide(req, state) {
		t.Fatal(`Rule "" must behave identically to "dpp"`)
	}
}
```
Confirm in situ: the request constructor (`newTestRequestInput` stand-in — use whatever existing reduced tests use to build a `*Request` with a non-empty uncached prompt, e.g. the helper behind `decodeState` tests), and that `edppClassState` has fields `zTTFT, zITL` (it does — `sim/edpp.go:267-271`). If the exact congested/idle magnitudes don't flip the decision with the real affine test model, adjust the snapshot numbers until `ttftD`/`ttftP` cross — the assertions (disagg when decode congested; local when prefill congested; z-invariance) are what must hold.

- [ ] **Step 2: Run to verify they fail**

Run: `go test ./sim/ -run 'TestDecideReduced_LeastTTFT|TestNewEDPPDecider_RejectsUnknownRule|TestDecideReduced_EmptyRuleEqualsDPP' -v`
Expected: FAIL — `Rule` field undefined (compile error).

- [ ] **Step 3: Add the `Rule` config field + decider field**

In `EDPPConfig` (`sim/edpp.go`, after `Joint bool` at line 82):
```go
	Rule              string                // reduced-path decision rule: "" / "dpp" (drift-plus-penalty, default) or "least-ttft" (disaggregate iff ttftP < ttftD; bypasses the drift/z/V machinery). Design 2026-07-15.
```
In the `EDPPDecider` struct (near `joint bool`, ~line 293):
```go
	// rule selects the reduced-path decision: "" / "dpp" => lhs > rhs (drift-plus-penalty);
	// "least-ttft" => ttftP < ttftD. Estimation is identical; only the final comparison differs.
	rule string
```

- [ ] **Step 4: Validate the rule + store it**

In `EDPPConfig.validate()` (`sim/edpp.go:155`), add (matching the method's existing panic-on-invalid style; confirm whether it panics or returns error and match it):
```go
	switch c.Rule {
	case "", "dpp", "least-ttft":
	default:
		panic(fmt.Sprintf("EDPPConfig.Rule must be \"\", \"dpp\", or \"least-ttft\", got %q", c.Rule))
	}
```
In `NewEDPPDecider`'s struct literal (beside `joint: cfg.Joint`):
```go
		rule: cfg.Rule,
```

- [ ] **Step 5: Add the decision branch**

In the reduced `Decide` path, replace the decision at `sim/edpp.go:623`. Currently:
```go
	dec := DisaggregationDecision{Disaggregate: lhs > rhs}
```
Change to compute the flag once (so the trace uses the same value) and branch on the rule:
```go
	disagg := lhs > rhs
	if d.rule == "least-ttft" {
		disagg = ttftP < ttftD // bypass drift/z/V; decide purely on predicted TTFT (ttftP already includes c_xfer)
	}
	dec := DisaggregationDecision{Disaggregate: disagg}
```
Then in the trace struct just below (`sim/edpp.go:634`), replace `Disaggregate: lhs > rhs,` with `Disaggregate: disagg,` so the trace reflects the active rule. (`ttftP` and `ttftD` are in scope — computed at lines 602-603.)

- [ ] **Step 6: Run to verify tests pass + no regression**

Run: `go test ./sim/ -run 'TestDecideReduced_LeastTTFT|TestNewEDPPDecider_RejectsUnknownRule|TestDecideReduced_EmptyRuleEqualsDPP' -v && go test ./sim/...`
Expected: PASS; all existing `sim` tests still green (byte-identical under `Rule ∈ {"", "dpp"}`). gofmt `sim/edpp.go` `sim/edpp_test.go`.

- [ ] **Step 7: Commit**

```bash
git add sim/edpp.go sim/edpp_test.go
git commit -m "feat(edpp): least-ttft reduced decision rule (ttftP < ttftD; bypasses drift/z/V)"
```

---

### Task 2: `--edpp-rule` CLI flag wiring (`cmd/root.go`, `sim/cluster`)

**Files:**
- Modify: `cmd/root.go` (flag var + registration + `DeploymentConfig` literal + the `least-ttft`+`--edpp-joint` reject); `sim/cluster/deployment.go` (new `EDPPRule` field); `sim/cluster/cluster.go` (`EDPPConfig{Rule: config.EDPPRule}`)
- Test: `cmd/` integration test

**Interfaces:**
- Consumes: `EDPPConfig.Rule` (Task 1).
- Produces: a running EDPP decider whose reduced rule is set from `--edpp-rule`.

- [ ] **Step 1: Write the failing integration test**

Add a `cmd/` test (mirror an existing small `cmd` run test) that runs `blis run` on a tiny 1P2D config with `--pd-decider edpp --edpp-rule least-ttft` and asserts it completes without error and produces metrics (i.e. the flag is accepted and threads through). Then a second case: `--edpp-rule least-ttft --edpp-joint` together must fail (non-zero exit / fatal). Skeleton (match the real cmd-test harness in situ — several `cmd/*_test.go` build and run the command in a child process or via `runCmd`):
```go
func TestEDPPRule_LeastTTFT_AcceptedAndJointRejected(t *testing.T) {
	// (1) --edpp-rule least-ttft with reduced EDPP: runs, produces metrics.
	// (2) --edpp-rule least-ttft --edpp-joint: rejected (fatal / non-zero).
	// Use the same run-invocation pattern as the sibling cmd tests.
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./cmd/ -run TestEDPPRule -v`
Expected: FAIL — `--edpp-rule` is an unknown flag.

- [ ] **Step 3: Add the flag var + registration**

In `cmd/root.go`, near `edppJoint` (line 166):
```go
	edppRule               string        // EDPP reduced-path decision rule (--edpp-rule): dpp (default) | least-ttft
```
Near the `--edpp-joint` registration (line 1348):
```go
	cmd.Flags().StringVar(&edppRule, "edpp-rule", "dpp", "EDPP reduced-path decision rule: dpp (drift-plus-penalty, default) | least-ttft (disaggregate iff predicted-TTFT-disagg < predicted-TTFT-local; bypasses the drift/z/V machinery). Only used with --pd-decider edpp; incompatible with --edpp-joint.")
```

- [ ] **Step 4: Reject the unsupported combination**

Where the run command validates EDPP flags (near where `edppJoint`/`edppTAdmEstimator` are consumed; grep for `edppJoint` usage in the run path, or place it just before building `DeploymentConfig`), add:
```go
	if edppRule == "least-ttft" && edppJoint {
		logrus.Fatalf("--edpp-rule least-ttft is a reduced-path baseline and cannot be combined with --edpp-joint")
	}
```

- [ ] **Step 5: Thread through DeploymentConfig → EDPPConfig**

In `sim/cluster/deployment.go`, next to `EDPPJoint` (line 90):
```go
	EDPPRule             string           // EDPP reduced-path decision rule: "" / "dpp" (default) | "least-ttft"
```
In `cmd/root.go`, on the `DeploymentConfig` literal (near line 1974, beside `EDPPTAdmEstimator: edppTAdmEstimator`):
```go
			EDPPRule:                        edppRule,
```
In `sim/cluster/cluster.go`, in the `NewEDPPDecider(sim.EDPPConfig{...})` literal (beside `Joint: config.EDPPJoint`, line 460):
```go
				Rule:              config.EDPPRule,
```

- [ ] **Step 6: Run tests + build**

Run: `go test ./cmd/ -run TestEDPPRule -v && go build -o blis main.go && go test ./sim/... ./cmd/...`
Expected: integration test PASSES; build green; no regression. gofmt changed files.

- [ ] **Step 7: Commit**

```bash
git add cmd/root.go sim/cluster/deployment.go sim/cluster/cluster.go cmd/<test file>
git commit -m "feat(edpp): --edpp-rule flag (least-ttft baseline) threaded to the decider"
```

---

## Notes for the implementer (confirm-in-situ)

- **Reduced-path test helpers (Task 1):** grep `sim/edpp_test.go` for the reduced-rule tests (e.g. around `decodeState`, `defaultTestEDPPConfig`, `newTestAffineModel`) and the `*Request` constructor they use; reuse them. The pseudo-helpers `newTestRequestInput` is a stand-in — use the real one. Set `RoutingSnapshot.GPUType` is NOT needed here (reduced, single global coeffs).
- **`EDPPConfig.validate()` shape (Task 1 Step 4):** confirm it panics vs returns error at `sim/edpp.go:155` and match that; `fmt` is already imported.
- **State magnitudes (Task 1 Step 1):** the affine test model may need different congested/idle numbers to make `ttftD`/`ttftP` cross; tune the snapshot batch/KV/queue values until the three behavioral assertions hold. Do not weaken the assertions.
- **`least-ttft`+`--edpp-joint` reject placement (Task 2 Step 4):** put it wherever the run path already guards EDPP flag combinations; a `logrus.Fatalf` at the CLI boundary is correct (R3: CLI → Fatalf).
- **cmd test harness (Task 2 Step 1):** reuse the existing pattern (child-process re-exec or direct `runCmd` invocation) used by sibling `cmd/*_test.go`; don't invent a new one.
