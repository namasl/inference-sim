# Per-Decision Counterfactual Regret Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how much better each routing decision could have been, by forcing per-request `(decode, prefill)` plans through the simulator and re-running one-request deviations — an exact, hindsight, per-decision regret diagnostic.

**Architecture:** A new **fixed-plan decider** (Go) forces a supplied `{request_id → (d,p)}` plan via the existing `DecodePodOverride` plus newly-honored `PrefillPodHint`. A Python **counterfactual driver** captures a policy's baseline plan from `--pd-outcome-trace`, verifies the decider replays it byte-identically, then sweeps single-request deviations and reports regret on total goodput.

**Tech Stack:** Go 1.22 (`sim/`, `sim/cluster/`, `cmd/`), Python 3 + pandas/numpy, bash.

## Global Constraints

- Branch: `feat/edpp-estimator-validation`.
- Spec: `docs/superpowers/specs/2026-07-07-edpp-counterfactual-regret-design.md`.
- Action set 𝒜 = ℳ × (𝒫 ∪ {local}); a plan is TOTAL — a request missing from the plan is a fatal error (R1, no silent fallback).
- Fixed-plan decider reads only the plan (instance IDs); never reads `Request.OutputTokens` (INV-9).
- INV-6 determinism: a forced plan runs byte-identical across invocations; INV-13: forced routing has run/replay parity.
- The decider forces `local` → `DisaggregationDecision{Disaggregate:false, DecodePodOverride:d}`; disagg → `{Disaggregate:true, DecodePodOverride:d, PrefillPodHint:p}`.
- Baseline plan captured by reusing `--pd-outcome-trace` (columns include `request_id, prefill_instance, decode_instance`); do not build new plan-capture instrumentation.
- Regret is measured on TOTAL trace goodput (the sim's SLO attainment), not the request's own outcome.
- Go tests: `go test ./sim/... -run <name>`; build `go build -o blis main.go`; gofmt before commit.

---

### Task 1: Fixed-plan decider + PrefillPodHint wiring + `--pd-plan` flag

**Files:**
- Create: `sim/fixed_plan_decider.go`
- Test: `sim/fixed_plan_decider_test.go`
- Modify: `sim/cluster/parent_request.go` (add a prefill-hint field), `sim/cluster/pd_events.go` (honor the hint in `PrefillRoutingEvent.Execute`), `sim/cluster/cluster.go` (store the hint on the parent at the disaggregation call site), the decider-construction site selected by `--pd-decider`/config, and `cmd/` flag registration.

**Interfaces:**
- Consumes: `sim.DisaggregationDecider` interface (`sim/disaggregation.go:52`), `DisaggregationDecision{Disaggregate, DecodePodOverride, PrefillPodHint}` (`sim/disaggregation.go:17`).
- Produces:
  - `sim.FixedPlanAction struct { DecodeInstance string; PrefillInstance string /* "" or "local" ⇒ local */ }`
  - `sim.NewFixedPlanDecider(plan map[string]FixedPlanAction) *FixedPlanDecider` with method `Decide(req *Request, state *RouterState) DisaggregationDecision`.
  - `sim.LoadFixedPlanCSV(path string) (map[string]FixedPlanAction, error)` (columns `request_id, decode_instance, prefill_instance`).

- [ ] **Step 1: Write the decider unit test (failing)**

```go
// sim/fixed_plan_decider_test.go
package sim

import "testing"

func TestFixedPlanDecider_Local(t *testing.T) {
	d := NewFixedPlanDecider(map[string]FixedPlanAction{
		"r1": {DecodeInstance: "M0", PrefillInstance: "local"},
	})
	got := d.Decide(&Request{ID: "r1"}, nil)
	if got.Disaggregate || got.DecodePodOverride != "M0" || got.PrefillPodHint != "" {
		t.Fatalf("local action: got %+v, want {Disaggregate:false, DecodePodOverride:M0, PrefillPodHint:''}", got)
	}
}

func TestFixedPlanDecider_Disagg(t *testing.T) {
	d := NewFixedPlanDecider(map[string]FixedPlanAction{
		"r2": {DecodeInstance: "M1", PrefillInstance: "P0"},
	})
	got := d.Decide(&Request{ID: "r2"}, nil)
	if !got.Disaggregate || got.DecodePodOverride != "M1" || got.PrefillPodHint != "P0" {
		t.Fatalf("disagg action: got %+v, want {Disaggregate:true, DecodePodOverride:M1, PrefillPodHint:P0}", got)
	}
}

func TestFixedPlanDecider_MissingRequestPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on request absent from plan (plans must be total, R1)")
		}
	}()
	d := NewFixedPlanDecider(map[string]FixedPlanAction{"r1": {DecodeInstance: "M0", PrefillInstance: "local"}})
	d.Decide(&Request{ID: "absent"}, nil)
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestFixedPlanDecider -v`
Expected: FAIL — `FixedPlanAction`/`NewFixedPlanDecider` undefined (compile error).

- [ ] **Step 3: Implement the decider**

```go
// sim/fixed_plan_decider.go
package sim

import "fmt"

// FixedPlanAction is the forced action for one request: which decode instance,
// and which prefill location ("" or "local" ⇒ prefill locally on the decode instance).
type FixedPlanAction struct {
	DecodeInstance  string
	PrefillInstance string
}

func (a FixedPlanAction) isLocal() bool {
	return a.PrefillInstance == "" || a.PrefillInstance == "local"
}

// FixedPlanDecider forces a supplied per-request (decode, prefill) plan. It is a
// measurement/evaluation tool (the counterfactual-regret harness and the offline
// yardstick), not a routing policy. The plan must be TOTAL: a request absent from
// the plan is a fatal error (R1 — no silent fallback). INV-9: reads only the plan.
type FixedPlanDecider struct {
	plan map[string]FixedPlanAction
}

func NewFixedPlanDecider(plan map[string]FixedPlanAction) *FixedPlanDecider {
	return &FixedPlanDecider{plan: plan}
}

func (d *FixedPlanDecider) Decide(req *Request, _ *RouterState) DisaggregationDecision {
	a, ok := d.plan[req.ID]
	if !ok {
		panic(fmt.Sprintf("fixed-plan decider: request %q absent from plan (plans must be total)", req.ID))
	}
	if a.isLocal() {
		return DisaggregationDecision{Disaggregate: false, DecodePodOverride: a.DecodeInstance}
	}
	return DisaggregationDecision{Disaggregate: true, DecodePodOverride: a.DecodeInstance, PrefillPodHint: a.PrefillInstance}
}
```

- [ ] **Step 4: Run to verify decider tests pass**

Run: `go test ./sim/ -run TestFixedPlanDecider -v`
Expected: PASS (all three).

- [ ] **Step 5: Write the CSV loader test (failing)**

```go
func TestLoadFixedPlanCSV(t *testing.T) {
	dir := t.TempDir()
	p := dir + "/plan.csv"
	if err := os.WriteFile(p, []byte("request_id,decode_instance,prefill_instance\nr1,M0,local\nr2,M1,P0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	plan, err := LoadFixedPlanCSV(p)
	if err != nil {
		t.Fatal(err)
	}
	if plan["r1"] != (FixedPlanAction{"M0", "local"}) || plan["r2"] != (FixedPlanAction{"M1", "P0"}) {
		t.Fatalf("parsed plan wrong: %+v", plan)
	}
}
```
Add `import ("os"; "testing")` to the test file.

- [ ] **Step 6: Run to verify it fails, then implement the loader**

Run: `go test ./sim/ -run TestLoadFixedPlanCSV -v` → FAIL (undefined).

Add to `sim/fixed_plan_decider.go`:
```go
import (
	"encoding/csv"
	"fmt"
	"os"
)

// LoadFixedPlanCSV reads a plan CSV with header columns
// request_id,decode_instance,prefill_instance (prefill_instance "" or "local" ⇒ local).
func LoadFixedPlanCSV(path string) (map[string]FixedPlanAction, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("fixed-plan CSV %s is empty", path)
	}
	hdr := map[string]int{}
	for i, name := range rows[0] {
		hdr[name] = i
	}
	for _, col := range []string{"request_id", "decode_instance", "prefill_instance"} {
		if _, ok := hdr[col]; !ok {
			return nil, fmt.Errorf("fixed-plan CSV %s missing column %q", path, col)
		}
	}
	plan := make(map[string]FixedPlanAction, len(rows)-1)
	for _, row := range rows[1:] {
		plan[row[hdr["request_id"]]] = FixedPlanAction{
			DecodeInstance:  row[hdr["decode_instance"]],
			PrefillInstance: row[hdr["prefill_instance"]],
		}
	}
	return plan, nil
}
```
(Consolidate the two `import` blocks into one.) Run the loader test → PASS.

- [ ] **Step 7: Honor `PrefillPodHint` in prefill routing**

The disaggregation decision is made in `executeDisaggregatedRouting` (`sim/cluster/cluster.go` ~2266) but prefill is routed later in `PrefillRoutingEvent.Execute` (`sim/cluster/pd_events.go:28`). Thread the hint via the parent:
1. In `sim/cluster/parent_request.go`, add a field `PrefillPodHint string` (documented: "forced prefill instance from a joint/fixed-plan decider; empty ⇒ normal scorer routing").
2. At the disaggregation call site in `cluster.go` (where `Disaggregate==true` builds the `ParentRequest` and schedules `PrefillRoutingEvent` — confirm exact line, near the `DecodePodOverride` handling at `cluster.go:2284`), set `parent.PrefillPodHint = disaggDecision.PrefillPodHint`.
3. In `pd_events.go:Execute`, before calling `policy.Route`, honor the hint:
```go
	var targetInstance string
	if e.parentReq.PrefillPodHint != "" {
		targetInstance = e.parentReq.PrefillPodHint
	} else {
		policy := cs.prefillRoutingPolicy
		if policy == nil {
			policy = cs.routingPolicy
		}
		decision := policy.Route(e.request, state)
		targetInstance = decision.TargetInstance
		if cs.routingTraceOn() {
			cs.recordRoutingDecisionTrace("prefill", e.parentReq.ID, decision, state.Snapshots)
		}
	}
	e.request.AssignedInstance = targetInstance
	e.parentReq.PrefillInstanceID = InstanceID(targetInstance)
	e.parentReq.PrefillEnqueueTime = e.time
```
Keep the existing trace/record code paths for the non-hint branch. (Confirm the exact surrounding lines in situ; the goal is: hint present ⇒ skip the scorer and use the hinted instance.)

- [ ] **Step 8: Wire `--pd-plan <csv>` flag + decider selection**

1. Register a string flag `--pd-plan` on `run` and `replay` (in the same place `--pd-decider` / `--edpp-coeffs` are registered — `registerSimConfigFlags`), carried on the deployment/sim config.
2. At the decider-construction site (where `--pd-decider` selects `NeverDisaggregate` / `AlwaysDisaggregate` / `NewEDPPDecider`): if `--pd-plan` is set, `LoadFixedPlanCSV` it and use `NewFixedPlanDecider(plan)` as the `DisaggregationDecider` (it takes precedence; a load error is `logrus.Fatalf`). Confirm the exact construction site (cluster construction, near where the EDPP decider is built, ~`cluster.go:448`).

- [ ] **Step 9: End-to-end forced-routing test**

Add a cluster-level test (in `sim/cluster/`) that builds a tiny 1P + 2M topology, injects 2 requests, forces a plan (`r1→(M0,local)`, `r2→(M1,P0)`) via `NewFixedPlanDecider`, runs, and asserts: `r1` decoded on `M0` with no prefill sub-request on P; `r2` prefilled on `P0` and decoded on `M1` (read `ParentRequest.PrefillInstanceID`/`DecodeInstanceID` or the metrics). This proves both overrides (decode + prefill-hint) take effect.

Run: `go test ./sim/... -run 'FixedPlan|ForcedRouting' -v` → PASS. Then `go build -o blis main.go && go test ./sim/...` → green. gofmt the changed files.

- [ ] **Step 10: Commit**

```bash
git add sim/fixed_plan_decider.go sim/fixed_plan_decider_test.go sim/cluster/parent_request.go sim/cluster/pd_events.go sim/cluster/cluster.go cmd/
git commit -m "feat(edpp): fixed-plan decider + --pd-plan flag + PrefillPodHint wiring

Forces a supplied per-request (decode,prefill) plan (DecodePodOverride +
newly-honored PrefillPodHint). Total-plan (missing request fatal, R1); reads
only the plan (INV-9). Enables the counterfactual-regret harness / offline yardstick."
```

---

### Task 2: Counterfactual driver + regret report

**Files:**
- Create: `campaigns/edpp-study/analyze/counterfactual_regret.py`
- Test: `campaigns/edpp-study/analyze/test_counterfactual_regret.py`

**Interfaces:**
- Consumes: the `blis` binary with `--pd-plan` (Task 1); `--pd-outcome-trace` CSV (has `request_id, disaggregated, prefill_instance, decode_instance`); the run metrics JSON (`slo_attainment` / `per_class`) via `--metrics-path`.
- Produces (CLI, mirrors the other analyze/ scripts):
  - `capture-plan --outcome <csv> --out <plan.csv>`: convert a `--pd-outcome-trace` into a fixed-plan CSV (`request_id, decode_instance, prefill_instance`).
  - `regret --sweep-dir <dir>`: read a directory of per-deviation run metrics (named `dev_<reqid>_<action>.json` + a `baseline.json`) and emit the regret report JSON.
  - a pure helper `goodput(metrics_json) -> float` (reads `slo_attainment`, summing a per-instance array if present).

- [ ] **Step 1: Write the self-check test (failing)**

```python
# campaigns/edpp-study/analyze/test_counterfactual_regret.py
"""Self-check for counterfactual_regret.py. Run: python3 .../test_counterfactual_regret.py"""
import json, os, subprocess, sys, tempfile, csv

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "counterfactual_regret.py")


def test_capture_plan_from_outcome():
    with tempfile.TemporaryDirectory() as d:
        oc = os.path.join(d, "outcome.csv")
        with open(oc, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["request_id", "disaggregated", "prefill_instance", "decode_instance"])
            w.writerow(["r1", "false", "", "M0"]); w.writerow(["r2", "true", "P0", "M1"])
        out = os.path.join(d, "plan.csv")
        r = subprocess.run([sys.executable, SCRIPT, "capture-plan", "--outcome", oc, "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rows = list(csv.DictReader(open(out)))
        assert rows[0] == {"request_id": "r1", "decode_instance": "M0", "prefill_instance": "local"}
        assert rows[1] == {"request_id": "r2", "decode_instance": "M1", "prefill_instance": "P0"}


def test_regret_aggregation():
    with tempfile.TemporaryDirectory() as d:
        def m(path, att): json.dump({"slo_attainment": att}, open(path, "w"))
        m(os.path.join(d, "baseline.json"), 0.80)
        # r1: one deviation improves to 0.90 -> regret 0.10; another 0.70
        m(os.path.join(d, "dev_r1_M1-local.json"), 0.90)
        m(os.path.join(d, "dev_r1_M0-P0.json"), 0.70)
        # r2: no deviation beats baseline -> regret 0
        m(os.path.join(d, "dev_r2_M0-local.json"), 0.75)
        out = os.path.join(d, "regret.json")
        r = subprocess.run([sys.executable, SCRIPT, "regret", "--sweep-dir", d, "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rep = json.load(open(out))
        by = {p["request_id"]: p for p in rep["per_request"]}
        assert abs(by["r1"]["regret"] - 0.10) < 1e-9 and by["r1"]["hindsight_best"] == "M1-local"
        assert abs(by["r2"]["regret"] - 0.0) < 1e-9
        assert abs(rep["total_regret"] - 0.10) < 1e-9 and rep["frac_positive_regret"] == 0.5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok ", name)
    print("all passed")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 campaigns/edpp-study/analyze/test_counterfactual_regret.py`
Expected: FAIL (script missing).

- [ ] **Step 3: Implement the driver**

```python
#!/usr/bin/env python3
"""Per-decision counterfactual regret (see FINDINGS "Counterfactual regret").

Subcommands:
  capture-plan --outcome <pd-outcome-trace.csv> --out <plan.csv>
      Convert a policy's --pd-outcome-trace into a fixed-plan CSV
      (request_id, decode_instance, prefill_instance; "local" when not disaggregated).
  regret --sweep-dir <dir> [--out <json>]
      Read baseline.json + dev_<reqid>_<action>.json run-metrics in <dir>, compute
      per-request regret = max_action goodput(dev) - goodput(baseline) on TOTAL goodput.
"""
import argparse, csv, glob, json, os, re, sys


def goodput(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    if isinstance(m, list):  # per-instance array: average attainment
        vals = [x.get("slo_attainment", 0.0) for x in m]
        return float(sum(vals) / len(vals)) if vals else 0.0
    return float(m.get("slo_attainment", 0.0))


def capture_plan(args):
    rows_out = []
    with open(args.outcome) as f:
        for row in csv.DictReader(f):
            disagg = str(row.get("disaggregated", "")).strip().lower() == "true"
            rows_out.append({
                "request_id": row["request_id"],
                "decode_instance": row["decode_instance"],
                "prefill_instance": row["prefill_instance"] if disagg and row.get("prefill_instance") else "local",
            })
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["request_id", "decode_instance", "prefill_instance"])
        w.writeheader(); w.writerows(rows_out)
    return 0


def regret(args):
    base = goodput(os.path.join(args.sweep_dir, "baseline.json"))
    devs = {}  # reqid -> {action: goodput}
    for path in glob.glob(os.path.join(args.sweep_dir, "dev_*.json")):
        mobj = re.match(r"dev_(.+)_([^_]+)\.json$", os.path.basename(path))
        if not mobj:
            continue
        rid, action = mobj.group(1), mobj.group(2)
        devs.setdefault(rid, {})[action] = goodput(path)
    per_request = []
    for rid, actions in sorted(devs.items()):
        best_action = max(actions, key=actions.get)
        best_g = actions[best_action]
        reg = max(0.0, best_g - base)
        per_request.append({"request_id": rid, "baseline_goodput": base,
                            "hindsight_best": best_action if reg > 0 else "baseline",
                            "hindsight_best_goodput": best_g, "regret": reg})
    n = len(per_request)
    pos = [p for p in per_request if p["regret"] > 0]
    report = {"baseline_goodput": base, "n_requests": n,
              "frac_positive_regret": (len(pos) / n) if n else 0.0,
              "total_regret": sum(p["regret"] for p in per_request),
              "mean_regret": (sum(p["regret"] for p in per_request) / n) if n else 0.0,
              "per_request": per_request}
    text = json.dumps(report, indent=2)
    if args.out:
        open(args.out, "w").write(text + "\n")
    print(text)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    cp = sub.add_parser("capture-plan"); cp.add_argument("--outcome", required=True); cp.add_argument("--out", required=True)
    cp.set_defaults(func=capture_plan)
    rg = sub.add_parser("regret"); rg.add_argument("--sweep-dir", required=True); rg.add_argument("--out", default="")
    rg.set_defaults(func=regret)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify tests pass**

Run: `python3 campaigns/edpp-study/analyze/test_counterfactual_regret.py`
Expected: PASS (`ok  test_capture_plan_from_outcome`, `ok  test_regret_aggregation`, `all passed`).

- [ ] **Step 5: Commit**

```bash
git add campaigns/edpp-study/analyze/counterfactual_regret.py campaigns/edpp-study/analyze/test_counterfactual_regret.py
git commit -m "feat(edpp): counterfactual-regret driver (capture-plan + regret aggregation)"
```

---

### Task 3: `repro_counterfactual.sh` + diagnosis run + findings

**Files:**
- Create: `campaigns/edpp-study/repro_counterfactual.sh`
- Modify: `campaigns/edpp-study/FINDINGS.md`, `campaigns/edpp-study/README.md`

**Interfaces:**
- Consumes: `blis --pd-plan` (Task 1), `counterfactual_regret.py` (Task 2), an existing synth spec + a small PD topology.

- [ ] **Step 1: Write the repro script**

`campaigns/edpp-study/repro_counterfactual.sh` (mirror the other repro scripts; `set -euo pipefail`, `cd "$(git rev-parse --show-toplevel)"`, build if `./blis` absent). It must:
1. Run the target policy (default `edpp`, configurable) on a saturating synth trace at a **1P2D** topology with `--pd-outcome-trace $OUT/outcome.csv` and `--metrics-path $OUT/baseline.json`.
2. `counterfactual_regret.py capture-plan --outcome $OUT/outcome.csv --out $OUT/plan.csv`.
3. **Self-consistency gate:** run `blis --pd-plan $OUT/plan.csv --metrics-path $OUT/replay.json`; assert `slo_attainment` in `replay.json` equals `baseline.json` (fail loudly otherwise — the decider must replay the plan exactly, INV-6).
4. Sample `K` request IDs from the plan (default `K=50`, configurable). For each sampled `r` and each action `a ∈ 𝒜 \ {plan(r)}` (enumerate decode instances × {local, each P}), write a deviated plan (`plan.csv` with `r`'s row replaced), run `blis --pd-plan <dev> --metrics-path $OUT/dev_<r>_<a>.json`.
5. `counterfactual_regret.py regret --sweep-dir $OUT --out $OUT/regret.json`.
Naming: action `a` encoded `<decode>-<prefill|local>` to match the driver's `dev_<reqid>_<action>.json` regex. Echo the headline (total/mean regret, frac positive) at the end. Note the cost: `K·(|𝒜|−1)` runs.

- [ ] **Step 2: Run the harness end-to-end (small K)**

Run: `bash campaigns/edpp-study/repro_counterfactual.sh` (optionally with a small `K` for the first run).
Expected: the self-consistency gate passes (replay == baseline), the sweep completes, `regret.json` is produced. **Report the actual regret distribution honestly** — this is the empirical verdict on reduced-EDPP; if regret is near zero the policy is locally fine, if large it is improvable. Do not tune anything.

- [ ] **Step 3: Hand-case sanity check**

Construct a tiny 2-request trace on 1P2D. Run the harness with a `never` baseline on an idle configuration: every hindsight-best must be `local` (regret 0), confirming the harness reproduces the known answer. Then an `always` baseline: expect positive regret (local would have been better with no contention). Record the two checks in the script's comments or a tiny fixture.

- [ ] **Step 4: Write FINDINGS + README**

- FINDINGS "Counterfactual regret" section: purpose (exact per-decision hindsight regret), the topology/trace, the self-consistency guarantee, the reduced-EDPP regret result (distribution, where it concentrates: kept-local vs disaggregated decisions), and the interpretation (does reduced-EDPP's pool-average structure cost goodput?). Reproduction: `bash campaigns/edpp-study/repro_counterfactual.sh`; note it is a diagnostic (local one-step-deviation), not the global optimum.
- README pointer (new subsection): what the harness is, the fixed-plan decider + `--pd-plan`, and that it is the shared infra for the later full-joint decider and MILP yardstick.

- [ ] **Step 5: Commit**

```bash
git add campaigns/edpp-study/repro_counterfactual.sh campaigns/edpp-study/FINDINGS.md campaigns/edpp-study/README.md
git commit -m "docs(edpp): counterfactual-regret repro + reduced-EDPP diagnosis findings"
```

---

## Notes for the implementer (confirm-in-situ)

- **Exact decider-construction site + flag plumbing (Task 1 steps 7-8):** the plan names `cluster.go:2284` (decode override), `cluster.go:~448` (decider build), `pd_events.go:28` (prefill route), and `registerSimConfigFlags` (flags) as anchors; confirm the exact lines and the `DeploymentConfig`/`EDPPConfig` field names in situ. The goal is invariant: `--pd-plan` present ⇒ the cluster's `DisaggregationDecider` is the fixed-plan decider, and a disagg decision's `PrefillPodHint` reaches `PrefillRoutingEvent.Execute`.
- **`--metrics-path` shape:** the counterfactual driver's `goodput()` tolerates both a single metrics object and a per-instance array (as the utilization-sweep analyzer does). Confirm `blis`'s `--metrics-path` output carries `slo_attainment` (it is populated when `--slo-ttft`/`--slo-itl` are set — pass those in the repro script).
- **INV-13:** the fixed-plan decider must behave identically on `run` and `replay`; if `--pd-plan` is only wired on `run`, note it and wire both (mirror `--pd-decider`).
- The sweep is `K·(|𝒜|−1)` sim runs; keep `K` small on the first end-to-end run, then scale.
