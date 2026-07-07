package sim

import (
	"encoding/csv"
	"fmt"
	"os"
)

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
