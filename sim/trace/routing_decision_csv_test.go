package trace

import (
	"bytes"
	"encoding/csv"
	"strings"
	"testing"
)

func TestWriteRoutingDecisionCSV(t *testing.T) {
	records := []RoutingDecisionTraceRecord{
		{
			Clock: 1000, Stage: "decode", RequestID: "req_0", ChosenInstance: "instance_3", Regret: 0.25,
			Candidates: []RoutingTraceCandidate{
				{InstanceID: "instance_2", IsChosen: false, CompositeScore: 0.5,
					ScorerScores: map[string]float64{"precise-prefix-cache": 1.0, "queue-depth": 0.0},
					QueueDepth:   4, BatchSize: 2, InFlightRequests: 4, KVUtilization: 0.8, FreeKVBlocks: 10},
				{InstanceID: "instance_3", IsChosen: true, CompositeScore: 0.75,
					ScorerScores: map[string]float64{"precise-prefix-cache": 0.5, "queue-depth": 1.0},
					QueueDepth:   0, BatchSize: 0, InFlightRequests: 0, KVUtilization: 0.1, FreeKVBlocks: 90},
			},
		},
	}

	var buf bytes.Buffer
	if err := WriteRoutingDecisionCSV(&buf, records); err != nil {
		t.Fatalf("WriteRoutingDecisionCSV: %v", err)
	}

	rows, err := csv.NewReader(strings.NewReader(buf.String())).ReadAll()
	if err != nil {
		t.Fatalf("output is not valid CSV: %v", err)
	}
	if len(rows) != 3 { // header + 2 candidates
		t.Fatalf("got %d rows, want 3 (header + 2 candidates)", len(rows))
	}

	// Header: fixed + sorted scorer union (precise-prefix-cache before queue-depth) + trailing.
	wantHeader := []string{
		"clock", "stage", "request_id", "chosen_instance",
		"candidate_instance", "is_chosen", "regret", "composite_score",
		"score_precise-prefix-cache", "score_queue-depth",
		"queue_depth", "batch_size", "inflight", "kv_utilization", "free_kv_blocks",
	}
	if strings.Join(rows[0], ",") != strings.Join(wantHeader, ",") {
		t.Errorf("header mismatch:\n got %v\nwant %v", rows[0], wantHeader)
	}

	col := func(row []string, name string) string {
		for i, h := range rows[0] {
			if h == name {
				return row[i]
			}
		}
		t.Fatalf("column %q not found", name)
		return ""
	}

	// Exactly one candidate row is_chosen=true, and it is instance_3.
	chosenCount := 0
	for _, r := range rows[1:] {
		if col(r, "is_chosen") == "true" {
			chosenCount++
			if col(r, "candidate_instance") != "instance_3" {
				t.Errorf("chosen candidate = %q, want instance_3", col(r, "candidate_instance"))
			}
		}
	}
	if chosenCount != 1 {
		t.Errorf("is_chosen=true count = %d, want 1", chosenCount)
	}

	// Per-scorer columns are populated from ScorerScores.
	for _, r := range rows[1:] {
		if col(r, "candidate_instance") == "instance_2" {
			if col(r, "score_precise-prefix-cache") != "1" {
				t.Errorf("instance_2 score_precise-prefix-cache = %q, want 1", col(r, "score_precise-prefix-cache"))
			}
			if col(r, "inflight") != "4" {
				t.Errorf("instance_2 inflight = %q, want 4", col(r, "inflight"))
			}
		}
	}
}

// TestWriteRoutingDecisionCSV_ScorerUnion verifies the header is the sorted union
// across records even when different decisions ran different scorers (e.g. prefill
// vs decode pools with different profiles); missing cells are blank.
func TestWriteRoutingDecisionCSV_ScorerUnion(t *testing.T) {
	records := []RoutingDecisionTraceRecord{
		{Clock: 1, Stage: "prefill", RequestID: "r", ChosenInstance: "p0", Candidates: []RoutingTraceCandidate{
			{InstanceID: "p0", IsChosen: true, ScorerScores: map[string]float64{"queue-depth": 1.0}},
		}},
		{Clock: 1, Stage: "decode", RequestID: "r", ChosenInstance: "d0", Candidates: []RoutingTraceCandidate{
			{InstanceID: "d0", IsChosen: true, ScorerScores: map[string]float64{"kv-utilization": 0.5}},
		}},
	}
	var buf bytes.Buffer
	if err := WriteRoutingDecisionCSV(&buf, records); err != nil {
		t.Fatalf("WriteRoutingDecisionCSV: %v", err)
	}
	rows, _ := csv.NewReader(strings.NewReader(buf.String())).ReadAll()
	header := strings.Join(rows[0], ",")
	if !strings.Contains(header, "score_kv-utilization") || !strings.Contains(header, "score_queue-depth") {
		t.Errorf("header missing union scorer columns: %s", header)
	}
	// kv-utilization sorts before queue-depth.
	if strings.Index(header, "score_kv-utilization") > strings.Index(header, "score_queue-depth") {
		t.Errorf("scorer columns not sorted: %s", header)
	}
}
