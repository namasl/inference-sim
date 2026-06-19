package trace

import (
	"bytes"
	"encoding/csv"
	"testing"
)

func TestWriteEDPPDecisionCSV(t *testing.T) {
	records := []EDPPDecisionRecord{
		{
			RequestID: "r1", Clock: 1000, Class: "standard",
			Ap: 800, Wp: 8000, DeltaPfChunk: 8000,
			QdRaw: 38664, QpRaw: 0, Qd: 0.3945, Qp: 0,
			TransferTerm: 0.05, TTFTTerm: 0, ITLTerm: 0,
			BalanceTermD: 0.0322, BalanceTermP: 0,
			LHS: 0.0322, RHS: 0.05, Disaggregate: false,
		},
		{RequestID: "r2", Clock: 2000, Class: "standard", SkipReason: "empty-prompt"},
	}

	var buf bytes.Buffer
	if err := WriteEDPPDecisionCSV(&buf, records); err != nil {
		t.Fatalf("WriteEDPPDecisionCSV: %v", err)
	}

	rows, err := csv.NewReader(&buf).ReadAll()
	if err != nil {
		t.Fatalf("parse CSV: %v", err)
	}
	if len(rows) != 3 { // header + 2 data rows
		t.Fatalf("rows = %d, want 3 (header + 2)", len(rows))
	}

	header := rows[0]
	// Build a column index so the test is order-independent.
	col := map[string]int{}
	for i, h := range header {
		col[h] = i
	}
	for _, want := range []string{"request_id", "clock", "class", "skip_reason", "ap", "wp", "lhs", "rhs", "transfer_term", "ttft_term", "itl_term", "disaggregate"} {
		if _, ok := col[want]; !ok {
			t.Errorf("missing expected column %q in header %v", want, header)
		}
	}

	r1 := rows[1]
	if r1[col["request_id"]] != "r1" {
		t.Errorf("row1 request_id = %q, want r1", r1[col["request_id"]])
	}
	if r1[col["disaggregate"]] != "false" {
		t.Errorf("row1 disaggregate = %q, want false", r1[col["disaggregate"]])
	}
	if r1[col["lhs"]] == "" || r1[col["rhs"]] == "" {
		t.Errorf("row1 lhs/rhs should be populated, got lhs=%q rhs=%q", r1[col["lhs"]], r1[col["rhs"]])
	}

	r2 := rows[2]
	if r2[col["skip_reason"]] != "empty-prompt" {
		t.Errorf("row2 skip_reason = %q, want empty-prompt", r2[col["skip_reason"]])
	}
}
