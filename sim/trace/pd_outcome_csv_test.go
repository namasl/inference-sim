package trace

import (
	"bytes"
	"strings"
	"testing"
)

func TestWritePDOutcomeCSV_HeaderAndRow(t *testing.T) {
	records := []PDOutcomeRecord{{
		RequestID: "r1", SLOClass: "standard", InputTokens: 512, Disaggregated: true,
		PrefillInstance: "instance_0", DecodeInstance: "instance_2",
		PrefillEnqueue: 100, PrefillSchedule: 140, PrefillTAdm: 40,
		DecodeEnqueue: 900, DecodeSchedule: 1200, DecodeTAdm: 300,
		LocalEnqueue: 0, LocalSchedule: 0, LocalTAdm: 0,
		RealizedTTFT: 1500, RealizedMeanITL: 30, RealizedE2E: 42000, Completed: true,
	}}
	var buf bytes.Buffer
	if err := WritePDOutcomeCSV(&buf, records); err != nil {
		t.Fatalf("WritePDOutcomeCSV: %v", err)
	}
	out := buf.String()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 2 {
		t.Fatalf("want header + 1 row, got %d lines:\n%s", len(lines), out)
	}
	if !strings.HasPrefix(lines[0], "request_id,slo_class,input_tokens,disaggregated,") {
		t.Fatalf("unexpected header: %s", lines[0])
	}
	if !strings.Contains(lines[1], "r1,standard,512,true,instance_0,instance_2,") {
		t.Fatalf("unexpected row: %s", lines[1])
	}
	if !strings.HasSuffix(lines[1], ",true") { // completed
		t.Fatalf("row should end with completed=true: %s", lines[1])
	}
}
