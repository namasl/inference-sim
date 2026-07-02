package trace

import (
	"bytes"
	"strings"
	"testing"
)

func TestWriteWorkTraceCSV_HeaderAndRow(t *testing.T) {
	recs := []WorkTraceRecord{{
		RequestID: "r1", SLOClass: "batch", Ar: 1000, ApRealized: 1000, ORealized: 50,
		PrefillChunks: 1, CacheHitFrac: 0.0,
		RealizedPrefillWork: 1506000, RealizedDecodeWork: 313725,
		WpClosed: 1506000, WdClosed: 313725, WpClosedNoCacheOld: 506000,
	}}
	var buf bytes.Buffer
	if err := WriteWorkTraceCSV(&buf, recs); err != nil {
		t.Fatalf("WriteWorkTraceCSV: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("want header + 1 row, got %d:\n%s", len(lines), buf.String())
	}
	if !strings.HasPrefix(lines[0], "request_id,slo_class,a_r,a_p_realized,o_r_realized,prefill_chunks,cache_hit_frac,") {
		t.Fatalf("unexpected header: %s", lines[0])
	}
	if !strings.Contains(lines[1], "r1,batch,1000,1000,50,1,0,") {
		t.Fatalf("unexpected row: %s", lines[1])
	}
}
