package trace

import (
	"bytes"
	"strings"
	"testing"
)

func TestWriteAdmissionCSV_HeaderAndRow(t *testing.T) {
	recs := []AdmissionRecord{{
		RequestID: "r1", Pool: "decode", RealizedTAdm: 1200,
		TAdmPredWaiting: 100, TAdmPredLittle: 200, TAdmPredFluid: 300,
		TAdmPredRollforward: 400, TAdmPredFluidOracle: 500, TAdmPredRollforwardOracle: 600,
	}}
	var buf bytes.Buffer
	if err := WriteAdmissionCSV(&buf, recs); err != nil {
		t.Fatalf("WriteAdmissionCSV: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("want header + 1 row, got %d:\n%s", len(lines), buf.String())
	}
	wantHeader := "request_id,pool,realized_t_adm,t_adm_pred_waiting,t_adm_pred_little,t_adm_pred_fluid,t_adm_pred_rollforward,t_adm_pred_fluid_oracle,t_adm_pred_rollforward_oracle"
	if lines[0] != wantHeader {
		t.Fatalf("unexpected header:\n got %s\nwant %s", lines[0], wantHeader)
	}
	if lines[1] != "r1,decode,1200,100,200,300,400,500,600" {
		t.Fatalf("unexpected row: %s", lines[1])
	}
}
