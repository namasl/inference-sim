package trace

import (
	"encoding/csv"
	"io"
	"strconv"
)

// admissionCSVHeader lists the per-request admission-trace columns in write order.
var admissionCSVHeader = []string{
	"request_id", "pool", "realized_t_adm",
	"t_adm_pred_waiting", "t_adm_pred_little", "t_adm_pred_fluid",
	"t_adm_pred_rollforward", "t_adm_pred_fluid_oracle", "t_adm_pred_rollforward_oracle",
}

// WriteAdmissionCSV writes realized-vs-predicted admission-delay records to w as CSV
// (header + one row per record). Callers pass records pre-sorted by request_id for
// deterministic output (INV-6). Consumed by --edpp-admission-trace / the
// estimator_validation.py analysis.
func WriteAdmissionCSV(w io.Writer, records []AdmissionRecord) error {
	cw := csv.NewWriter(w)
	if err := cw.Write(admissionCSVHeader); err != nil {
		return err
	}
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, r := range records {
		row := []string{
			r.RequestID, r.Pool, f(r.RealizedTAdm),
			f(r.TAdmPredWaiting), f(r.TAdmPredLittle), f(r.TAdmPredFluid),
			f(r.TAdmPredRollforward), f(r.TAdmPredFluidOracle), f(r.TAdmPredRollforwardOracle),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
