package trace

import (
	"encoding/csv"
	"io"
	"strconv"
)

// pdOutcomeCSVHeader lists the per-request outcome columns in write order.
var pdOutcomeCSVHeader = []string{
	"request_id", "slo_class", "input_tokens", "disaggregated",
	"prefill_instance", "decode_instance",
	"prefill_enqueue", "prefill_schedule", "prefill_t_adm",
	"decode_enqueue", "decode_schedule", "decode_t_adm",
	"local_enqueue", "local_schedule", "local_t_adm",
	"realized_ttft", "realized_mean_itl", "realized_e2e", "completed",
}

// WritePDOutcomeCSV writes realized per-request outcome records to w as CSV
// (header + one row per record). Callers pass records pre-sorted by request_id
// for deterministic output (INV-6). Consumed by --pd-outcome-trace / the
// estimator_validation.py analysis.
func WritePDOutcomeCSV(w io.Writer, records []PDOutcomeRecord) error {
	cw := csv.NewWriter(w)
	if err := cw.Write(pdOutcomeCSVHeader); err != nil {
		return err
	}
	i := func(v int64) string { return strconv.FormatInt(v, 10) }
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, r := range records {
		row := []string{
			r.RequestID, r.SLOClass, strconv.Itoa(r.InputTokens), strconv.FormatBool(r.Disaggregated),
			r.PrefillInstance, r.DecodeInstance,
			i(r.PrefillEnqueue), i(r.PrefillSchedule), i(r.PrefillTAdm),
			i(r.DecodeEnqueue), i(r.DecodeSchedule), i(r.DecodeTAdm),
			i(r.LocalEnqueue), i(r.LocalSchedule), i(r.LocalTAdm),
			f(r.RealizedTTFT), f(r.RealizedMeanITL), f(r.RealizedE2E), strconv.FormatBool(r.Completed),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
