package trace

import (
	"encoding/csv"
	"io"
	"strconv"
)

var workTraceCSVHeader = []string{
	"request_id", "slo_class", "a_r", "a_p_realized", "o_r_realized",
	"prefill_chunks", "cache_hit_frac",
	"realized_prefill_work", "realized_decode_work",
	"wp_closed", "wd_closed", "wp_closed_nocache_old",
}

// WriteWorkTraceCSV writes realized-vs-closed work records to w as CSV (header +
// one row per record). Callers pass records pre-sorted by request_id for
// deterministic output (INV-6). Consumed by --edpp-work-trace / work_model_validation.py.
func WriteWorkTraceCSV(w io.Writer, records []WorkTraceRecord) error {
	cw := csv.NewWriter(w)
	if err := cw.Write(workTraceCSVHeader); err != nil {
		return err
	}
	i := func(v int64) string { return strconv.FormatInt(v, 10) }
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, r := range records {
		row := []string{
			r.RequestID, r.SLOClass, i(r.Ar), i(r.ApRealized), i(r.ORealized),
			strconv.Itoa(r.PrefillChunks), f(r.CacheHitFrac),
			f(r.RealizedPrefillWork), f(r.RealizedDecodeWork),
			f(r.WpClosed), f(r.WdClosed), f(r.WpClosedNoCacheOld),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
