package trace

import (
	"encoding/csv"
	"io"
	"sort"
	"strconv"
)

// routingDecisionFixedHeader lists the non-scorer CSV columns, in write order.
// Per-scorer columns ("score_<name>") are inserted after composite_score, in the
// sorted union of scorer names across all records (see WriteRoutingDecisionCSV).
var routingDecisionFixedHeader = []string{
	"clock", "stage", "request_id", "chosen_instance",
	"candidate_instance", "is_chosen", "regret", "composite_score",
}

var routingDecisionTrailingHeader = []string{
	"queue_depth", "batch_size", "inflight", "kv_utilization", "free_kv_blocks",
}

// WriteRoutingDecisionCSV writes per-candidate routing-decision records to w as
// CSV in long format: one row per (decision × candidate). The header is
// deterministic — fixed columns, then one "score_<name>" column per scorer in the
// sorted union of scorer names across all records (blank when a candidate lacks
// that scorer), then the trailing per-candidate state columns. Floats use the
// shortest round-trippable form. Used by the --routing-decision-trace output path.
func WriteRoutingDecisionCSV(w io.Writer, records []RoutingDecisionTraceRecord) error {
	// Collect the sorted union of scorer names across all candidates.
	nameSet := map[string]struct{}{}
	for _, rec := range records {
		for _, c := range rec.Candidates {
			for name := range c.ScorerScores {
				nameSet[name] = struct{}{}
			}
		}
	}
	scorerNames := make([]string, 0, len(nameSet))
	for name := range nameSet {
		scorerNames = append(scorerNames, name)
	}
	sort.Strings(scorerNames)

	cw := csv.NewWriter(w)

	header := make([]string, 0, len(routingDecisionFixedHeader)+len(scorerNames)+len(routingDecisionTrailingHeader))
	header = append(header, routingDecisionFixedHeader...)
	for _, name := range scorerNames {
		header = append(header, "score_"+name)
	}
	header = append(header, routingDecisionTrailingHeader...)
	if err := cw.Write(header); err != nil {
		return err
	}

	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, rec := range records {
		for _, c := range rec.Candidates {
			row := make([]string, 0, len(header))
			row = append(row,
				strconv.FormatInt(rec.Clock, 10), rec.Stage, rec.RequestID, rec.ChosenInstance,
				c.InstanceID, strconv.FormatBool(c.IsChosen), f(rec.Regret), f(c.CompositeScore),
			)
			for _, name := range scorerNames {
				if c.ScorerScores != nil {
					if v, ok := c.ScorerScores[name]; ok {
						row = append(row, f(v))
						continue
					}
				}
				row = append(row, "") // candidate did not run this scorer
			}
			row = append(row,
				strconv.Itoa(c.QueueDepth), strconv.Itoa(c.BatchSize), strconv.Itoa(c.InFlightRequests),
				f(c.KVUtilization), strconv.FormatInt(c.FreeKVBlocks, 10),
			)
			if err := cw.Write(row); err != nil {
				return err
			}
		}
	}
	cw.Flush()
	return cw.Error()
}
