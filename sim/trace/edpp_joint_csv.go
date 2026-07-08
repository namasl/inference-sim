package trace

import (
	"encoding/csv"
	"io"
	"strconv"
)

// edppJointCSVHeader lists the scorer-vs-joint divergence CSV columns in write order.
var edppJointCSVHeader = []string{
	"request_id", "clock", "class",
	"scorer_d", "joint_d", "scorer_p", "joint_p",
	"agree_d", "agree_p",
	"j_scorer", "j_joint", "disaggregate",
}

// WriteEDPPJointDecisionCSV writes the scorer-vs-joint divergence records to w as CSV
// (header + one row per record). Floats use the shortest round-trippable form. Used by the
// --edpp-joint-trace output path; analysis tools consume the result to quantify how often
// and by how much the joint objective overrides the composable scorer.
func WriteEDPPJointDecisionCSV(w io.Writer, records []EDPPJointDecisionRecord) error {
	cw := csv.NewWriter(w)
	if err := cw.Write(edppJointCSVHeader); err != nil {
		return err
	}
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, r := range records {
		row := []string{
			r.RequestID, strconv.FormatInt(r.Clock, 10), r.Class,
			r.ScorerD, r.JointD, r.ScorerP, r.JointP,
			strconv.FormatBool(r.AgreeD), strconv.FormatBool(r.AgreeP),
			f(r.JScorer), f(r.JJoint), strconv.FormatBool(r.Disaggregate),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
