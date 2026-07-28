#!/usr/bin/env python3
"""SLO-miss decomposition + placement analysis from pd-outcome traces (2026-07-28).

For each trace CSV (one per cell x policy at the operating point), reports:
  - goodput recomputed from the trace (cross-check vs metrics)
  - miss fraction per SLO dimension (ttft / itl / e2e), split by serving instance
  - placement shares: disagg fraction, decode-instance shares
Requires the local-record fix (class/instance on non-disaggregated rows).

Usage: decomposition.py SLO_TTFT_MS SLO_ITL_MS SLO_E2E_MS trace1.csv [trace2.csv ...]
"""
import csv, sys, collections, os

def analyze(path, T, I, E):
    rows = [r for r in csv.DictReader(open(path)) if r["slo_class"] == "standard"]
    n = len(rows)
    if n == 0:
        return f"{os.path.basename(path)}: no standard rows"
    good = 0
    disagg = 0
    share = collections.Counter()
    miss = collections.defaultdict(collections.Counter)  # instance -> dim -> count
    for r in rows:
        inst = r["decode_instance"] or "?"
        share[inst] += 1
        if r["disaggregated"] == "true":
            disagg += 1
        t, i, e = (float(r["realized_ttft"]), float(r["realized_mean_itl"]), float(r["realized_e2e"]))
        ok = r["completed"] == "true"
        if t > T * 1e3: miss[inst]["ttft"] += 1; ok = False
        if i > I * 1e3: miss[inst]["itl"] += 1; ok = False
        if e > E * 1e3 or e == 0: miss[inst]["e2e"] += 1; ok = False
        if ok: good += 1
    out = [f"{os.path.basename(path)}: n={n} goodput={good/n:.3f} disagg={disagg/n:.2f}"]
    out.append("  placement: " + ", ".join(f"{k}={v} ({v/n:.0%})" for k, v in sorted(share.items())))
    for inst in sorted(miss):
        d = miss[inst]
        tot = share[inst]
        out.append("  misses on " + inst + ": " + ", ".join(
            f"{dim}={d[dim]} ({d[dim]/tot:.0%} of its {tot})" for dim in ("ttft", "itl", "e2e") if d[dim]))
    if not miss:
        out.append("  misses: none")
    return "\n".join(out)

if __name__ == "__main__":
    T, I, E = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    for p in sys.argv[4:]:
        print(analyze(p, T, I, E))
