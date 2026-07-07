#!/usr/bin/env python3
"""Per-decision counterfactual regret (see FINDINGS "Counterfactual regret").

Subcommands:
  capture-plan --outcome <pd-outcome-trace.csv> --out <plan.csv>
      Convert a policy's --pd-outcome-trace into a fixed-plan CSV
      (request_id, decode_instance, prefill_instance; "local" when not disaggregated).
  regret --sweep-dir <dir> [--out <json>]
      Read baseline.json + dev_<reqid>_<action>.json run-metrics in <dir>, compute
      per-request regret = max_action goodput(dev) - goodput(baseline) on TOTAL goodput.
"""
import argparse, csv, glob, json, os, re, sys


def goodput(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    if isinstance(m, list):  # per-instance array: average attainment
        vals = [x.get("slo_attainment", 0.0) for x in m]
        return float(sum(vals) / len(vals)) if vals else 0.0
    return float(m.get("slo_attainment", 0.0))


def capture_plan(args):
    rows_out = []
    with open(args.outcome) as f:
        for row in csv.DictReader(f):
            disagg = str(row.get("disaggregated", "")).strip().lower() == "true"
            rows_out.append({
                "request_id": row["request_id"],
                "decode_instance": row["decode_instance"],
                "prefill_instance": row["prefill_instance"] if disagg and row.get("prefill_instance") else "local",
            })
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["request_id", "decode_instance", "prefill_instance"])
        w.writeheader(); w.writerows(rows_out)
    return 0


def regret(args):
    base = goodput(os.path.join(args.sweep_dir, "baseline.json"))
    devs = {}  # reqid -> {action: goodput}
    for path in glob.glob(os.path.join(args.sweep_dir, "dev_*.json")):
        mobj = re.match(r"dev_(.+)_([^_]+)\.json$", os.path.basename(path))
        if not mobj:
            continue
        rid, action = mobj.group(1), mobj.group(2)
        devs.setdefault(rid, {})[action] = goodput(path)
    per_request = []
    for rid, actions in sorted(devs.items()):
        best_action = max(actions, key=actions.get)
        best_g = actions[best_action]
        reg = max(0.0, best_g - base)
        per_request.append({"request_id": rid, "baseline_goodput": base,
                            "hindsight_best": best_action if reg > 0 else "baseline",
                            "hindsight_best_goodput": best_g, "regret": reg})
    n = len(per_request)
    pos = [p for p in per_request if p["regret"] > 0]
    report = {"baseline_goodput": base, "n_requests": n,
              "frac_positive_regret": (len(pos) / n) if n else 0.0,
              "total_regret": sum(p["regret"] for p in per_request),
              "mean_regret": (sum(p["regret"] for p in per_request) / n) if n else 0.0,
              "per_request": per_request}
    text = json.dumps(report, indent=2)
    if args.out:
        open(args.out, "w").write(text + "\n")
    print(text)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    cp = sub.add_parser("capture-plan"); cp.add_argument("--outcome", required=True); cp.add_argument("--out", required=True)
    cp.set_defaults(func=capture_plan)
    rg = sub.add_parser("regret"); rg.add_argument("--sweep-dir", required=True); rg.add_argument("--out", default="")
    rg.set_defaults(func=regret)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
