"""EDPP study report: tier-1 outcomes (summary.csv), detective regret join
(regret.csv), and outcome-vs-load plots. Run from the analyze/ directory.

Filenames: results_<wl>_rate<R>_<decider>_<tag>.json and
decisions_<wl>_rate<R>_edpp_<tag>.csv, where tag in {agg,1P3D,2P2D,3P1D}.
Decider names contain hyphens, tags do not, so split on the LAST underscore.
"""
import json, glob, csv, re, os
from window import steady_window, offered_work

# Per-(workload, slo_class) SLO targets in microseconds — must match the
# --slo-ttft / --edpp-tau-ttft-classes values passed in sweep.sh.
SLO_TTFT_US = {
    ("rag", "standard"): 500_000, ("rag", "batch"): 5_000_000,
    ("synth", "batch"): 2_000_000,
}
SLO_ITL_US = {
    ("rag", "standard"): 150_000, ("rag", "batch"): 200_000,
    ("synth", "batch"): 150_000,
}
DEFAULT_TTFT_US = 500_000  # fallback for unknown (workload,class)


DEFAULT_ITL_US = 150_000


def ttft_target(wl, r):
    return SLO_TTFT_US.get((wl, r.get("slo_class", "")), DEFAULT_TTFT_US)


def itl_target(wl, r):
    return SLO_ITL_US.get((wl, r.get("slo_class", "")), DEFAULT_ITL_US)


def pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, int(p * len(xs)) - 1)] if xs else 0.0


def trim_tail(rs, frac=0.05):
    return rs[:int(len(rs) * (1 - frac))] if len(rs) > 20 else rs


def load_cell(path):
    rs = json.load(open(path))
    start, _ = steady_window(rs)
    return trim_tail(rs[start:])  # steady-state head trim + drain tail trim


def tier1(wl, rs):
    ttft = [r["ttft_us"] for r in rs]
    itl = [r.get("itl_mean_us", 0) for r in rs]
    disagg = sum(1 for r in rs if r.get("was_disaggregated"))
    # SLO attainment uses each request's own per-class target (standard vs batch).
    ttft_ok = sum(r["ttft_us"] <= ttft_target(wl, r) for r in rs)
    # ITL attainment counts only requests with a recorded ITL (>0; needs ≥2 output tokens).
    itl_rs = [r for r in rs if r.get("itl_mean_us", 0) > 0]
    itl_ok = sum(r["itl_mean_us"] <= itl_target(wl, r) for r in itl_rs)
    return {
        "n": len(rs),
        "ttft_p50": pct(ttft, .50), "ttft_p99": pct(ttft, .99),
        "itl_p99": pct(itl, .99),
        "slo_ttft_attain": ttft_ok / len(rs) if rs else 0,
        "slo_itl_attain": itl_ok / len(itl_rs) if itl_rs else 0,
        "disagg_frac": disagg / len(rs) if rs else 0,
    }


def parse_name(f):
    m = re.match(r".*results_(\w+)_rate([\d.]+)_(.+)\.json", f)
    wl, rate, suffix = m.group(1), m.group(2), m.group(3)
    decider, tag = suffix.rsplit("_", 1)  # "edpp_2P2D"->("edpp","2P2D"); "never_agg"->("never","agg")
    return wl, rate, decider, tag


def build_summary():
    rows = []
    for f in sorted(glob.glob("../out/results_*.json")):
        wl, rate, dec, split = parse_name(f)
        rs = load_cell(f)
        work = offered_work(f"../specs/{wl}_rate{rate}.yaml")
        rows.append({"workload": wl, "rate": rate, "decider": dec, "split": split,
                     **work, **tier1(wl, rs)})
    if not rows:
        print("build_summary: no results_*.json found — run sweep.sh first"); return
    with open("../out/summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote summary.csv:", len(rows), "cells")


def regret_join():
    """EDPP requests that missed the TTFT SLO, joined to the dominant rule term
    in the decision trace — explains WHY EDPP chose its side for each miss."""
    out = []
    for f in sorted(glob.glob("../out/results_*_edpp_*.json")):
        wl, rate, dec, split = parse_name(f)
        # Join contract: replay request IDs are exactly "request_<int>". SimResult
        # stores the int (request_ stripped); the decision CSV stores the raw req.ID
        # ("request_<int>"). Re-prefix the SimResult int so both sides match.
        rs = {f'request_{r["request_id"]}': r for r in json.load(open(f))}
        dec_path = f"../out/decisions_{wl}_rate{rate}_edpp_{split}.csv"
        if not os.path.exists(dec_path):
            continue
        for d in csv.DictReader(open(dec_path)):
            r = rs.get(d["request_id"])
            if not r or r["ttft_us"] <= ttft_target(wl, r):
                continue  # only per-class SLO misses
            terms = {k: abs(float(d[k])) for k in
                     ("balance_term_d", "balance_term_p", "transfer_term", "ttft_term", "itl_term")
                     if d.get(k)}
            dom = max(terms, key=terms.get) if terms else "?"
            out.append({"workload": wl, "rate": rate, "split": split,
                        "request_id": d["request_id"], "ttft_us": r["ttft_us"],
                        "disaggregated": d.get("disaggregate"), "dominant_term": dom})
    with open("../out/regret.csv", "w", newline="") as fh:
        if out:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
    print("wrote regret.csv:", len(out), "SLO-miss decisions")


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open("../out/summary.csv")))
    SPLITS = ("1P3D", "2P2D", "3P1D")

    def series(wl, dec, split):
        return sorted((float(r["prefill_tok_s"]), float(r["ttft_p99"]) / 1e6) for r in rows
                      if r["workload"] == wl and r["decider"] == dec and r["split"] == split)

    for wl in ("rag", "synth"):
        never = sorted((float(r["prefill_tok_s"]), float(r["ttft_p99"]) / 1e6) for r in rows
                       if r["workload"] == wl and r["decider"] == "never")
        for split in SPLITS:
            plt.figure()
            for dec in ("edpp", "always", "prefix-threshold"):
                pts = series(wl, dec, split)
                if pts:
                    xs, ys = zip(*pts)
                    plt.plot(xs, ys, marker="o", label=f"{dec} {split}")
            if never:
                xs, ys = zip(*never)
                plt.plot(xs, ys, marker="s", ls="--", color="gray", label="never (4, all-local)")
            plt.xlabel("offered prefill work (tok/s)"); plt.ylabel("TTFT p99 (s)")
            plt.title(f"{wl} @ {split}: TTFT p99 vs offered prefill work"); plt.legend()
            plt.savefig(f"../out/ttft_p99_{wl}_{split}.png", dpi=120); plt.close()
    print("wrote plots")


def main():
    build_summary()
    regret_join()
    plots()


if __name__ == "__main__":
    main()
