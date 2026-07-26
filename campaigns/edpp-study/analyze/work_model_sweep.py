#!/usr/bin/env python3
"""Fig 1 (work model): aggregate per-rate Stage B bias reports into a load-on-x figure.

Reads the <wl>_rate<r>_bias.json reports written by repro_work_model_sweep.sh and
plots work-model relative error vs offered load, one series per workload, in three
panels: single-chunk prefill, chunked prefill, decode.

Each panel plots |median rel err| as the line with a [p90, p99] shaded band, both on
a symlog y-axis (errors span float-eps to ~10%). The single-chunk and decode panels
should hug the floor at every load (load-invariant, float-exact closed form); the
chunked panel carries the documented C_attn*(a_p^2 - sum s_r^2)/2 residual, which may
rise with load as batches chunk more aggressively.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (json key in the bias report, human panel title)
PANELS = [
    ("prefill_work_single_chunk", "prefill (single-chunk)"),
    ("prefill_work_chunked", "prefill (chunked)"),
    ("decode_work", "decode"),
]


def _load_points(sweep_dir):
    """Return {workload: [(rate, report), ...] sorted by rate}."""
    pat = re.compile(r"^(?P<wl>.+)_rate(?P<rate>[0-9.]+)_bias\.json$")
    by_wl = {}
    for path in glob.glob(os.path.join(sweep_dir, "*_bias.json")):
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            report = json.load(f)
        by_wl.setdefault(m["wl"], []).append((float(m["rate"]), report))
    for wl in by_wl:
        by_wl[wl].sort(key=lambda t: t[0])
    return by_wl


def _series(points, key):
    """Extract (rates, |median|, p90, p99) for one panel key, skipping empty groups."""
    rates, med, p90, p99 = [], [], [], []
    for rate, report in points:
        stat = report.get(key, {})
        if not stat.get("n"):
            continue
        rates.append(rate)
        med.append(abs(stat.get("median_rel_err", 0.0)))
        p90.append(stat.get("abs_rel_err_p90", 0.0))
        p99.append(stat.get("abs_rel_err_p99", 0.0))
    return rates, med, p90, p99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, help="dir of <wl>_rate<r>_bias.json reports")
    ap.add_argument("--plots", required=True, help="output PNG path")
    args = ap.parse_args()

    by_wl = _load_points(args.sweep_dir)
    if not by_wl:
        raise SystemExit(f"no *_bias.json reports found in {args.sweep_dir}")

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 4.2), sharey=True)
    colors = {wl: c for wl, c in zip(sorted(by_wl), plt.cm.tab10.colors)}

    for ax, (key, title) in zip(axes, PANELS):
        for wl in sorted(by_wl):
            rates, med, p90, p99 = _series(by_wl[wl], key)
            if not rates:
                continue
            c = colors[wl]
            ax.plot(rates, med, marker="o", color=c, label=wl)
            ax.fill_between(rates, p90, p99, color=c, alpha=0.18)
        ax.set_yscale("symlog", linthresh=1e-15)
        ax.set_xlabel("offered load (aggregate_rate)")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel("|relative error|  (median line, p90–p99 band)")
    axes[0].legend(title="workload", fontsize=8)
    fig.suptitle("Work-model accuracy vs load: realized vs closed-form per-request work")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.plots, dpi=140)
    print(f"wrote {args.plots}")


if __name__ == "__main__":
    raise SystemExit(main())
