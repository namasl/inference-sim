#!/usr/bin/env python3
"""Grid-v2 report: goodput table, worst-case regret, and the floor A/B, from
campaigns/edpp-study/out/grid_v2/*.json (written by repro_grid_v2.sh).

File naming: {cell}_{load}_{arm}_{seed}.json with arms
  nv a p k{beta} l lj dp vv vg vgo vvnf vgonf
Kairos is reported as the per-seed best over betas (matching the paper protocol).
"""
import json, glob, os, re, sys
from collections import defaultdict

OUT = sys.argv[1] if len(sys.argv) > 1 else "campaigns/edpp-study/out/grid_v2"
SEEDS = ["42", "7", "123"]
ARMS = ["nv", "a", "p", "k", "l", "lj", "dp", "vv", "vg", "vgo", "vvnf", "vgonf"]
LABEL = {"nv": "never", "a": "always", "p": "prefix16", "k": "kairos*", "l": "least-ttft",
         "lj": "lt-joint", "dp": "dpp", "vv": "dpVaR", "vg": "gp+orc", "vgo": "gp-dep",
         "vvnf": "dpVaR-nofloor", "vgonf": "gp-dep-nofloor"}
# regret is computed over the DEPLOYABLE, mutually-exclusive policy set (one floor
# convention at a time); the no-floor arms are the A/B comparison, and vg is oracle.
REGRET_ARMS = ["nv", "a", "p", "k", "l", "lj", "dp", "vv", "vgo"]

gp = {}
for f in glob.glob(f"{OUT}/*.json"):
    m = re.match(r"(.+)_(0\.7|1\.0)_([a-z0-9.]+)_(\d+)\.json$", os.path.basename(f))
    if not m:
        continue
    cell, load, arm, seed = m.groups()
    try:
        v = json.load(open(f))["per_class"]["standard"]["slo_attainment"]
    except Exception:
        continue
    gp[(cell, load, arm, seed)] = v

cells = sorted({(c, l) for (c, l, _, _) in gp})

def arm_seeds(cell, load, arm):
    if arm == "k":
        out = []
        for s in SEEDS:
            bs = [v for (c, l, a, sd), v in gp.items()
                  if c == cell and l == load and sd == s and a.startswith("k") and a not in ("l",)
                  and re.match(r"k[\d.]+$", a)]
            if bs:
                out.append(max(bs))
        return out
    return [gp[(cell, load, arm, s)] for s in SEEDS if (cell, load, arm, s) in gp]

print(f"{'cell':<14}{'load':<6}" + "".join(f"{LABEL[a]:>15}" for a in ARMS))
table = {}
for cell, load in cells:
    row = {}
    for a in ARMS:
        vs = arm_seeds(cell, load, a)
        row[a] = (sum(vs) / len(vs), min(vs)) if vs else (None, None)
    table[(cell, load)] = row
    print(f"{cell:<14}{load:<6}" + "".join(
        f"{row[a][0]:>10.3f}({row[a][1]:.2f})" if row[a][0] is not None else f"{'--':>15}" for a in ARMS))

print("\nWORST-CASE REGRET across cells x loads (reference = best of", ", ".join(LABEL[a] for a in REGRET_ARMS), ")")
regret = defaultdict(float)
where = {}
for (cell, load), row in table.items():
    avail = {a: row[a][0] for a in REGRET_ARMS if row[a][0] is not None}
    if not avail:
        continue
    ref = max(avail.values())
    for a, v in avail.items():
        r = ref - v
        if r > regret[a] - 1e-12 and r > regret[a]:
            where[a] = f"{cell}@{load}"
        regret[a] = max(regret[a], r)
for a in sorted(REGRET_ARMS, key=lambda x: regret[x]):
    print(f"  {LABEL[a]:<14} {regret[a]:.3f}   (at {where.get(a,'-')})")

print("\nFLOOR A/B (floored minus no-floor, mean goodput):")
for (cell, load), row in table.items():
    for base, nf in [("vv", "vvnf"), ("vgo", "vgonf")]:
        if row[base][0] is not None and row[nf][0] is not None:
            d = row[base][0] - row[nf][0]
            if abs(d) > 0.005:
                print(f"  {cell}@{load} {LABEL[base]:<8} {d:+.3f}")
