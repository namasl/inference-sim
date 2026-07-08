#!/usr/bin/env python3
"""Scorer-vs-joint divergence summary for the EDPP joint P/D mechanism.

Reads a `--edpp-joint-trace` CSV (columns: request_id, clock, class, scorer_d,
joint_d, scorer_p, joint_p, agree_d, agree_p, j_scorer, j_joint, disaggregate)
and reports how often the joint drift-plus-penalty argmin overrides the
composable scorer, plus the DIRECTION of the divergence on the rows where it does.

Metrics
-------
  n                     total decision rows
  d_divergence_rate     share of rows where joint_d != scorer_d (agree_d == false)
  p_divergence_rate     share of rows where joint_p != scorer_p (agree_p == false)
  any_divergence_rate   share of rows where d OR p diverged
  On the DIVERGENT rows (any):
    dir_lower_J          share where j_joint <  j_scorer (joint strictly improved objective)
    dir_tie_J            share where j_joint == j_scorer (deterministic tie-break override)
    dir_higher_J         share where j_joint >  j_scorer  (MUST be ~0: argmin invariant;
                                                           nonzero => a bug in the joint decider)
    disagg_share         share of divergent rows that ended disaggregated

DIRECTION CAVEAT (honest scope): the `--edpp-joint-trace` schema carries only the
two objective values (j_scorer, j_joint) and the chosen instances, NOT the
per-candidate occupancy or a_p. So the direction we can attribute from this trace
is "did joint pick the lower-J candidate" (the argmin, by construction) vs a
tie-break. Finer "lower-occupancy / lower-a_p" attribution would need extra trace
columns; the reduction and lower-occupancy/cache-warmth behaviours are covered by
the Go unit tests (sim/edpp_joint_test.go: TestJoint_PicksLowerOccupancyDecode,
TestJoint_PrefersCacheWarmOverIdleCold). This analyzer is a LOCAL diagnostic.

Usage:
  joint_divergence.py summary --trace <edpp-joint-trace.csv> [--out <json>]
  joint_divergence.py selftest
"""
import argparse, csv, json, sys


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _b(s):
    return str(s).strip().lower() == "true"


def summarize(rows, tol=1e-12):
    # tol absorbs float-summation dust: J values within tol are a tie (the joint
    # then falls back to the deterministic lower-index tie-break). Without this,
    # candidates whose true J == 0 log as ~1e-21 and spuriously classify as
    # "higher J", faking an argmin-invariant violation.
    n = len(rows)
    d_div = [r for r in rows if not _b(r.get("agree_d", "true"))]
    p_div = [r for r in rows if not _b(r.get("agree_p", "true"))]
    div = [r for r in rows if (not _b(r.get("agree_d", "true"))) or (not _b(r.get("agree_p", "true")))]
    nd = len(div)
    lower = sum(1 for r in div if _f(r.get("j_joint")) < _f(r.get("j_scorer")) - tol)
    higher = sum(1 for r in div if _f(r.get("j_joint")) > _f(r.get("j_scorer")) + tol)
    tie = nd - lower - higher
    disagg = sum(1 for r in div if _b(r.get("disaggregate", "false")))
    frac = lambda k, d: (k / d) if d else 0.0
    return {
        "n": n,
        "d_divergence_rate": frac(len(d_div), n),
        "p_divergence_rate": frac(len(p_div), n),
        "any_divergence_rate": frac(nd, n),
        "n_divergent": nd,
        "direction_on_divergent": {
            "dir_lower_J": frac(lower, nd),
            "dir_tie_J": frac(tie, nd),
            "dir_higher_J": frac(higher, nd),
            "disagg_share": frac(disagg, nd),
        },
        "counts": {"d_div": len(d_div), "p_div": len(p_div),
                   "lower_J": lower, "tie_J": tie, "higher_J": higher},
    }


def cmd_summary(args):
    with open(args.trace) as f:
        rows = list(csv.DictReader(f))
    rep = summarize(rows)
    text = json.dumps(rep, indent=2)
    if args.out:
        open(args.out, "w").write(text + "\n")
    print(text)
    if rep["direction_on_divergent"]["dir_higher_J"] > 1e-9:
        print("WARNING: joint chose a STRICTLY HIGHER J on some divergent rows "
              "(argmin invariant violated).", file=sys.stderr)
    return 0


def cmd_selftest(_args):
    # Synthetic trace with known divergences:
    #   r0: d diverges, joint strictly lower J          -> lower_J
    #   r1: agree on both                               -> not divergent
    #   r2: p diverges, both J == 0 (tie-break)         -> tie_J, disagg
    #   r3: d diverges, j_joint == j_scorer (nonzero)   -> tie_J
    rows = [
        {"request_id": "r0", "agree_d": "false", "agree_p": "true",
         "j_scorer": "0.01", "j_joint": "0.004", "disaggregate": "false"},
        {"request_id": "r1", "agree_d": "true", "agree_p": "true",
         "j_scorer": "0.02", "j_joint": "0.02", "disaggregate": "false"},
        {"request_id": "r2", "agree_d": "true", "agree_p": "false",
         "j_scorer": "0", "j_joint": "0", "disaggregate": "true"},
        {"request_id": "r3", "agree_d": "false", "agree_p": "true",
         "j_scorer": "0.05", "j_joint": "0.05", "disaggregate": "false"},
    ]
    rep = summarize(rows)
    assert rep["n"] == 4, rep
    assert rep["counts"]["d_div"] == 2, rep
    assert rep["counts"]["p_div"] == 1, rep
    assert rep["n_divergent"] == 3, rep  # r0, r2, r3 (r1 agrees on both)
    assert abs(rep["d_divergence_rate"] - 0.5) < 1e-12, rep
    assert abs(rep["p_divergence_rate"] - 0.25) < 1e-12, rep
    assert abs(rep["any_divergence_rate"] - 0.75) < 1e-12, rep
    d = rep["direction_on_divergent"]
    assert rep["counts"]["lower_J"] == 1, rep   # r0
    assert rep["counts"]["tie_J"] == 2, rep     # r2, r3
    assert rep["counts"]["higher_J"] == 0, rep
    assert abs(d["dir_lower_J"] - 1 / 3) < 1e-12, rep
    assert abs(d["dir_tie_J"] - 2 / 3) < 1e-12, rep
    assert abs(d["disagg_share"] - 1 / 3) < 1e-12, rep  # only r2 disagg among the 3
    # Empty trace: all rates 0, no crash.
    empty = summarize([])
    assert empty["n"] == 0 and empty["any_divergence_rate"] == 0.0, empty
    print("joint_divergence selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("summary"); s.add_argument("--trace", required=True); s.add_argument("--out", default="")
    s.set_defaults(func=cmd_summary)
    t = sub.add_parser("selftest"); t.set_defaults(func=cmd_selftest)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
