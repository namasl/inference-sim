"""Steady-state windowing + offered-work helpers for the EDPP study.

steady_window: marks steady state when the running mean of per-request output
length stops moving (N̂_out has converged); returns the start index + a report.
offered_work: computes offered prefill/decode token rates from a spec variant.
"""
import math
import yaml


def steady_window(results, eps=0.02, win=200):
    """Mark steady state when the running mean of output_tokens stops moving.

    results: list of SimResult dicts, in completion order.
    Returns (start_index, report). start_index=0 if it never converges.
    """
    outs = [r["output_tokens"] for r in results]
    if len(outs) < 2 * win:
        return 0, {"converged": False, "reason": "too_few", "n": len(outs)}
    run_mean = []
    s = 0.0
    for i, v in enumerate(outs):
        s += v
        run_mean.append(s / (i + 1))
    start = 0
    for i in range(win, len(run_mean)):
        prev, cur = run_mean[i - win], run_mean[i]
        if prev > 0 and abs(cur - prev) / prev < eps:
            start = i
            break
    return start, {"converged": start > 0, "start": start,
                   "final_mean_out": run_mean[-1], "n": len(outs)}


def _lognormal_mean(params):
    """Mean of a lognormal given mu/sigma, clamped to [min,max] if present."""
    mu = params.get("mu")
    if mu is None:
        return params.get("mean", 0.0)
    sig = params.get("sigma", 0.0)
    return math.exp(mu + sig * sig / 2.0)


def offered_work(spec_path):
    """Offered prefill/decode token rates (tok/s) from a spec variant."""
    spec = yaml.safe_load(open(spec_path))
    rate = spec["aggregate_rate"]
    pf = dec = 0.0
    for c in spec["clients"]:
        f = c.get("rate_fraction", 1.0)
        idist = c["input_distribution"]
        if idist["type"] == "lognormal":
            mean_in = _lognormal_mean(idist["params"])
        else:
            mean_in = idist["params"].get("mean", 0.0)
        mean_in += c.get("prefix_length", 0)
        out = c["output_distribution"]["params"].get("mean", 0.0)
        pf += rate * f * mean_in
        dec += rate * f * out
    return {"prefill_tok_s": pf, "decode_tok_s": dec}
