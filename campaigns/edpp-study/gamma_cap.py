#!/usr/bin/env python3
"""Analytical goodput ceiling Gamma^cap per regime (paper Theorem 2 / eq:gcap).

Policy-independent: Monte-Carlo the least-work distribution W^min(r) from the
workload generator and frozen coefficients, then solve lambda * m(gamma) = |I|,
where m(gamma) is the lower partial mean of W^min. No simulation of any policy.

W^min(r) = min_d Wd_r(theta_d) + min_l Wp_r(theta_l)   (eq:wmin)
with, on the homogeneous H100 fleet of every main regime, both minima taken at
the single frozen coefficient set. No prefix recurs so uncached = full prompt.
The heterogeneous regime takes the cheaper (H100) instance for both stages.

Work model (sim/edpp_coeffs.go):
  Wp(ap, ar) = CPf*ap + CAttn*ap*(ar - ap/2)          # ap = ar (no cache)
  Wd(ar, o)  = C0*o  + C1*o*(ar + (o-1)/2)            # o > 0
All in microseconds; converted to seconds against the per-second arrival rate.
"""
import json
import math
import sys

import numpy as np

COEFF_PATH = "scripts/calibration/coeffs-llama70b-h100-tp4.json"


def load_coeffs(path):
    j = json.load(open(path))
    return dict(
        CPf=j["prefill"]["c_pf_us_per_token"],
        CAttn=j["prefill"]["c_attn_us_per_unit"],
        C0=j["decode"]["c0_us_per_req"],
        C1=j["decode"]["c1_us_per_token"],
    )


def Wp(c, ap, ar):
    return c["CPf"] * ap + c["CAttn"] * ap * (ar - ap / 2.0)


def Wd(c, ar, o):
    o = np.asarray(o, dtype=float)
    return np.where(o > 0, c["C0"] * o + c["C1"] * o * (ar + (o - 1.0) / 2.0), 0.0)


def sample_outputs(mean_o, sigma, n, rng):
    """Lognormal outputs matching the grid's outdist(): E[out]=mean_o, floored/capped."""
    if sigma == 0:
        return np.full(n, float(mean_o))
    mu = math.log(mean_o) - sigma * sigma / 2.0
    hi = int(mean_o * 8) + 16
    s = rng.lognormal(mu, sigma, size=n)
    return np.clip(s, 4, hi)


def gamma_cap(wmin_s, lam, n_inst):
    """Solve lambda * m(gamma) = |I|; return (Gamma^cap, rho) where rho is the
    marginal-capacity utilization lambda*E[W^min]/|I| (>=1 means the knee binds)."""
    w = np.sort(wmin_s)
    n = len(w)
    partial_mean = np.cumsum(w) / n           # m(k/n) = (1/n) sum of k smallest
    e_w = partial_mean[-1]                     # m(1) = E[W^min]
    rho = lam * e_w / n_inst
    if lam * e_w <= n_inst:
        return 1.0, rho
    # largest fraction k/n with lambda * m(k/n) <= |I|
    ok = lam * partial_mean <= n_inst
    k = int(np.count_nonzero(ok))
    return k / n, rho


def main():
    c = load_coeffs(COEFF_PATH)
    N = 2_000_000
    SIGMA = 0.4
    N_INST = 3  # 1P2D topology in every regime
    rng = np.random.default_rng(20260727)

    # name, input tokens, mean output, arrival rate (req/s)
    regimes = [
        ("decode", 256, 512, 16),
        ("mixed", 2048, 128, 16),
        ("prefill_lean", 8192, 64, 16),
        ("prefill_bound", 16000, 16, 8),
        ("heterogeneous", 256, 64, 10),
    ]

    print(f"{'regime':<14} {'lambda':>7} {'E[Wmin]ms':>10} {'rho':>7} {'Gamma^cap':>10}")
    for name, ar, mean_o, lam in regimes:
        o = sample_outputs(mean_o, SIGMA, N, rng)
        wmin_us = Wp(c, ar, ar) + Wd(c, ar, o)   # homogeneous / cheapest-instance
        wmin_s = wmin_us / 1e6
        gc, rho = gamma_cap(wmin_s, lam, N_INST)
        print(f"{name:<14} {lam:>7d} {wmin_us.mean()/1e3:>10.2f} {rho:>7.3f} {gc:>10.4f}")

    # Per-pool diagnostic (sharper than the lumped |I| form): decode work only on
    # |D|=2, prefill work only on |P|=1. Reported to stderr, not the Table.
    print("\n# per-pool marginal utilization (diagnostic; theorem uses lumped |I|)",
          file=sys.stderr)
    print(f"{'regime':<14} {'rho_dec(|D|=2)':>15} {'rho_pf(|P|=1)':>15}", file=sys.stderr)
    for name, ar, mean_o, lam in regimes:
        o = sample_outputs(mean_o, SIGMA, N, rng)
        rho_d = lam * (Wd(c, ar, o).mean() / 1e6) / 2.0
        rho_p = lam * (Wp(c, ar, ar) / 1e6) / 1.0
        print(f"{name:<14} {rho_d:>15.3f} {rho_p:>15.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
