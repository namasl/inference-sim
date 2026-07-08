#!/usr/bin/env python3
import sys, pathlib, re

BASE = {
    "rag":   "inference-perf-batch-summarization-rag.yaml",
    "synth": "inference-perf-batch-synthetic-data-generation.yaml",
}
RATES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # provisional; Task 5 fixes final points around the knee
OUT = pathlib.Path("campaigns/edpp-study/specs"); OUT.mkdir(parents=True, exist_ok=True)

def rewrite(text, rate, name, nreq=5000):
    text = re.sub(r"^aggregate_rate:.*$", f"aggregate_rate: {rate}", text, flags=re.M)
    text = re.sub(r"^num_requests:.*$", f"num_requests: {nreq}", text, flags=re.M)
    # RAG: split the two clients into distinct SLO classes so per-class SLO/τ are
    # meaningful. vector-qa (first client) is interactive; doc-read stays batch.
    # synth has a single client; leave its slo_class untouched.
    if name == "rag":
        text = text.replace('slo_class: "batch"', 'slo_class: "standard"', 1)
    return text

for name, base in BASE.items():
    src = pathlib.Path(base).read_text()
    for r in RATES:
        (OUT / f"{name}_rate{r}.yaml").write_text(rewrite(src, r, name))
        print(f"wrote {name}_rate{r}.yaml")

# synth_cf.yaml: small saturating synth trace for the counterfactual-regret harness
# (repro_counterfactual.sh). 800 requests at rate 2.0 saturates 1P2D (baseline
# goodput ~0.98, not 1.0) while keeping each single-request deviation's aggregate
# leverage measurable and K*(|A|-1) full sims fast; the 5000-req trace dilutes a
# one-request deviation to ~0. Same decode-bound workload as synth_rate2.0.yaml.
_synth = pathlib.Path(BASE["synth"]).read_text()
(OUT / "synth_cf.yaml").write_text(rewrite(_synth, 2.0, "synth", nreq=800))
print("wrote synth_cf.yaml")

# synth_asym.yaml: CACHE-ASYMMETRIC counterpart to synth_cf for the joint-mechanism
# sweep (repro_joint.sh). synth_cf shares a 2000-token system prompt across every
# request (prefix_group) so both decode nodes stay ~equally cache-warm and the
# `precise-prefix-cache` scorer term is near-constant. synth_asym instead gives
# every request a UNIQUE large prompt (no prefix_group => no shared prefix) with a
# large input distribution, so as the run proceeds the two decode nodes genuinely
# diverge in cache warmth and a_p differs across candidates — the regime where the
# joint objective's cache/occupancy lever should bite differently than the reduced
# fixed-decode rule. Same rate/size as synth_cf for a controlled comparison.
_asym = rewrite(pathlib.Path(BASE["synth"]).read_text(), 2.0, "synth", nreq=800)
# Drop the shared prefix: remove prefix_group + prefix_length lines entirely.
_asym = re.sub(r"^\s*prefix_group:.*$\n?", "", _asym, flags=re.M)
_asym = re.sub(r"^\s*prefix_length:.*$\n?", "", _asym, flags=re.M)
# Large unique inputs: raise the lognormal input mean/spread so per-request KV
# footprint is large and unique (median ~2000 tok, heavy tail).
_asym = _asym.replace("mu: 5.259", "mu: 7.6").replace("sigma: 1.3824", "sigma: 1.0")
(OUT / "synth_asym.yaml").write_text(_asym)
print("wrote synth_asym.yaml")
