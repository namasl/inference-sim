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
