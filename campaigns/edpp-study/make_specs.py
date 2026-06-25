#!/usr/bin/env python3
import sys, pathlib, re

BASE = {
    "rag":   "inference-perf-batch-summarization-rag.yaml",
    "synth": "inference-perf-batch-synthetic-data-generation.yaml",
}
RATES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # provisional; Task 5 fixes final points around the knee
OUT = pathlib.Path("campaigns/edpp-study/specs"); OUT.mkdir(parents=True, exist_ok=True)

def rewrite(text, rate, nreq=20000):
    text = re.sub(r"^aggregate_rate:.*$", f"aggregate_rate: {rate}", text, flags=re.M)
    text = re.sub(r"^num_requests:.*$", f"num_requests: {nreq}", text, flags=re.M)
    return text

for name, base in BASE.items():
    src = pathlib.Path(base).read_text()
    for r in RATES:
        (OUT / f"{name}_rate{r}.yaml").write_text(rewrite(src, r))
        print(f"wrote {name}_rate{r}.yaml")
