# Workload Concerns

Known limitations and open issues with the inference-perf catalog workloads in this directory.
Concerns are categorized by root cause: BLIS schema gaps, inference-perf catalog gaps, or
target-model constraints.

---

## BLIS Schema Limitations

### Fixed `max_rounds` causes artificial session synchronization

**Affects:** interactive-chat, code-generation, deep-research

The catalog specifies distributed turn counts per session (lognormal for interactive-chat and
code-generation, normal for deep-research). BLIS only supports a single fixed `max_rounds`
value, so all sessions run for exactly the same number of turns. In practice this means
sessions that started at similar times all terminate together, creating artificial periodic
load bursts rather than the smooth steady-state produced by variable session lengths.

Fix requires a `max_rounds_distribution` field in `MultiTurnSpec`.

### Gaussian approximation of uniform output for batch-synthetic-data-generation

**Affects:** inference-perf-batch-synthetic-data-generation.yaml

The catalog specifies `distribution_type: uniform` for output tokens. The inference-perf tool
correctly implements this. BLIS has no uniform distribution type, so we approximate with
`gaussian(mean=4000, std=2500, [500, 8000])`. The gaussian bell-shapes around 4000 and
generates short outputs that wouldn't occur in the real workload, understating the sustained
near-maximum decode pressure the catalog intends ("Uniformly High — the model is prompted to
hit maximum context limits").

Fix requires adding `uniform` to BLIS's supported distribution types.

---

## Inference-Perf Catalog Gaps

### No production request rates specified

**Affects:** all workloads

The catalog defines concurrency *stress-test* stages (e.g. 10→50 concurrent sessions) rather
than characterizing steady-state production traffic rates. The aggregate rates in these BLIS
workloads (`aggregate_rate: 5.0`, `2.0`, etc.) are arbitrary defaults with no grounding in
real traffic data. A user running these workloads doesn't know whether they are in a
sub-saturated, saturated, or over-saturated regime without first sweeping rates manually.

**Recommendation:** sweep `aggregate_rate` to find the saturation point for your cluster
configuration rather than treating the defaults as representative. The catalog should be
extended with production rate guidance or reference concurrency-to-rate conversion factors.

### Bimodal distributions specified but not parameterized

**Affects:** deep-research (output), batch-summarization-rag (input)

The catalog correctly identifies bimodal distributions for these workloads (config.json:
`distribution_type: "bimodal"`) but provides only aggregate statistics (combined mean and
std_dev), not per-mode parameters (mode locations, mixing ratio). Without per-mode parameters,
no tool — including BLIS — can faithfully implement the bimodal shape. The inference-perf tool
approximates with `normal`; BLIS approximates with `lognormal` (deep-research output) and a
two-client decomposition with mode locations estimated from README examples
(batch-summarization-rag input).

**Recommendation:** the catalog should add per-mode parameters, e.g.:
```json
"distribution_type": "bimodal",
"modes": [
  {"mean": 100, "std_dev": 50, "weight": 0.85},
  {"mean": 3000, "std_dev": 800, "weight": 0.15}
]
```

---

## Target Model Constraints

### Prefix length significantly understated for code-generation and deep-research

**Affects:** inference-perf-code-generation.yaml, inference-perf-deep-research.yaml

The catalog targets 256K+ models (code-generation even sets `max_model_len: 262144`
explicitly). The catalog's dynamic system prompts have median ~90K tokens (code-generation)
and mean ~45K tokens (deep-research). Against llama-3.3-70b-instruct at 128K context, we
cap prefix lengths at 30K and 25K respectively. KV cache utilization and prefill compute in
these simulations will be materially lower than a real deployment against an appropriate
model.

No fix is possible with llama-3.3-70b-instruct. Use a 256K+ context model for full-fidelity
simulation of these workloads.
