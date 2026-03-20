# BLIS CLI Flags Reference

This document provides a comprehensive reference for all `blis run` command-line flags, including their purpose, valid inputs, default values, and interactions with other flags.

## Table of Contents

- [Basic Simulation Flags](#basic-simulation-flags)
- [Model and Hardware Configuration](#model-and-hardware-configuration)
- [Workload Generation](#workload-generation)
- [KV Cache Configuration](#kv-cache-configuration)
- [Cluster Configuration](#cluster-configuration)
- [Policy Configuration](#policy-configuration)
- [Tiered KV Cache](#tiered-kv-cache)
- [PD Disaggregation](#pd-disaggregation)
- [Decision Tracing](#decision-tracing)
- [Results Output](#results-output)

---

## Basic Simulation Flags

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--seed` | int64 | 42 | Any int64 | Random seed for request generation and simulation | Overrides `workload-spec` seed when both provided |
| `--horizon` | int64 | MaxInt64 | > 0 | Total simulation time in ticks | Required if `num-requests` is 0 or unset |
| `--log` | string | "warn" | trace, debug, info, warn, error, fatal, panic | Log level for diagnostic messages (stderr). Simulation results always print to stdout | None |
| `--defaults-filepath` | string | "defaults.yaml" | Valid file path | Path to defaults file containing trained coefficients and presets | Used by all latency models for coefficient loading |

---

## Model and Hardware Configuration

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--model` | string | "" (required) | Any string | LLM model name (e.g., "qwen/qwen3-14b"). Normalized to lowercase internally | **Required**. Used for coefficient lookup and HF config resolution |
| `--hardware` | string | "" | Valid GPU type from hardware_config.json | GPU type (e.g., "A100", "H100") | **Required** for `roofline`, `crossmodel`, `trained-roofline` backends |
| `--tp` | int | 0 | > 0 | Tensor parallelism degree | **Required** for analytical latency models (`roofline`, `crossmodel`, `trained-roofline`) |
| `--vllm-version` | string | "" | Version string | vLLM version for coefficient lookup | Optional; defaults loaded from defaults.yaml if available |
| `--latency-model` | string | "" (blackbox) | blackbox, roofline, crossmodel, trained-roofline | Latency estimation backend | `roofline`/`crossmodel`/`trained-roofline` require `--hardware` and `--tp`. Auto-fetches HF configs when needed |
| `--max-model-len` | int64 | 0 (unlimited) | >= 0 | Max total sequence length (input + output). 0 = unlimited | Auto-derived from HF config for analytical backends when not set. Capped at KV-feasible maximum |
| `--model-config-folder` | string | "" | Valid directory path | Path to folder containing HuggingFace config.json | Auto-resolved for analytical backends via cache → HF fetch → bundled fallback |
| `--hardware-config` | string | "" | Valid file path | Path to hardware_config.json | Auto-resolved for analytical backends (uses bundled default) |

### Latency Model Interactions

- **Blackbox mode** (default): Uses `--alpha-coeffs` and `--beta-coeffs` from defaults.yaml or CLI
- **Roofline mode**: Requires `--hardware`, `--tp`, and HF config. Replaces beta coefficients with analytical step time. Still uses alpha coefficients for queueing time
- **Crossmodel mode**: Requires `--hardware`, `--tp`, and HF config. Uses 7 global coefficients (4 beta + 3 alpha) from `crossmodel_defaults` in defaults.yaml
- **Trained-roofline mode**: Requires `--hardware`, `--tp`, and HF config. Uses 10 global coefficients (7 beta + 3 alpha) from `trained_roofline_defaults` in defaults.yaml

---

## Workload Generation

### Workload Source Flags

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--workload-spec` | string | "" | Valid YAML file path | Path to v2 workload specification file | **Takes precedence** over `--workload`. Overrides all distribution flags |
| `--workload` | string | "distribution" | chatbot, summarization, contentgen, multidoc, distribution, traces | Workload preset or generation mode | Ignored if `--workload-spec` is set |
| `--workload-traces-filepath` | string | "" | Valid CSV file path | Path to CSV trace file | **Required** when `--workload=traces` |

### Distribution Parameters

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--rate` | float64 | 1.0 | > 0, finite | Request arrival rate (requests/second) | Required for `distribution` mode and presets. Ignored when using `--workload-spec` |
| `--num-requests` | int | 100 | >= 0 | Total number of requests to generate | Overrides `workload-spec` num_requests when both provided. Required if `--horizon` is MaxInt64 |
| `--prefix-tokens` | int | 0 | >= 0 | Number of shared prefix tokens | Used in `distribution` mode |
| `--prompt-tokens` | int | 512 | [prompt-tokens-min, prompt-tokens-max] | Mean prompt token count | Used in `distribution` mode |
| `--prompt-tokens-stdev` | int | 256 | >= 0, [prompt-tokens-min, prompt-tokens-max] | Prompt token standard deviation | Used in `distribution` mode |
| `--prompt-tokens-min` | int | 2 | >= 1 | Minimum prompt tokens | Must be <= `prompt-tokens-max` |
| `--prompt-tokens-max` | int | 7000 | >= 1 | Maximum prompt tokens | Must be >= `prompt-tokens-min` |
| `--output-tokens` | int | 512 | [output-tokens-min, output-tokens-max] | Mean output token count | Used in `distribution` mode |
| `--output-tokens-stdev` | int | 256 | >= 0, [output-tokens-min, output-tokens-max] | Output token standard deviation | Used in `distribution` mode |
| `--output-tokens-min` | int | 2 | >= 1 | Minimum output tokens | Must be <= `output-tokens-max` |
| `--output-tokens-max` | int | 7000 | >= 1 | Maximum output tokens | Must be >= `output-tokens-min` |

### Workload Validation Rules

- `prompt-tokens-min` must be <= `prompt-tokens-max`
- `output-tokens-min` must be <= `output-tokens-max`
- `prompt-tokens` and `prompt-tokens-stdev` must be in range [min, max]
- `output-tokens` and `output-tokens-stdev` must be in range [min, max]
- Either `num-requests` > 0 OR `horizon` < MaxInt64 must be set to bound generation

---

## KV Cache Configuration

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--total-kv-blocks` | int64 | 1000000 | > 0 | Total GPU KV cache blocks | Auto-calculated for analytical backends when not set. Used for capacity validation |
| `--block-size-in-tokens` | int64 | 16 | > 0 | Tokens per KV cache block | Used in KV capacity calculations and auto-derivation |
| `--max-num-running-reqs` | int64 | 256 | > 0 | Maximum concurrent running requests | Batch formation constraint |
| `--max-num-scheduled-tokens` | int64 | 2048 | > 0 | Maximum total new tokens across running requests | Batch formation constraint |
| `--long-prefill-token-threshold` | int64 | 0 | >= 0 | Prefill length threshold for chunked prefill. 0 = disabled | Enables chunked prefill when request prefill exceeds this value |
| `--alpha-coeffs` | []float64 | [0.0, 0.0, 0.0] | Any float64 values | Alpha coefficients for queueing time estimation | Loaded from defaults.yaml when not provided. Used by all latency models |
| `--beta-coeffs` | []float64 | [0.0, 0.0, 0.0] | Any float64 values | Beta coefficients for step time estimation | Loaded from defaults.yaml for blackbox mode. Replaced by analytical computation in roofline/crossmodel/trained-roofline |

### KV Cache Auto-Calculation

When using analytical latency models (`roofline`, `crossmodel`, `trained-roofline`) and `--total-kv-blocks` is not explicitly set:
- Blocks are auto-calculated from model architecture + GPU memory
- Formula matches llm-d-benchmark `capacity_planner.py`
- Supports dense and MoE models
- Requires `MemoryGiB` field in hardware_config.json

---

## Cluster Configuration

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--num-instances` | int | 1 | >= 1 | Number of instances in the cluster | Single-instance mode when 1, cluster mode when > 1 |

---

## Policy Configuration

### Admission Control

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--admission-policy` | string | "always-admit" | always-admit, token-bucket, reject-all | Admission control policy | `token-bucket` requires `--token-bucket-capacity` and `--token-bucket-refill-rate` |
| `--admission-latency` | int64 | 0 | >= 0 | Admission decision latency in microseconds | Applied to all admission decisions |
| `--token-bucket-capacity` | float64 | 10000 | > 0, finite | Token bucket capacity | **Required** when `--admission-policy=token-bucket` |
| `--token-bucket-refill-rate` | float64 | 1000 | > 0, finite | Token bucket refill rate (tokens/second) | **Required** when `--admission-policy=token-bucket` |

### Routing

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--routing-policy` | string | "round-robin" | round-robin, least-loaded, weighted, always-busiest | Request routing policy | `weighted` uses `--routing-scorers` |
| `--routing-latency` | int64 | 0 | >= 0 | Routing decision latency in microseconds | Applied to all routing decisions |
| `--routing-scorers` | string | "" | Comma-separated name:weight pairs | Scorer configuration for weighted routing (e.g., "queue-depth:2,kv-utilization:2") | **Only applies** to `--routing-policy=weighted`. Default: "prefix-affinity:3,queue-depth:2,kv-utilization:2" |

Valid scorer names: `queue-depth`, `kv-utilization`, `load-balance`, `prefix-affinity`

### Priority and Scheduling

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--priority-policy` | string | "constant" | constant, slo-based, inverted-slo | Request priority assignment policy | Used by priority-aware schedulers |
| `--scheduler` | string | "fcfs" | fcfs, priority-fcfs, sjf, reverse-priority | Instance-level scheduling policy | `priority-fcfs` and `reverse-priority` use `--priority-policy` |

### Policy Bundle

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--policy-config` | string | "" | Valid YAML file path | Path to policy bundle YAML file | CLI flags override YAML values when both provided |

Policy bundle can specify: admission policy, routing policy, priority policy, scheduler, and scorer configurations.

---

## Tiered KV Cache

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--kv-cpu-blocks` | int64 | 0 | >= 0 | CPU tier KV cache blocks. 0 = disabled (single-tier mode) | Typical: 1/3 of `--total-kv-blocks`. Enables GPU↔CPU offloading |
| `--kv-offload-threshold` | float64 | 0.9 | [0, 1] | GPU utilization threshold for offloading to CPU | Only applies when `--kv-cpu-blocks > 0` |
| `--kv-transfer-bandwidth` | float64 | 100.0 | > 0, finite | CPU↔GPU transfer rate in blocks/tick | **Required** when `--kv-cpu-blocks > 0` |
| `--kv-transfer-base-latency` | int64 | 0 | >= 0 | Fixed per-transfer latency in ticks | Applied to all CPU↔GPU transfers |
| `--snapshot-refresh-interval` | int64 | 0 | >= 0 | Prometheus snapshot refresh interval in microseconds. 0 = immediate | Affects routing signal freshness (INV-7) |

---

## PD Disaggregation

### Pool Configuration

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--prefill-instances` | int | 0 | >= 0 | Number of prefill-dedicated instances. 0 = disabled | **Both** `--prefill-instances` and `--decode-instances` must be > 0 to enable PD mode |
| `--decode-instances` | int | 0 | >= 0 | Number of decode-dedicated instances. 0 = disabled | **Both** `--prefill-instances` and `--decode-instances` must be > 0 to enable PD mode |
| `--pd-decider` | string | "never" | never, always, prefix-threshold, direct-to-decode | Disaggregation decision policy | `prefix-threshold` uses `--pd-prefix-threshold`. `direct-to-decode` uses `--pd-direct-decode-threshold`. Requires PD mode enabled |
| `--pd-prefix-threshold` | int | 512 | >= 0 | Non-cached token threshold for prefix-threshold decider | **Only applies** when `--pd-decider=prefix-threshold` |
| `--pd-direct-decode-threshold` | int | 256 | >= 0 | Input token threshold for direct-to-decode decider | **Only applies** when `--pd-decider=direct-to-decode`. Requests with < threshold tokens go direct to decode |

### KV Transfer Configuration

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--pd-transfer-bandwidth` | float64 | 25.0 | > 0, finite | Inter-instance KV transfer bandwidth in GB/s | Used for transfer duration calculation |
| `--pd-transfer-base-latency` | float64 | 0.05 | >= 0, finite | Inter-instance KV transfer base latency in ms | Fixed overhead per transfer |
| `--pd-kv-bytes-per-token` | int | 512 | > 0 | KV cache bytes per token | Used for transfer size calculation |
| `--pd-transfer-contention` | bool | false | true/false | Enable fair-share bandwidth contention model | When true, concurrent transfers share bandwidth equally (INV-P2-2) |

### Interference Configuration

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--pd-interference-prefill` | float64 | 0 | [0, 100] | Interference factor for prefill-dominant batches. 0 = disabled | Multiplier = 1 + factor × (minority/total). Factor=0.5 at even split → 1.25× slowdown |
| `--pd-interference-decode` | float64 | 0 | [0, 100] | Interference factor for decode-dominant batches. 0 = disabled | Multiplier = 1 + factor × (minority/total). Factor=0.5 at even split → 1.25× slowdown |

### Per-Pool Routing

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--prefill-routing-scorers` | string | "" | Comma-separated name:weight pairs | Scorer weights for prefill pool routing | Uses global `--routing-scorers` when not set |
| `--decode-routing-scorers` | string | "" | Comma-separated name:weight pairs | Scorer weights for decode pool routing | Uses global `--routing-scorers` when not set |

### Per-Pool Hardware Overrides

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--prefill-tp` | int | 0 | > 0 when set | Tensor parallelism for prefill pool. 0 = use global `--tp` | Triggers per-pool KV auto-calculation when differs from global |
| `--decode-tp` | int | 0 | > 0 when set | Tensor parallelism for decode pool. 0 = use global `--tp` | Triggers per-pool KV auto-calculation when differs from global |
| `--prefill-hardware` | string | "" | Valid GPU type | GPU type for prefill pool. Empty = use global `--hardware` | Triggers per-pool KV auto-calculation when differs from global |
| `--decode-hardware` | string | "" | Valid GPU type | GPU type for decode pool. Empty = use global `--hardware` | Triggers per-pool KV auto-calculation when differs from global |
| `--prefill-latency-model` | string | "" | blackbox, roofline, crossmodel, trained-roofline | Latency backend for prefill pool. Empty = use global `--latency-model` | Must be valid latency backend |
| `--decode-latency-model` | string | "" | blackbox, roofline, crossmodel, trained-roofline | Latency backend for decode pool. Empty = use global `--latency-model` | Must be valid latency backend |
| `--prefill-max-model-len` | int64 | 0 | > 0 when set | Max model length for prefill pool. 0 = use global `--max-model-len` | Auto-capped when pool KV capacity < global |
| `--decode-max-model-len` | int64 | 0 | > 0 when set | Max model length for decode pool. 0 = use global `--max-model-len` | Auto-capped when pool KV capacity < global |

### PD Mode Validation Rules

- Pool topology: `prefill-instances + decode-instances <= num-instances`
- Both `--prefill-instances` and `--decode-instances` must be > 0 to enable PD mode
- `--pd-decider` (except "never") requires PD mode enabled
- Per-pool flags are ignored when PD mode is disabled

---

## Decision Tracing

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--trace-level` | string | "none" | none, decisions | Decision trace verbosity | "decisions" enables admission, routing, and disaggregation tracing |
| `--counterfactual-k` | int | 0 | >= 0 | Number of counterfactual candidates per routing decision | **Only applies** when `--trace-level=decisions`. 0 = disabled |
| `--summarize-trace` | bool | false | true/false | Print trace summary after simulation | **Only applies** when `--trace-level=decisions` |

---

## Fitness Evaluation

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--fitness-weights` | string | "" | Comma-separated key:value pairs | Fitness function weights (e.g., "throughput:0.5,p99_ttft:0.3") | Prints fitness score and components when set |

Valid fitness keys: `throughput`, `p50_ttft`, `p95_ttft`, `p99_ttft`, `p50_tpot`, `p95_tpot`, `p99_tpot`, `p50_e2e`, `p95_e2e`, `p99_e2e`, `rejection_rate`, `fairness`

---

## Results Output

| Flag | Type | Default | Valid Range | Description | Interactions |
|------|------|---------|-------------|-------------|--------------|
| `--results-path` | string | "" | Valid file path | File path to save JSON results | Results always print to stdout; this flag also saves to file |

---

## Flag Precedence Rules

When multiple configuration sources are provided, the precedence order is:

1. **CLI flags** (highest priority)
2. **Workload spec file** (`--workload-spec`)
3. **Policy bundle file** (`--policy-config`)
4. **Defaults file** (`--defaults-filepath`)
5. **Built-in defaults** (lowest priority)

### Examples

- `--seed` CLI flag overrides `workload-spec` seed
- `--num-requests` CLI flag overrides `workload-spec` num_requests
- `--admission-policy` CLI flag overrides `policy-config` admission policy
- `--routing-scorers` CLI flag overrides `policy-config` scorer configuration

---

## Common Flag Combinations

### Blackbox Mode (Default)
```bash
blis run --model qwen/qwen3-14b --num-requests 100 --rate 10
```
Uses trained alpha/beta coefficients from defaults.yaml.

### Roofline Mode
```bash
blis run --model qwen/qwen3-14b --latency-model roofline --hardware A100 --tp 1 --num-requests 100 --rate 10
```
Auto-fetches HF config, uses analytical step time estimation.

### Cluster with Weighted Routing
```bash
blis run --model qwen/qwen3-14b --num-instances 4 --routing-policy weighted --routing-scorers "queue-depth:2,kv-utilization:2" --num-requests 100 --rate 10
```

### PD Disaggregation
```bash
blis run --model qwen/qwen3-14b --num-instances 4 --prefill-instances 2 --decode-instances 2 --pd-decider prefix-threshold --pd-prefix-threshold 512 --num-requests 100 --rate 10
```

### Tiered KV Cache
```bash
blis run --model qwen/qwen3-14b --kv-cpu-blocks 333333 --kv-offload-threshold 0.9 --kv-transfer-bandwidth 100 --num-requests 100 --rate 10
```

### Decision Tracing with Counterfactuals
```bash
blis run --model qwen/qwen3-14b --num-instances 4 --trace-level decisions --counterfactual-k 3 --summarize-trace --num-requests 100 --rate 10
```

---

## Validation Errors

Common validation errors and their causes:

- **"--latency-model roofline requires --hardware"**: Must specify GPU type for analytical backends
- **"--latency-model roofline requires --tp > 0"**: Must specify tensor parallelism for analytical backends
- **"--pd-decider requires both --prefill-instances and --decode-instances"**: PD mode requires both pools configured
- **"--token-bucket-capacity must be > 0"**: Token bucket policy requires positive capacity
- **"prompt-tokens-min must be <= prompt-tokens-max"**: Token range validation failed
- **"num-instances must be >= 1"**: At least one instance required
- **"--total-kv-blocks must be > 0"**: KV cache capacity must be positive
- **"Invalid PD pool topology"**: Pool instance counts exceed total instances

---

## Notes

- All time values are in **microseconds** unless otherwise specified (e.g., `--horizon` is in ticks, transfer latencies may be in ms)
- All bandwidth values are in their specified units (GB/s for PD transfers, blocks/tick for tiered KV)
- Flag names use kebab-case (e.g., `--num-instances`, not `--numInstances`)
- Boolean flags don't require a value: `--summarize-trace` is equivalent to `--summarize-trace=true`
- Comma-separated lists (scorers, coefficients) must not contain spaces
- File paths can be relative or absolute
