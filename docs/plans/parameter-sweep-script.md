# Parameter Sweep Script Implementation Plan

## Executive Summary

Create a simple, self-contained Python script (`sweep_parameters.py`) that performs 1D parameter sweeps for BLIS simulations. The script will have all configuration at the top, no CLI interface, and output results as CSV files. This is a lightweight tool for quick experimentation and sanity checking BLIS behavior.

## Requirements

### Functional Requirements

1. **Sweep Parameters** (1D sweeps, one at a time):
   - ISL (Input Sequence Length): `--prompt-tokens-mean`
   - OSL (Output Sequence Length): `--output-tokens-mean`
   - TP (Tensor Parallelism): `--tp`
   - Batch Size: `--max-running-reqs` or `--max-scheduled-tokens`

2. **Collected Metrics**:
   - Throughput: `tokens_per_sec`, `throughput_per_gpu`
   - TTFT: `ttft_mean_ms`, `ttft_p95_ms`
   - ITL: `itl_mean_ms`
   - E2E: `e2e_mean_ms`, `e2e_p95_ms`
   - Completion: `completed_requests`, `dropped_unservable`

3. **Configuration**: All settings at top of script (no CLI, no external files)

4. **Output**: CSV file with sweep results

### Non-Functional Requirements

1. **Simple**: Single Python file, ~200-300 lines
2. **Self-contained**: No external dependencies except pandas
3. **Clear**: Inline comments explain usage
4. **Quick**: For experimentation, not production

## Design

### Script Structure (Simplified)

```python
#!/usr/bin/env python3
"""
BLIS Parameter Sweep Script

Simple script to sweep BLIS parameters and collect performance metrics.
Edit the configuration section below to customize sweeps.
"""

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Which parameter to sweep (uncomment one)
SWEEP_PARAM = "isl"  # Options: "isl", "osl", "tp", "max_running_reqs", "max_scheduled_tokens"

# Base configuration (fixed parameters)
BASE_CONFIG = {
    "model": "qwen/qwen3-14b",
    "hardware": "A100-80GB",
    "tp": 1,
    "num_instances": 1,
    "rate": 5.0,
    "num_requests": 100,
    "seed": 42,
    "prompt_tokens_mean": 512,
    "prompt_tokens_stdev": 128,
    "output_tokens_mean": 256,
    "output_tokens_stdev": 64,
    "max_running_reqs": 256,
    "max_scheduled_tokens": 8192,
}

# Sweep ranges for each parameter
SWEEP_RANGES = {
    "isl": [128, 256, 512, 1024, 2048],
    "osl": [64, 128, 256, 512, 1024],
    "tp": [1, 2, 4, 8],
    "max_running_reqs": [32, 64, 128, 256, 512],
    "max_scheduled_tokens": [2048, 4096, 8192, 16384],
}

# Output file
OUTPUT_CSV = f"sweep_results_{SWEEP_PARAM}.csv"

# ============================================================================
# Implementation (no need to edit below)
# ============================================================================

import subprocess
import json
import pandas as pd
import os
import sys

def find_blis_binary():
    """Find BLIS binary (env var, repo root, or auto-build)."""
    # ... implementation ...

def run_blis(config):
    """Run BLIS with given config, return (metrics_dict, error)."""
    # ... implementation ...

def sweep_parameter(param_name, values, base_config):
    """Run sweep for one parameter."""
    # ... implementation ...

if __name__ == "__main__":
    results_df = sweep_parameter(SWEEP_PARAM, SWEEP_RANGES[SWEEP_PARAM], BASE_CONFIG)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")
    print(results_df)
```

## Implementation Tasks

### Task 1: Create script skeleton
- Create `sweep_parameters.py` with configuration section
- Add imports: pandas, subprocess, json, os, sys
- Add docstring explaining usage

### Task 2: Implement BLIS runner
- `find_blis_binary()`: Auto-discover BLIS binary
- `run_blis(config)`: Execute BLIS subprocess, parse JSON output
- Return (metrics_dict, error_string)

### Task 3: Implement sweep function
- `sweep_parameter(param_name, values, base_config)`: Loop over values
- Build BLIS command for each point
- Collect results into list of dicts
- Return pandas DataFrame

### Task 4: Add parameter-specific logic
- ISL/OSL: Auto-compute min/max bounds for Gaussian validation
- TP: Pass through directly
- Batch size: Pass through directly

### Task 5: Add main execution
- Read SWEEP_PARAM from config
- Call sweep_parameter()
- Save DataFrame to CSV
- Print summary

## Success Criteria

1. ✅ Script is self-contained (~200-300 lines)
2. ✅ All configuration at top of file
3. ✅ No CLI arguments needed
4. ✅ Outputs CSV with sweep results
5. ✅ Clear inline comments explain usage
6. ✅ Works for all 5 sweep types

## Estimated Effort

- Task 1: 15 minutes
- Task 2: 30 minutes
- Task 3: 30 minutes
- Task 4: 15 minutes
- Task 5: 15 minutes
- **Total**: ~2 hours

## Next Steps

1. Create `sweep_parameters.py` with configuration section
2. Implement BLIS runner (adapt from demo/pd-disaggregation/blis_runner.py)
3. Implement sweep logic
4. Test with one sweep type
5. Verify CSV output format