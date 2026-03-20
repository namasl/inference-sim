import subprocess
import json
import os
from typing import Optional, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

# Timeout per BLIS run (seconds)
TIMEOUT = 300

def find_blis_binary() -> str:
    """Find the BLIS binary in the repository root.
    
    Returns:
        str: Absolute path to the BLIS binary.
        
    Raises:
        FileNotFoundError: If the binary is not found.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise FileNotFoundError("Not in a git repository")
    
    binary_path = os.path.join(result.stdout.strip(), "blis")
    if not os.path.isfile(binary_path):
        raise FileNotFoundError(f"BLIS binary not found at {binary_path}")
    
    return binary_path


def parse_json_output(stdout: str) -> Dict[str, Any]:
    """Extract the last JSON metrics block from BLIS stdout.
    
    Multi-instance mode prints per-instance metrics first, then aggregated
    cluster metrics last. Each section is preceded by '=== Simulation Metrics ==='.
    This function extracts the final aggregated metrics block.
    
    Args:
        stdout: The complete stdout output from a BLIS run.
        
    Returns:
        Dict[str, Any]: Parsed JSON metrics dictionary containing performance data.
        
    Raises:
        ValueError: If no metrics section is found or JSON is malformed.
        json.JSONDecodeError: If the JSON cannot be parsed.
    """
    sections = stdout.split("=== Simulation Metrics ===")
    if len(sections) < 2:
        raise ValueError("No '=== Simulation Metrics ===' section found in output")

    last_section = sections[-1]

    # Find the JSON object in the last section
    brace_start = last_section.find("{")
    if brace_start == -1:
        raise ValueError("No JSON object found after metrics header")

    # Find matching closing brace
    depth = 0
    for i in range(brace_start, len(last_section)):
        if last_section[i] == "{":
            depth += 1
        elif last_section[i] == "}":
            depth -= 1
            if depth == 0:
                json_str = last_section[brace_start : i + 1]
                return json.loads(json_str)

    raise ValueError("Unterminated JSON object in metrics output")


def run_blis(config: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run a single BLIS simulation with the given configuration.
    
    Constructs and executes a BLIS command with the provided parameters,
    then parses the output metrics. Adds derived metrics like throughput_per_gpu.
    
    Args:
        config: Dictionary containing BLIS parameters including model, hardware,
            tp, num_instances, rate, num_requests, seed, latency_model, and
            token distribution parameters.
    
    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]: A tuple containing:
            - On success: (metrics dictionary, None)
            - On failure: (None, error message string)
    """
    binary = find_blis_binary()
    
    # Build command
    cmd = [
        binary,
        "run",
        "--model", config["model"],
        "--hardware", config["hardware"],
        "--tp", str(config["tp"]),
        "--num-instances", str(config["num_instances"]),
        "--rate", str(config["rate"]),
        "--num-requests", str(config["num_requests"]),
        "--seed", str(config["seed"]),
        "--log", config["log"],
        "--latency-model", config["latency_model"],
        "--prompt-tokens", str(config["prompt_tokens"]),
        "--prompt-tokens-stdev", str(config["prompt_tokens_stdev"]),
        "--output-tokens", str(config["output_tokens"]),
        "--output-tokens-stdev", str(config["output_tokens_stdev"]),
    ]
    
    # Derive min/max bounds for Gaussian validation
    # BLIS requires: min <= mean <= max AND min <= stdev <= max
    prompt_max = max(config["prompt_tokens"], config["prompt_tokens_stdev"]) + 3 * config["prompt_tokens_stdev"]
    output_max = max(config["output_tokens"], config["output_tokens_stdev"]) + 3 * config["output_tokens_stdev"]
    
    cmd.extend([
        "--prompt-tokens-min", "1",
        "--prompt-tokens-max", str(int(prompt_max)),
        "--output-tokens-min", "1",
        "--output-tokens-max", str(int(output_max)),
        "--max-num-running-reqs", str(config["max_num_running_reqs"]),
        "--max-num-scheduled-tokens", str(config["max_num_scheduled_tokens"]),
    ])
    
    # Execute BLIS
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None, f"BLIS timed out after {TIMEOUT}s"
    
    if result.returncode != 0:
        return None, f"BLIS exited with code {result.returncode}: {result.stderr}"
    
    # Parse JSON output
    try:
        metrics = parse_json_output(result.stdout)
    except (ValueError, json.JSONDecodeError) as e:
        return None, f"Failed to parse output: {e}"
    
    # Add derived metrics
    num_gpus = config["num_instances"] * config["tp"]
    tokens_per_sec = metrics.get("tokens_per_sec", 0.0)
    metrics["throughput_per_gpu"] = tokens_per_sec / num_gpus if num_gpus > 0 else 0.0
    
    return metrics, None


def sweep_parameter(param_name: str, values: List[Any], base_config: Dict[str, Any]) -> pd.DataFrame:
    """Run a 1D parameter sweep for a single parameter.
    
    Iterates through the provided values for the specified parameter while
    keeping all other parameters fixed. Collects metrics from each BLIS run
    and returns them as a DataFrame.
    
    Args:
        param_name: Name of the parameter being swept (e.g., "isl", "tp",
            "max_num_running_reqs"). Must be a key in the internal param_map.
        values: List of values to sweep for the parameter.
        base_config: Dictionary of fixed BLIS parameters that remain constant
            across all sweep points.
    
    Returns:
        pd.DataFrame: DataFrame with columns including sweep_param, param_value,
            and all metrics returned by BLIS (e.g., tokens_per_sec, ttft_mean_ms).
            
    Raises:
        ValueError: If param_name is not a valid sweep parameter.
        RuntimeError: If all sweep points fail to execute.
    """
    results = []
    
    # Map parameter names to config keys
    param_map = {
        "isl": "prompt_tokens",
        "osl": "output_tokens",
        "tp": "tp",
        "max_num_running_reqs": "max_num_running_reqs",
        "max_num_scheduled_tokens": "max_num_scheduled_tokens",
    }
    
    if param_name not in param_map:
        raise ValueError(f"Unknown parameter: {param_name}. Valid options: {list(param_map.keys())}")
    
    config_key = param_map[param_name]
    
    print(f"Starting sweep: {param_name}")
    print(f"Values: {values}")
    print(f"Base config: {base_config}")
    print("-" * 80)
    
    for i, value in enumerate(values):
        print(f"[{i+1}/{len(values)}] Running {param_name}={value}...", end=" ", flush=True)
        
        # Create config for this point
        config = base_config.copy()
        config[config_key] = value
        
        # Run BLIS
        metrics, error = run_blis(config)
        
        if metrics:
            row = {"param_value": value, **metrics}
            results.append(row)
            print(f"✓ (throughput: {metrics.get('tokens_per_sec', 0):.1f} tok/s)")
        else:
            print(f"✗ Error: {error}")
    
    print("-" * 80)
    
    if not results:
        raise RuntimeError("All sweep points failed. Check BLIS configuration.")
    
    df = pd.DataFrame(results)
    df.insert(0, "sweep_param", param_name)
    return df


# qwen/qwen3-14b
# meta-llama/llama-3.3-70b-instruct
# deepseek-ai/deepseek-r1
config = {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "hardware": "A100-80",
    "tp": 1,
    "num_instances": 1,
    "rate": 5.0,
    "num_requests": 100,
    "seed": 42,
    "prompt_tokens": 512,
    "prompt_tokens_stdev": 128,
    "output_tokens": 256,
    "output_tokens_stdev": 64,
    "max_num_running_reqs": 256,
    "max_num_scheduled_tokens": 8192,
    "latency_model": "roofline",
    "log": "error",
}


sweep_param = "isl"

# Sweep ranges for each parameter
sweep_range = {
    "isl": [128, 256, 512, 1024, 2048],
    "osl": [64, 128, 256, 512, 1024],
    "tp": [1, 2, 4, 8],
    "max_num_running_reqs": [32, 64, 128, 256, 512],
    "max_num_scheduled_tokens": [2048, 4096, 8192, 16384, 32768],
}

df = sweep_parameter(
    param_name=sweep_param,
    values=sweep_range[sweep_param], 
    base_config=config
)

y_axis = df.ttft_mean_ms

plt.plot(df.param_value, y_axis, marker="o")
plt.xlabel(df.sweep_param[0], fontsize='16')
plt.ylabel(y_axis.name, fontsize='16')
plt.grid(True, linewidth=1, ls='--', color='gray')
plt.show()
