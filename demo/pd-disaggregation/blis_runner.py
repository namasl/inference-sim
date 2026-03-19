"""BLIS subprocess wrapper for the PD disaggregation demo.

Handles binary discovery, CLI command construction, subprocess execution,
and JSON output parsing.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class BlisResult:
    """Parsed results from a single BLIS simulation run."""

    rate: float
    mode: str  # "Aggregate" or "Disaggregated"
    num_gpus: int  # num_instances * tp
    tokens_per_sec: float
    responses_per_sec: float
    ttft_p95_ms: float
    ttft_mean_ms: float
    itl_mean_ms: float  # TPOT
    e2e_p95_ms: float
    e2e_mean_ms: float
    completed_requests: int
    dropped_unservable: int
    throughput_per_gpu: float  # tokens_per_sec / num_gpus
    throughput_per_user: float  # tokens_per_sec / rate
    effective_concurrency: float  # rate * (e2e_mean_ms / 1000)

    # PD-specific metrics (0/0.0/None when not a PD run)
    config_name: str = ""
    pd_disaggregated_count: int = 0
    pd_dropped_at_decode_kv: int = 0
    pd_prefill_throughput: float = 0.0
    pd_decode_throughput: float = 0.0
    pd_load_imbalance: float = 0.0
    pd_transfer_duration_mean_us: float = 0.0
    pd_transfer_duration_p95_us: float = 0.0
    pd_peak_concurrent_transfers: float = 0.0
    pd_mean_transfer_queue_depth: float = 0.0


def find_blis_binary() -> str:
    """Discover the BLIS binary using a priority chain.

    Order:
    1. BLIS_BINARY env var
    2. ./blis in repo root (via git rev-parse --show-toplevel)
    3. Auto-build: go build -o blis main.go
    4. Fail with instructions
    """
    # 1. Environment variable
    env_path = os.environ.get("BLIS_BINARY")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Repo root detection
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            repo_root = result.stdout.strip()
            binary_path = os.path.join(repo_root, "blis")
            if os.path.isfile(binary_path):
                return binary_path

            # 3. Auto-build
            build_result = subprocess.run(
                ["go", "build", "-o", "blis", "main.go"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if build_result.returncode == 0 and os.path.isfile(binary_path):
                return binary_path
            raise FileNotFoundError(
                f"Auto-build failed: {build_result.stderr}"
            )
    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        pass

    # 4. Fail with instructions
    raise FileNotFoundError(
        "Could not find BLIS binary. Options:\n"
        "  1. Set BLIS_BINARY=/path/to/blis\n"
        "  2. Run 'go build -o blis main.go' from the repo root\n"
        "  3. Ensure you are inside the inference-sim git repository"
    )


def _parse_cluster_metrics(stdout: str) -> dict:
    """Extract the last JSON metrics block from BLIS stdout.

    Multi-instance mode prints per-instance metrics first, then aggregated
    cluster metrics last. Each section is preceded by '=== Simulation Metrics ==='.
    """
    sections = stdout.split("=== Simulation Metrics ===")
    if len(sections) < 2:
        raise ValueError(
            "No '=== Simulation Metrics ===' section found in output"
        )

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


def _parse_pd_metrics(stdout: str) -> dict:
    """Parse the '=== PD Metrics ===' text section from BLIS stdout.

    Returns a dict with keys matching BlisResult pd_* fields, or empty dict
    if the section is not present (aggregate runs, no PD activity).
    """
    if "=== PD Metrics ===" not in stdout:
        return {}

    # Extract the PD Metrics section
    sections = stdout.split("=== PD Metrics ===")
    if len(sections) < 2:
        return {}
    section = sections[-1]

    result = {}

    # Disaggregated Requests: 150
    m = re.search(r"Disaggregated Requests:\s*(\d+)", section)
    if m:
        result["pd_disaggregated_count"] = int(m.group(1))

    # Dropped at Decode KV: 3  (only present when > 0)
    m = re.search(r"Dropped at Decode KV:\s*(\d+)", section)
    result["pd_dropped_at_decode_kv"] = int(m.group(1)) if m else 0

    # Prefill Throughput: 12.3456 sub-req/s
    m = re.search(r"Prefill Throughput:\s*([\d.]+)", section)
    if m:
        result["pd_prefill_throughput"] = float(m.group(1))

    # Decode Throughput: 11.9876 sub-req/s
    m = re.search(r"Decode Throughput:\s*([\d.]+)", section)
    if m:
        result["pd_decode_throughput"] = float(m.group(1))

    # Load Imbalance Ratio: 1.0234  OR  Load Imbalance Ratio: inf (one pool idle)
    m = re.search(r"Load Imbalance Ratio:\s*([\d.]+|inf)", section)
    if m:
        val = m.group(1)
        result["pd_load_imbalance"] = float("inf") if val == "inf" else float(val)

    # KV Transfer Duration (μs): mean=1234.5 p50=1100.2 p95=2345.6 p99=2890.1
    m = re.search(
        r"KV Transfer Duration \(.\u03bcs\):[^\n]*mean=([\d.]+)[^\n]*p95=([\d.]+)",
        section,
    )
    if not m:
        # Fallback: handle plain ASCII rendering without unicode
        m = re.search(
            r"KV Transfer Duration \([^)]*\):[^\n]*mean=([\d.]+)[^\n]*p95=([\d.]+)",
            section,
        )
    if m:
        result["pd_transfer_duration_mean_us"] = float(m.group(1))
        result["pd_transfer_duration_p95_us"] = float(m.group(2))
    else:
        result["pd_transfer_duration_mean_us"] = 0.0
        result["pd_transfer_duration_p95_us"] = 0.0

    # Peak Concurrent Transfers: 3  (only when contention enabled)
    m = re.search(r"Peak Concurrent Transfers:\s*([\d.]+)", section)
    result["pd_peak_concurrent_transfers"] = float(m.group(1)) if m else 0.0

    # Mean Transfer Queue Depth: 1.75  (only when contention enabled)
    m = re.search(r"Mean Transfer Queue Depth:\s*([\d.]+)", section)
    result["pd_mean_transfer_queue_depth"] = float(m.group(1)) if m else 0.0

    return result


def run_blis(
    binary: str,
    model: str,
    gpu: str,
    tp: int,
    num_instances: int,
    rate: float,
    num_requests: int,
    seed: int,
    prompt_tokens_mean: int,
    prompt_tokens_stdev: int,
    output_tokens_mean: int,
    output_tokens_stdev: int,
    latency_model: str,
    mode: str,
    prefill_instances: int = 0,
    decode_instances: int = 0,
    pd_decider: str = "always",
    pd_prefix_threshold: Optional[int] = None,
    pd_direct_decode_threshold: Optional[int] = None,
    pd_transfer_bandwidth: Optional[float] = None,
    pd_transfer_base_latency: Optional[float] = None,
    pd_kv_bytes_per_token: Optional[int] = None,
    pd_transfer_contention: Optional[bool] = None,
    pd_interference_prefill: Optional[float] = None,
    pd_interference_decode: Optional[float] = None,
    prefill_routing_scorers: Optional[str] = None,
    decode_routing_scorers: Optional[str] = None,
    prefill_tp: Optional[int] = None,
    decode_tp: Optional[int] = None,
    prefill_hardware: Optional[str] = None,
    decode_hardware: Optional[str] = None,
    prefill_latency_model: Optional[str] = None,
    decode_latency_model: Optional[str] = None,
    config_name: str = "",
    num_gpus_override: Optional[int] = None,
    timeout: int = 300,
) -> tuple[Optional[BlisResult], Optional[str]]:
    """Run a single BLIS simulation and parse the results.

    Returns (BlisResult, None) on success, or (None, error_message) on failure.
    """
    cmd = [
        binary,
        "run",
        "--model", model,
        "--hardware", gpu,
        "--tp", str(tp),
        "--num-instances", str(num_instances),
        "--rate", str(rate),
        "--num-requests", str(num_requests),
        "--seed", str(seed),
        "--log", "error",
        "--prompt-tokens", str(prompt_tokens_mean),
        "--prompt-tokens-stdev", str(prompt_tokens_stdev),
        "--output-tokens", str(output_tokens_mean),
        "--output-tokens-stdev", str(output_tokens_stdev),
    ]

    # Derive min/max bounds so BLIS validation accepts any mean/stdev values.
    # BLIS requires: min <= mean <= max AND min <= stdev <= max.
    prompt_max = max(prompt_tokens_mean, prompt_tokens_stdev) + 3 * prompt_tokens_stdev
    output_max = max(output_tokens_mean, output_tokens_stdev) + 3 * output_tokens_stdev
    cmd.extend([
        "--prompt-tokens-min", "1",
        "--prompt-tokens-max", str(int(prompt_max)),
        "--output-tokens-min", "1",
        "--output-tokens-max", str(int(output_max)),
    ])

    if latency_model:
        cmd.extend(["--latency-model", latency_model])

    if mode == "Disaggregated":
        cmd.extend([
            "--prefill-instances", str(prefill_instances),
            "--decode-instances", str(decode_instances),
            "--pd-decider", pd_decider,
        ])

        # Optional PD flags (only appended when explicitly set)
        if pd_prefix_threshold is not None:
            cmd.extend(["--pd-prefix-threshold", str(pd_prefix_threshold)])
        if pd_direct_decode_threshold is not None:
            cmd.extend(["--pd-direct-decode-threshold", str(pd_direct_decode_threshold)])
        if pd_transfer_bandwidth is not None:
            cmd.extend(["--pd-transfer-bandwidth", str(pd_transfer_bandwidth)])
        if pd_transfer_base_latency is not None:
            cmd.extend(["--pd-transfer-base-latency", str(pd_transfer_base_latency)])
        if pd_kv_bytes_per_token is not None:
            cmd.extend(["--pd-kv-bytes-per-token", str(pd_kv_bytes_per_token)])
        if pd_transfer_contention is not None and pd_transfer_contention:
            cmd.append("--pd-transfer-contention")
        if pd_interference_prefill is not None:
            cmd.extend(["--pd-interference-prefill", str(pd_interference_prefill)])
        if pd_interference_decode is not None:
            cmd.extend(["--pd-interference-decode", str(pd_interference_decode)])
        if prefill_routing_scorers is not None:
            cmd.extend(["--prefill-routing-scorers", prefill_routing_scorers])
        if decode_routing_scorers is not None:
            cmd.extend(["--decode-routing-scorers", decode_routing_scorers])
        if prefill_tp is not None:
            cmd.extend(["--prefill-tp", str(prefill_tp)])
        if decode_tp is not None:
            cmd.extend(["--decode-tp", str(decode_tp)])
        if prefill_hardware is not None:
            cmd.extend(["--prefill-hardware", prefill_hardware])
        if decode_hardware is not None:
            cmd.extend(["--decode-hardware", decode_hardware])
        if prefill_latency_model is not None:
            cmd.extend(["--prefill-latency-model", prefill_latency_model])
        if decode_latency_model is not None:
            cmd.extend(["--decode-latency-model", decode_latency_model])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"BLIS timed out after {timeout}s (rate={rate}, mode={mode})"

    if result.returncode != 0:
        return None, f"BLIS exited with code {result.returncode}: {result.stderr}"

    try:
        metrics = _parse_cluster_metrics(result.stdout)
    except (ValueError, json.JSONDecodeError) as e:
        return None, f"Failed to parse output: {e}\nRaw stdout:\n{result.stdout[:500]}"

    num_gpus = num_gpus_override if num_gpus_override is not None else num_instances * tp
    tokens_per_sec = metrics.get("tokens_per_sec", 0.0)
    e2e_mean_ms = metrics.get("e2e_mean_ms", 0.0)

    throughput_per_gpu = tokens_per_sec / num_gpus if num_gpus > 0 else 0.0
    throughput_per_user = tokens_per_sec / rate if rate > 0 else 0.0
    effective_concurrency = rate * (e2e_mean_ms / 1000.0)

    pd_metrics = _parse_pd_metrics(result.stdout)

    return BlisResult(
        rate=rate,
        mode=mode,
        num_gpus=num_gpus,
        tokens_per_sec=tokens_per_sec,
        responses_per_sec=metrics.get("responses_per_sec", 0.0),
        ttft_p95_ms=metrics.get("ttft_p95_ms", 0.0),
        ttft_mean_ms=metrics.get("ttft_mean_ms", 0.0),
        itl_mean_ms=metrics.get("itl_mean_ms", 0.0),
        e2e_p95_ms=metrics.get("e2e_p95_ms", 0.0),
        e2e_mean_ms=e2e_mean_ms,
        completed_requests=metrics.get("completed_requests", 0),
        dropped_unservable=metrics.get("dropped_unservable", 0),
        throughput_per_gpu=throughput_per_gpu,
        throughput_per_user=throughput_per_user,
        effective_concurrency=effective_concurrency,
        config_name=config_name,
        **pd_metrics,
    ), None
