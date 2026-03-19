"""Sweep execution helpers for the PD Disaggregation Demo."""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import asdict

import pandas as pd

from blis_runner import run_blis
from config_ui import SERIES_COLORS, SimConfig


def run_config(
    binary: str,
    cfg: SimConfig,
    model: str,
    gpu: str,
    latency_model: str,
    rate: float,
    num_requests: int,
    seed: int,
    prompt_tokens_mean: int,
    prompt_tokens_stdev: int,
    output_tokens_mean: int,
    output_tokens_stdev: int,
):
    """Run one config at one point. Returns (BlisResult|None, error|None)."""
    # Always use the caller's gpu for the global --hardware flag.
    # Per-pool overrides (--prefill-hardware / --decode-hardware) are passed via to_run_kwargs().
    kwargs = cfg.to_run_kwargs()
    return run_blis(
        binary=binary,
        model=model,
        gpu=gpu,
        latency_model=latency_model,
        rate=rate,
        num_requests=num_requests,
        seed=seed,
        prompt_tokens_mean=prompt_tokens_mean,
        prompt_tokens_stdev=prompt_tokens_stdev,
        output_tokens_mean=output_tokens_mean,
        output_tokens_stdev=output_tokens_stdev,
        num_gpus_override=cfg.num_gpus,  # correct GPU count for heterogeneous TP
        **kwargs,
    )


def run_sweep_parallel(
    binary: str,
    configs: list,
    model: str,
    gpu: str,
    latency_model: str,
    sweep_values: list,
    sweep_key: str,
    num_requests: int,
    seed: int,
    prompt_tokens_mean: int,
    prompt_tokens_stdev: int,
    output_tokens_mean: int,
    output_tokens_stdev: int,
    rate: float = 5.0,
    progress_bar=None,
) -> tuple:
    """Run all configs x sweep values in parallel.

    Returns (list of result dicts with sweep_key and color added, list of errors).
    """
    result_dicts: list = []
    errors: list = []

    for i, val in enumerate(sweep_values):
        if progress_bar:
            progress_bar.progress(
                i / len(sweep_values),
                text=f"Point {i + 1}/{len(sweep_values)} ({sweep_key}={val:.1f})...",
            )

        # Resolve per-point parameters
        if sweep_key == "rate":
            point_rate = float(val)
            point_prompt_mean = prompt_tokens_mean
            point_output_mean = output_tokens_mean
        elif sweep_key == "prompt_tokens_mean":
            point_rate = rate
            point_prompt_mean = int(val)
            point_output_mean = output_tokens_mean
        else:  # output_tokens_mean
            point_rate = rate
            point_prompt_mean = prompt_tokens_mean
            point_output_mean = int(val)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as pool:
            futures = {
                pool.submit(
                    run_config,
                    binary=binary,
                    cfg=cfg,
                    model=model,
                    gpu=gpu,
                    latency_model=latency_model,
                    rate=point_rate,
                    num_requests=num_requests,
                    seed=seed,
                    prompt_tokens_mean=point_prompt_mean,
                    prompt_tokens_stdev=prompt_tokens_stdev,
                    output_tokens_mean=point_output_mean,
                    output_tokens_stdev=output_tokens_stdev,
                ): cfg
                for cfg in configs
            }
            for future, cfg in futures.items():
                res, err = future.result()
                if res is not None:
                    row = asdict(res)
                    row[sweep_key] = float(val)
                    row["color"] = cfg.color
                    result_dicts.append(row)
                elif err is not None:
                    errors.append(f"[{cfg.name} {sweep_key}={val:.1f}] {err}")

    if progress_bar:
        progress_bar.progress(1.0, text="Sweep complete!")

    return result_dicts, errors


def result_dicts_to_df(result_dicts: list, sweep_key: str) -> pd.DataFrame:
    """Convert list of result dicts to DataFrame."""
    if not result_dicts:
        return pd.DataFrame()
    df = pd.DataFrame(result_dicts)
    return df


def run_sweep_2d(
    binary: str,
    cfg_a: "SimConfig",
    cfg_b: "SimConfig",
    model: str,
    gpu: str,
    latency_model: str,
    input_values: list,
    output_values: list,
    num_requests: int,
    seed: int,
    prompt_stdev: int,
    output_stdev: int,
    rate: float,
    progress_bar=None,
) -> tuple:
    """Run 2D grid sweep comparing two configs.

    Returns (results dict, errors list).
    results maps (input_val, output_val) -> {config_name: BlisResult or None}.
    """
    results: dict = {}
    errors: list = []
    total = len(input_values) * len(output_values)
    step = 0

    for out_val in output_values:
        for in_val in input_values:
            if progress_bar:
                progress_bar.progress(
                    step / total,
                    text=f"Point {step + 1}/{total} (in={int(in_val)}, out={int(out_val)})...",
                )

            cell: dict = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(
                        run_config,
                        binary=binary,
                        cfg=cfg,
                        model=model,
                        gpu=gpu,
                        latency_model=latency_model,
                        rate=rate,
                        num_requests=num_requests,
                        seed=seed,
                        prompt_tokens_mean=int(in_val),
                        prompt_tokens_stdev=prompt_stdev,
                        output_tokens_mean=int(out_val),
                        output_tokens_stdev=output_stdev,
                    ): cfg
                    for cfg in [cfg_a, cfg_b]
                }
                for future, cfg in futures.items():
                    res, err = future.result()
                    cell[cfg.name] = res
                    if err:
                        errors.append(
                            f"[{cfg.name} in={int(in_val)} out={int(out_val)}] {err}"
                        )

            results[(in_val, out_val)] = cell
            step += 1

    if progress_bar:
        progress_bar.progress(1.0, text="2D sweep complete!")

    return results, errors


def compute_ratio_matrix(
    results: dict,
    input_values: list,
    output_values: list,
    cfg_a_name: str,
    cfg_b_name: str,
    metric: str,
) -> tuple:
    """Compute ratio matrix (A/B) from 2D sweep results.

    Returns (matrix, missing_count). Matrix is indexed [output_idx][input_idx].
    metric: "Throughput/GPU", "TTFT P95", or "E2E P95".
    """
    METRIC_ATTR = {
        "Throughput/GPU": "throughput_per_gpu",
        "TTFT P95": "ttft_p95_ms",
        "E2E P95": "e2e_p95_ms",
    }
    attr = METRIC_ATTR.get(metric, "throughput_per_gpu")

    matrix = []
    missing = 0

    for out_val in output_values:
        row = []
        for in_val in input_values:
            cell = results.get((in_val, out_val), {})
            res_a = cell.get(cfg_a_name)
            res_b = cell.get(cfg_b_name)
            if res_a is not None and res_b is not None:
                val_b = getattr(res_b, attr)
                val_a = getattr(res_a, attr)
                ratio = val_a / val_b if val_b > 0 else math.nan
            else:
                ratio = math.nan
                missing += 1
            row.append(ratio)
        matrix.append(row)

    return matrix, missing


def enumerate_pd_splits(budget: int, tp_p: int, tp_d: int) -> list:
    """Enumerate (prefill, decode) instance pairs where P*tp_p + D*tp_d == budget.

    Falls back to at-most-budget pairs if no exact-budget splits exist.
    """
    if tp_p <= 0 or tp_d <= 0:
        return []
    splits = []
    for p in range(1, budget // tp_p + 1):
        remaining = budget - p * tp_p
        if remaining >= tp_d and remaining % tp_d == 0:
            d = remaining // tp_d
            splits.append((p, d))

    if not splits:
        # Fallback: all valid pairs that fit within budget
        for p in range(1, budget // max(tp_p, 1) + 1):
            for d in range(1, budget // max(tp_d, 1) + 1):
                if p * tp_p + d * tp_d <= budget:
                    splits.append((p, d))
        splits = sorted(set(splits))

    return splits


def run_pd_split_sweep(
    binary: str,
    model: str,
    gpu: str,
    latency_model: str,
    num_requests: int,
    seed: int,
    prompt_mean: int,
    prompt_stdev: int,
    output_mean: int,
    output_stdev: int,
    rate: float,
    budget: int,
    tp_p: int,
    tp_d: int,
    agg_tp: int,
    pd_decider: str,
    progress_bar=None,
) -> tuple:
    """Sweep all valid P/D splits for a GPU budget.

    Returns (split_result_dicts, errors, agg_result_dict or None).
    Each split result dict has 'prefill_count', 'decode_count', 'config_name', and all BlisResult fields.
    """
    splits = enumerate_pd_splits(budget, tp_p, tp_d)
    result_dicts = []
    errors = []
    total = len(splits) + 1  # +1 for aggregate

    for i, (p, d) in enumerate(splits):
        if progress_bar:
            progress_bar.progress(
                i / total,
                text=f"Split {i + 1}/{total}: {p}P+{d}D...",
            )
        cfg = SimConfig(
            name=f"{p}P+{d}D",
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
            mode="Disaggregated",
            num_instances=p + d,
            tp=tp_p,
            prefill_instances=p,
            decode_instances=d,
            pd_decider=pd_decider,
            prefill_tp=tp_p,
            decode_tp=tp_d,
        )
        res, err = run_config(
            binary=binary, cfg=cfg, model=model, gpu=gpu,
            latency_model=latency_model, rate=rate,
            num_requests=num_requests, seed=seed,
            prompt_tokens_mean=prompt_mean, prompt_tokens_stdev=prompt_stdev,
            output_tokens_mean=output_mean, output_tokens_stdev=output_stdev,
        )
        if res is not None:
            row = asdict(res)
            row["prefill_count"] = p
            row["decode_count"] = d
            row["color"] = cfg.color
            result_dicts.append(row)
        else:
            errors.append(f"[{p}P+{d}D] {err}")

    # Aggregate baseline
    agg_result_dict = None
    agg_instances = max(1, budget // agg_tp)
    if progress_bar:
        progress_bar.progress(len(splits) / total, text="Running aggregate baseline...")
    agg_cfg = SimConfig(
        name=f"{agg_instances}x Aggregate (TP={agg_tp})",
        color="#888888",
        mode="Aggregate",
        num_instances=agg_instances,
        tp=agg_tp,
    )
    res, err = run_config(
        binary=binary, cfg=agg_cfg, model=model, gpu=gpu,
        latency_model=latency_model, rate=rate,
        num_requests=num_requests, seed=seed,
        prompt_tokens_mean=prompt_mean, prompt_tokens_stdev=prompt_stdev,
        output_tokens_mean=output_mean, output_tokens_stdev=output_stdev,
    )
    if res is not None:
        agg_result_dict = asdict(res)
    else:
        errors.append(f"[Aggregate baseline] {err}")

    if progress_bar:
        progress_bar.progress(1.0, text="Done!")

    return result_dicts, errors, agg_result_dict


def run_tp_sweep(
    binary: str,
    model: str,
    gpu: str,
    latency_model: str,
    num_requests: int,
    seed: int,
    prompt_mean: int,
    prompt_stdev: int,
    output_mean: int,
    output_stdev: int,
    rate: float,
    p_inst: int,
    d_inst: int,
    tp_values: list,
    apply_to: str,
    progress_bar=None,
) -> tuple:
    """Sweep TP values for a fixed P/D instance count.

    apply_to: "Both pools", "Prefill only", "Decode only"
    Returns (result_dicts, errors).
    Each result dict has 'tp_value' and 'series' ("PD" or "Aggregate") added.
    """
    tp_values = sorted(set(int(t) for t in tp_values))  # sort for deterministic "min TP" behavior
    if not tp_values:
        return [], []

    result_dicts = []
    errors = []
    total = len(tp_values) * 2

    for i, tp in enumerate(tp_values):
        tp = int(tp)
        # Determine per-pool TPs based on apply_to
        if apply_to == "Both pools":
            p_tp, d_tp = tp, tp
        elif apply_to == "Prefill only":
            p_tp, d_tp = tp, tp_values[0]  # decode stays at min TP
        else:  # Decode only
            p_tp, d_tp = tp_values[0], tp

        # PD config
        if progress_bar:
            progress_bar.progress(i * 2 / total, text=f"PD TP={tp} ({i*2+1}/{total})...")
        pd_cfg = SimConfig(
            name=f"PD TP={tp}",
            color=SERIES_COLORS[0],
            mode="Disaggregated",
            num_instances=p_inst + d_inst,
            tp=tp,
            prefill_instances=p_inst,
            decode_instances=d_inst,
            pd_decider="always",
            prefill_tp=p_tp,
            decode_tp=d_tp,
        )
        res, err = run_config(
            binary=binary, cfg=pd_cfg, model=model, gpu=gpu,
            latency_model=latency_model, rate=rate,
            num_requests=num_requests, seed=seed,
            prompt_tokens_mean=prompt_mean, prompt_tokens_stdev=prompt_stdev,
            output_tokens_mean=output_mean, output_tokens_stdev=output_stdev,
        )
        if res is not None:
            row = asdict(res)
            row["tp_value"] = tp
            row["series"] = "PD"
            row["color"] = SERIES_COLORS[0]
            result_dicts.append(row)
        else:
            errors.append(f"[PD TP={tp}] {err}")

        # Matching aggregate (same total GPU budget)
        total_gpus = p_inst * p_tp + d_inst * d_tp
        agg_instances = max(1, total_gpus // tp)
        if progress_bar:
            progress_bar.progress((i * 2 + 1) / total, text=f"Agg TP={tp} ({i*2+2}/{total})...")
        agg_cfg = SimConfig(
            name=f"Aggregate TP={tp}",
            color=SERIES_COLORS[1],
            mode="Aggregate",
            num_instances=agg_instances,
            tp=tp,
        )
        res, err = run_config(
            binary=binary, cfg=agg_cfg, model=model, gpu=gpu,
            latency_model=latency_model, rate=rate,
            num_requests=num_requests, seed=seed,
            prompt_tokens_mean=prompt_mean, prompt_tokens_stdev=prompt_stdev,
            output_tokens_mean=output_mean, output_tokens_stdev=output_stdev,
        )
        if res is not None:
            row = asdict(res)
            row["tp_value"] = tp
            row["series"] = "Aggregate"
            row["color"] = SERIES_COLORS[1]
            result_dicts.append(row)
        else:
            errors.append(f"[Aggregate TP={tp}] {err}")

    if progress_bar:
        progress_bar.progress(1.0, text="Done!")

    return result_dicts, errors


def run_replica_sweep(
    binary: str,
    model: str,
    gpu: str,
    latency_model: str,
    num_requests: int,
    seed: int,
    prompt_mean: int,
    prompt_stdev: int,
    output_mean: int,
    output_stdev: int,
    rate: float,
    p_tp: int,
    d_tp: int,
    p_ratio: int,
    d_ratio: int,
    total_instances_list: list,
    agg_tp: int,
    pd_decider: str,
    progress_bar=None,
) -> tuple:
    """Sweep total instance count with a fixed P:D ratio.

    Returns (result_dicts, errors).
    Each result dict has 'total_instances' and 'series' ("PD" or "Aggregate") added.
    """
    result_dicts = []
    errors = []
    total = len(total_instances_list) * 2

    for i, n in enumerate(total_instances_list):
        n = int(n)
        ratio_sum = p_ratio + d_ratio
        p_inst = max(1, round(n * p_ratio / ratio_sum))
        d_inst = max(1, n - p_inst)

        if progress_bar:
            progress_bar.progress(i * 2 / total, text=f"PD total={n} ({i*2+1}/{total})...")

        pd_cfg = SimConfig(
            name=f"PD {p_inst}P+{d_inst}D",
            color=SERIES_COLORS[0],
            mode="Disaggregated",
            num_instances=p_inst + d_inst,
            tp=p_tp,
            prefill_instances=p_inst,
            decode_instances=d_inst,
            pd_decider=pd_decider,
            prefill_tp=p_tp,
            decode_tp=d_tp,
        )
        res, err = run_config(
            binary=binary, cfg=pd_cfg, model=model, gpu=gpu,
            latency_model=latency_model, rate=rate,
            num_requests=num_requests, seed=seed,
            prompt_tokens_mean=prompt_mean, prompt_tokens_stdev=prompt_stdev,
            output_tokens_mean=output_mean, output_tokens_stdev=output_stdev,
        )
        if res is not None:
            row = asdict(res)
            row["total_instances"] = n
            row["series"] = "PD"
            row["color"] = SERIES_COLORS[0]
            result_dicts.append(row)
        else:
            errors.append(f"[PD n={n}] {err}")

        # Matching aggregate
        if progress_bar:
            progress_bar.progress(
                (i * 2 + 1) / total, text=f"Agg total={n} ({i*2+2}/{total})..."
            )
        agg_cfg = SimConfig(
            name=f"Aggregate {n}x",
            color=SERIES_COLORS[1],
            mode="Aggregate",
            num_instances=n,
            tp=agg_tp,
        )
        res, err = run_config(
            binary=binary, cfg=agg_cfg, model=model, gpu=gpu,
            latency_model=latency_model, rate=rate,
            num_requests=num_requests, seed=seed,
            prompt_tokens_mean=prompt_mean, prompt_tokens_stdev=prompt_stdev,
            output_tokens_mean=output_mean, output_tokens_stdev=output_stdev,
        )
        if res is not None:
            row = asdict(res)
            row["total_instances"] = n
            row["series"] = "Aggregate"
            row["color"] = SERIES_COLORS[1]
            result_dicts.append(row)
        else:
            errors.append(f"[Aggregate n={n}] {err}")

    if progress_bar:
        progress_bar.progress(1.0, text="Done!")

    return result_dicts, errors
