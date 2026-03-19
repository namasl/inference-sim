"""PD Disaggregation Demo — Multi-configuration exploration app.

Three exploration modes rendered as tabs:
  Rate Sweep          — Compare configs as arrival rate increases
  Workload Sweep      — Find where PD wins as prompt/output length grows
  Parallelism Explorer — Optimal GPU allocation across P/D split, TP, and replica count
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from blis_runner import find_blis_binary
from charts import (
    build_comparison_chart,
    build_concurrency_chart,
    build_heatmap_chart,
    build_latency_chart,
    build_pd_metrics_chart,
    compute_pareto_front,
    show_results_tables,
)
from config_ui import (
    PD_DECIDERS,
    SERIES_COLORS,
    SimConfig,
    render_sidebar,
)
from sweep import (
    compute_ratio_matrix,
    enumerate_pd_splits,
    result_dicts_to_df,
    run_config,
    run_pd_split_sweep,
    run_replica_sweep,
    run_sweep_2d,
    run_sweep_parallel,
    run_tp_sweep,
)


# ---------------------------------------------------------------------------
# Tab: Rate Sweep
# ---------------------------------------------------------------------------

def tab_rate_sweep(
    binary: str, model: str, gpu: str, latency_model: str,
    configs: list, num_requests: int, seed: int,
    prompt_mean: int, prompt_stdev: int, output_mean: int, output_stdev: int,
) -> None:
    """Rate Sweep tab: compare configs at increasing load."""
    st.subheader("Rate Sweep")
    st.caption(
        "Compare configurations as arrival rate increases. "
        "Shows the throughput-per-GPU vs throughput-per-user tradeoff curve."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        min_rate = st.number_input(
            "Min rate (req/s)", min_value=0.1, value=1.0, step=0.5, key="rs_min_rate",
        )
    with col2:
        max_rate = st.number_input(
            "Max rate (req/s)", min_value=0.1, value=50.0, step=1.0, key="rs_max_rate",
        )
    with col3:
        num_points = st.slider(
            "Number of points", min_value=2, max_value=20, value=8, key="rs_num_points",
        )

    if st.button("Run Rate Sweep", type="primary", key="rs_run"):
        if min_rate >= max_rate:
            st.error("Min rate must be less than max rate.")
            return
        rates = list(np.linspace(float(min_rate), float(max_rate), num_points))
        progress = st.progress(0, text="Starting rate sweep...")
        result_dicts, errors = run_sweep_parallel(
            binary=binary, configs=configs, model=model, gpu=gpu,
            latency_model=latency_model, sweep_values=rates, sweep_key="rate",
            num_requests=num_requests, seed=seed,
            prompt_tokens_mean=prompt_mean, prompt_tokens_stdev=prompt_stdev,
            output_tokens_mean=output_mean, output_tokens_stdev=output_stdev,
            progress_bar=progress,
        )
        if errors:
            st.warning(f"{len(errors)} of {len(rates) * len(configs)} runs failed. Expand for details.")
            with st.expander("Failed runs", expanded=False):
                for e in errors:
                    st.warning(e)
        if result_dicts:
            st.session_state["rate_sweep_results"] = result_dicts
        else:
            st.error("All runs failed.")

    if "rate_sweep_results" in st.session_state:
        st.info("Showing cached results. Click 'Run Rate Sweep' to regenerate with current settings.")
        result_dicts = st.session_state["rate_sweep_results"]
        df = result_dicts_to_df(result_dicts, "rate")
        pareto_df = compute_pareto_front(df)
        st.plotly_chart(
            build_comparison_chart(df, pareto_df, "rate", "Rate (req/s)"),
            use_container_width=True,
        )
        st.plotly_chart(
            build_latency_chart(df, "rate", "Rate (req/s)"),
            use_container_width=True,
        )
        st.plotly_chart(
            build_concurrency_chart(df, "rate", "Rate (req/s)"),
            use_container_width=True,
        )
        pd_chart = build_pd_metrics_chart(df, "rate", "Rate (req/s)")
        if pd_chart:
            st.plotly_chart(pd_chart, use_container_width=True)
        show_results_tables(df, pareto_df)


# ---------------------------------------------------------------------------
# Tab: Workload Sweep
# ---------------------------------------------------------------------------

def tab_workload_sweep(
    binary: str, model: str, gpu: str, latency_model: str,
    configs: list, num_requests: int, seed: int,
    prompt_mean: int, prompt_stdev: int, output_mean: int, output_stdev: int,
) -> None:
    """Workload Sweep tab: 1D crossover sweep or 2D heatmap comparison."""
    st.subheader("Workload Sweep")

    mode = st.radio("Mode", ["1D Sweep", "2D Heatmap"], horizontal=True, key="ws_mode")

    if mode == "1D Sweep":
        st.caption(
            "Find where PD disaggregation starts winning as prompt or output token length grows."
        )
        col1, col2 = st.columns(2)
        with col1:
            sweep_dim = st.radio(
                "Sweep dimension",
                ["Input token mean", "Output token mean"],
                key="ws_dim",
            )
            fixed_rate = st.number_input(
                "Fixed rate (req/s)", min_value=0.1, value=10.0, step=1.0, key="ws_rate",
            )
        with col2:
            ws_min = st.number_input("Min value (tokens)", min_value=1, value=128, key="ws_min")
            ws_max = st.number_input("Max value (tokens)", min_value=1, value=2048, key="ws_max")
            ws_points = st.slider(
                "Number of points", min_value=2, max_value=15, value=7, key="ws_points",
            )

        if st.button("Run Workload Sweep", type="primary", key="ws_run"):
            if ws_min >= ws_max:
                st.error("Min must be less than max.")
                return

            sweep_key = "prompt_tokens_mean" if sweep_dim == "Input token mean" else "output_tokens_mean"
            x_label = "Input Token Mean" if sweep_dim == "Input token mean" else "Output Token Mean"
            sweep_values = list(np.linspace(float(ws_min), float(ws_max), ws_points))

            progress = st.progress(0, text="Starting workload sweep...")
            result_dicts, errors = run_sweep_parallel(
                binary=binary, configs=configs, model=model, gpu=gpu,
                latency_model=latency_model, sweep_values=sweep_values, sweep_key=sweep_key,
                num_requests=num_requests, seed=seed,
                prompt_tokens_mean=prompt_mean, prompt_tokens_stdev=prompt_stdev,
                output_tokens_mean=output_mean, output_tokens_stdev=output_stdev,
                rate=float(fixed_rate),
                progress_bar=progress,
            )
            if errors:
                st.warning(f"{len(errors)} of {len(sweep_values) * len(configs)} runs failed. Expand for details.")
                with st.expander("Failed runs", expanded=False):
                    for e in errors:
                        st.warning(e)
            if result_dicts:
                st.session_state["ws_results"] = (result_dicts, sweep_key, x_label)
            else:
                st.error("All runs failed.")

        if "ws_results" in st.session_state:
            st.info("Showing cached results. Click 'Run Workload Sweep' to regenerate with current settings.")
            result_dicts, sweep_key, x_label = st.session_state["ws_results"]
            df = result_dicts_to_df(result_dicts, sweep_key)
            pareto_df = compute_pareto_front(df)
            st.plotly_chart(
                build_comparison_chart(df, pareto_df, sweep_key, x_label),
                use_container_width=True,
            )
            st.plotly_chart(
                build_latency_chart(df, sweep_key, x_label),
                use_container_width=True,
            )
            st.plotly_chart(
                build_concurrency_chart(df, sweep_key, x_label),
                use_container_width=True,
            )
            pd_chart = build_pd_metrics_chart(df, sweep_key, x_label)
            if pd_chart:
                st.plotly_chart(pd_chart, use_container_width=True)
            show_results_tables(df, pareto_df)

    else:  # 2D Heatmap
        st.caption(
            "Compare two configs across an input × output token grid. "
            "Heatmap shows ratio A/B — red means config A wins, blue means config B wins."
        )

        if len(configs) < 2:
            st.warning("Add at least 2 configurations in the sidebar to use 2D Heatmap mode.")
            return

        config_names = [c.name for c in configs]
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            cfg_a_name = st.selectbox("Config A", config_names, key="ws2d_cfg_a")
        with col_sel2:
            default_b_idx = min(1, len(config_names) - 1)
            cfg_b_name = st.selectbox(
                "Config B", config_names, index=default_b_idx, key="ws2d_cfg_b",
            )

        if cfg_a_name == cfg_b_name:
            st.warning("Config A and Config B must be different.")
            return

        cfg_a = next(c for c in configs if c.name == cfg_a_name)
        cfg_b = next(c for c in configs if c.name == cfg_b_name)

        col1, col2, col3 = st.columns(3)
        with col1:
            in_min = st.number_input("Input tokens min", min_value=1, value=128, key="ws2d_in_min")
            in_max = st.number_input("Input tokens max", min_value=1, value=4096, key="ws2d_in_max")
            in_pts = st.slider("Input points", min_value=2, max_value=8, value=5, key="ws2d_in_pts")
        with col2:
            out_min = st.number_input("Output tokens min", min_value=1, value=128, key="ws2d_out_min")
            out_max = st.number_input("Output tokens max", min_value=1, value=4096, key="ws2d_out_max")
            out_pts = st.slider("Output points", min_value=2, max_value=8, value=5, key="ws2d_out_pts")
        with col3:
            metric = st.selectbox(
                "Metric", ["Throughput/GPU", "TTFT P95", "E2E P95"], key="ws2d_metric",
            )
            fixed_rate_2d = st.number_input(
                "Fixed rate (req/s)", min_value=0.1, value=10.0, step=1.0, key="ws2d_rate",
            )

        total_runs = in_pts * out_pts * 2
        st.caption(
            f"Estimated runs: {in_pts} input × {out_pts} output × 2 configs = {total_runs}"
        )

        if st.button("Run 2D Heatmap", type="primary", key="ws2d_run"):
            if in_min >= in_max:
                st.error("Input tokens: min must be less than max.")
                return
            if out_min >= out_max:
                st.error("Output tokens: min must be less than max.")
                return

            input_values = list(np.linspace(float(in_min), float(in_max), in_pts))
            output_values = list(np.linspace(float(out_min), float(out_max), out_pts))

            progress = st.progress(0, text="Starting 2D heatmap sweep...")
            results, errors = run_sweep_2d(
                binary=binary, cfg_a=cfg_a, cfg_b=cfg_b,
                model=model, gpu=gpu, latency_model=latency_model,
                input_values=input_values, output_values=output_values,
                num_requests=num_requests, seed=seed,
                prompt_stdev=prompt_stdev, output_stdev=output_stdev,
                rate=float(fixed_rate_2d),
                progress_bar=progress,
            )
            if errors:
                with st.expander(f"{len(errors)} failed runs", expanded=False):
                    for e in errors:
                        st.warning(e)
            if results:
                st.session_state["ws2d_results"] = (
                    results, input_values, output_values, cfg_a_name, cfg_b_name, metric
                )
            else:
                st.error("All runs failed.")

        if "ws2d_results" in st.session_state:
            st.info("Showing cached results. Click 'Run 2D Heatmap' to regenerate.")
            (results, input_values, output_values,
             a_name, b_name, saved_metric) = st.session_state["ws2d_results"]
            ratio_matrix, missing = compute_ratio_matrix(
                results, input_values, output_values, a_name, b_name, saved_metric,
            )
            if missing > 0:
                st.warning(f"{missing} grid cells are missing data (failed runs).")
            fig = build_heatmap_chart(
                ratio_matrix, input_values, output_values, a_name, b_name, saved_metric,
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Parallelism Explorer
# ---------------------------------------------------------------------------

def tab_parallelism_explorer(
    binary: str, model: str, gpu: str, latency_model: str,
    configs: list, num_requests: int, seed: int,
    prompt_mean: int, prompt_stdev: int, output_mean: int, output_stdev: int,
) -> None:
    """Parallelism Explorer tab: find optimal GPU allocation across 3 exploration modes."""
    st.subheader("Parallelism Explorer")

    sub_mode = st.radio(
        "Exploration mode",
        ["P/D Split Sweep", "TP Sweep", "Replica Sweep"],
        horizontal=True,
        key="pe_mode",
    )

    if sub_mode == "P/D Split Sweep":
        st.caption(
            "Enumerate all valid prefill/decode GPU splits for a fixed budget. "
            "Which split maximizes throughput per GPU?"
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            budget = st.number_input("GPU Budget", min_value=2, value=8, key="pe_budget")
            fixed_rate = st.number_input(
                "Fixed rate (req/s)", min_value=0.1, value=10.0, step=1.0, key="pe_rate",
            )
        with col2:
            tp_p = st.number_input("Prefill TP", min_value=1, max_value=8, value=1, key="pe_tp_p")
            tp_d = st.number_input("Decode TP", min_value=1, max_value=8, value=1, key="pe_tp_d")
            agg_tp = st.number_input(
                "Aggregate TP", min_value=1, max_value=8, value=1, key="pe_agg_tp",
                help="TP for the aggregate baseline"
            )
        with col3:
            pe_decider = st.selectbox("PD Decider", PD_DECIDERS, key="pe_decider")
            splits_preview = enumerate_pd_splits(int(budget), int(tp_p), int(tp_d))
            st.caption(
                f"{len(splits_preview)} valid splits found "
                f"(+1 aggregate baseline = {len(splits_preview) + 1} runs)"
            )

        if st.button("Run P/D Split Sweep", type="primary", key="pe_split_run"):
            progress = st.progress(0, text="Starting P/D split sweep...")
            result_dicts, errors, agg_result_dict = run_pd_split_sweep(
                binary=binary, model=model, gpu=gpu, latency_model=latency_model,
                num_requests=num_requests, seed=seed,
                prompt_mean=prompt_mean, prompt_stdev=prompt_stdev,
                output_mean=output_mean, output_stdev=output_stdev,
                rate=float(fixed_rate), budget=int(budget),
                tp_p=int(tp_p), tp_d=int(tp_d), agg_tp=int(agg_tp),
                pd_decider=pe_decider, progress_bar=progress,
            )
            if errors:
                with st.expander(f"{len(errors)} warnings", expanded=False):
                    for e in errors:
                        st.warning(e)
            if result_dicts:
                st.session_state["pe_split_results"] = (result_dicts, agg_result_dict)
            else:
                st.error("All runs failed.")

        if "pe_split_results" in st.session_state:
            st.info("Showing cached results. Click 'Run P/D Split Sweep' to regenerate.")
            result_dicts, agg_result_dict = st.session_state["pe_split_results"]

            tpg = [r["throughput_per_gpu"] for r in result_dicts]
            ttft = [r["ttft_p95_ms"] for r in result_dicts]
            names = [r["config_name"] for r in result_dicts]
            colors = [r.get("color", "#636EFA") for r in result_dicts]
            best_idx = int(np.argmax(tpg)) if tpg else 0

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[f"{r['prefill_count']}P+{r['decode_count']}D" for r in result_dicts],
                y=tpg,
                marker_color=["gold" if i == best_idx else c for i, c in enumerate(colors)],
                text=[f"* {names[best_idx]}" if i == best_idx else "" for i in range(len(names))],
                textposition="outside",
                hovertemplate="Split: %{x}<br>Throughput/GPU: %{y:.1f}<extra></extra>",
            ))
            if agg_result_dict:
                fig_bar.add_hline(
                    y=agg_result_dict["throughput_per_gpu"],
                    line_dash="dash", line_color="gray",
                    annotation_text="Aggregate baseline",
                    annotation_position="bottom right",
                )
            fig_bar.update_layout(
                title="P/D Split Sweep: Throughput/GPU by Split",
                xaxis_title="P/D Split",
                yaxis_title="Throughput per GPU (tokens/s/GPU)",
                height=450,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(
                x=[f"{r['prefill_count']}P+{r['decode_count']}D" for r in result_dicts],
                y=ttft, mode="lines+markers",
                name="TTFT P95 (ms)", marker=dict(color="#636EFA", size=9),
            ))
            if agg_result_dict:
                fig_lat.add_hline(
                    y=agg_result_dict["ttft_p95_ms"],
                    line_dash="dash", line_color="gray",
                    annotation_text="Aggregate TTFT P95",
                )
            fig_lat.update_layout(
                title="TTFT P95 by P/D Split",
                xaxis_title="P/D Split",
                yaxis_title="TTFT P95 (ms)", height=350,
            )
            st.plotly_chart(fig_lat, use_container_width=True)

            rows = [{"Split": f"{r['prefill_count']}P+{r['decode_count']}D",
                     "Throughput/GPU": round(r["throughput_per_gpu"], 2),
                     "TTFT P95": round(r["ttft_p95_ms"], 1),
                     "E2E P95": round(r["e2e_p95_ms"], 1),
                     "Completed": r["completed_requests"]} for r in result_dicts]
            if agg_result_dict:
                rows.insert(0, {"Split": agg_result_dict["config_name"],
                                "Throughput/GPU": round(agg_result_dict["throughput_per_gpu"], 2),
                                "TTFT P95": round(agg_result_dict["ttft_p95_ms"], 1),
                                "E2E P95": round(agg_result_dict["e2e_p95_ms"], 1),
                                "Completed": agg_result_dict["completed_requests"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif sub_mode == "TP Sweep":
        st.caption(
            "Fix P/D instance counts, sweep TP values. "
            "Compare PD vs aggregate at same GPU budget."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            p_inst = st.number_input("Prefill Instances", min_value=1, value=2, key="pe_tp_p_inst")
            d_inst = st.number_input("Decode Instances", min_value=1, value=6, key="pe_tp_d_inst")
            fixed_rate_tp = st.number_input(
                "Fixed rate (req/s)", min_value=0.1, value=10.0, step=1.0, key="pe_tp_rate",
            )
        with col2:
            tp_min = st.number_input("TP min", min_value=1, max_value=8, value=1, key="pe_tp_min")
            tp_max = st.number_input("TP max", min_value=1, max_value=8, value=4, key="pe_tp_max")
            tp_pts = st.slider("TP points", min_value=2, max_value=8, value=4, key="pe_tp_pts")
        with col3:
            apply_to = st.selectbox(
                "Apply TP to", ["Both pools", "Prefill only", "Decode only"], key="pe_tp_apply",
            )
            st.caption(
                f"~{tp_pts * 2} runs "
                f"({tp_pts} TP values × 2 series)"
            )

        if st.button("Run TP Sweep", type="primary", key="pe_tp_run"):
            tp_values = list(np.linspace(int(tp_min), int(tp_max), tp_pts))
            tp_values = sorted(set(int(t) for t in tp_values))
            progress = st.progress(0, text="Starting TP sweep...")
            result_dicts, errors = run_tp_sweep(
                binary=binary, model=model, gpu=gpu, latency_model=latency_model,
                num_requests=num_requests, seed=seed,
                prompt_mean=prompt_mean, prompt_stdev=prompt_stdev,
                output_mean=output_mean, output_stdev=output_stdev,
                rate=float(fixed_rate_tp),
                p_inst=int(p_inst), d_inst=int(d_inst),
                tp_values=tp_values, apply_to=apply_to,
                progress_bar=progress,
            )
            if errors:
                with st.expander(f"{len(errors)} warnings", expanded=False):
                    for e in errors:
                        st.warning(e)
            if result_dicts:
                st.session_state["pe_tp_results"] = result_dicts
            else:
                st.error("All runs failed.")

        if "pe_tp_results" in st.session_state:
            st.info("Showing cached results. Click 'Run TP Sweep' to regenerate.")
            result_dicts = st.session_state["pe_tp_results"]
            df_tp = pd.DataFrame(result_dicts)

            fig_tpg = go.Figure()
            fig_lat_tp = go.Figure()
            for series in ["PD", "Aggregate"]:
                sdf = df_tp[df_tp["series"] == series].sort_values("tp_value")
                if sdf.empty:
                    continue
                color = sdf.iloc[0]["color"]
                dash = "solid" if series == "PD" else "dash"
                fig_tpg.add_trace(go.Scatter(
                    x=sdf["tp_value"], y=sdf["throughput_per_gpu"],
                    mode="lines+markers", name=series,
                    marker=dict(color=color, size=9),
                    line=dict(color=color, dash=dash),
                ))
                fig_lat_tp.add_trace(go.Scatter(
                    x=sdf["tp_value"], y=sdf["ttft_p95_ms"],
                    mode="lines+markers", name=f"{series} TTFT P95",
                    marker=dict(color=color, size=9),
                    line=dict(color=color, dash=dash),
                ))
            fig_tpg.update_layout(
                title="Throughput/GPU vs TP",
                xaxis_title="TP Value",
                yaxis_title="Throughput per GPU (tokens/s/GPU)",
                hovermode="x unified", height=400,
            )
            fig_lat_tp.update_layout(
                title="TTFT P95 vs TP",
                xaxis_title="TP Value",
                yaxis_title="TTFT P95 (ms)",
                hovermode="x unified", height=350,
            )
            st.plotly_chart(fig_tpg, use_container_width=True)
            st.plotly_chart(fig_lat_tp, use_container_width=True)

    else:  # Replica Sweep
        st.caption(
            "Fix P:D ratio and TP, sweep total instance count. "
            "Shows how PD and aggregate scale as you add more replicas."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            p_tp_rep = st.number_input(
                "Prefill TP", min_value=1, max_value=8, value=1, key="pe_rep_p_tp",
            )
            d_tp_rep = st.number_input(
                "Decode TP", min_value=1, max_value=8, value=1, key="pe_rep_d_tp",
            )
            agg_tp_rep = st.number_input(
                "Aggregate TP", min_value=1, max_value=8, value=1, key="pe_rep_agg_tp",
            )
        with col2:
            p_ratio = st.number_input(
                "Prefill ratio", min_value=1, value=1, key="pe_rep_p_ratio",
                help="Prefill share of total instances (e.g., 1 in 1:3)",
            )
            d_ratio = st.number_input(
                "Decode ratio", min_value=1, value=3, key="pe_rep_d_ratio",
                help="Decode share of total instances (e.g., 3 in 1:3)",
            )
            fixed_rate_rep = st.number_input(
                "Fixed rate (req/s)", min_value=0.1, value=10.0, step=1.0, key="pe_rep_rate",
            )
        with col3:
            rep_min = st.number_input(
                "Total instances min", min_value=2, value=4, key="pe_rep_min",
            )
            rep_max = st.number_input(
                "Total instances max", min_value=2, value=16, key="pe_rep_max",
            )
            rep_step = st.number_input(
                "Step size", min_value=1, value=4, key="pe_rep_step",
            )
            pe_rep_decider = st.selectbox("PD Decider", PD_DECIDERS, key="pe_rep_decider")

        if st.button("Run Replica Sweep", type="primary", key="pe_rep_run"):
            total_list = list(range(int(rep_min), int(rep_max) + 1, int(rep_step)))
            if not total_list:
                st.error("No valid instance counts in range.")
                return
            progress = st.progress(0, text="Starting replica sweep...")
            result_dicts, errors = run_replica_sweep(
                binary=binary, model=model, gpu=gpu, latency_model=latency_model,
                num_requests=num_requests, seed=seed,
                prompt_mean=prompt_mean, prompt_stdev=prompt_stdev,
                output_mean=output_mean, output_stdev=output_stdev,
                rate=float(fixed_rate_rep),
                p_tp=int(p_tp_rep), d_tp=int(d_tp_rep),
                p_ratio=int(p_ratio), d_ratio=int(d_ratio),
                total_instances_list=total_list,
                agg_tp=int(agg_tp_rep),
                pd_decider=pe_rep_decider,
                progress_bar=progress,
            )
            if errors:
                with st.expander(f"{len(errors)} warnings", expanded=False):
                    for e in errors:
                        st.warning(e)
            if result_dicts:
                st.session_state["pe_rep_results"] = result_dicts
            else:
                st.error("All runs failed.")

        if "pe_rep_results" in st.session_state:
            st.info("Showing cached results. Click 'Run Replica Sweep' to regenerate.")
            result_dicts = st.session_state["pe_rep_results"]
            df_rep = pd.DataFrame(result_dicts)

            fig_rep_tpg = go.Figure()
            fig_rep_lat = go.Figure()
            for series in ["PD", "Aggregate"]:
                sdf = df_rep[df_rep["series"] == series].sort_values("total_instances")
                if sdf.empty:
                    continue
                color = sdf.iloc[0]["color"]
                dash = "solid" if series == "PD" else "dash"
                fig_rep_tpg.add_trace(go.Scatter(
                    x=sdf["total_instances"], y=sdf["throughput_per_gpu"],
                    mode="lines+markers", name=series,
                    marker=dict(color=color, size=9),
                    line=dict(color=color, dash=dash),
                ))
                fig_rep_lat.add_trace(go.Scatter(
                    x=sdf["total_instances"], y=sdf["ttft_p95_ms"],
                    mode="lines+markers", name=f"{series} TTFT P95",
                    marker=dict(color=color, size=9),
                    line=dict(color=color, dash=dash),
                ))
            fig_rep_tpg.update_layout(
                title="Throughput/GPU vs Total Instances",
                xaxis_title="Total Instances",
                yaxis_title="Throughput per GPU (tokens/s/GPU)",
                hovermode="x unified", height=400,
            )
            fig_rep_lat.update_layout(
                title="TTFT P95 vs Total Instances",
                xaxis_title="Total Instances",
                yaxis_title="TTFT P95 (ms)",
                hovermode="x unified", height=350,
            )
            st.plotly_chart(fig_rep_tpg, use_container_width=True)
            st.plotly_chart(fig_rep_lat, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="PD Disaggregation Explorer", layout="wide")
    st.title("PD Disaggregation Explorer")
    st.caption(
        "Compare LLM serving configurations: aggregate vs prefill-decode disaggregated. "
        "Explore rate sweeps, workload crossover points, and optimal GPU topology splits."
    )

    # Discover binary once at startup
    try:
        binary = find_blis_binary()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.caption(f"BLIS binary: `{binary}`")

    (model, gpu, latency_model, num_requests, seed,
     prompt_mean, prompt_stdev, output_mean, output_stdev, configs) = render_sidebar()

    tab_rate, tab_workload, tab_parallelism = st.tabs(
        ["Rate Sweep", "Workload Sweep", "Parallelism Explorer"]
    )

    common = dict(
        binary=binary, model=model, gpu=gpu, latency_model=latency_model,
        configs=configs, num_requests=num_requests, seed=seed,
        prompt_mean=prompt_mean, prompt_stdev=prompt_stdev,
        output_mean=output_mean, output_stdev=output_stdev,
    )

    with tab_rate:
        tab_rate_sweep(**common)

    with tab_workload:
        tab_workload_sweep(**common)

    with tab_parallelism:
        tab_parallelism_explorer(**common)


if __name__ == "__main__":
    main()
