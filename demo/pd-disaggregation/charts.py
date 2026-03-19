"""Visualization helpers for the PD Disaggregation Demo."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto front across all configs: no other point dominates on BOTH metrics."""
    if df.empty:
        return df
    sorted_df = df.sort_values("throughput_per_user", ascending=False).reset_index(drop=True)
    pareto_indices = []
    max_tpg = -float("inf")
    for i, row in sorted_df.iterrows():
        if row["throughput_per_gpu"] >= max_tpg:
            pareto_indices.append(i)
            max_tpg = row["throughput_per_gpu"]
    return sorted_df.loc[pareto_indices].reset_index(drop=True)


def build_comparison_chart(
    df: pd.DataFrame, pareto_df: pd.DataFrame, x_col: str, x_label: str
) -> go.Figure:
    """Multi-series throughput tradeoff chart. x_col is the sweep axis."""
    fig = go.Figure()

    for config_name in df["config_name"].unique():
        config_df = df[df["config_name"] == config_name].sort_values(x_col)
        if config_df.empty:
            continue
        color = config_df.iloc[0]["color"] if "color" in config_df.columns else "#888888"

        hover_text = [
            f"<b>{config_name}</b><br>"
            f"{x_label}: {r[x_col]:.1f}<br>"
            f"Tokens/s: {r['tokens_per_sec']:.0f}<br>"
            f"Throughput/GPU: {r['throughput_per_gpu']:.1f}<br>"
            f"Throughput/User: {r['throughput_per_user']:.1f}<br>"
            f"Eff. Concurrency: {r['effective_concurrency']:.1f}<br>"
            f"TTFT P95: {r['ttft_p95_ms']:.1f} ms<br>"
            f"TPOT: {r['itl_mean_ms']:.1f} ms<br>"
            f"E2E P95: {r['e2e_p95_ms']:.1f} ms<br>"
            f"Completed: {r['completed_requests']}"
            for _, r in config_df.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=config_df["throughput_per_user"],
            y=config_df["throughput_per_gpu"],
            mode="lines+markers",
            name=config_name,
            marker=dict(color=color, size=9),
            line=dict(color=color, width=2),
            hovertext=hover_text,
            hoverinfo="text",
        ))

    if not pareto_df.empty:
        fig.add_trace(go.Scatter(
            x=pareto_df["throughput_per_user"],
            y=pareto_df["throughput_per_gpu"],
            mode="markers",
            name="Pareto Front",
            marker=dict(
                color="gold", symbol="star", size=16,
                line=dict(color="black", width=1),
            ),
            hovertext=[
                f"[Pareto] {r['config_name']}<br>"
                f"Throughput/User: {r['throughput_per_user']:.1f}<br>"
                f"Throughput/GPU: {r['throughput_per_gpu']:.1f}"
                for _, r in pareto_df.iterrows()
            ],
            hoverinfo="text",
        ))

    fig.update_layout(
        title="Throughput Tradeoff: GPU Efficiency vs User Experience",
        xaxis_title="Throughput per User (tokens/s/user)",
        yaxis_title="Throughput per GPU (tokens/s/GPU)",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
    )
    return fig


def build_latency_chart(df: pd.DataFrame, x_col: str, x_label: str) -> go.Figure:
    """TTFT P95 and E2E P95 vs sweep axis, per config."""
    fig = go.Figure()

    for config_name in df["config_name"].unique():
        config_df = df[df["config_name"] == config_name].sort_values(x_col)
        if config_df.empty:
            continue
        color = config_df.iloc[0]["color"] if "color" in config_df.columns else "#888888"

        fig.add_trace(go.Scatter(
            x=config_df[x_col],
            y=config_df["ttft_p95_ms"],
            customdata=config_df[["effective_concurrency"]],
            mode="lines+markers",
            name=f"{config_name} TTFT P95",
            marker=dict(color=color, symbol="circle", size=8),
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{config_name} TTFT P95</b><br>"
                + x_label + ": %{x:.1f}<br>"
                "TTFT P95: %{y:.1f} ms<br>"
                "Eff. Concurrency: %{customdata[0]:.1f}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=config_df[x_col],
            y=config_df["e2e_p95_ms"],
            customdata=config_df[["effective_concurrency"]],
            mode="lines+markers",
            name=f"{config_name} E2E P95",
            marker=dict(color=color, symbol="square", size=8),
            line=dict(color=color, width=2, dash="dash"),
            hovertemplate=(
                f"<b>{config_name} E2E P95</b><br>"
                + x_label + ": %{x:.1f}<br>"
                "E2E P95: %{y:.1f} ms<br>"
                "Eff. Concurrency: %{customdata[0]:.1f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Latency vs " + x_label,
        xaxis_title=x_label,
        yaxis_title="Latency (ms)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=400,
    )
    return fig


def build_pd_metrics_chart(
    df: pd.DataFrame, x_col: str, x_label: str
) -> Optional[go.Figure]:
    """Transfer duration vs sweep axis for PD configs only."""
    if "pd_disaggregated_count" not in df.columns:
        return None
    pd_df = df[df["pd_disaggregated_count"] > 0]
    if pd_df.empty:
        return None

    fig = go.Figure()
    for config_name in pd_df["config_name"].unique():
        config_df = pd_df[pd_df["config_name"] == config_name].sort_values(x_col)
        color = config_df.iloc[0]["color"] if "color" in config_df.columns else "#888888"

        fig.add_trace(go.Bar(
            x=config_df[x_col],
            y=config_df["pd_transfer_duration_mean_us"] / 1000.0,  # us -> ms
            customdata=config_df[["effective_concurrency"]],
            name=f"{config_name} Transfer Mean (ms)",
            marker_color=color,
            opacity=0.7,
            hovertemplate=(
                f"<b>{config_name}</b><br>"
                + x_label + ": %{x:.1f}<br>"
                "Transfer Duration: %{y:.2f} ms<br>"
                "Eff. Concurrency: %{customdata[0]:.1f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="KV Transfer Duration (PD configs)",
        xaxis_title=x_label,
        yaxis_title="KV Transfer Duration Mean (ms)",
        barmode="group",
        height=350,
    )
    return fig


def build_concurrency_chart(df: pd.DataFrame, x_col: str, x_label: str) -> go.Figure:
    """Effective concurrency vs sweep axis, per config."""
    fig = go.Figure()
    for config_name in df["config_name"].unique():
        config_df = df[df["config_name"] == config_name].sort_values(x_col)
        if config_df.empty:
            continue
        color = config_df.iloc[0]["color"] if "color" in config_df.columns else "#888888"
        fig.add_trace(go.Scatter(
            x=config_df[x_col],
            y=config_df["effective_concurrency"],
            mode="lines+markers",
            name=config_name,
            marker=dict(color=color, size=8),
            line=dict(color=color, width=2),
        ))
    fig.update_layout(
        title="Effective Concurrency vs " + x_label,
        xaxis_title=x_label,
        yaxis_title="Effective Concurrency (users)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=400,
    )
    return fig


def _compute_crossover_line(
    ratio_matrix: list, input_values: list, output_values: list
) -> tuple:
    """Find points where the ratio crosses 1.0 for crossover boundary annotation."""
    cx, cy = [], []
    for i, out_val in enumerate(output_values):
        row = ratio_matrix[i]
        for j in range(len(input_values) - 1):
            r1 = row[j]
            r2 = row[j + 1]
            if r1 is None or r2 is None or np.isnan(r1) or np.isnan(r2):
                continue
            if (r1 - 1.0) * (r2 - 1.0) <= 0 and r1 != r2:
                t = (1.0 - r1) / (r2 - r1)
                cx.append(input_values[j] + t * (input_values[j + 1] - input_values[j]))
                cy.append(out_val)
    return cx, cy


def build_heatmap_chart(
    ratio_matrix: list,
    input_values: list,
    output_values: list,
    config_a_name: str,
    config_b_name: str,
    metric: str,
) -> go.Figure:
    """2D heatmap of metric ratio (A / B). Color = ratio: red means A wins, blue means B wins.

    For throughput: ratio > 1 means A has higher throughput (A wins).
    For latency: ratio < 1 means A is faster (A wins), so the display is consistent.
    """
    z = ratio_matrix
    fig = go.Figure()

    # Main heatmap
    fig.add_trace(go.Heatmap(
        z=z,
        x=input_values,
        y=output_values,
        colorscale="RdBu",
        zmid=1.0,
        colorbar=dict(title="Ratio A/B"),
        hovertemplate=(
            "Input: %{x:.0f}<br>"
            "Output: %{y:.0f}<br>"
            "Ratio: %{z:.2f}<extra></extra>"
        ),
    ))

    # Crossover boundary line (ratio = 1.0)
    cx, cy = _compute_crossover_line(z, input_values, output_values)
    if cx:
        fig.add_trace(go.Scatter(
            x=cx, y=cy,
            mode="markers+lines",
            marker=dict(color="black", size=5),
            line=dict(color="black", width=2, dash="dash"),
            name="Crossover (ratio=1)",
            hoverinfo="skip",
            showlegend=True,
        ))

    is_latency = metric in ("TTFT P95", "E2E P95")
    win_note = (
        "Red: A faster (lower latency). Blue: B faster." if is_latency
        else "Red: A more throughput. Blue: B more throughput."
    )
    fig.update_layout(
        title=(
            f"2D Workload Heatmap: {config_a_name} vs {config_b_name}<br>"
            f"<sup>{metric} ratio (A/B). {win_note}</sup>"
        ),
        xaxis_title="Input Token Mean",
        yaxis_title="Output Token Mean",
        height=500,
    )
    return fig


def show_results_tables(df: pd.DataFrame, pareto_df: pd.DataFrame) -> None:
    """Display Pareto front and full results tables with CSV download."""
    display_cols = [
        "config_name", "rate", "effective_concurrency",
        "throughput_per_user", "throughput_per_gpu",
        "ttft_p95_ms", "itl_mean_ms", "e2e_p95_ms",
        "completed_requests", "dropped_unservable",
    ]
    col_labels = {
        "config_name": "Config",
        "rate": "Rate (req/s)",
        "effective_concurrency": "Eff. Concurrency",
        "throughput_per_user": "Throughput/User",
        "throughput_per_gpu": "Throughput/GPU",
        "ttft_p95_ms": "TTFT P95 (ms)",
        "itl_mean_ms": "TPOT (ms)",
        "e2e_p95_ms": "E2E P95 (ms)",
        "completed_requests": "Completed",
        "dropped_unservable": "Dropped",
    }

    # Add PD columns if any PD results exist
    if "pd_disaggregated_count" in df.columns and df["pd_disaggregated_count"].sum() > 0:
        display_cols += ["pd_disaggregated_count", "pd_transfer_duration_mean_us", "pd_load_imbalance"]
        col_labels.update({
            "pd_disaggregated_count": "Disagg. Count",
            "pd_transfer_duration_mean_us": "Transfer Mean (us)",
            "pd_load_imbalance": "Load Imbalance",
        })

    # Keep only cols that exist in df
    display_cols = [c for c in display_cols if c in df.columns]

    st.subheader("Pareto Front")
    st.caption("Points where no other configuration dominates on both throughput metrics.")
    if not pareto_df.empty:
        pareto_display_cols = [c for c in display_cols if c in pareto_df.columns]
        st.dataframe(
            pareto_df[pareto_display_cols].rename(columns=col_labels),
            use_container_width=True, hide_index=True,
        )

    with st.expander("All Results"):
        all_disp = df[display_cols].rename(columns=col_labels)
        st.dataframe(all_disp, use_container_width=True, hide_index=True)
        csv = all_disp.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "results.csv", "text/csv")
