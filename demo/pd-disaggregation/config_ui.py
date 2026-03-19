"""SimConfig dataclass, constants, defaults, and sidebar for the PD Disaggregation Demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DEFAULTS = {
    "meta-llama/llama-3.1-8b-instruct": {"gpu": "H100", "tp": 1},
    "meta-llama/llama-3.3-70b-instruct": {"gpu": "H100", "tp": 4},
    "qwen/qwen3-14b": {"gpu": "H100", "tp": 1},
    "qwen/qwen2.5-7b-instruct": {"gpu": "H100", "tp": 1},
    "qwen/qwen2.5-3b-instruct": {"gpu": "H100", "tp": 1},
    "ibm-granite/granite-3.1-8b-instruct": {"gpu": "H100", "tp": 4},
    "microsoft/phi-4": {"gpu": "H100", "tp": 2},
    "mistralai/mistral-small-24b-instruct-2501": {"gpu": "H100", "tp": 2},
    "codellama/codellama-34b-instruct-hf": {"gpu": "H100", "tp": 1},
    "nvidia/llama-3.1-nemotron-70b-instruct-hf": {"gpu": "H100", "tp": 4},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"gpu": "H100", "tp": 4},
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": {"gpu": "H100", "tp": 8},
}

MODELS = list(MODEL_DEFAULTS.keys())
GPUS = ["H100", "A100-SXM", "A100-80"]
LATENCY_MODELS = ["blackbox", "roofline", "crossmodel", "trained-roofline"]
PD_DECIDERS = ["always", "prefix-threshold", "direct-to-decode", "never"]
SERIES_COLORS = ["#EF553B", "#636EFA", "#00CC96", "#AB63FA"]


# ---------------------------------------------------------------------------
# SimConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Complete configuration for one BLIS simulation series."""
    name: str
    color: str
    mode: str  # "Aggregate" or "Disaggregated"
    num_instances: int
    tp: int
    # PD fields (None = not applicable / use defaults)
    prefill_instances: Optional[int] = None
    decode_instances: Optional[int] = None
    pd_decider: Optional[str] = None
    pd_prefix_threshold: Optional[int] = None
    pd_direct_decode_threshold: Optional[int] = None
    pd_transfer_bandwidth: Optional[float] = None
    pd_transfer_base_latency: Optional[float] = None
    pd_kv_bytes_per_token: Optional[int] = None
    pd_transfer_contention: Optional[bool] = None
    pd_interference_prefill: Optional[float] = None
    pd_interference_decode: Optional[float] = None
    prefill_tp: Optional[int] = None
    decode_tp: Optional[int] = None
    prefill_hardware: Optional[str] = None
    decode_hardware: Optional[str] = None
    prefill_latency_model: Optional[str] = None
    decode_latency_model: Optional[str] = None
    prefill_routing_scorers: Optional[str] = None
    decode_routing_scorers: Optional[str] = None

    @property
    def num_gpus(self) -> int:
        """Total GPU count for this configuration."""
        if self.mode == "Disaggregated" and self.prefill_tp and self.decode_tp:
            p = self.prefill_instances or 0
            d = self.decode_instances or 0
            return p * self.prefill_tp + d * self.decode_tp
        return self.num_instances * self.tp

    def to_run_kwargs(self) -> dict:
        """Build kwargs for run_blis() from this config (excludes gpu, which is separate)."""
        kwargs: dict = dict(
            num_instances=self.num_instances,
            tp=self.tp,
            mode=self.mode,
            config_name=self.name,
        )
        if self.mode == "Disaggregated":
            kwargs.update(
                prefill_instances=self.prefill_instances or 0,
                decode_instances=self.decode_instances or 0,
                pd_decider=self.pd_decider or "always",
                pd_prefix_threshold=self.pd_prefix_threshold,
                pd_direct_decode_threshold=self.pd_direct_decode_threshold,
                pd_transfer_bandwidth=self.pd_transfer_bandwidth,
                pd_transfer_base_latency=self.pd_transfer_base_latency,
                pd_kv_bytes_per_token=self.pd_kv_bytes_per_token,
                pd_transfer_contention=self.pd_transfer_contention,
                pd_interference_prefill=self.pd_interference_prefill,
                pd_interference_decode=self.pd_interference_decode,
                prefill_routing_scorers=self.prefill_routing_scorers,
                decode_routing_scorers=self.decode_routing_scorers,
                prefill_tp=self.prefill_tp,
                decode_tp=self.decode_tp,
                prefill_hardware=self.prefill_hardware,
                decode_hardware=self.decode_hardware,
                prefill_latency_model=self.prefill_latency_model,
                decode_latency_model=self.decode_latency_model,
            )
        return kwargs


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def get_default_configs(model: str) -> list:
    """Return 3 sensible starting configurations for comparison."""
    tp = MODEL_DEFAULTS.get(model, {}).get("tp", 1)
    total_instances = max(2, 8 // tp)  # 8 GPU budget
    prefill_inst = max(1, total_instances // 4)
    decode_inst = total_instances - prefill_inst

    return [
        SimConfig(
            name=f"Aggregate {total_instances}\u00d7TP={tp}",
            color=SERIES_COLORS[0],
            mode="Aggregate",
            num_instances=total_instances,
            tp=tp,
        ),
        SimConfig(
            name=f"PD {prefill_inst}P+{decode_inst}D (always)",
            color=SERIES_COLORS[1],
            mode="Disaggregated",
            num_instances=total_instances,
            tp=tp,
            prefill_instances=prefill_inst,
            decode_instances=decode_inst,
            pd_decider="always",
            prefill_tp=tp,
            decode_tp=tp,
            pd_interference_prefill=0.15,
            pd_interference_decode=0.20,
        ),
        SimConfig(
            name=f"PD {prefill_inst}P+{decode_inst}D (direct-decode)",
            color=SERIES_COLORS[2],
            mode="Disaggregated",
            num_instances=total_instances,
            tp=tp,
            prefill_instances=prefill_inst,
            decode_instances=decode_inst,
            pd_decider="direct-to-decode",
            pd_direct_decode_threshold=256,
            prefill_tp=tp,
            decode_tp=tp,
            pd_interference_prefill=0.15,
            pd_interference_decode=0.20,
        ),
    ]


# ---------------------------------------------------------------------------
# Sidebar: Configuration Editor
# ---------------------------------------------------------------------------

def _pd_advanced_editor(cfg_idx: int, cfg: SimConfig) -> None:
    """Render PD Advanced expander for one config. Mutates cfg in-place.

    Prefill/decode instance counts and per-pool TP are set by the caller
    (card body) before this function is invoked; this function handles all
    other PD parameters.
    """
    with st.expander("PD Advanced", expanded=False):
        decider = st.selectbox(
            "Decider", PD_DECIDERS,
            index=PD_DECIDERS.index(cfg.pd_decider) if cfg.pd_decider in PD_DECIDERS else 0,
            key=f"pd_decider_{cfg_idx}",
        )

        col1, col2 = st.columns(2)
        with col1:
            prefix_thresh = st.number_input(
                "Prefix Threshold (tokens)", min_value=1,
                value=cfg.pd_prefix_threshold or 512,
                key=f"prefix_thresh_{cfg_idx}",
                help="Only used with prefix-threshold decider",
            )
            direct_thresh = st.number_input(
                "Direct-Decode Threshold (tokens)", min_value=1,
                value=cfg.pd_direct_decode_threshold or 256,
                key=f"direct_thresh_{cfg_idx}",
                help="Only used with direct-to-decode decider",
            )
            bw = st.number_input(
                "Transfer Bandwidth (GB/s)", min_value=0.1,
                value=cfg.pd_transfer_bandwidth or 25.0,
                key=f"bw_{cfg_idx}",
            )
            base_lat = st.number_input(
                "Transfer Base Latency (ms)", min_value=0.0,
                value=cfg.pd_transfer_base_latency or 0.05,
                step=0.01, format="%.3f", key=f"base_lat_{cfg_idx}",
            )
        with col2:
            bytes_per_token = st.number_input(
                "KV Bytes/Token", min_value=1,
                value=cfg.pd_kv_bytes_per_token or 512,
                key=f"kv_bytes_{cfg_idx}",
            )
            contention = st.checkbox(
                "Transfer Contention", value=cfg.pd_transfer_contention or False,
                key=f"contention_{cfg_idx}",
                help="Fair-share bandwidth when transfers overlap",
            )
            interf_prefill = st.number_input(
                "Interference: Prefill Factor", min_value=0.0, max_value=2.0,
                value=cfg.pd_interference_prefill or 0.0, step=0.05,
                key=f"interf_p_{cfg_idx}",
                help="Co-location slowdown multiplier for prefill batches (0 = disabled)",
            )
            interf_decode = st.number_input(
                "Interference: Decode Factor", min_value=0.0, max_value=2.0,
                value=cfg.pd_interference_decode or 0.0, step=0.05,
                key=f"interf_d_{cfg_idx}",
                help="Co-location slowdown multiplier for decode batches (0 = disabled)",
            )

        with st.expander("Per-Pool Hardware Overrides", expanded=False):
            col5, col6 = st.columns(2)
            with col5:
                st.caption("Prefill Pool")
                p_hw = st.text_input(
                    "Prefill Hardware", value=cfg.prefill_hardware or "",
                    key=f"p_hw_{cfg_idx}", placeholder="e.g. H100",
                )
                p_lm = st.selectbox(
                    "Prefill Latency Model", [""] + LATENCY_MODELS,
                    index=([""] + LATENCY_MODELS).index(cfg.prefill_latency_model)
                    if cfg.prefill_latency_model in LATENCY_MODELS else 0,
                    key=f"p_lm_{cfg_idx}",
                )
                p_rs = st.text_input(
                    "Prefill Routing Scorers", value=cfg.prefill_routing_scorers or "",
                    key=f"p_rs_{cfg_idx}",
                    placeholder="e.g. queue-depth:2,kv-utilization:1",
                )
            with col6:
                st.caption("Decode Pool")
                d_hw = st.text_input(
                    "Decode Hardware", value=cfg.decode_hardware or "",
                    key=f"d_hw_{cfg_idx}", placeholder="e.g. A100-SXM",
                )
                d_lm = st.selectbox(
                    "Decode Latency Model", [""] + LATENCY_MODELS,
                    index=([""] + LATENCY_MODELS).index(cfg.decode_latency_model)
                    if cfg.decode_latency_model in LATENCY_MODELS else 0,
                    key=f"d_lm_{cfg_idx}",
                )
                d_rs = st.text_input(
                    "Decode Routing Scorers", value=cfg.decode_routing_scorers or "",
                    key=f"d_rs_{cfg_idx}",
                    placeholder="e.g. queue-depth:2,kv-utilization:1",
                )

        cfg.pd_decider = decider
        cfg.pd_prefix_threshold = prefix_thresh if decider == "prefix-threshold" else None
        cfg.pd_direct_decode_threshold = direct_thresh if decider == "direct-to-decode" else None
        cfg.pd_transfer_bandwidth = bw
        cfg.pd_transfer_base_latency = base_lat
        cfg.pd_kv_bytes_per_token = bytes_per_token
        cfg.pd_transfer_contention = contention if contention else None
        cfg.pd_interference_prefill = interf_prefill if interf_prefill > 0 else None
        cfg.pd_interference_decode = interf_decode if interf_decode > 0 else None
        cfg.prefill_hardware = p_hw if p_hw else None
        cfg.decode_hardware = d_hw if d_hw else None
        cfg.prefill_latency_model = p_lm if p_lm else None
        cfg.decode_latency_model = d_lm if d_lm else None
        cfg.prefill_routing_scorers = p_rs if p_rs else None
        cfg.decode_routing_scorers = d_rs if d_rs else None


def _clear_config_widget_keys() -> None:
    """Delete all per-config widget keys from session_state so that fresh
    default values (set via `value=`) are picked up on the next render."""
    for i in range(4):  # max 4 configs
        for key in [
            # core
            f"cfg_name_{i}", f"cfg_mode_{i}", f"cfg_inst_{i}", f"cfg_tp_{i}",
            # PD pool instances / TP
            f"p_inst_{i}", f"p_tp_{i}", f"d_inst_{i}", f"d_tp_{i}",
            # PD advanced
            f"pd_decider_{i}", f"prefix_thresh_{i}", f"direct_thresh_{i}",
            f"bw_{i}", f"base_lat_{i}", f"kv_bytes_{i}", f"contention_{i}",
            f"interf_p_{i}", f"interf_d_{i}",
            # per-pool hardware
            f"p_hw_{i}", f"p_lm_{i}", f"p_rs_{i}",
            f"d_hw_{i}", f"d_lm_{i}", f"d_rs_{i}",
            # remove button
            f"remove_{i}",
        ]:
            st.session_state.pop(key, None)


def render_sidebar():
    """Render sidebar and return (model, gpu, latency_model, num_requests, seed,
    prompt_mean, prompt_stdev, output_mean, output_stdev, configs)."""
    with st.sidebar:
        st.header("Model & Hardware")
        model = st.selectbox("Model", MODELS, index=0)
        defaults = MODEL_DEFAULTS[model]
        gpu = st.selectbox("GPU", GPUS, index=GPUS.index(defaults["gpu"]))
        latency_model = st.selectbox("Latency Model", LATENCY_MODELS, index=0)

        st.header("Workload")
        col1, col2 = st.columns(2)
        with col1:
            prompt_mean = st.number_input(
                "Prompt tokens (mean)", min_value=1, value=1024, key="prompt_mean",
            )
            output_mean = st.number_input(
                "Output tokens (mean)", min_value=1, value=512, key="output_mean",
            )
        with col2:
            prompt_stdev = st.number_input(
                "Prompt tokens (stdev)", min_value=0, value=256, key="prompt_stdev",
            )
            output_stdev = st.number_input(
                "Output tokens (stdev)", min_value=0, value=256, key="output_stdev",
            )

        st.header("Run Settings")
        num_requests = st.number_input(
            "Requests per Point", min_value=50, value=200, key="num_requests",
        )
        seed = st.number_input("Seed", min_value=0, value=42, key="seed")

        st.header("Configurations")

        # Initialize configs in session state if not present or model changed
        if "configs" not in st.session_state or st.session_state.get("_cfg_model") != model:
            _clear_config_widget_keys()
            st.session_state["configs"] = get_default_configs(model)
            st.session_state["_cfg_model"] = model

        configs = st.session_state["configs"]

        # Add configuration button (max 4)
        if len(configs) < 4:
            if st.button("+ Add Configuration"):
                new_idx = len(configs)
                tp = MODEL_DEFAULTS.get(model, {}).get("tp", 1)
                new_cfg = SimConfig(
                    name=f"Config {new_idx + 1}",
                    color=SERIES_COLORS[new_idx % len(SERIES_COLORS)],
                    mode="Aggregate",
                    num_instances=4,
                    tp=tp,
                )
                configs.append(new_cfg)
                st.session_state["configs"] = configs
                st.rerun()

        to_remove = []
        for idx, cfg in enumerate(configs):
            with st.expander(
                f"**{cfg.name}** ({cfg.mode}, {cfg.num_gpus} GPUs)",
                expanded=(idx == 0),
            ):
                cfg.name = st.text_input("Name", value=cfg.name, key=f"cfg_name_{idx}")
                cfg.mode = st.radio(
                    "Mode", ["Aggregate", "Disaggregated"],
                    index=0 if cfg.mode == "Aggregate" else 1,
                    key=f"cfg_mode_{idx}", horizontal=True,
                )

                if cfg.mode == "Aggregate":
                    col_a, col_b = st.columns(2)
                    with col_a:
                        cfg.num_instances = st.number_input(
                            "Instances", min_value=1, value=cfg.num_instances,
                            key=f"cfg_inst_{idx}",
                        )
                    with col_b:
                        cfg.tp = st.number_input(
                            "TP", min_value=1, max_value=8, value=cfg.tp,
                            key=f"cfg_tp_{idx}",
                        )
                    st.caption(
                        f"Total: {cfg.num_instances} \u00d7 TP{cfg.tp} = {cfg.num_gpus} GPUs"
                    )
                else:
                    # Disaggregated: show P/D instances and per-pool TP in card body
                    default_p = cfg.prefill_instances or max(1, cfg.num_instances // 4)
                    default_d = cfg.decode_instances or max(1, cfg.num_instances - default_p)
                    default_p_tp = cfg.prefill_tp or cfg.tp
                    default_d_tp = cfg.decode_tp or cfg.tp

                    col_p, col_ptp, col_d, col_dtp = st.columns(4)
                    with col_p:
                        p_inst = st.number_input(
                            "P Instances", min_value=1,
                            value=default_p,
                            key=f"p_inst_{idx}",
                        )
                    with col_ptp:
                        p_tp = st.number_input(
                            "P TP", min_value=1, max_value=8,
                            value=default_p_tp,
                            key=f"p_tp_{idx}",
                        )
                    with col_d:
                        d_inst = st.number_input(
                            "D Instances", min_value=1,
                            value=default_d,
                            key=f"d_inst_{idx}",
                        )
                    with col_dtp:
                        d_tp = st.number_input(
                            "D TP", min_value=1, max_value=8,
                            value=default_d_tp,
                            key=f"d_tp_{idx}",
                        )

                    cfg.prefill_instances = p_inst
                    cfg.decode_instances = d_inst
                    cfg.num_instances = p_inst + d_inst
                    cfg.prefill_tp = p_tp
                    cfg.decode_tp = d_tp
                    cfg.tp = p_tp  # global TP fallback (per-pool flags override)

                    st.caption(
                        f"Total: {cfg.num_gpus} GPUs "
                        f"({p_inst}\u00d7TP{p_tp} prefill + {d_inst}\u00d7TP{d_tp} decode)"
                    )

                    _pd_advanced_editor(idx, cfg)

                if len(configs) > 1:
                    if st.button("Remove", key=f"remove_{idx}", type="secondary"):
                        to_remove.append(idx)

        for idx in reversed(to_remove):
            configs.pop(idx)
        if to_remove:
            st.session_state["configs"] = configs
            st.rerun()

        st.session_state["configs"] = configs

    return (
        model, gpu, latency_model, int(num_requests), int(seed),
        int(prompt_mean), int(prompt_stdev), int(output_mean), int(output_stdev),
        configs,
    )
