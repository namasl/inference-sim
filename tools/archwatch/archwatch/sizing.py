"""Parameter-count estimation from a HuggingFace ``config.json``.

Why archwatch needs this
------------------------
The significance gate (S1) asks "is this model big enough to matter?". HuggingFace
does not put a parameter count in ``config.json``, model cards lie or omit it, and
fetching the safetensors index for every candidate would cost another request per
repo. So we reconstruct the count from the shape fields we already have.

Scope: this is a *sizing estimate*, not an accounting of BLIS's weight model. It aims at
the number the lab would print in its own release post ("671B total, 37B active"),
because that is what S1 is really thresholding on.

**Three different figures exist for one model** and they must not be conflated:

1. the **published** count — what the lab announces. This module targets it.
2. **archwatch's** estimate — this module's output, ``est_total_params``. It is a
   config-only reconstruction, so it misses anything the config does not declare (see
   the Llama-4 shared expert below).
3. **BLIS's** weight accounting — ``computeModelWeightBytes``. Deliberately *not*
   reproduced: where BLIS's arithmetic is wrong about a real model (its
   ``swiGLUActivations`` set treats bare ``gelu`` as ungated; it sizes Q/O projections as
   ``hidden**2`` rather than ``n_heads*head_dim``), matching it would import the error.
   A divergence between (2) and (3) is itself a finding, which is why they stay separate.

The formula
-----------
All terms are parameter *counts* (not bytes). Fields are read after pivoting
``text_config`` onto the top level, exactly as BLIS's ``ParseHFConfig`` does
(``sim/latency/config.go:213-218``), so a multimodal repo is sized on its text tower.

**That pivot is not an edge case.** Frontier releases increasingly ship a config whose
top level holds only ``architectures``, ``model_type``, ``dtype``, ``text_config`` and
``vision_config`` — every shape field lives inside ``text_config``. Reading top-level
keys only would return ``None`` for precisely the large models the significance gate
exists to catch, and the symptom would look like "the filter mysteriously drops
frontier models" rather than like a bug here.

Let ``H`` = ``hidden_size``, ``L`` = ``num_hidden_layers``, ``V`` = ``vocab_size``,
``n_h`` = ``num_attention_heads``, ``n_kv`` = ``num_key_value_heads`` (defaulting to
``n_h``), ``d_h`` = explicit ``head_dim`` else ``H // n_h``.

1. **Embeddings** ``V * H``, plus an ``lm_head`` of ``V * H`` unless
   ``tie_word_embeddings`` is true.

2. **Attention, per layer.** Two branches.

   *Standard MHA/GQA/MQA* (no ``kv_lora_rank``)::

       q_proj = H * (n_h * d_h)
       k_proj = H * (n_kv * d_h)
       v_proj = H * (n_kv * d_h)
       o_proj = (n_h * d_h) * H

   *MLA* (``kv_lora_rank > 0``; DeepSeek-V2/V3, Kimi-K2/K3, GLM-5.x)::

       qk_h  = qk_head_dim, else qk_nope_head_dim + qk_rope_head_dim, else d_h
       v_h   = v_head_dim, else d_h
       q     = H * q_lora_rank + q_lora_rank * n_h * qk_h   (when q_lora_rank > 0)
             = H * n_h * qk_h                               (otherwise)
       kv_a  = H * (kv_lora_rank + qk_rope_head_dim)
       kv_b  = kv_lora_rank * n_h * (qk_nope_head_dim + v_h)
       o     = n_h * v_h * H

3. **MLP, per layer.** ``m`` matrices at the layer's FFN dim, where ``m = 3`` for a
   gated (SwiGLU-family) ``hidden_act`` and ``m = 2`` otherwise.

   *Dense layer*: ``m * H * ffn`` with ``ffn`` = ``intermediate_size_mlp`` when set
   (Llama-4 style split dims), else ``intermediate_size``/``ffn_hidden_size``.

   *MoE layer*::

       routed = m * H * moe_ffn * n_experts
       shared = m * H * shared_expert_intermediate_size
                (or n_shared_experts * moe_ffn when only the count is given)
       router = n_experts * H

   ``moe_ffn`` is ``moe_intermediate_size`` when present, else ``intermediate_size``
   (the Mixtral convention, where ``intermediate_size`` *is* the per-expert dim).
   ``n_experts`` is resolved over the same alias chain BLIS uses
   (``num_experts``, ``moe_num_experts``, ``n_routed_experts``, ``num_local_experts``,
   ``num_routed_experts``) and must be >= 2 to count as MoE — a 1-expert config is
   dense-equivalent.

4. **Which layers are MoE.** ``first_k_dense_replace`` layers form a dense prefix;
   layers listed in ``mlp_only_layers`` (Qwen3-MoE) are dense; of what remains, every
   ``interleave_moe_layer_step``-th layer is MoE (step 1 or absent => all of them).

5. **Norms** ``2 * H`` per layer plus a final ``H``.

**Active parameters** replace the routed-expert term with only the experts a token
actually visits: ``num_experts_per_tok`` (or ``num_experts_per_token``) instead of
``n_experts``. Everything else — attention, norms, shared experts, router,
embeddings, ``lm_head``, dense-layer MLPs — is charged in full. For a dense model
active == total.

Deliberate omissions (each recorded in ``ParamEstimate.notes`` when it applies)
-------------------------------------------------------------------------------
- **MTP / speculative-decode heads** (``num_nextn_predict_layers``) are excluded.
  Labs quote the main model: DeepSeek-V3 is "671B" without its MTP module.
- **Linear-attention layers** in a hybrid model (``linear_attn_config``) are charged
  with the full-attention projection arithmetic. Their real weights (short conv,
  gates, recurrent state) differ, but the q/k/v/o projections dominate, so the error
  is small — unlike the *KV cache*, which differs by construction.
- **Biases, LayerScale, q/k norms, rotary buffers, vision towers, adapters.**
  Sub-1% terms.
- **Quantization and dtype.** Counts are parameters, not bytes, so
  ``quantization_config`` and the precision fields are irrelevant here by design —
  which also means this module is immune to the 2026 configs that spell it ``dtype``
  instead of ``torch_dtype``. (BLIS reads ``torch_dtype`` with a ``dtype`` fallback at
  ``sim/latency/config.go:334-336``; that matters for its byte accounting, not ours.)

Accuracy against published counts (measured; see ``tests/test_sizing.py``)::

    model                    estimated total/active   published    error total/active
    Llama-3.1-70B-Instruct        70.55B / 70.55B     70.55B/70.55B   -0.0% /  -0.0%
    Mixtral-8x7B-v0.1             46.70B / 12.88B     46.7B / 12.9B   +0.0% /  -0.2%
    Qwen3-30B-A3B                 30.53B /  3.35B     30.5B /  3.3B   +0.1% /  +1.6%
    DeepSeek-V3                  671.03B / 37.55B     671B  / 37B     +0.0% /  +1.5%
    DeepSeek-V2-Lite              15.71B /  2.66B     15.7B /  2.4B   +0.0% / +10.9%
    Llama-4-Scout-17B-16E        101.73B / 11.13B     109B  / 17B     -6.7% / -34.5%

Totals land within 0.1% on every model whose shared-expert structure is declared in
``config.json``. The two outliers are known and bounded:

- **DeepSeek-V2-Lite active (+10.9%)** — DeepSeek's "2.4B activated" appears to exclude
  a term this uniform accounting includes (most plausibly the 0.21B embedding).
- **Llama-4 Scout (-6.7% total, -34.5% active)** — Llama-4 runs one always-on shared
  expert per MoE layer, but *nothing in the config declares it* (no
  ``n_shared_experts``, and ``intermediate_size_mlp`` belongs to the dense layers).
  No config-only estimator can see it. The total is still far above any plausible S1
  threshold, so the gate is unaffected; the active figure should not be quoted for
  Llama-4-family models.

Returning ``None``
------------------
When a required field is missing or non-positive the estimate is ``None`` and
``ParamEstimate.missing`` names the fields. We never guess a default for
``hidden_size`` or ``num_hidden_layers``: a fabricated parameter count silently
mis-fires the significance gate in both directions, which is worse than no number.

The alias chains below cover the spellings seen in real HuggingFace *language-model*
configs (they are BLIS's and vLLM's chains, plus the long-standing GPT-2/T5 names).
A config that invents its own vocabulary — ``ParallaxOpen/Vela-Lumen-31M`` declares
``d_model``/``n_layers``/``n_heads``/``n_kv_heads`` and no ``architectures[]`` at all —
returns ``None`` with ``missing`` naming exactly what could not be read. That is the
intended outcome: BLIS would parse none of those fields either, so the honest answer is
"unknown", and the named gap shows up in the run log where a human can decide whether
the spelling is common enough to add.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ParamEstimate",
    "estimate_params",
    "estimate_total_params",
    "estimate_active_params",
    "format_params",
    "pivot_text_config",
    "resolve_num_experts",
]

# ---------------------------------------------------------------------------
# Field alias chains
# ---------------------------------------------------------------------------

# Total routed-expert count. Same chain, same order, as BLIS's
# latency.moeExpertCountFields / vLLM's get_num_experts.
EXPERT_COUNT_FIELDS: tuple[str, ...] = (
    "num_experts",         # Jamba
    "moe_num_experts",     # Dbrx
    "n_routed_experts",    # DeepSeek
    "num_local_experts",   # Mixtral
    "num_routed_experts",  # BLIS-historical alias
)

# Experts visited per token. DeepSeek/GLM/Qwen spell it ..._tok; Kimi-K3 ..._token.
ACTIVE_EXPERT_FIELDS: tuple[str, ...] = ("num_experts_per_tok", "num_experts_per_token")

# Always-on expert count.
SHARED_EXPERT_COUNT_FIELDS: tuple[str, ...] = ("n_shared_experts", "num_shared_experts")

HIDDEN_SIZE_FIELDS: tuple[str, ...] = ("hidden_size", "n_embd", "d_model")
NUM_LAYERS_FIELDS: tuple[str, ...] = ("num_hidden_layers", "n_layer", "num_layers")
NUM_HEADS_FIELDS: tuple[str, ...] = ("num_attention_heads", "n_head", "num_heads")
KV_HEADS_FIELDS: tuple[str, ...] = (
    "num_key_value_heads",
    "num_kv_heads",           # Falcon
    "multi_query_group_num",  # GLM
)
INTERMEDIATE_FIELDS: tuple[str, ...] = ("intermediate_size", "ffn_hidden_size")

# A gated MLP has three weight matrices (gate, up, down); an ungated one has two.
#
# ``hidden_act`` names the *nonlinearity*, not the MLP topology, so this set is a prior
# over the population rather than a decoding of the field. A spelling is listed only when
# a real model that declares it is demonstrably gated in its own modeling code:
#
#   silu                  Llama/Qwen/Mistral: down(silu(gate(x)) * up(x)) — SwiGLU
#   gelu                  Spark2_5 (XHToken/Spark-X2.5-4B): modeling_spark.py has
#                         Spark2_5MLP = down_proj(act_fn(gate_proj(x)) * up_proj(x)) —
#                         a GEGLU. "gelu" names the gate nonlinearity there exactly as
#                         Llama's "silu" names SwiGLU's.
#   gelu_pytorch_tanh     Gemma 2/3: GeGLU.
#   gelu_tanh             the same function under its shorter alias.
#   swiglu/geglu/reglu/glu/gelu_pytorch_tanh_glu/situ
#                         self-describing; the name IS the gated form.
#
# Deliberately EXCLUDED, because no known gated model declares them: ``gelu_new``
# (GPT-2/GPT-Neo/GPT-J lineage, all ungated), ``quick_gelu`` (CLIP vision), ``gelu_fast``,
# ``relu``. Adding a spelling on suspicion rather than on a gated exemplar would make
# this set a constant and the distinction pointless.
#
# ``gelu`` is the genuinely ambiguous one: BERT-family encoders declare it and are NOT
# gated, so listing it over-counts them. That is the right direction to be wrong in.
# ``est_total_params`` feeds the S1 scale gate, so an under-count DROPS a model
# (unrecoverable), while an over-count only inflates a number: every gelu encoder is a
# ~100-400M model that stays far below the 3B threshold either way, whereas a gelu-gated
# 3.5B decoder under-counted by 23% falls out of the report entirely. This is the bug that
# occasioned the note — Spark2_5 was reported at 3.17B against a real 4.11B.
#
# NOTE this intentionally diverges from BLIS's own swiGLUActivations
# (sim/latency/kv_capacity.go), which lists silu/swiglu/geglu/situ and so treats bare
# "gelu" as ungated. That is the same bug on BLIS's side; sizing targets the *published*
# parameter count, not BLIS's arithmetic, so it does not reproduce BLIS's error.
GATED_ACTIVATIONS: frozenset[str] = frozenset(
    {
        "silu", "swish", "swiglu", "geglu", "reglu", "glu", "situ",
        "gelu", "gelu_pytorch_tanh", "gelu_tanh", "gelu_pytorch_tanh_glu",
    }
)

# An expert count below this is dense-equivalent (BLIS's sim.MoEMinExperts).
MOE_MIN_EXPERTS = 2

_REQUIRED = (
    ("hidden_size", HIDDEN_SIZE_FIELDS),
    ("num_hidden_layers", NUM_LAYERS_FIELDS),
    ("vocab_size", ("vocab_size",)),
    ("num_attention_heads", NUM_HEADS_FIELDS),
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ParamEstimate:
    """Outcome of sizing one config.

    ``total``/``active`` are ``None`` together — either the config had what the
    formula needs or it did not.
    """

    total: int | None = None
    active: int | None = None
    is_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    # Approximations that were actually applied to *this* config.
    notes: list[str] = field(default_factory=list)
    # Required fields that were absent or non-positive; non-empty <=> total is None.
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total is not None


# ---------------------------------------------------------------------------
# Config access helpers
# ---------------------------------------------------------------------------


def pivot_text_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``text_config`` onto the top level, mirroring BLIS's ``ParseHFConfig``.

    A multimodal repo (Llama-4, Gemma-3, Qwen-VL) keeps every shape field the text
    tower needs inside ``text_config``; the inner values win on collision, exactly as
    in ``sim/latency/config.go``. Returns a copy — the input is never mutated.
    """
    if not isinstance(config, dict):
        return {}
    merged = dict(config)
    inner = config.get("text_config")
    if isinstance(inner, dict):
        merged.update(inner)
    return merged


def _as_int(value: Any) -> int | None:
    """Coerce a JSON scalar to a non-negative-capable int, or None.

    ``config.json`` numbers arrive as ``int`` or ``float``; ``null`` (DeepSeek's
    ``q_lora_rank``) and strings must read as "absent", not as 0, so that "field
    missing" and "field is zero" stay distinguishable upstream.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _first_int(config: dict[str, Any], keys: tuple[str, ...], *, minimum: int = 1) -> int:
    """First key in ``keys`` whose value is an int >= ``minimum``; else 0."""
    for key in keys:
        got = _as_int(config.get(key))
        if got is not None and got >= minimum:
            return got
    return 0


def resolve_num_experts(config: dict[str, Any]) -> int:
    """Total routed-expert count, or 0 for a dense (or dense-equivalent) model.

    Walks ``EXPERT_COUNT_FIELDS`` in order and returns the first value >=
    ``MOE_MIN_EXPERTS``, matching BLIS's ``ResolveNumExperts`` so archwatch and BLIS
    agree on which models are MoE at all.
    """
    return _first_int(config, EXPERT_COUNT_FIELDS, minimum=MOE_MIN_EXPERTS)


def _list_len(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    return len(value) if isinstance(value, list) else 0


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


def estimate_params(config: dict[str, Any] | None) -> ParamEstimate:
    """Estimate total and active parameter counts. See the module docstring.

    Never raises on a malformed config: anything it cannot read becomes an entry in
    ``ParamEstimate.missing`` and the counts stay ``None``.
    """
    est = ParamEstimate()
    if not isinstance(config, dict) or not config:
        est.missing = [name for name, _ in _REQUIRED]
        return est

    cfg = pivot_text_config(config)
    if isinstance(config.get("text_config"), dict):
        est.notes.append("sized on text_config (multimodal repo; vision tower excluded)")

    hidden = _first_int(cfg, HIDDEN_SIZE_FIELDS)
    layers = _first_int(cfg, NUM_LAYERS_FIELDS)
    vocab = _first_int(cfg, ("vocab_size",))
    heads = _first_int(cfg, NUM_HEADS_FIELDS)

    for name, keys in _REQUIRED:
        if _first_int(cfg, keys) == 0:
            est.missing.append(name)

    n_experts = resolve_num_experts(cfg)
    moe_ffn = _first_int(cfg, ("moe_intermediate_size",))
    dense_ffn = _first_int(cfg, INTERMEDIATE_FIELDS)
    if n_experts and moe_ffn == 0:
        # Mixtral convention: intermediate_size IS the per-expert dim.
        moe_ffn = dense_ffn
        if moe_ffn:
            est.notes.append(
                "no moe_intermediate_size; per-expert FFN dim taken from "
                "intermediate_size (Mixtral convention)"
            )
    if dense_ffn == 0 and moe_ffn == 0:
        est.missing.append("intermediate_size")

    if est.missing:
        return est

    # --- layer typing -----------------------------------------------------
    kv_heads = _first_int(cfg, KV_HEADS_FIELDS) or heads
    head_dim = _first_int(cfg, ("head_dim",)) or (hidden // heads)
    if head_dim == 0:
        est.missing.append("head_dim")
        return est

    gated = str(cfg.get("hidden_act") or cfg.get("hidden_activation") or "silu").lower()
    mlp_matrices = 3 if gated in GATED_ACTIVATIONS else 2
    if mlp_matrices == 2:
        est.notes.append(f"hidden_act={gated!r} treated as ungated (2-matrix MLP)")
    elif gated.startswith("gelu"):
        # Worth surfacing: a gelu spelling is ambiguous in a way silu is not, and an
        # encoder that declares it is over-counted by half its MLP.
        est.notes.append(
            f"hidden_act={gated!r} treated as GATED (3-matrix MLP); correct for a "
            "gelu-gated decoder such as Spark2_5, an over-count for a BERT-family encoder"
        )

    est.is_moe = n_experts >= MOE_MIN_EXPERTS
    est.num_experts = n_experts

    # --- attention --------------------------------------------------------
    kv_lora_rank = _first_int(cfg, ("kv_lora_rank",))
    if kv_lora_rank:
        attn_per_layer = _mla_attention_params(cfg, hidden, heads, head_dim, kv_lora_rank)
        est.notes.append(
            f"MLA attention (kv_lora_rank={kv_lora_rank}); "
            "q/kv down+up projections sized explicitly"
        )
    else:
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        attn_per_layer = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden

    full_attn_layers = 0
    lac = cfg.get("linear_attn_config")
    if isinstance(lac, dict):
        full_attn_layers = _list_len(lac, "full_attn_layers")
        est.notes.append(
            f"hybrid attention: {full_attn_layers or '?'}/{layers} full-attention "
            "layers; linear-attention layers charged with full-attention projections"
        )

    # --- MLP: how many layers are MoE ------------------------------------
    dense_prefix = max(0, min(_first_int(cfg, ("first_k_dense_replace",), minimum=0), layers))
    mlp_only = _list_len(cfg, "mlp_only_layers")
    moe_layers = 0
    if est.is_moe:
        remaining = layers - min(layers, dense_prefix + mlp_only)
        step = _first_int(cfg, ("interleave_moe_layer_step",)) or 1
        moe_layers = remaining // step
        dense_layers = layers - moe_layers
        if step > 1:
            est.notes.append(f"interleave_moe_layer_step={step}: {moe_layers}/{layers} MoE layers")
        elif dense_layers:
            est.notes.append(f"dense prefix / mlp_only: {dense_layers}/{layers} layers dense")
    else:
        # Dense model: every layer carries an MLP. (A dense_prefix/mlp_only value on a
        # non-MoE config is meaningless, so it is ignored rather than subtracted.)
        dense_layers = layers

    # --- MLP parameter terms ---------------------------------------------
    dense_layer_ffn = _first_int(cfg, ("intermediate_size_mlp",)) or dense_ffn
    if dense_layer_ffn == 0:
        dense_layer_ffn = moe_ffn
    dense_mlp_per_layer = mlp_matrices * hidden * dense_layer_ffn

    routed_total_per_layer = 0
    routed_active_per_layer = 0
    fixed_moe_per_layer = 0
    if est.is_moe:
        per_expert = mlp_matrices * hidden * moe_ffn
        routed_total_per_layer = per_expert * n_experts

        per_tok = _first_int(cfg, ACTIVE_EXPERT_FIELDS)
        if per_tok == 0:
            # BLIS treats this as fatal (the MoE-consistency guard); for sizing we
            # fall back to "all experts active" and say so, so a caller reading
            # active == total on an MoE model is not misled into thinking it is dense.
            per_tok = n_experts
            est.notes.append(
                "num_experts_per_tok absent on a MoE config; active params fall back "
                "to all experts (upper bound)"
            )
        per_tok = min(per_tok, n_experts)
        est.num_experts_per_tok = per_tok
        routed_active_per_layer = per_expert * per_tok

        shared_dim = _first_int(cfg, ("shared_expert_intermediate_size",))
        if shared_dim == 0:
            n_shared = _first_int(cfg, SHARED_EXPERT_COUNT_FIELDS)
            shared_dim = n_shared * moe_ffn
        # Shared experts run on every token; the router is a tiny always-on matrix.
        fixed_moe_per_layer = mlp_matrices * hidden * shared_dim + n_experts * hidden

    # --- assemble ---------------------------------------------------------
    embeddings = vocab * hidden
    lm_head = 0 if bool(cfg.get("tie_word_embeddings")) else vocab * hidden
    norms = layers * 2 * hidden + hidden
    attention = layers * attn_per_layer
    dense_mlp = dense_layers * dense_mlp_per_layer

    base = embeddings + lm_head + norms + attention + dense_mlp
    est.total = base + moe_layers * (routed_total_per_layer + fixed_moe_per_layer)
    est.active = base + moe_layers * (routed_active_per_layer + fixed_moe_per_layer)

    mtp = _first_int(cfg, ("num_nextn_predict_layers",))
    if mtp:
        est.notes.append(
            f"num_nextn_predict_layers={mtp}: MTP module excluded (labs quote the main model)"
        )

    return est


def _mla_attention_params(
    cfg: dict[str, Any], hidden: int, heads: int, head_dim: int, kv_lora_rank: int
) -> int:
    """Per-layer attention params for Multi-head Latent Attention.

    MLA replaces the flat q/k/v projections with low-rank down+up pairs, so the
    standard ``4 * H^2``-shaped estimate is wrong in both directions: the KV path
    shrinks to a ``kv_lora_rank``-wide latent while the Q path can *grow* (DeepSeek-V3
    carries 128 heads of 192-wide qk against a 7168 hidden).
    """
    qk_head_dim = _first_int(cfg, ("qk_head_dim",))
    qk_nope = _first_int(cfg, ("qk_nope_head_dim",))
    qk_rope = _first_int(cfg, ("qk_rope_head_dim",))
    if qk_head_dim == 0:
        qk_head_dim = (qk_nope + qk_rope) or head_dim
    if qk_nope == 0:
        qk_nope = max(qk_head_dim - qk_rope, 0) or head_dim
    v_head_dim = _first_int(cfg, ("v_head_dim",)) or head_dim

    q_lora_rank = _first_int(cfg, ("q_lora_rank",))
    if q_lora_rank:
        q = hidden * q_lora_rank + q_lora_rank * heads * qk_head_dim
    else:
        q = hidden * heads * qk_head_dim

    kv_a = hidden * (kv_lora_rank + qk_rope)
    kv_b = kv_lora_rank * heads * (qk_nope + v_head_dim)
    o = heads * v_head_dim * hidden
    return q + kv_a + kv_b + o


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def estimate_total_params(config: dict[str, Any] | None) -> int | None:
    """Total parameter count, or ``None`` when the config lacks what it needs."""
    return estimate_params(config).total


def estimate_active_params(config: dict[str, Any] | None) -> int | None:
    """Per-token active parameter count (== total for a dense model), or ``None``."""
    return estimate_params(config).active


def format_params(n: int | None) -> str:
    """Human-readable count for issue text: ``70554____`` -> ``70.6B``."""
    if n is None:
        return "unknown"
    for scale, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M")):
        if abs(n) >= scale:
            return f"{n / scale:.3g}{suffix}"
    return str(n)
