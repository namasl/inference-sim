"""Tests for archwatch.sizing.

The load-bearing test is :func:`test_matches_published_parameter_counts`: the
significance gate thresholds on this number, so it has to land near what the lab
published. Everything else guards the "return None rather than guess" contract and
the config-shape branches (MoE alias chains, MLA, hybrid, multimodal pivot).

Pure: fixtures only, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archwatch.sizing import (
    GATED_ACTIVATIONS,
    MOE_MIN_EXPERTS,
    ParamEstimate,
    estimate_active_params,
    estimate_params,
    estimate_total_params,
    format_params,
    pivot_text_config,
    resolve_num_experts,
)

FIXTURES = Path(__file__).parent / "fixtures" / "novelty"

B = 1_000_000_000


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


# ---------------------------------------------------------------------------
# Accuracy against published counts
# ---------------------------------------------------------------------------

# (fixture, published total, published active, tolerance as a fraction)
#
# Published numbers are the labs' own headline figures. Tolerances are per-case
# because the sources of error differ: DeepSeek-V2-Lite's "2.4B activated" excludes
# something our uniform accounting includes (most likely the embedding term), and
# Llama-4 Scout declares no shared expert in config.json although its modeling code
# has one (covered separately in test_llama4_scout_shared_expert_is_undercounted).
PUBLISHED = [
    ("dense_llama31_70b", 70.554 * B, 70.554 * B, 0.01),
    ("moe_mixtral_8x7b", 46.7 * B, 12.9 * B, 0.02),
    ("moe_qwen3_30b_a3b", 30.5 * B, 3.3 * B, 0.03),
    ("mla_moe_deepseek_v3", 671.0 * B, 37.0 * B, 0.02),
    ("mla_moe_deepseek_v2_lite", 15.7 * B, 2.4 * B, 0.12),
]


@pytest.mark.parametrize("name,total,active,tol", PUBLISHED)
def test_matches_published_parameter_counts(name, total, active, tol):
    est = estimate_params(load(name))
    assert est.total is not None and est.active is not None
    assert est.total == pytest.approx(total, rel=tol), (
        f"{name}: estimated {format_params(est.total)}, published {format_params(int(total))}"
    )
    assert est.active == pytest.approx(active, rel=tol), (
        f"{name}: estimated {format_params(est.active)} active, "
        f"published {format_params(int(active))}"
    )


def test_dense_model_active_equals_total():
    est = estimate_params(load("dense_llama31_70b"))
    assert est.is_moe is False
    assert est.total == est.active


@pytest.mark.parametrize(
    "name",
    [
        "dense_llama31_70b",
        "moe_mixtral_8x7b",
        "moe_qwen3_30b_a3b",
        "mla_moe_deepseek_v3",
        "mla_moe_deepseek_v2_lite",
        "hybrid_mla_linear_attn",
        "novel_arch_large",
        "multimodal_llama4_scout",
    ],
)
def test_active_never_exceeds_total(name):
    est = estimate_params(load(name))
    assert est.active is not None and est.total is not None
    assert est.active <= est.total


def test_llama4_scout_shared_expert_is_undercounted():
    """Documents a known limitation rather than hiding it behind a loose tolerance.

    Llama-4 runs one always-on shared expert per MoE layer, but nothing in
    ``config.json`` declares it (no ``n_shared_experts``), so no config-only estimator
    can see it. Scout is published at 109B total / 17B active; we land low. If someone
    teaches the estimator about it, this test fails and the module docstring's accuracy
    table needs updating.
    """
    est = estimate_params(load("multimodal_llama4_scout"))
    assert est.total is not None and est.active is not None
    assert 95 * B < est.total < 109 * B, "expected a modest under-count of 109B"
    assert est.active < 17 * B, "expected an under-count of the 17B active figure"
    # Still comfortably over the S1 threshold, which is all the gate needs.
    assert est.total > 30 * B


# ---------------------------------------------------------------------------
# "Return None rather than guess"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config,missing",
    [
        (None, "hidden_size"),
        ({}, "hidden_size"),
        ({"num_hidden_layers": 32, "vocab_size": 32000, "num_attention_heads": 32,
          "intermediate_size": 11008}, "hidden_size"),
        ({"hidden_size": 4096, "vocab_size": 32000, "num_attention_heads": 32,
          "intermediate_size": 11008}, "num_hidden_layers"),
        ({"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
          "intermediate_size": 11008}, "vocab_size"),
        ({"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000,
          "intermediate_size": 11008}, "num_attention_heads"),
        ({"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000,
          "num_attention_heads": 32}, "intermediate_size"),
    ],
)
def test_missing_required_field_returns_none(config, missing):
    est = estimate_params(config)
    assert est.total is None
    assert est.active is None
    assert missing in est.missing


def test_non_positive_shape_field_returns_none():
    cfg = {"hidden_size": 0, "num_hidden_layers": 32, "vocab_size": 32000,
           "num_attention_heads": 32, "intermediate_size": 11008}
    assert estimate_total_params(cfg) is None


@pytest.mark.parametrize(
    "garbage",
    [
        {"hidden_size": "not-a-number", "num_hidden_layers": 32, "vocab_size": 32000,
         "num_attention_heads": 32, "intermediate_size": 11008},
        {"hidden_size": None, "num_hidden_layers": None, "vocab_size": None,
         "num_attention_heads": None, "intermediate_size": None},
        {"hidden_size": [4096], "num_hidden_layers": {"a": 1}, "vocab_size": 32000,
         "num_attention_heads": 32, "intermediate_size": 11008},
        {"architectures": "NotAList", "hidden_size": True},
    ],
)
def test_never_raises_on_malformed_config(garbage):
    est = estimate_params(garbage)
    assert isinstance(est, ParamEstimate)
    assert est.total is None


def test_float_valued_fields_are_accepted():
    """HF configs occasionally carry shape fields as JSON floats."""
    cfg = {"hidden_size": 4096.0, "num_hidden_layers": 32.0, "vocab_size": 32000.0,
           "num_attention_heads": 32.0, "intermediate_size": 11008.0,
           "hidden_act": "silu", "tie_word_embeddings": False}
    total = estimate_total_params(cfg)
    assert total is not None
    assert 6.5 * B < total < 7.5 * B  # Llama-2-7B shape


# ---------------------------------------------------------------------------
# MoE resolution
# ---------------------------------------------------------------------------


def test_expert_alias_chain_order_matches_blis():
    """Same chain and order as BLIS's ResolveNumExperts / vLLM's get_num_experts."""
    assert resolve_num_experts({"num_experts": 8, "n_routed_experts": 64}) == 8
    assert resolve_num_experts({"moe_num_experts": 16, "num_local_experts": 8}) == 16
    assert resolve_num_experts({"n_routed_experts": 256}) == 256
    assert resolve_num_experts({"num_local_experts": 8}) == 8
    assert resolve_num_experts({"num_routed_experts": 4}) == 4
    assert resolve_num_experts({}) == 0


def test_single_expert_config_is_dense_equivalent():
    """One expert is not a mixture; BLIS's MoEMinExperts guard says so too."""
    assert MOE_MIN_EXPERTS == 2
    cfg = {"hidden_size": 2048, "num_hidden_layers": 24, "vocab_size": 32000,
           "num_attention_heads": 16, "intermediate_size": 5632,
           "num_local_experts": 1, "num_experts_per_tok": 1, "hidden_act": "silu"}
    est = estimate_params(cfg)
    assert est.is_moe is False
    assert est.total == est.active


def test_moe_without_experts_per_tok_falls_back_to_all_experts():
    """BLIS treats this as fatal; sizing degrades to an upper bound and says so."""
    cfg = {"hidden_size": 2048, "num_hidden_layers": 24, "vocab_size": 32000,
           "num_attention_heads": 16, "intermediate_size": 5632,
           "n_routed_experts": 8, "moe_intermediate_size": 1408, "hidden_act": "silu"}
    est = estimate_params(cfg)
    assert est.is_moe is True
    assert est.total == est.active
    assert any("num_experts_per_tok absent" in n for n in est.notes)


def test_kimi_style_experts_per_token_spelling_is_honored():
    """Kimi-K3 spells it num_experts_per_token; missing it would size active wrong."""
    est = estimate_params(load("hybrid_mla_linear_attn"))
    assert est.num_experts_per_tok == 8
    assert est.active is not None and est.total is not None
    assert est.active < est.total / 10


def test_shared_expert_count_and_explicit_dim_agree():
    base = {"hidden_size": 2048, "num_hidden_layers": 4, "vocab_size": 1000,
            "num_attention_heads": 16, "intermediate_size": 5632,
            "n_routed_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 1408, "hidden_act": "silu"}
    by_count = estimate_total_params({**base, "n_shared_experts": 2})
    explicit = estimate_total_params({**base, "shared_expert_intermediate_size": 2816})
    assert by_count == explicit


def test_mixtral_convention_uses_intermediate_size_per_expert():
    est = estimate_params(load("moe_mixtral_8x7b"))
    assert any("Mixtral convention" in n for n in est.notes)


def test_interleave_moe_layer_step_reduces_expert_layers():
    base = {"hidden_size": 4096, "num_hidden_layers": 48, "vocab_size": 32000,
            "num_attention_heads": 32, "intermediate_size": 8192,
            "num_local_experts": 16, "num_experts_per_tok": 1, "hidden_act": "silu"}
    every = estimate_total_params(base)
    alternate = estimate_total_params({**base, "interleave_moe_layer_step": 2})
    assert every is not None and alternate is not None
    assert alternate < every


def test_mlp_only_layers_are_dense():
    base = {"hidden_size": 2048, "num_hidden_layers": 48, "vocab_size": 151936,
            "num_attention_heads": 32, "intermediate_size": 6144,
            "num_experts": 128, "num_experts_per_tok": 8,
            "moe_intermediate_size": 768, "hidden_act": "silu"}
    all_moe = estimate_total_params(base)
    some_dense = estimate_total_params({**base, "mlp_only_layers": [0, 1, 2, 3]})
    assert all_moe is not None and some_dense is not None
    assert some_dense < all_moe


def test_first_k_dense_replace_is_honored():
    base = load("mla_moe_deepseek_v3")
    with_prefix = estimate_total_params(base)
    without = estimate_total_params({**base, "first_k_dense_replace": 0})
    assert with_prefix is not None and without is not None
    # Three dense layers are far cheaper than three 256-expert MoE layers.
    assert with_prefix < without


# ---------------------------------------------------------------------------
# Attention branches
# ---------------------------------------------------------------------------


def test_mla_branch_is_used_and_differs_from_gqa():
    cfg = load("mla_moe_deepseek_v3")
    est = estimate_params(cfg)
    assert any("MLA attention" in n for n in est.notes)
    gqa = estimate_params({k: v for k, v in cfg.items() if k != "kv_lora_rank"})
    assert not any("MLA attention" in n for n in gqa.notes)
    assert gqa.total != est.total


def test_explicit_head_dim_beats_hidden_over_heads():
    """Qwen3-MoE declares head_dim=128 against hidden/heads=64; using the wrong one
    halves the attention term."""
    cfg = load("moe_qwen3_30b_a3b")
    assert cfg["head_dim"] * cfg["num_attention_heads"] != cfg["hidden_size"]
    with_explicit = estimate_total_params(cfg)
    without = estimate_total_params({k: v for k, v in cfg.items() if k != "head_dim"})
    assert with_explicit is not None and without is not None
    assert with_explicit > without


def test_gqa_kv_head_aliases_are_honored():
    base = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000,
            "num_attention_heads": 32, "intermediate_size": 11008, "hidden_act": "silu"}
    mha = estimate_total_params(base)
    for alias in ("num_key_value_heads", "num_kv_heads", "multi_query_group_num"):
        gqa = estimate_total_params({**base, alias: 8})
        assert gqa is not None and mha is not None
        assert gqa < mha, f"{alias} was ignored"


def test_hybrid_linear_attention_is_flagged():
    est = estimate_params(load("hybrid_mla_linear_attn"))
    note = next((n for n in est.notes if "hybrid attention" in n), None)
    assert note is not None
    assert "24/93" in note


def test_mtp_module_is_excluded_and_noted():
    est = estimate_params(load("mla_moe_deepseek_v3"))
    assert any("MTP module excluded" in n for n in est.notes)


# ---------------------------------------------------------------------------
# MLP / embedding details
# ---------------------------------------------------------------------------


def test_ungated_activation_uses_two_matrices():
    """``relu``, not ``gelu``: bare ``gelu`` is a *gated* spelling in modern decoders (see
    test_gelu_is_gated_three_matrix_on_a_real_config). Using gelu here was the bug."""
    base = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000,
            "num_attention_heads": 32, "intermediate_size": 11008}
    gated = estimate_total_params({**base, "hidden_act": "silu"})
    ungated = estimate_total_params({**base, "hidden_act": "relu"})
    assert gated is not None and ungated is not None
    assert ungated < gated
    assert ungated == gated - 32 * 4096 * 11008, "exactly one matrix per layer"
    assert any("ungated" in n for n in estimate_params({**base, "hidden_act": "relu"}).notes)


def test_tie_word_embeddings_drops_lm_head():
    base = {"hidden_size": 2048, "num_hidden_layers": 24, "vocab_size": 151936,
            "num_attention_heads": 16, "intermediate_size": 5632, "hidden_act": "silu"}
    untied = estimate_total_params({**base, "tie_word_embeddings": False})
    tied = estimate_total_params({**base, "tie_word_embeddings": True})
    assert untied is not None and tied is not None
    assert untied - tied == base["vocab_size"] * base["hidden_size"]


# ---------------------------------------------------------------------------
# text_config pivot
# ---------------------------------------------------------------------------


def test_pivot_text_config_matches_blis_semantics():
    outer = {"hidden_size": 1408, "model_type": "llama4", "text_config": {"hidden_size": 5120}}
    merged = pivot_text_config(outer)
    assert merged["hidden_size"] == 5120, "inner text_config must win"
    assert merged["model_type"] == "llama4", "outer-only keys survive"
    assert outer["hidden_size"] == 1408, "input must not be mutated"


def test_multimodal_config_is_sized_on_the_text_tower():
    cfg = load("multimodal_llama4_scout")
    est = estimate_params(cfg)
    assert any("text_config" in n for n in est.notes)
    assert est.total is not None and est.total > 50 * B
    # Sizing the outer map alone (no shape fields at the top level) yields nothing.
    outer_only = {k: v for k, v in cfg.items() if k not in ("text_config", "vision_config")}
    assert estimate_total_params(outer_only) is None


def test_pivot_tolerates_non_dict_text_config():
    assert pivot_text_config({"text_config": "nope", "hidden_size": 8}) == {
        "text_config": "nope",
        "hidden_size": 8,
    }
    assert pivot_text_config(None) == {}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [
        (None, "unknown"),
        (0, "0"),
        (999, "999"),
        (70_554_000_000, "70.6B"),
        (3_350_000_000, "3.35B"),
        (671_000_000_000, "671B"),
        (1_200_000_000_000, "1.2T"),
        (450_000_000, "450M"),
    ],
)
def test_format_params(n, expected):
    assert format_params(n) == expected


def test_convenience_wrappers_agree_with_estimate_params():
    cfg = load("moe_qwen3_30b_a3b")
    est = estimate_params(cfg)
    assert estimate_total_params(cfg) == est.total
    assert estimate_active_params(cfg) == est.active


# ---------------------------------------------------------------------------
# Real-world config shapes reported by the HF connector
# ---------------------------------------------------------------------------


def test_frontier_config_with_shape_fields_only_under_text_config():
    """The now-common frontier shape: the top level holds only architectures,
    model_type, dtype, text_config and vision_config. Reading top-level keys alone
    would return None for exactly the models the S1 gate exists to catch.
    """
    cfg = load("textconfig_only_frontier")
    assert set(cfg) == {"architectures", "model_type", "dtype", "text_config", "vision_config"}
    est = estimate_params(cfg)
    assert est.missing == []
    assert est.total is not None and est.total > 500 * B
    assert est.active is not None and est.active < est.total / 10
    assert est.is_moe and est.num_experts == 384
    assert any("text_config" in n for n in est.notes)


def test_vision_tower_is_not_what_gets_sized():
    """text_config must win over an identically-named vision_config field."""
    cfg = load("textconfig_only_frontier")
    assert pivot_text_config(cfg)["hidden_size"] == cfg["text_config"]["hidden_size"]
    assert cfg["vision_config"]["hidden_size"] != cfg["text_config"]["hidden_size"]


def test_dtype_spelling_is_irrelevant_to_a_parameter_count():
    """2026 configs spell it ``dtype``, not ``torch_dtype``. Counts are parameters, not
    bytes, so neither spelling may change the answer."""
    base = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000,
            "num_attention_heads": 32, "intermediate_size": 11008, "hidden_act": "silu",
            "tie_word_embeddings": False}
    plain = estimate_total_params(base)
    assert plain is not None
    assert estimate_total_params({**base, "dtype": "bfloat16"}) == plain
    assert estimate_total_params({**base, "torch_dtype": "bfloat16"}) == plain
    assert estimate_total_params({**base, "dtype": "fp8"}) == plain
    assert estimate_total_params(
        {**base, "quantization_config": {"quant_method": "fp8"}}
    ) == plain


def test_nonstandard_field_names_return_none_and_name_the_gap():
    """A config that invents its own vocabulary (d_model/n_layers/n_heads, no
    architectures[]) must not be guessed at. BLIS would parse none of those fields
    either, so the honest answer is "unknown" — with ``missing`` naming what could not
    be read, so the gap is visible in the run log.
    """
    est = estimate_params(load("nonstandard_field_names"))
    assert est.total is None
    assert est.active is None
    assert est.missing == ["num_hidden_layers", "num_attention_heads", "intermediate_size"]


# ---------------------------------------------------------------------------
# Gated vs ungated MLP: the gelu family
# ---------------------------------------------------------------------------
# hidden_act names the nonlinearity, not the MLP topology. Bare "gelu" was treated as
# ungated, which under-counted a real gelu-gated decoder by 23%. Since est_total_params
# feeds the S1 scale gate, an under-count can drop a model out of the report entirely.


def test_gelu_is_gated_three_matrix_on_a_real_config():
    """``XHToken/Spark-X2.5-4B`` (``Spark2_5ForCausalLM``), the config that found the bug.

    Its shipped ``modeling_spark.py`` defines
    ``Spark2_5MLP = down_proj(act_fn(gate_proj(x)) * up_proj(x))`` — a three-matrix GEGLU.
    ``gelu`` names the gate nonlinearity there exactly as Llama's ``silu`` names SwiGLU's.

    **``intermediate_size / hidden_size == 4.0`` here, and that is a convention, NOT
    evidence of an ungated MLP.** 4.0 is the classic BERT/GPT ratio and gated models more
    often use ~2.67, so the ratio reads as "ungated" and is simply wrong: only the
    modeling code settles it. The stage-2 classifier misread it this way first. Do not
    "fix" this test back on the strength of the ratio.
    """
    cfg = load("dense_gelu_gated_spark")
    assert cfg["hidden_act"] == "gelu"
    assert cfg["intermediate_size"] / cfg["hidden_size"] == 4.0  # convention, not evidence

    est = estimate_params(cfg)
    assert est.total == 4_110_604_800, "3-matrix GEGLU; published is ~4.112G"
    assert est.total == pytest.approx(4.112 * B, rel=0.001)

    # The pre-fix 2-matrix figure, for the record: 23% low, and the exact number the
    # classifier reproduced.
    two_matrix = 3_166_886_400
    assert est.total - two_matrix == (
        cfg["num_hidden_layers"] * cfg["hidden_size"] * cfg["intermediate_size"]
    ), "the difference is exactly one gate matrix per layer"
    assert two_matrix / est.total == pytest.approx(0.77, abs=0.01)


def test_gelu_and_silu_size_identically():
    """Both name a gate nonlinearity, so neither may change the matrix count."""
    base = {"hidden_size": 2560, "num_hidden_layers": 36, "vocab_size": 131072,
            "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
            "intermediate_size": 10240, "tie_word_embeddings": True}
    assert estimate_total_params({**base, "hidden_act": "gelu"}) == \
        estimate_total_params({**base, "hidden_act": "silu"})


@pytest.mark.parametrize(
    "act",
    ["silu", "swish", "swiglu", "geglu", "reglu", "glu", "situ",
     "gelu", "gelu_pytorch_tanh", "gelu_tanh", "gelu_pytorch_tanh_glu"],
)
def test_every_gated_spelling_has_a_gated_exemplar(act):
    """Each entry is listed because a real model declaring it is gated in its own
    modeling code — silu (Llama), gelu (Spark2_5), gelu_pytorch_tanh (Gemma 2/3), and the
    self-describing names. The set is a prior over the population, so it must not grow on
    suspicion."""
    assert act in GATED_ACTIVATIONS


@pytest.mark.parametrize("act", ["gelu_new", "quick_gelu", "gelu_fast", "relu"])
def test_spellings_with_no_gated_exemplar_stay_ungated(act):
    """``gelu_new`` is the GPT-2/GPT-Neo/GPT-J spelling and those MLPs are ungated;
    ``quick_gelu`` is CLIP's vision activation. Listing them on the strength of the
    over-count-is-safer argument alone would make the set a constant."""
    assert act not in GATED_ACTIVATIONS


def test_a_gelu_encoder_is_over_counted_and_that_is_the_safe_direction():
    """The cost of treating gelu as gated: a BERT-family encoder gains half its MLP.

    Recorded rather than hidden, because it is a real inaccuracy. It is the right
    direction to be wrong in: est_total_params feeds S1, and every gelu encoder stays far
    below the 3B threshold either way, whereas under-counting a gelu-gated 3.5B decoder
    drops it out of the report.
    """
    bert_base = {"architectures": ["BertForMaskedLM"], "hidden_size": 768,
                 "num_hidden_layers": 12, "vocab_size": 30522, "num_attention_heads": 12,
                 "intermediate_size": 3072, "hidden_act": "gelu",
                 "tie_word_embeddings": True}
    est = estimate_params(bert_base)
    assert est.total is not None
    assert est.total < 3 * B, "still nowhere near the S1 gate, so the gate is unaffected"
    assert any("over-count for a BERT-family encoder" in n for n in est.notes)


def test_the_gated_set_deliberately_diverges_from_blis():
    """BLIS's swiGLUActivations lists silu/swiglu/geglu/situ, so it treats bare gelu as
    ungated — the same bug, reported separately. Sizing targets the published parameter
    count, so it must not reproduce BLIS's error."""
    blis_swiglu_set = {"silu", "swiglu", "geglu", "situ"}
    assert blis_swiglu_set < GATED_ACTIVATIONS
    assert "gelu" in GATED_ACTIVATIONS - blis_swiglu_set


def test_a_missing_activation_still_does_not_widen_guesses():
    """This fix is about a known-gated activation being misclassified, not about guessing
    more. A config lacking required fields still returns None."""
    assert estimate_total_params({"hidden_act": "gelu"}) is None
    assert estimate_total_params({"hidden_act": "gelu", "hidden_size": 2560}) is None
