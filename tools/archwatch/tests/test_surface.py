"""Tests for component B — the support surface.

No network. Every config comes from ``tests/fixtures/surface/``; two of those fixtures
are verbatim copies of configs BLIS itself commits under ``model_configs/``, so the
"plain Llama parses cleanly" claim is tested against a config BLIS is actually validated
on rather than against something written to pass.

Three groups of assertions matter most:

* **The PLAN.md acceptance criteria** — ``test_acceptance_*``.
* **The severity split** — Bucket 0 must mean "BLIS would not run". A condition that
  BLIS survives while reporting a wrong number is T1 evidence, and mislabelling one as
  the other would put a false verdict in every emitted issue.
* **The ``text_config`` pivot** — frontier releases put every shape field under
  ``text_config``. Getting the pivot wrong fires T1 and Bucket 0 on exactly the models
  the tool exists to judge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from archwatch.config import IGNORED_CONFIG_KEYS
from archwatch.surface import (
    BUCKET0_SEVERITIES,
    FATAL,
    MIXED,
    MOE_MIN_EXPERTS,
    SILENT,
    SILENT_SEVERITIES,
    SWIGLU_ACTIVATIONS,
    Surface,
    _VALIDATOR_IMPLS,
    first_nonzero,
    get_int,
    get_string,
    linear_attn_full_layer_count,
    load_surface,
    pivot_text_config,
    resolve_num_experts,
)

FIXTURES = Path(__file__).parent / "fixtures" / "surface"
#: BLIS's Go source, read-only, in the parent worktree. Absent in a bare checkout of
#: just this tool, so the drift checks that need it skip rather than fail.
BLIS_ROOT = Path(__file__).resolve().parents[3]


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def surface() -> Surface:
    return load_surface()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_surface_works_from_a_clean_checkout(surface: Surface) -> None:
    """The default dir resolves relative to the package, not to the cwd."""
    assert surface.parsed_fields
    assert surface.hard_validators
    assert surface.gaps
    assert surface.known_architectures


def test_load_surface_accepts_an_explicit_dir() -> None:
    from archwatch.surface import DEFAULT_SURFACE_DIR

    assert load_surface(DEFAULT_SURFACE_DIR).parsed_fields
    assert load_surface(str(DEFAULT_SURFACE_DIR)).parsed_fields  # str is accepted too


def test_load_surface_raises_on_a_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_surface(tmp_path)


def test_parsed_field_names_flattens_canonical_names_and_aliases(surface: Surface) -> None:
    assert "num_key_value_heads" in surface.parsed_field_names
    assert "num_kv_heads" in surface.parsed_field_names  # Falcon
    assert "multi_query_group_num" in surface.parsed_field_names  # GLM
    assert "ffn_hidden_size" in surface.parsed_field_names  # Falcon/GLM
    assert "num_experts_per_token" in surface.parsed_field_names  # Kimi-K3
    assert "dtype" in surface.parsed_field_names  # GLM-5 and every 2026 release
    # And a field BLIS demonstrably does not read must not be in there.
    assert "n_group" not in surface.parsed_field_names
    assert "sliding_window" not in surface.parsed_field_names


def test_plan_minimum_field_coverage(surface: Surface) -> None:
    """PLAN.md's "cover at minimum" list, field by field."""
    required = {
        "num_hidden_layers", "hidden_size", "vocab_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "num_experts_per_tok", "moe_intermediate_size", "interleave_moe_layer_step",
        "first_k_dense_replace", "kv_lora_rank", "qk_rope_head_dim",
        "linear_attn_config", "torch_dtype", "quantization_config", "hidden_act",
        "tie_word_embeddings", "max_position_embeddings", "rope_scaling",
        # ResolveNumExperts's whole alias set
        "num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts",
        "num_routed_experts",
        # shared experts, both spellings plus the explicit dim
        "n_shared_experts", "num_shared_experts", "shared_expert_intermediate_size",
    }
    assert required <= surface.parsed_field_names


def test_parsed_and_ignored_sets_are_disjoint(surface: Surface) -> None:
    """Enforced by the loader too; asserted here so the message is obvious."""
    assert not (surface.parsed_field_names & surface.ignored_field_names)


def test_every_parsed_field_cites_a_source_ref(surface: Surface) -> None:
    for pf in surface.parsed_fields:
        assert pf.source_ref, pf.name
        assert re.match(r"^[\w./-]+\.(go|md):\d+(-\d+)?$", pf.source_ref), (
            pf.name, pf.source_ref
        )


def test_roles_come_from_the_documented_vocabulary(surface: Surface) -> None:
    allowed = {"shape", "attention", "moe", "precision", "rope", "other"}
    assert {pf.role for pf in surface.parsed_fields} <= allowed


# ---------------------------------------------------------------------------
# Go parse semantics
# ---------------------------------------------------------------------------


def test_get_int_mirrors_hfconfig_getint() -> None:
    """Only a JSON number counts. Everything else is BLIS's "absent" value, 0."""
    assert get_int({"a": 8}, "a") == 8
    assert get_int({"a": 8.0}, "a") == 8  # Go decodes every JSON number as float64
    assert get_int({"a": 8.7}, "a") == 8  # int(float64) truncates toward zero
    assert get_int({"a": -8.7}, "a") == -8
    assert get_int({}, "a") == 0
    assert get_int({"a": "8"}, "a") == 0  # a numeric STRING reads as absent
    assert get_int({"a": None}, "a") == 0
    assert get_int({"a": [8]}, "a") == 0
    # Python's bool is an int subclass; Go's float64 assertion rejects a JSON bool.
    assert get_int({"a": True}, "a") == 0
    assert get_int({"a": False}, "a") == 0


def test_first_nonzero_mirrors_mustgetintfallback() -> None:
    assert first_nonzero({"b": 4}, "a", "b") == 4
    assert first_nonzero({"a": 0, "b": 4}, "a", "b") == 4  # 0 is skipped, not returned
    assert first_nonzero({"a": 2, "b": 4}, "a", "b") == 2  # first non-zero wins
    assert first_nonzero({}, "a", "b") == 0


def test_get_string_requires_an_actual_string() -> None:
    assert get_string({"hidden_act": "silu"}, "hidden_act") == "silu"
    assert get_string({"hidden_act": 3}, "hidden_act") == ""
    assert get_string({}, "hidden_act") == ""


def test_resolve_num_experts_uses_the_moe_min_experts_threshold() -> None:
    """Not "first field present" — first field at or above the threshold."""
    assert resolve_num_experts({"num_experts": 128}) == 128
    assert resolve_num_experts({"n_routed_experts": 256}) == 256
    assert resolve_num_experts({"num_local_experts": 8}) == 8
    assert resolve_num_experts({}) == 0
    # A single-expert config is dense-equivalent in BLIS, so the scan continues past it.
    assert MOE_MIN_EXPERTS == 2
    assert resolve_num_experts({"num_experts": 1}) == 0
    assert resolve_num_experts({"num_experts": 1, "n_routed_experts": 64}) == 64
    # Resolution ORDER matters: num_experts is tried before n_routed_experts.
    assert resolve_num_experts({"num_experts": 4, "n_routed_experts": 64}) == 4


def test_linear_attn_full_layer_count() -> None:
    assert linear_attn_full_layer_count({}) == 0
    assert linear_attn_full_layer_count({"linear_attn_config": {}}) == 0
    # Present but unusable falls back to 0 (BLIS warns and sizes over all layers).
    assert linear_attn_full_layer_count({"linear_attn_config": {"full_attn_layers": []}}) == 0
    assert linear_attn_full_layer_count(
        {"linear_attn_config": {"full_attn_layers": "nope"}}
    ) == 0
    assert linear_attn_full_layer_count(
        {"linear_attn_config": {"full_attn_layers": [9, 19, 29]}}
    ) == 3


def test_pivot_text_config_copies_inner_keys_and_overwrites() -> None:
    config = {"model_type": "kimi_k3", "text_config": {"model_type": "kimi_k3_text", "hidden_size": 7168}}
    pivoted = pivot_text_config(config)
    assert pivoted["hidden_size"] == 7168
    assert pivoted["model_type"] == "kimi_k3_text"  # the pivot overwrites, as Go does
    assert config["model_type"] == "kimi_k3"  # caller's dict is never mutated


def test_pivot_ignores_a_non_dict_text_config() -> None:
    assert pivot_text_config({"text_config": None, "hidden_size": 4096})["hidden_size"] == 4096
    assert pivot_text_config({"text_config": "x"}) == {"text_config": "x"}


# ---------------------------------------------------------------------------
# unparsed_fields
# ---------------------------------------------------------------------------


def test_acceptance_plain_llama_has_no_unparsed_fields(surface: Surface) -> None:
    """PLAN.md acceptance. The fixture is BLIS's own committed llama-2-7b config.

    If this ever regresses, T1 fires on every dense model ever published.
    """
    assert surface.unparsed_fields(fixture("llama-2-7b")) == []


def test_acceptance_deepseek_v3_kv_lora_rank_is_parsed_not_unparsed(
    surface: Surface,
) -> None:
    """PLAN.md acceptance: the MLA shape fields are parsed, so they are not T1 evidence."""
    unparsed = surface.unparsed_fields(fixture("deepseek-v3"))
    for parsed in ("kv_lora_rank", "qk_rope_head_dim", "first_k_dense_replace",
                   "moe_intermediate_size", "n_routed_experts", "n_shared_experts",
                   "num_experts_per_tok", "quantization_config", "rope_scaling"):
        assert parsed not in unparsed, parsed
    # It still has plenty of genuinely-unread fields — that is the point of T1.
    assert {"n_group", "topk_group", "num_nextn_predict_layers", "q_lora_rank"} <= set(unparsed)


def test_unparsed_fields_excludes_ignored_config_keys(surface: Surface) -> None:
    config = dict(fixture("llama-2-7b"))
    config.update({key: "x" for key in ("_name_or_path", "auto_map", "tokenizer_class")})
    assert surface.unparsed_fields(config) == []
    assert {"_name_or_path", "auto_map", "tokenizer_class"} <= IGNORED_CONFIG_KEYS


def test_unparsed_fields_excludes_inert_values(surface: Surface) -> None:
    """A key set to null / false / 0 / empty declares a feature OFF, so it is not evidence."""
    base = fixture("llama-2-7b")
    for inert in (None, False, 0, 0.0, "", [], {}):
        config = dict(base, some_brand_new_mechanism=inert)
        assert surface.unparsed_fields(config) == [], inert
    # ...but an engaged one is reported.
    assert surface.unparsed_fields(dict(base, some_brand_new_mechanism=True)) == [
        "some_brand_new_mechanism"
    ]
    assert surface.unparsed_fields(dict(base, some_brand_new_mechanism=7)) == [
        "some_brand_new_mechanism"
    ]


def test_unparsed_fields_never_descends_into_nested_dicts(surface: Surface) -> None:
    """quantization_config's inner keys are BLIS's business, not the diff's."""
    unparsed = surface.unparsed_fields(fixture("deepseek-v3"))
    for nested_only in ("weight_block_size", "activation_scheme", "quant_method",
                        "factor", "original_max_position_embeddings", "fmt"):
        assert nested_only not in unparsed, nested_only


def test_unparsed_fields_is_sorted_and_deterministic(surface: Surface) -> None:
    config = fixture("deepseek-v3")
    first = surface.unparsed_fields(config)
    assert first == sorted(first)
    assert first == surface.unparsed_fields(config)


def test_unparsed_fields_does_not_mutate_the_config(surface: Surface) -> None:
    config = fixture("nested-text-config-frontier")
    before = json.dumps(config, sort_keys=True)
    surface.unparsed_fields(config)
    surface.check_hard_validators(config)
    surface.match_gaps(config)
    assert json.dumps(config, sort_keys=True) == before


# --- the text_config pivot: load-bearing, not an edge case -------------------


def test_nested_text_config_shape_fields_are_not_reported_as_unparsed(
    surface: Surface,
) -> None:
    """A frontier release with every shape field under text_config.

    Without the pivot, unparsed_fields() would report num_hidden_layers, hidden_size and
    the whole shape set — firing T1 on essentially every multimodal-capable frontier
    release. Shaped after foranyone2026/Kimi-K3.
    """
    config = fixture("nested-text-config-frontier")
    # The premise: the top level really is nearly empty.
    assert set(config) == {
        "architectures", "model_type", "dtype", "transformers_version",
        "text_config", "vision_config", "quantization_config",
    }
    unparsed = surface.unparsed_fields(config)
    for pivoted in ("num_hidden_layers", "hidden_size", "num_attention_heads",
                    "num_key_value_heads", "intermediate_size", "vocab_size",
                    "hidden_act", "head_dim", "kv_lora_rank", "qk_rope_head_dim",
                    "num_experts", "num_experts_per_token", "num_shared_experts",
                    "linear_attn_config", "max_position_embeddings",
                    "tie_word_embeddings", "first_k_dense_replace",
                    "moe_intermediate_size"):
        assert pivoted not in unparsed, pivoted
    # What it SHOULD report: the genuinely-unread mechanisms, and the vision tower.
    assert set(unparsed) == {
        "activation_situ_beta", "n_group", "num_nextn_predict_layers", "q_lora_rank",
        "qk_nope_head_dim", "routed_expert_hidden_size", "topk_group", "v_head_dim",
        "vision_config",
    }


def test_nested_text_config_is_not_bucket_zero(surface: Surface) -> None:
    """The same fixture must read as a model BLIS would happily run."""
    assert surface.check_hard_validators(fixture("nested-text-config-frontier")) == []


def test_nested_text_config_resolves_shape_through_the_pivot(surface: Surface) -> None:
    """Belt and braces: the validators see the pivoted values, not zeros."""
    pivoted = pivot_text_config(fixture("nested-text-config-frontier"))
    assert get_int(pivoted, "num_hidden_layers") == 93
    assert get_int(pivoted, "hidden_size") == 7168
    assert resolve_num_experts(pivoted) == 384
    assert first_nonzero(pivoted, "num_experts_per_tok", "num_experts_per_token") == 8
    assert linear_attn_full_layer_count(pivoted) == 9
    # dtype survives from the top level, since text_config declares none.
    assert get_string(pivoted, "dtype") == "bfloat16"


# ---------------------------------------------------------------------------
# check_hard_validators — Bucket 0
# ---------------------------------------------------------------------------


def test_acceptance_gelu_activation_is_bucket_zero(surface: Surface) -> None:
    """PLAN.md acceptance."""
    failures = surface.check_hard_validators(fixture("bucket0-gelu-activation"))
    assert len(failures) == 1
    assert "swiglu_family_hidden_act" in failures[0]
    assert "gelu" in failures[0]


def test_acceptance_unrecognized_torch_dtype_is_bucket_zero(surface: Surface) -> None:
    """PLAN.md acceptance. Traced fatal: BytesPerParam=0 aborts on both paths."""
    failures = surface.check_hard_validators(fixture("bucket0-novel-dtype"))
    assert len(failures) == 1
    assert "recognized_torch_dtype" in failures[0]
    assert "float4_e2m1" in failures[0]


def test_swiglu_set_matches_blis(surface: Surface) -> None:
    assert SWIGLU_ACTIVATIONS == {"silu", "swiglu", "geglu", "situ", ""}
    base = fixture("llama-2-7b")
    for ok in ("silu", "swiglu", "geglu", "situ"):
        assert surface.check_hard_validators(dict(base, hidden_act=ok)) == []
    # Absent or empty is accepted (BLIS assumes SwiGLU and logs an info line).
    assert surface.check_hard_validators(dict(base, hidden_act="")) == []
    assert surface.check_hard_validators(
        {k: v for k, v in base.items() if k != "hidden_act"}
    ) == []
    # The lookup is a plain Go map, so it is case-SENSITIVE.
    assert surface.check_hard_validators(dict(base, hidden_act="SiLU"))
    for bad in ("gelu", "relu", "gelu_pytorch_tanh", "gelu_new", "quick_gelu"):
        assert surface.check_hard_validators(dict(base, hidden_act=bad)), bad


def test_modern_dtype_only_config_is_not_bucket_zero(surface: Surface) -> None:
    """A 2026-era config carrying only `dtype` is handled by BLIS, so it is NOT a gap.

    ``sim/latency/config.go:334-338`` reads torch_dtype then falls back to dtype. A
    validator keyed on torch_dtype alone would be dead code against modern configs.
    """
    config = fixture("modern-dtype-only")
    assert "torch_dtype" not in config
    assert config["dtype"] == "bfloat16"
    assert surface.check_hard_validators(config) == []
    assert surface.check_silent_validators(config) == []


@pytest.mark.parametrize("dtype_value", sorted(
    {"float32", "float16", "bfloat16", "int8", "uint8", "fp8", "int4", "nf4"}
))
def test_every_recognized_dtype_passes_under_either_spelling(
    surface: Surface, dtype_value: str
) -> None:
    base = {k: v for k, v in fixture("llama-2-7b").items() if k != "torch_dtype"}
    assert surface.check_hard_validators(dict(base, torch_dtype=dtype_value)) == []
    assert surface.check_hard_validators(dict(base, dtype=dtype_value)) == []


def test_dtype_absent_from_both_keys_is_bucket_zero(surface: Surface) -> None:
    base = {k: v for k, v in fixture("llama-2-7b").items() if k != "torch_dtype"}
    failures = surface.check_hard_validators(base)
    assert len(failures) == 1
    assert "recognized_torch_dtype" in failures[0]
    assert "neither torch_dtype nor dtype" in failures[0]


def test_unrecognized_torch_dtype_short_circuits_a_sane_dtype_alias(
    surface: Surface,
) -> None:
    """The Go code is else-IF, not first-valid-value, so `dtype` is never consulted.

    A vendor shipping a new format under the legacy key while keeping a readable
    ``dtype`` alongside it still aborts. This is a real Bucket-0 path, not a nicety.
    """
    config = fixture("novel-dtype-with-sane-dtype-alias")
    assert config["dtype"] == "bfloat16"  # readable, and ignored
    failures = surface.check_hard_validators(config)
    assert len(failures) == 1
    assert "recognized_torch_dtype" in failures[0]
    assert "mxfp4" in failures[0]


def test_nonstandard_field_names_are_bucket_zero_and_loud(surface: Surface) -> None:
    """ParallaxOpen/Vela-Lumen-31M: d_model / n_layers / n_heads, no architectures[].

    BLIS reads none of those names, so every shape field resolves to 0. The answer to
    "fatal or silent" is FATAL — the run aborts before simulating, which is the good
    outcome: loud, not a wrong number. All five failures are Bucket 0.
    """
    config = fixture("nonstandard-field-names")
    assert "architectures" not in config
    assert config["model_type"] == "small_lm"

    failures = surface.check_hard_validators(config)
    ids = {f.split("]")[0].lstrip("[") for f in failures}
    assert ids == {
        "positive_shape_fields", "positive_intermediate_size", "positive_vocab_size"
    }
    assert len(failures) == 5  # 3 core shape fields + intermediate_size + vocab_size
    assert all("absent" in f for f in failures)
    # Nothing silent: it cannot mis-size a model it refuses to load.
    assert surface.check_silent_validators(config) == []
    # And the renamed fields are T1 evidence in their own right.
    assert {"d_model", "n_layers", "n_heads", "n_kv_heads"} <= set(
        surface.unparsed_fields(config)
    )


def test_positive_shape_fields_reports_non_numeric_reads(surface: Surface) -> None:
    """A shape field present as a string reads as 0 and fails identically."""
    config = dict(fixture("llama-2-7b"), hidden_size="4096")
    failures = surface.check_hard_validators(config)
    assert len(failures) == 1
    assert "positive_shape_fields" in failures[0]
    assert "not a JSON number" in failures[0]


def test_intermediate_size_resolves_through_its_falcon_alias(surface: Surface) -> None:
    base = {k: v for k, v in fixture("llama-2-7b").items() if k != "intermediate_size"}
    assert surface.check_hard_validators(base)  # missing entirely -> Bucket 0
    assert surface.check_hard_validators(dict(base, ffn_hidden_size=11008)) == []


def test_nonnegative_shape_fields(surface: Surface) -> None:
    base = fixture("llama-2-7b")
    for negatable in ("head_dim", "kv_lora_rank", "qk_rope_head_dim",
                      "first_k_dense_replace"):
        failures = surface.check_hard_validators(dict(base, **{negatable: -1}))
        assert any("nonnegative_shape_fields" in f and negatable in f for f in failures), (
            negatable, failures
        )
    # Zero is the documented "absent" value and must stay legal.
    assert surface.check_hard_validators(dict(base, head_dim=0, kv_lora_rank=0)) == []


def test_hidden_size_divisible_by_heads(surface: Surface) -> None:
    base = fixture("llama-2-7b")
    bad = dict(base, hidden_size=4100, num_attention_heads=32)
    failures = surface.check_hard_validators(bad)
    assert any("hidden_size_divisible_by_heads" in f for f in failures)
    # An explicit head_dim is used directly and needs no divisibility.
    assert surface.check_hard_validators(dict(bad, head_dim=128)) == []
    # The MLA latent path never forms the quotient.
    assert surface.check_hard_validators(dict(bad, kv_lora_rank=512)) == []


def test_moe_active_expert_count_required_is_fatal_on_both_backends(
    surface: Surface,
) -> None:
    config = dict(fixture("llama-2-7b"), num_experts=128, moe_intermediate_size=768)
    failures = surface.check_hard_validators(config)
    assert any("moe_active_expert_count_required" in f for f in failures)


def test_healthy_moe_configs_are_not_bucket_zero(surface: Surface) -> None:
    for name in ("qwen3-30b-a3b", "deepseek-v3", "nested-text-config-frontier"):
        assert surface.check_hard_validators(fixture(name)) == [], name


def test_committed_blis_configs_all_pass_bucket_zero(surface: Surface) -> None:
    """Every config BLIS ships must, by construction, be runnable."""
    for name in ("llama-2-7b", "qwen3-30b-a3b"):
        assert surface.check_hard_validators(fixture(name)) == [], name


# ---------------------------------------------------------------------------
# The severity split: "would not run" vs "runs, silently wrong"
# ---------------------------------------------------------------------------


def test_severity_vocabulary_and_partition(surface: Surface) -> None:
    assert BUCKET0_SEVERITIES == {FATAL}
    assert SILENT_SEVERITIES == {SILENT, MIXED}
    assert not (BUCKET0_SEVERITIES & SILENT_SEVERITIES)
    assert {v.severity for v in surface.hard_validators} <= {FATAL, SILENT, MIXED}


def test_every_validator_records_what_the_user_observes(surface: Surface) -> None:
    for v in surface.hard_validators:
        assert v.observed, v.id
        assert v.source_ref, v.id
        assert v.rule, v.id
        if v.severity == FATAL:
            assert v.fatal_at, f"{v.id}: a fatal severity must name the abort site"
        else:
            assert v.silent_when, f"{v.id}: a non-fatal severity must say when it is silent"


def test_every_yaml_validator_has_an_implementation(surface: Surface) -> None:
    """Drift between the documented surface and the executed one, made visible."""
    assert surface.unimplemented_validators == []
    assert {v.id for v in surface.hard_validators} == set(_VALIDATOR_IMPLS)


def test_bucket_zero_is_a_strict_subset_of_the_validators(surface: Surface) -> None:
    """If every validator were fatal, the severity work would be pointless."""
    bucket0 = [v for v in surface.hard_validators if v.is_bucket0]
    non_fatal = [v for v in surface.hard_validators if not v.is_bucket0]
    assert bucket0 and non_fatal
    assert {v.id for v in non_fatal} == {
        "kv_head_count_unreadable",
        "moe_expert_count_resolvable",
        "moe_active_not_exceeding_total",
        "moe_total_required_when_active_present",
    }


def test_moe_without_resolvable_expert_count_is_silent_not_bucket_zero(
    surface: Surface,
) -> None:
    """The highest-value T1 case in the surface, and PLAN.md classified it wrong.

    On the CLI default backend (trained-physics) ExtractKVCapacityParams' error is only
    warned (cmd/root.go:951-956); GetModelConfigFromHF does not error at all; and the
    trained-physics constructor's only MoE guard is
    ``NumLocalExperts > 1 && NumExpertsPerTok <= 0``, which 0 does not trip. So a sparse
    model whose expert total uses an unknown spelling is simulated as a DENSE model.
    """
    config = fixture("moe-no-resolvable-expert-count")
    assert resolve_num_experts(config) == 0  # the premise: no known spelling resolves
    assert config["num_experts_per_tok"] == 8  # yet it is plainly MoE

    assert surface.check_hard_validators(config) == []  # NOT Bucket 0
    silent = surface.check_silent_validators(config)
    assert any("moe_expert_count_resolvable" in f for f in silent)
    assert any("moe_total_required_when_active_present" in f for f in silent)
    # T1 also sees the unknown spelling itself.
    assert "expert_count_total" in surface.unparsed_fields(config)


def test_moe_expert_count_validator_severity_is_mixed(surface: Surface) -> None:
    v = next(x for x in surface.hard_validators if x.id == "moe_expert_count_resolvable")
    assert v.severity == MIXED
    assert not v.is_bucket0
    assert "trained-physics" in v.silent_when
    assert "roofline" in v.fatal_at


def test_recognized_torch_dtype_severity_is_fatal(surface: Surface) -> None:
    """The other condition PLAN.md called Bucket 0 — and this one really is."""
    v = next(x for x in surface.hard_validators if x.id == "recognized_torch_dtype")
    assert v.severity == FATAL
    assert v.is_bucket0
    assert "logrus.Fatalf" in v.fatal_at or "panic" in v.fatal_at


def test_kv_head_count_as_a_string_is_silent(surface: Surface) -> None:
    """A GQA model sized and timed as full MHA, with no error and no warning."""
    config = fixture("kv-heads-as-string")
    assert config["num_key_value_heads"] == "8"
    assert surface.check_hard_validators(config) == []
    silent = surface.check_silent_validators(config)
    assert len(silent) == 1
    assert "kv_head_count_unreadable" in silent[0]
    assert "64-way MHA" in silent[0]  # num_attention_heads is 64


def test_negative_kv_head_count_is_fatal(surface: Surface) -> None:
    """Distinct severity from the unreadable case, hence a distinct validator."""
    config = dict(fixture("llama-2-7b"), num_key_value_heads=-8)
    failures = surface.check_hard_validators(config)
    assert any("nonnegative_kv_head_count" in f for f in failures)


def test_fractional_kv_head_count_is_silent(surface: Surface) -> None:
    config = dict(fixture("llama-2-7b"), num_key_value_heads=8.5)
    assert surface.check_hard_validators(config) == []
    silent = surface.check_silent_validators(config)
    assert any("truncates it to 8" in f for f in silent)


def test_tp_divisibility_validator_is_not_evaluated_at_tp_one(surface: Surface) -> None:
    """It cannot fire, is documented as such, and stays live for a future real TP."""
    from archwatch.surface import ASSUMED_TP

    assert ASSUMED_TP == 1
    v = next(x for x in surface.hard_validators if x.id == "head_counts_tp_divisible")
    assert v.evaluated_by_archwatch is False
    # A head count that a TP=8 deployment would reject still passes here.
    config = dict(fixture("llama-2-7b"), num_attention_heads=33, hidden_size=4224,
                  num_key_value_heads=3)
    assert not any("head_counts_tp_divisible" in f
                   for f in surface.check_hard_validators(config))


def test_check_all_validators_tags_each_message_with_its_severity(
    surface: Surface,
) -> None:
    config = fixture("moe-no-resolvable-expert-count")
    pairs = surface.check_all_validators(config)
    assert pairs
    assert {sev for sev, _ in pairs} == {MIXED}
    fatal_config = fixture("bucket0-gelu-activation")
    assert {sev for sev, _ in surface.check_all_validators(fatal_config)} == {FATAL}


def test_bucket_zero_and_silent_are_disjoint_for_every_fixture(surface: Surface) -> None:
    for path in sorted(FIXTURES.glob("*.json")):
        config = json.loads(path.read_text())
        hard = set(surface.check_hard_validators(config))
        silent = set(surface.check_silent_validators(config))
        assert not (hard & silent), path.name
        assert hard | silent == {m for _, m in surface.check_all_validators(config)}


# ---------------------------------------------------------------------------
# match_gaps
# ---------------------------------------------------------------------------


def test_gap_ids_are_unique_and_well_formed(surface: Surface) -> None:
    ids = [g.id for g in surface.gaps]
    assert len(ids) == len(set(ids))
    for g in surface.gaps:
        assert re.match(r"^[a-z0-9_]+$", g.id), g.id
        assert g.mechanism and g.impact and g.keywords and g.seam_refs, g.id
        assert g.direction in {"pessimistic", "optimistic", "unknown"}, g.id
        assert g.documented_in, g.id


def test_plan_minimum_gap_coverage(surface: Surface) -> None:
    """PLAN.md's "cover at minimum" gap list."""
    required = {
        "mla_step_time_kv_read",              # MLA step-time pessimism
        "hybrid_linear_attn_weights",         # hybrid layers charged full attention
        "mtp_module_weights_unmodeled",       # MTP / speculative decode
        "blockwise_fp8_flattened",            # block-wise FP8 flattened
        "first_k_dense_replace_capacity_only",
        "explicit_head_dim_step_time_blind",
        "novel_moe_routing_unparsed",         # n_group / topk_group unparsed
    }
    assert required <= {g.id for g in surface.gaps}


def test_match_gaps_on_an_mla_moe_config(surface: Surface) -> None:
    matched = {g.id for g in surface.match_gaps(fixture("deepseek-v3"))}
    assert "mla_step_time_kv_read" in matched
    assert "novel_moe_routing_unparsed" in matched          # n_group / topk_group
    assert "first_k_dense_replace_capacity_only" in matched
    assert "mtp_module_weights_unmodeled" in matched        # num_nextn_predict_layers
    assert "blockwise_fp8_flattened" in matched             # nested weight_block_size
    # Not this one: DeepSeek-V3 is not hybrid.
    assert "hybrid_linear_attn_weights" not in matched


def test_match_gaps_finds_keywords_nested_inside_sub_dicts(surface: Surface) -> None:
    """weight_block_size lives inside quantization_config, and must still match."""
    config = fixture("deepseek-v3")
    assert "weight_block_size" in config["quantization_config"]
    assert "blockwise_fp8_flattened" in {g.id for g in surface.match_gaps(config)}


def test_match_gaps_matches_string_values_as_substrings(surface: Surface) -> None:
    """quant_method: "fp8" and topk_method: "noaux_tc" are value-side evidence."""
    assert "blockwise_fp8_flattened" in {
        g.id for g in surface.match_gaps(
            {"quantization_config": {"quant_method": "fp8"}}
        )
    }
    assert "novel_moe_routing_unparsed" in {
        g.id for g in surface.match_gaps({"topk_method": "noaux_tc"})
    }


def test_match_gaps_matches_config_keys_exactly_not_as_substrings(
    surface: Surface,
) -> None:
    """Substring key matching would fire the head_dim gap on every MLA config.

    DeepSeek-V3 declares qk_rope_head_dim / qk_nope_head_dim / v_head_dim but no
    explicit head_dim, so the step-time head_dim gap must NOT match it.
    """
    config = fixture("deepseek-v3")
    assert "head_dim" not in config
    assert any(k.endswith("head_dim") for k in config)
    assert "explicit_head_dim_step_time_blind" not in {
        g.id for g in surface.match_gaps(config)
    }
    # A config that really does declare head_dim matches.
    assert "explicit_head_dim_step_time_blind" in {
        g.id for g in surface.match_gaps(fixture("modern-dtype-only"))
    }


def test_match_gaps_ignores_inert_keys(surface: Surface) -> None:
    """"sliding_window": null describes a model without sliding-window attention."""
    base = fixture("llama-2-7b")
    assert "sliding_window_unmodeled" not in {
        g.id for g in surface.match_gaps(dict(base, sliding_window=None))
    }
    assert "sliding_window_unmodeled" not in {
        g.id for g in surface.match_gaps(dict(base, sliding_window=0))
    }
    assert "sliding_window_unmodeled" in {
        g.id for g in surface.match_gaps(dict(base, sliding_window=4096))
    }


def test_match_gaps_on_a_hybrid_config(surface: Surface) -> None:
    matched = {g.id for g in surface.match_gaps(fixture("nested-text-config-frontier"))}
    assert "hybrid_linear_attn_weights" in matched   # linear_attn_config
    assert "mla_step_time_kv_read" in matched        # kv_lora_rank
    assert "vision_tower_unmodeled" in matched       # vision_config


def test_match_gaps_on_an_ssm_and_sliding_window_config(surface: Surface) -> None:
    matched = {g.id for g in surface.match_gaps(fixture("mamba-hybrid-sliding-window"))}
    assert "mamba_ssm_state_unmodeled" in matched
    assert "sliding_window_unmodeled" in matched


# --- direction correctness: the one defect a reader cannot detect unaided ----------


def test_hybrid_linear_attn_weight_direction_is_optimistic(surface: Surface) -> None:
    """The KDA weight charge is an UNDER-count, against BLIS's own docs.

    BLIS charges every layer 2h^2 + 2h*kv_dim (kv_capacity.go:624) over all layers
    (:700). A gated-delta-rule layer carries ~5 full-width projections at an EXPANDED
    inner width once use_full_rank_gate is set, so it has MORE weights than standard
    attention, not fewer. Kimi-K3: ~19.03G charged vs ~36.19G real = 0.53x.

    docs/reference/models.md:45 calls this a "pessimism". The arithmetic says otherwise,
    and this entry deliberately disagrees with the doc.
    """
    gap = surface.gap_by_id("hybrid_linear_attn_weights")
    assert gap.direction == "optimistic"
    assert "0.53x" in gap.impact
    assert "UNDER-count" in gap.impact
    assert "DIRECTION CORRECTED" in gap.notes


def test_hybrid_gap_names_the_step_time_seam_too(surface: Surface) -> None:
    """A maintainer following only the capacity seams would fix half the bug.

    trained_physics_model.go:693 charges all L layers the same uniform attention shape on
    the STEP-TIME path.
    """
    gap = surface.gap_by_id("hybrid_linear_attn_weights")
    assert "sim/latency/trained_physics_model.go:693" in gap.seam_refs
    assert "sim/latency/kv_capacity.go:700" in gap.seam_refs
    assert "sim/latency/kv_capacity.go:624" in gap.seam_refs


def test_every_gap_direction_is_justified_in_prose(surface: Surface) -> None:
    """A stated direction must SAY which way BLIS errs, not just assert a field value.

    Enforces the derivation discipline in known-gaps.yaml's header. The requirement is a
    directional statement a reader can check, because the `direction` field alone is
    unfalsifiable — which is how hybrid_linear_attn_weights stayed inverted.

    Deliberately NOT requiring a numeric ratio: for several gaps the magnitude honestly
    depends on runtime context (sliding_window_unmodeled scales with context/window) or on
    a model-specific component (vision_tower_unmodeled). Demanding a number there would
    invite fabricating one, which PLAN.md rules out for good reason. Magnitude is asserted
    separately, only where one was actually measured.
    """
    directional = (
        "under-count", "over-count", "under-estim", "over-estim", "understat", "overstat",
        "absent from", "not counted", "not charged", "contributes no", "no weight",
        "low weight", "below", "above", "halve", "optimistic", "pessimistic",
    )
    for gap in surface.gaps:
        if gap.direction == "unknown":
            continue  # honest abstention needs no justification
        evidence = f"{gap.impact} {gap.scope} {gap.notes}".lower()
        assert any(t in evidence for t in directional), (
            f"{gap.id}: direction={gap.direction} is asserted but never justified in prose"
        )


def test_measured_gap_ratios_are_recorded_and_do_not_drift(surface: Surface) -> None:
    """The gaps whose magnitude WAS derived must keep the number, so the sign is auditable.

    These four are the ones with a real measurement behind them. Pinning the ratios means a
    future edit cannot quietly restate a direction without also moving the arithmetic that
    supports it.
    """
    for gap_id, ratio in (
        ("hybrid_linear_attn_weights", "0.53x"),
        ("attention_projection_width_ignores_head_dim", "0.70x"),
        ("latent_moe_expert_hidden_size", "1.997x"),
        ("mla_step_time_kv_read", "21x"),
    ):
        gap = surface.gap_by_id(gap_id)
        assert gap is not None, gap_id
        assert ratio in gap.impact, f"{gap_id}: lost its measured ratio {ratio}"


def test_a_ratio_below_one_is_never_labelled_pessimistic(surface: Surface) -> None:
    """Sign consistency: BLIS charging LESS than reality is optimistic, by definition.

    A cheap mechanical guard on the exact class of defect that shipped — a ratio of the
    form 0.NNx in the impact text is BLIS below reality, so the entry cannot claim
    pessimism.
    """
    import re

    for gap in surface.gaps:
        for match in re.finditer(r"(\d+\.\d+)x", gap.impact):
            if float(match.group(1)) < 1.0:
                assert gap.direction == "optimistic", (
                    f"{gap.id}: impact states {match.group(0)} (BLIS below reality) "
                    f"but direction={gap.direction}"
                )


# --- the two gaps the first real stage-2 run found missing --------------------


def test_latent_moe_expert_hidden_size_gap(surface: Surface) -> None:
    """Kimi-K3 runs routed experts at half width; BLIS charges full hidden_size.

    The largest single finding of the stage-2 run: routed experts dominate a big MoE's
    bytes, so a 2x error here outweighs every other weight gap in the surface.
    """
    gap = surface.gap_by_id("latent_moe_expert_hidden_size")
    assert gap is not None
    assert gap.direction == "pessimistic"          # BLIS charges 2x reality
    assert "sim/latency/kv_capacity.go:657" in gap.seam_refs
    # Wrong on BOTH paths, unlike the MLA gaps.
    assert "sim/latency/trained_physics_model.go:699" in gap.seam_refs
    assert "BOTH" in gap.scope

    config = fixture("nested-text-config-frontier")
    tc = config["text_config"]
    assert tc["routed_expert_hidden_size"] == 3584
    assert tc["hidden_size"] == 7168               # experts run at half width
    assert "latent_moe_expert_hidden_size" in {g.id for g in surface.match_gaps(config)}
    # BLIS reads none of it, so it is T1 evidence as well.
    assert "routed_expert_hidden_size" in surface.unparsed_fields(config)


def test_kv_cache_dtype_ignored_by_step_time_gap(surface: Surface) -> None:
    """const bytesPerKVElement = 2.0 never consults EffectiveKVBytesPerParam."""
    gap = surface.gap_by_id("kv_cache_dtype_ignored_by_step_time")
    assert gap is not None
    assert gap.direction == "pessimistic"
    assert "sim/latency/trained_physics_model.go:491" in gap.seam_refs
    assert "sim/latency/trained_physics_model.go:671" in gap.seam_refs   # decode KV read
    # The part worth emphasising: it needs no flag on a 1-byte-compute model.
    assert "no flag" in gap.scope


def test_attention_projection_width_gap(surface: Surface) -> None:
    """Q and O stay hidden^2; only K/V go through kv_dim."""
    gap = surface.gap_by_id("attention_projection_width_ignores_head_dim")
    assert gap is not None
    assert gap.direction == "optimistic"
    assert "sim/latency/kv_capacity.go:624" in gap.seam_refs
    assert "0.70x" in gap.impact
    assert "explicit_head_dim_step_time_blind" in gap.scope  # cross-referenced


def test_head_dim_gap_scope_no_longer_over_claims(surface: Surface) -> None:
    """It used to say weight sizing "DOES use" head_dim, unqualified. It reaches kv_dim only."""
    gap = surface.gap_by_id("explicit_head_dim_step_time_blind")
    assert "kv_dim" in gap.scope
    assert "attention_projection_width_ignores_head_dim" in gap.scope
    # And for an MLA model it never touches KV capacity: the MLA branch returns first.
    assert "sim/latency/kv_capacity.go:151" in gap.scope


def test_blockwise_fp8_matches_the_compressed_tensors_spelling(surface: Surface) -> None:
    """`ignore` is what compressed-tensors emits; without it match_gaps missed real configs."""
    gap = surface.gap_by_id("blockwise_fp8_flattened")
    assert "ignore" in gap.keywords
    assert "modules_to_not_convert" in gap.keywords
    config = fixture("nested-text-config-frontier")
    assert config["quantization_config"]["quant_method"] == "compressed-tensors"
    assert "ignore" in config["quantization_config"]
    assert "blockwise_fp8_flattened" in {g.id for g in surface.match_gaps(config)}
    # The mechanism is method-independent, so the impact text must not read as fp8-only.
    assert "quantization method" in gap.impact or "ANY" in gap.impact


def test_match_gaps_is_empty_for_a_plain_dense_model(surface: Surface) -> None:
    assert surface.match_gaps(fixture("llama-2-7b")) == []


def test_match_gaps_is_deterministic_and_in_yaml_order(surface: Surface) -> None:
    config = fixture("deepseek-v3")
    matched = surface.match_gaps(config)
    assert matched == surface.match_gaps(config)
    order = [g.id for g in surface.gaps]
    assert [order.index(g.id) for g in matched] == sorted(
        order.index(g.id) for g in matched
    )


def test_gap_by_id(surface: Surface) -> None:
    assert surface.gap_by_id("mla_step_time_kv_read").direction == "pessimistic"
    assert surface.gap_by_id("no_such_gap") is None


# ---------------------------------------------------------------------------
# is_known_architecture
# ---------------------------------------------------------------------------


def test_known_architectures_seed_set_is_a_few_hundred_names(surface: Surface) -> None:
    """PLAN.md: "expect a few hundred names"."""
    assert 300 <= len(surface.known_architectures) <= 2000


def test_is_known_architecture_is_case_insensitive(surface: Surface) -> None:
    for name in ("LlamaForCausalLM", "llamaforcausallm", "LLAMAFORCAUSALLM",
                 "  LlamaForCausalLM  "):
        assert surface.is_known_architecture(name), name


def test_frontier_architectures_blis_ships_configs_for_are_known(
    surface: Surface,
) -> None:
    for name in ("LlamaForCausalLM", "MixtralForCausalLM", "Qwen2ForCausalLM",
                 "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "DeepseekV2ForCausalLM",
                 "DeepseekV3ForCausalLM", "GlmMoeDsaForCausalLM", "MistralForCausalLM",
                 "Llama4ForConditionalGeneration"):
        assert surface.is_known_architecture(name), name


def test_unseen_architecture_is_not_known(surface: Surface) -> None:
    for name in ("NovelMoeForCausalLM", "TotallyMadeUpV9ForCausalLM",
                 "KimiK4ForCausalLM", "GLM6ForCausalLM", "", "   "):
        assert not surface.is_known_architecture(name), name


def test_seed_set_suppresses_architectures_vllm_already_ships(surface: Surface) -> None:
    """The suppressor's whole job, and a trap for anyone naming a fixture.

    vLLM's registry already carries KimiK3ForConditionalGeneration and
    DeepseekV4ForCausalLM, so those are NOT novel however new they feel. The
    nested-text-config fixture is named after Kimi-K3 for its config SHAPE; a
    Signal carrying that architecture string would be dropped by the suppressor before
    any of the T1 machinery ran. Component F should join on the architecture, not on how
    recent the model sounds.
    """
    known_but_feels_new = fixture("nested-text-config-frontier")["architectures"][0]
    assert known_but_feels_new == "KimiK3ForConditionalGeneration"
    assert surface.is_known_architecture(known_but_feels_new)
    assert surface.is_known_architecture("DeepseekV4ForCausalLM")
    # The purely-synthetic fixtures are genuinely unseen, so they exercise T1.
    for name in ("NovelMoeForCausalLM", "StringyGqaForCausalLM",
                 "NovelHybridSsmForCausalLM", "NovelPrecisionForCausalLM"):
        assert not surface.is_known_architecture(name), name


# --- architecture matching is EXACT: the decision, pinned ---------------------
#
# The task suffix (ForCausalLM / ForConditionalGeneration / ...) is NOT normalized away.
# See Surface.is_known_architecture's docstring for the full argument. These tests exist
# so the decision cannot be reversed by accident, in either direction.


def test_architecture_matching_is_exact_not_suffix_normalized(surface: Surface) -> None:
    """A name absent from the registry stays unknown even if its family sibling is known.

    This is the load-bearing half of the decision. Normalizing would make an unlisted
    variant look supported, and suppression is unrecoverable in a stateless pipeline.
    """
    assert surface.is_known_architecture("MistralForCausalLM")
    # Same family, a head vLLM does not list -> must NOT be suppressed.
    assert not surface.is_known_architecture("MistralForConditionalGeneration")
    assert surface.is_known_architecture("Qwen3ForCausalLM")
    assert not surface.is_known_architecture("Qwen3ForConditionalGeneration")


def test_both_variants_are_known_when_vllm_lists_both(surface: Surface) -> None:
    """Why normalization is unnecessary: the registry is already exhaustive on variants.

    Every family that has both heads appears twice in vLLM's registry, so the outcome
    normalization was meant to produce is already achieved by the DATA. The feared
    duplicate issue per multimodal variant cannot arise from the seed set.
    """
    for stem in ("Gemma3", "Gemma3n", "Gemma4", "Llama4", "DeepseekV4", "Glm5Next",
                 "Qwen3_5", "Qwen3_5Moe", "Qwen4Exp", "MiniMaxM3Sparse"):
        assert surface.is_known_architecture(f"{stem}ForCausalLM"), stem
        assert surface.is_known_architecture(f"{stem}ForConditionalGeneration"), stem


def test_exact_matching_never_collapses_distinct_architectures(surface: Surface) -> None:
    """The over-stripping trap, closed: MoE and dense siblings stay separate keys.

    Qwen3MoeForCausalLM and Qwen3ForCausalLM are genuinely different architectures — one
    is a 128-expert MoE — and must never share a key.
    """
    assert surface.is_known_architecture("Qwen3MoeForCausalLM")
    assert surface.is_known_architecture("Qwen3ForCausalLM")
    # Neither is the other's family sibling, so even the informational lookup keeps them
    # apart.
    assert "qwen3forcausallm" not in surface.related_known_architectures(
        "Qwen3MoeForCausalLM"
    )
    assert "qwen3moeforcausallm" not in surface.related_known_architectures(
        "Qwen3ForCausalLM"
    )


def test_related_known_architectures_reports_family_without_suppressing(
    surface: Surface,
) -> None:
    """The non-destructive alternative: context for a report, never a suppression."""
    siblings = surface.related_known_architectures("Llama4ForConditionalGeneration")
    assert "llama4forcausallm" in siblings
    # An exact self-match is excluded — is_known_architecture already covers that.
    assert "llama4forconditionalgeneration" not in siblings
    # An unrelated invented name has no family at all.
    assert surface.related_known_architectures("TotallyMadeUpV9ForCausalLM") == []
    # And knowing the family does NOT make an unlisted variant known. The Mistral family
    # has two listed heads, so this also shows the lookup returns all of them.
    mistral_family = surface.related_known_architectures("MistralForConditionalGeneration")
    assert mistral_family == ["mistralforcausallm", "mistralmodel"]
    assert not surface.is_known_architecture("MistralForConditionalGeneration")


def test_family_stem_does_not_strip_a_bare_suffix_to_nothing() -> None:
    """An empty stem would match every other bare-suffix name."""
    from archwatch.surface import _family_stem

    assert _family_stem("Llama4ForConditionalGeneration") == "llama4"
    assert _family_stem("LlamaForCausalLM") == "llama"
    assert _family_stem("GPT2LMHeadModel") == "gpt2"
    assert _family_stem("Qwen3MoeForCausalLM") == "qwen3moe"
    assert _family_stem("Qwen3ForCausalLM") == "qwen3"
    assert _family_stem("Model") == "model"  # nothing but a suffix: left intact
    assert _family_stem("GritLM") == "gritlm"  # no recognized suffix
    assert _family_stem("") == ""


def test_multimodal_variant_carries_findings_its_causal_twin_does_not(
    surface: Surface,
) -> None:
    """Concrete reason one key per variant is right, not just safe.

    BLIS's committed Llama-4-Scout config nests its whole text tower under text_config
    and adds a vision tower. The text-tower SHAPE resolves identically through the pivot
    — so BLIS's numbers agree, as the normalization argument claims — but the multimodal
    variant has an unmodeled vision tower that the causal twin does not. Collapsing the
    two keys would attach that finding to the wrong name, or drop it.
    """
    mm = fixture("llama-4-scout-multimodal")
    text_only = {k: v for k, v in mm.items() if k not in ("vision_config", "text_config")}
    text_only.update(mm["text_config"])

    # The text-tower shape BLIS reads really is identical.
    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads",
                "intermediate_size", "vocab_size"):
        assert get_int(pivot_text_config(mm), key) == get_int(text_only, key), key
    assert surface.check_hard_validators(mm) == surface.check_hard_validators(text_only)

    # But only the multimodal variant reports the vision tower.
    assert "vision_config" in surface.unparsed_fields(mm)
    assert "vision_config" not in surface.unparsed_fields(text_only)


def test_blis_validated_set_is_a_subset_of_the_seed_set(surface: Surface) -> None:
    for name in surface.blis_validated_architectures:
        assert surface.is_known_architecture(name), name


def test_seeded_from_records_provenance_with_a_date(surface: Surface) -> None:
    assert surface.seeded_from
    sources = {block["source"] for block in surface.seeded_from}
    assert {"vllm-registry", "blis-validated"} <= sources
    for block in surface.seeded_from:
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", str(block["fetched_at"])), block
        assert block["url"] and block["method"] and block["count"]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_reports_counts_for_the_cli(surface: Surface) -> None:
    s = surface.summary()
    assert s["parsed_fields"] == len(surface.parsed_fields)
    assert s["hard_validators"] == len(surface.hard_validators)
    assert s["gaps"] == len(surface.gaps)
    assert s["known_architectures"] == len(surface.known_architectures)
    assert sum(s["hard_validators_by_severity"].values()) == s["hard_validators"]
    assert s["bucket0_validators"] == s["hard_validators_by_severity"].get(FATAL, 0)
    assert s["unimplemented_validators"] == []
    assert sum(s["parsed_fields_by_role"].values()) == s["parsed_fields"]


def test_field_by_name_resolves_aliases(surface: Surface) -> None:
    assert surface.field_by_name("multi_query_group_num").name == "num_key_value_heads"
    assert surface.field_by_name("n_routed_experts").name == "num_experts"
    assert surface.field_by_name("dtype").name == "torch_dtype"
    assert surface.field_by_name("not_a_field") is None


# ---------------------------------------------------------------------------
# Drift: do the source_refs still point at real lines?
# ---------------------------------------------------------------------------

_REF_RE = re.compile(r"^(?P<path>[\w./-]+\.(?:go|md)):(?P<start>\d+)(?:-(?P<end>\d+))?$")


def _all_refs(surface: Surface) -> list[str]:
    refs: list[str] = []
    for pf in surface.parsed_fields:
        refs.append(pf.source_ref)
    for v in surface.hard_validators:
        refs.append(v.source_ref)
        refs.extend(v.also_at)
    for g in surface.gaps:
        refs.extend(g.seam_refs)
        if g.documented_in != "code-only":
            refs.append(g.documented_in)
    return [r for r in refs if r]


@pytest.mark.skipif(
    not (BLIS_ROOT / "sim" / "latency" / "config.go").exists(),
    reason="BLIS source not present (tool checked out on its own)",
)
def test_every_source_ref_points_at_a_real_in_bounds_line(surface: Surface) -> None:
    """Catches exactly the staleness found in PLAN.md's own line numbers.

    Read-only, and skipped when the parent worktree is not there. It cannot verify that
    a line still says what we claim — only that it exists — which is still enough to
    catch a file being truncated or a ref pointing past the end.
    """
    problems: list[str] = []
    line_counts: dict[str, int] = {}
    for ref in _all_refs(surface):
        m = _REF_RE.match(ref)
        if m is None:
            problems.append(f"malformed ref: {ref}")
            continue
        rel = m.group("path")
        path = BLIS_ROOT / rel
        if not path.exists():
            problems.append(f"missing file: {ref}")
            continue
        if rel not in line_counts:
            line_counts[rel] = len(path.read_text().splitlines())
        total = line_counts[rel]
        start = int(m.group("start"))
        end = int(m.group("end") or start)
        if not (1 <= start <= end <= total):
            problems.append(f"out of bounds ({total} lines): {ref}")
    assert problems == [], problems


@pytest.mark.skipif(
    not (BLIS_ROOT / "sim" / "latency" / "config.go").exists(),
    reason="BLIS source not present (tool checked out on its own)",
)
def test_mirrored_go_constants_still_match_blis(surface: Surface) -> None:
    """The values surface.py hardcodes are copies. Assert they are still true copies."""
    from archwatch.surface import (
        MOE_ACTIVE_EXPERT_FIELDS,
        MOE_EXPERT_COUNT_FIELDS,
        MOE_SHARED_EXPERT_FIELDS,
        PRECISION_TO_BYTES_PER_PARAM,
    )

    config_go = (BLIS_ROOT / "sim" / "latency" / "config.go").read_text()
    kv_go = (BLIS_ROOT / "sim" / "latency" / "kv_capacity.go").read_text()
    mhc_go = (BLIS_ROOT / "sim" / "model_hardware_config.go").read_text()

    for name in MOE_EXPERT_COUNT_FIELDS + MOE_ACTIVE_EXPERT_FIELDS + MOE_SHARED_EXPERT_FIELDS:
        assert f'"{name}"' in config_go, name
    for dtype in PRECISION_TO_BYTES_PER_PARAM:
        assert f'"{dtype}":' in config_go, dtype
    for act in SWIGLU_ACTIVATIONS - {""}:
        assert f'"{act}":' in kv_go, act
    assert f"MoEMinExperts = {MOE_MIN_EXPERTS}" in mhc_go
    # Every field archwatch claims BLIS parses must appear literally in the source that
    # reads it. Guards against inventing a field, or keeping one BLIS dropped.
    blis_source = config_go + kv_go + (BLIS_ROOT / "cmd" / "root.go").read_text()
    for pf in surface.parsed_fields:
        for name in pf.all_names:
            assert f'"{name}"' in blis_source, f"{pf.name}: {name} not read anywhere"


@pytest.mark.skipif(
    not (BLIS_ROOT / "sim").exists(),
    reason="BLIS source not present (tool checked out on its own)",
)
def test_fields_claimed_unparsed_are_really_unread_by_blis(surface: Surface) -> None:
    """The other direction: a gap must not claim BLIS ignores something it reads."""
    go_source = "\n".join(
        p.read_text()
        for p in list((BLIS_ROOT / "sim").rglob("*.go")) + list((BLIS_ROOT / "cmd").rglob("*.go"))
        if not p.name.endswith("_test.go")
    )
    # Sampled from the gaps' "BLIS never reads this" claims.
    for unread in ("sliding_window", "n_group", "topk_group", "num_nextn_predict_layers",
                   "weight_block_size", "index_topk", "index_n_heads", "mlp_only_layers",
                   "decoder_sparse_step", "vision_config", "layer_types", "mamba_d_state"):
        assert f'"{unread}"' not in go_source, (
            f"{unread} is claimed unparsed but appears in BLIS source"
        )
