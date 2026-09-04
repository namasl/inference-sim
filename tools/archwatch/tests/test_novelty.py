"""Tests for archwatch.novelty — the suppressors, triggers, and significance gate.

Pure unit tests: fixture configs plus synthetic Signals, no network and (except for
the two dedup tests, which use ``tmp_path``) no filesystem.

``archwatch.surface`` is owned by another component, so these tests drive a
:class:`FakeSurface` implementing exactly the interface PLAN.md section B fixes.
That keeps the filter's logic under test rather than the surface's YAML.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from archwatch.config import DERIVATIVE_PATTERNS, DetectorConfig, Thresholds
from archwatch.connectors.base import Candidate, Signal
from archwatch.novelty import (
    ALIAS_JOIN_MARKER,
    CORE_TRIGGER_IDS,
    KNOWN_ARCH_TRIGGER,
    SIGNIFICANCE_IDS,
    TRIGGER_IDS,
    LmShapeEvidence,
    MIN_FAMILY_KEY_LEN,
    _strength,
    coherent_arch_ids,
    evaluate,
    evaluate_detailed,
    join_signals,
    lm_shape_evidence,
    normalize_family_key,
    normalize_repo_key,
    signal_edges,
    matched_gaps,
    normalize_arch_key,
    structural_identity,
)
from archwatch.sizing import pivot_text_config

FIXTURES = Path(__file__).parent / "fixtures" / "novelty"
NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


# ---------------------------------------------------------------------------
# Surface stub (PLAN.md section B interface)
# ---------------------------------------------------------------------------

# What BLIS actually reads out of config.json, harvested from sim/latency/config.go
# and sim/latency/kv_capacity.go. The real surface loads this from
# support-surface/parsed-fields.yaml.
PARSED_FIELDS = frozenset(
    {
        "num_hidden_layers", "hidden_size", "vocab_size", "num_attention_heads",
        "num_key_value_heads", "num_kv_heads", "multi_query_group_num", "head_dim",
        "intermediate_size", "ffn_hidden_size", "intermediate_size_mlp",
        "num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts",
        "num_routed_experts", "num_experts_per_tok", "num_experts_per_token",
        "n_shared_experts", "num_shared_experts", "shared_expert_intermediate_size",
        "moe_intermediate_size", "interleave_moe_layer_step", "first_k_dense_replace",
        "kv_lora_rank", "qk_rope_head_dim", "linear_attn_config",
        "hidden_act", "quantization_config", "tie_word_embeddings",
        "max_position_embeddings", "rope_scaling",
    }
)

# Fields that are real but architecturally inert for a step-time/capacity model:
# training hyperparameters, numeric epsilons, routing loss weights, generation knobs.
# NOTE for component B / the frozen config.py: archwatch.config.IGNORED_CONFIG_KEYS
# does NOT contain these, so unless surface.py carries its own inert list, T1 fires on
# every plain Llama config and PLAN.md's own acceptance criterion ("unparsed_fields()
# returns [] for a plain Llama config") cannot hold.
INERT_FIELDS = frozenset(
    {
        "attention_bias", "attention_dropout", "initializer_range", "mlp_bias",
        "pretraining_tp", "rms_norm_eps", "rope_theta", "sliding_window",
        "use_sliding_window", "max_window_layers", "output_router_logits",
        "router_aux_loss_coef", "router_jitter_noise", "norm_topk_prob",
        "decoder_sparse_step", "mlp_only_layers", "aux_loss_alpha", "seq_aux",
        "scoring_func", "topk_method", "topk_group", "n_group",
        "routed_scaling_factor", "moe_layer_freq", "ep_size",
        "q_lora_rank", "qk_nope_head_dim", "v_head_dim", "qk_head_dim",
        "num_nextn_predict_layers", "attention_chunk_size", "no_rope_layers",
        "use_qk_norm", "for_llm_compressor", "boi_token_index", "eoi_token_index",
        "image_token_index", "vision_config", "text_config", "index_head_dim",
        "index_n_heads", "index_topk",
    }
)


class FakeSurface:
    """Stub for ``archwatch.surface.Surface``.

    Implements ``unparsed_fields``, ``check_hard_validators``,
    ``is_known_architecture`` and ``match_gaps`` and nothing else, so a test can only
    depend on the contract PLAN.md fixes.
    """

    def __init__(
        self,
        known: tuple[str, ...] = (),
        *,
        unparsed_override: list[str] | None = None,
        validator_failures: list[str] | None = None,
        silent_failures: list[str] | None = None,
        gaps: tuple[str, ...] = (),
    ) -> None:
        self.known = {k.lower() for k in known}
        self.unparsed_override = unparsed_override
        self.validator_failures = validator_failures or []
        self.silent_failures = silent_failures
        self.gaps = list(gaps)

    def unparsed_fields(self, config: dict) -> list[str]:
        if self.unparsed_override is not None:
            return list(self.unparsed_override)
        merged = pivot_text_config(config)
        return sorted(
            k for k in merged if k not in PARSED_FIELDS and k not in INERT_FIELDS
        )

    def check_hard_validators(self, config: dict) -> list[str]:
        return list(self.validator_failures)

    def check_silent_validators(self, config: dict) -> list[str]:
        """Shapes BLIS misreads without failing: the run completes, the number is wrong.

        Reproduces the real surface's highest-value rule rather than returning a canned
        string, so the tests exercise a config shape that actually occurs: an expert
        count spelled outside BLIS's alias set makes a sparse MoE simulate as dense,
        behind a single ``logrus.Warnf``.
        """
        if self.silent_failures is not None:
            return list(self.silent_failures)
        cfg = pivot_text_config(config)

        def get_int(*keys):
            for key in keys:
                value = cfg.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
            return 0

        active = get_int("num_experts_per_tok", "num_experts_per_token")
        total = get_int(*RECOGNIZED_EXPERT_COUNT_FIELDS)
        out = []
        if active and total < 2:
            out.append(
                "[moe_expert_count_resolvable] num_experts_per_tok=%d signals MoE but no "
                "total expert count resolved >= 2 from any known spelling -- warned and "
                "degraded on trained-physics (sim/latency/kv_capacity.go:779-786)" % active
            )
            out.append(
                "[moe_total_required_when_active_present] active experts per token (%d) is "
                "set but the resolved total expert count is 0 (sim/latency/config.go:575-579)"
                % active
            )
        return out

    def is_known_architecture(self, arch_id: str) -> bool:
        return bool(arch_id) and arch_id.lower() in self.known

    def match_gaps(self, config: dict) -> list[str]:
        return list(self.gaps)


# BLIS's recognized total-expert-count spellings (latency.moeExpertCountFields). A
# config signalling MoE through any OTHER spelling is the silent-misread case.
RECOGNIZED_EXPERT_COUNT_FIELDS = (
    "num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts",
    "num_routed_experts",
)

SEED_SET = (
    "LlamaForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "MixtralForCausalLM",
    "DeepseekV3ForCausalLM",
)


def surface(**kwargs) -> FakeSurface:
    kwargs.setdefault("known", SEED_SET)
    return FakeSurface(**kwargs)


def cfg(**kwargs) -> DetectorConfig:
    """DetectorConfig with the shipped defaults unless a test overrides one.

    Use this when the test does not depend on a particular threshold. When it *does*,
    use :func:`strict_cfg` or pass the value explicitly — the defaults are backtest
    outputs and move, and a test that silently inherits one can have its assertion
    inverted by a calibration change rather than by a code change.
    """
    return DetectorConfig(**kwargs)


def strict_cfg(**kwargs) -> DetectorConfig:
    """Config pinned to the pre-calibration behaviour, for tests that assert on it.

    ``recheck_known_architectures=False`` so "a known architecture is suppressed" means
    what it says, and ``min_total_params=30B`` so scale-gate tests are not carried by a
    lowered threshold. Both are stated here rather than inherited.
    """
    kwargs.setdefault("recheck_known_architectures", False)
    kwargs.setdefault("thresholds", Thresholds(min_total_params=30_000_000_000))
    return DetectorConfig(**kwargs)


def sig(
    source: str = "hf",
    *,
    arch: str | list[str] | None = None,
    model_ids: tuple[str, ...] = (),
    org: str | None = None,
    display_name: str = "",
    config: dict | None = None,
    extra: dict | None = None,
    observed_at: datetime | None = None,
    raw_ref: str = "",
) -> Signal:
    arch_ids = [arch] if isinstance(arch, str) else list(arch or [])
    return Signal(
        source=source,
        observed_at=observed_at or NOW,
        arch_ids=arch_ids,
        model_ids=list(model_ids),
        org=org,
        display_name=display_name,
        config=config,
        extra=dict(extra or {}),
        raw_ref=raw_ref,
    )


def never_reported(_arch_id: str) -> bool:
    """Dedup predicate for tests that must not touch the filesystem."""
    return False


def run(signals, *, surf=None, conf=None, **kwargs):
    """join + evaluate_detailed in one step, with the filesystem dedup stubbed out."""
    kwargs.setdefault("already_reported", never_reported)
    return evaluate_detailed(
        join_signals(signals), surf or surface(), conf or cfg(), **kwargs
    )


def reasons(report) -> list[str]:
    return [s.reason for s in report.dropped]


# ---------------------------------------------------------------------------
# join_signals
# ---------------------------------------------------------------------------


def test_join_groups_by_primary_arch():
    report = join_signals(
        [
            sig(arch="KimiK3ForCausalLM", model_ids=("moonshotai/Kimi-K3-Base",)),
            sig(arch="KimiK3ForCausalLM", model_ids=("moonshotai/Kimi-K3-Instruct",)),
            sig(arch="GlmMoeDsaForCausalLM", model_ids=("zai-org/GLM-5.2",)),
        ]
    )
    assert [c.arch_id for c in report] == ["KimiK3ForCausalLM", "GlmMoeDsaForCausalLM"]
    assert len(report[0].signals) == 2


def test_join_is_case_and_punctuation_insensitive():
    """Three sources' spelling conventions must not become three issues."""
    cands = join_signals(
        [
            sig(arch="KimiK3ForCausalLM"),
            sig("vllm", arch="kimik3forcausallm"),
            sig("sglang", arch="Kimi_K3_ForCausalLM"),
        ]
    )
    assert len(cands) == 1
    assert cands[0].arch_id == "KimiK3ForCausalLM", "canonical spelling wins"
    assert cands[0].sources == ["hf", "sglang", "vllm"]


def test_every_arch_id_is_a_join_edge_not_just_the_primary():
    """A framework PR mining both ``KimiK3ForCausalLM`` and ``KimiK3MTPModel`` from one
    patch must join either spelling seen elsewhere — the MTP head ships with the
    release, it is not a second architecture."""
    cands = join_signals(
        [
            sig(arch=["KimiK3ForCausalLM", "KimiK3MTPModel"]),
            sig("vllm", arch=["KimiK3MTPModel"], raw_ref="1"),
        ]
    )
    assert len(cands) == 1
    assert cands[0].arch_id == "KimiK3ForCausalLM", "hf primary architectures[0] wins"
    assert "arch:kimik3mtpmodel" in cands[0].join_edges


def test_join_falls_back_to_the_display_name_when_no_source_has_an_arch():
    """A framework PR and an InferenceX row both name a model, not a class. They join on
    the family edge and corroborate each other."""
    cands = join_signals(
        [
            sig("vllm", display_name="MiniMax-M3", raw_ref="12345"),
            sig("inferencex", display_name="MiniMax-M3", raw_ref="abc123"),
        ]
    )
    assert len(cands) == 1
    cand = cands[0]
    assert cand.arch_id == "MiniMax-M3"
    assert ALIAS_JOIN_MARKER in cand.triggers
    assert cand.corroborated


def test_an_org_qualified_display_name_does_not_match_a_bare_one_by_family():
    """A documented limitation of keeping the org prefix. ``MiniMaxAI/MiniMax-M3`` and a
    bare ``MiniMax-M3`` share no family key, because dropping the org from a display
    name is exactly what lets two orgs' identically named models collapse. Both real
    connectors also emit ``model_ids``, so in practice they meet on the repo edge (see
    test_repo_edge_joins_an_org_qualified_name_to_a_bare_one) — this records why the
    family edge alone does not."""
    cands = join_signals(
        [
            sig("vllm", display_name="MiniMaxAI/MiniMax-M3", raw_ref="1"),
            sig("inferencex", display_name="MiniMax-M3", raw_ref="2"),
        ]
    )
    assert len(cands) == 2


def test_repo_edge_joins_an_org_qualified_name_to_a_bare_one():
    cands = join_signals(
        [
            sig("vllm", display_name="MiniMaxAI/MiniMax-M3",
                model_ids=("MiniMaxAI/MiniMax-M3",), raw_ref="1"),
            sig("inferencex", display_name="MiniMax-M3",
                model_ids=("MiniMaxAI/MiniMax-M3",), raw_ref="2"),
        ]
    )
    assert len(cands) == 1
    assert "repo:minimaxai/minimax-m3" in cands[0].join_edges


def test_alias_marker_is_not_counted_as_a_trigger():
    assert ALIAS_JOIN_MARKER not in TRIGGER_IDS


def test_alias_label_falls_back_to_the_model_id():
    by_model = join_signals([sig("hf", model_ids=("someorg/Mystery-8B",))])
    assert by_model[0].arch_id == "Mystery-8B"
    assert ALIAS_JOIN_MARKER in by_model[0].triggers


def test_a_signal_with_no_arch_no_repo_and_no_name_is_dropped():
    """"vLLM PR 9876" is not an identity. Previously this became a candidate named
    ``vllm:9876``, i.e. junk in the issue list; now it offers no join edge and is
    dropped with a debug log. Both real framework connectors always set a display name
    from the PR title, so this is a degenerate case, not a live path."""
    assert join_signals([sig("vllm", raw_ref="9876")]) == []


def test_real_arch_string_promotes_over_an_alias_key():
    """An alias-keyed candidate upgrades when a real architectures[] entry joins it."""
    cands = join_signals(
        [
            sig("inferencex", display_name="MiniMax-M3"),
            sig("hf", arch="MiniMax_M3", model_ids=("MiniMaxAI/MiniMax-M3",)),
        ]
    )
    assert len(cands) == 1
    assert cands[0].arch_id == "MiniMax_M3"
    assert ALIAS_JOIN_MARKER not in cands[0].triggers


def test_join_drops_a_signal_with_no_usable_key():
    assert join_signals([sig(source="", raw_ref="")]) == []


def test_join_prefers_a_populated_display_name():
    cands = join_signals(
        [sig(arch="XForCausalLM"), sig("vllm", arch="XForCausalLM", display_name="Model X")]
    )
    assert cands[0].display_name == "Model X"


# ---------------------------------------------------------------------------
# Suppressors
# ---------------------------------------------------------------------------


def test_known_architecture_is_suppressed():
    """A LlamaForCausalLM fine-tune is the firehose; it must never reach a trigger.

    Pinned to ``recheck_known_architectures=False``: this asserts the suppressor itself,
    not the shipped default, which the backtest has since flipped to True.
    """
    report = run([sig(arch="LlamaForCausalLM", model_ids=("someorg/my-llama-tune",),
                      config=load("dense_llama31_70b"), org="meta-llama")],
                 conf=strict_cfg())
    assert report.passed == []
    assert reasons(report) == ["known_architecture"]


def test_gguf_repo_is_suppressed_at_the_primary_key():
    """Even with a novel-looking arch string, an all-derivative repo set is dropped."""
    report = run(
        [
            sig(arch="NovelThingForCausalLM", model_ids=("bartowski/NovelThing-8B-GGUF",),
                config=load("novel_arch_large")),
            sig(arch="NovelThingForCausalLM", model_ids=("TheBloke/NovelThing-8B-AWQ",)),
        ]
    )
    assert report.passed == []
    assert reasons(report) == ["all_model_ids_derivative"]
    assert "gguf" in report.dropped[0].detail


@pytest.mark.parametrize("pattern", ["gguf", "awq", "gptq", "int4", "-fp8", "-merge",
                                     "lora", "abliterated", "mlx-", "-onnx", "smashed"])
def test_every_derivative_pattern_suppresses(pattern):
    report = run([sig(arch="ZForCausalLM", model_ids=(f"org/Model{pattern}Thing",),
                      config=load("novel_arch_large"))])
    assert reasons(report) == ["all_model_ids_derivative"], pattern


def test_one_clean_repo_rescues_the_architecture():
    """The suppressor is "*every* model id", not "any": a real release plus its
    community quants must still get through."""
    report = run(
        [
            sig(arch="NovelThingForCausalLM", model_ids=("bartowski/NovelThing-GGUF",)),
            sig(arch="NovelThingForCausalLM", model_ids=("someorg/NovelThing-100B",),
                config=load("novel_arch_large")),
        ]
    )
    assert [c.arch_id for c in report.passed] == ["NovelThingForCausalLM"]


def test_framework_only_candidate_is_not_dropped_by_vacuous_truth():
    """A vLLM PR carries no model ids; ``all()`` over an empty list must not suppress."""
    report = run(
        [
            sig("vllm", arch="KimiK3ForCausalLM", raw_ref="30012"),
            sig("sglang", arch="KimiK3ForCausalLM", raw_ref="4411"),
        ]
    )
    assert [c.arch_id for c in report.passed] == ["KimiK3ForCausalLM"]


def test_quant_repack_under_a_renamed_arch_is_suppressed():
    """The residual firehose case: a repacker invents a class name so the seed-set
    check misses, but the config is a known architecture plus a quant block."""
    report = run(
        [
            sig(arch="Qwen3MoeFp8ForCausalLM", org="qwen",
                model_ids=("randomorg/Qwen3-30B-A3B-quantised",),
                config=load("quant_repack_renamed_arch"),
                extra={"downloads": 500_000, "likes": 900}),
        ]
    )
    assert report.passed == []
    assert reasons(report) == ["structurally_identical"]
    assert "Qwen3MoeForCausalLM" in report.dropped[0].detail


def test_quant_repack_listing_the_known_base_class_is_suppressed():
    report = run([sig(arch="AwqWrappedForCausalLM", org="meta-llama",
                      model_ids=("wrapper/llama-70b-w4",),
                      config=load("quant_repack_known_family"))])
    assert report.passed == []
    assert reasons(report) == ["structurally_identical"]
    assert "LlamaForCausalLM" in report.dropped[0].detail


def test_a_quantized_release_with_real_novelty_survives():
    """Shipping in FP8 must not launder away genuinely new config fields."""
    report = run([sig(arch="MysteryMoeForCausalLM", org="someorg",
                      model_ids=("someorg/Mystery-240B-base",),
                      config=load("quant_plus_real_novelty"))])
    assert [c.arch_id for c in report.passed] == ["MysteryMoeForCausalLM"]
    assert "T1" in report.passed[0].triggers
    assert "latent_router_rank" in report.passed[0].unparsed_fields


def test_quant_repack_suppressed_even_when_the_base_arch_has_unparsed_fields():
    """The regression that matters. A repack inherits its base architecture's fields,
    including ones BLIS does not parse: the real support surface reports
    ``['decoder_sparse_step', 'norm_topk_prob']`` as unparsed for *any* Qwen3-MoE
    config. A "novelty must be confined to quantization_config" rule would therefore
    let every FP8 repack of a frontier MoE model through on T1+T3 / S1+S2.
    """
    surf = surface(unparsed_override=["decoder_sparse_step", "norm_topk_prob"])
    report = run(
        [sig(arch="Qwen3MoeFp8ForCausalLM", org="qwen",
             model_ids=("randomorg/Qwen3-30B-A3B-quantised",),
             config=load("quant_repack_renamed_arch"))],
        surf=surf,
    )
    assert report.passed == []
    assert reasons(report) == ["structurally_identical"]


def test_quantizer_rename_is_recognized_without_a_config():
    """Branch A keys off the class name, so it works on a config-less signal too."""
    assert structural_identity("Qwen3MoeFp8ForCausalLM", None, [], surface()) is not None
    assert structural_identity("Qwen3MoeAwqForCausalLM", None, [], surface()) is not None
    assert structural_identity("GriffinMoeForCausalLM", None, [], surface()) is None


def test_architectures_branch_needs_a_quantization_block():
    """Branch B is the conservative one: with no quant block there is nothing to claim
    the difference is confined to."""
    cfg_no_quant = {k: v for k, v in load("quant_repack_known_family").items()
                    if k != "quantization_config"}
    assert structural_identity("AwqWrappedForCausalLM", cfg_no_quant, [], surface()) is None


def test_architectures_branch_keeps_real_novelty():
    """A wrapper class listed beside a known one, but with genuinely new fields, is
    real novelty and must survive."""
    cfg_novel = dict(load("quant_repack_known_family"), latent_router_rank=64)
    assert structural_identity(
        "WrapperForCausalLM", cfg_novel, ["latent_router_rank"], surface()
    ) is None


def test_structural_identity_needs_an_identifiable_known_base():
    """Without condition 3 this suppressor would eat a brand-new frontier
    architecture that merely ships FP8 and uses only fields BLIS already parses."""
    novel = dict(load("quant_repack_renamed_arch"), architectures=["DeepseekV4ForCausalLM"])
    assert structural_identity("DeepseekV4ForCausalLM", novel, [], surface()) is None
    report = run([sig(arch="DeepseekV4ForCausalLM", org="deepseek-ai",
                      model_ids=("deepseek-ai/DeepSeek-V4",), config=novel)])
    assert [c.arch_id for c in report.passed] == ["DeepseekV4ForCausalLM"]


def test_no_config_and_uncorroborated_is_suppressed_for_hf_only_candidates():
    """Scoped to HF: a curated source with no config is exempt (see
    test_vllm_only_signal_with_no_config_fires_t2)."""
    report = run([sig("hf", arch="MysteryForCausalLM", org="someone",
                      model_ids=("someone/mystery",))])
    assert report.passed == []
    assert reasons(report) == ["no_config_uncorroborated"]


def test_no_config_but_corroborated_survives():
    report = run(
        [
            sig("vllm", arch="MysteryForCausalLM", raw_ref="1"),
            sig("inferencex", arch="MysteryForCausalLM", raw_ref="deadbeef"),
        ]
    )
    passed = report.passed
    assert [c.arch_id for c in passed] == ["MysteryForCausalLM"]
    assert passed[0].triggers == ["T2", "T4", "T5"]
    assert passed[0].significance == ["S3"]
    assert passed[0].est_total_params is None


# ---------------------------------------------------------------------------
# Dedup — goes through the emitter's sanitizer, never an f-string path
# ---------------------------------------------------------------------------


def test_existing_stub_suppresses(tmp_path):
    from archwatch.emitter import issue_path

    arch = "NovelThingForCausalLM"
    issue_path(arch, tmp_path).write_text("# already reported\n")
    report = evaluate_detailed(
        join_signals([sig(arch=arch, model_ids=("org/NovelThing",),
                          config=load("novel_arch_large"))]),
        surface(), cfg(), issues_dir=tmp_path,
    )
    assert report.passed == []
    assert reasons(report) == ["already_reported"]


def test_dedup_uses_the_emitter_sanitizer_for_awkward_arch_ids(tmp_path):
    """An arch id needing sanitization does NOT live at ``issues/<arch_id>.md``.
    Building that path here would silently break the dedup forever.
    """
    from archwatch.emitter import issue_path

    arch = "Weird Arch/Name:v2"
    path = issue_path(arch, tmp_path)
    assert path.name != f"{arch}.md", "fixture must exercise the sanitizer"
    path.write_text("# already reported\n")

    report = evaluate_detailed(
        join_signals([sig(arch=arch, model_ids=("org/weird",),
                          config=load("novel_arch_large"))]),
        surface(), cfg(), issues_dir=tmp_path,
    )
    assert reasons(report) == ["already_reported"]


def test_absent_stub_does_not_suppress(tmp_path):
    report = evaluate_detailed(
        join_signals([sig(arch="NovelThingForCausalLM", model_ids=("org/NovelThing",),
                          config=load("novel_arch_large"))]),
        surface(), cfg(), issues_dir=tmp_path,
    )
    assert [c.arch_id for c in report.passed] == ["NovelThingForCausalLM"]


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


def test_t1_novel_arch_with_a_big_config_passes_on_scale_alone():
    """PLAN.md acceptance: a synthetic novel arch with a big config and 100B+ params
    passes with exactly ["T1"] / ["S1"]."""
    report = run([sig(arch="GriffinMoeForCausalLM", org="someuniversity",
                      model_ids=("someuniversity/griffin-158b",),
                      config=load("novel_arch_large"))])
    assert len(report.passed) == 1
    cand = report.passed[0]
    assert cand.triggers == ["T1"]
    assert cand.significance == ["S1"]
    assert cand.est_total_params is not None and cand.est_total_params > 100_000_000_000
    assert cand.est_active_params is not None
    assert cand.est_active_params < cand.est_total_params
    assert cand.unparsed_fields == [
        "gate_projection_rank", "recurrent_chunk_size",
        "recurrent_state_dim", "router_temperature",
    ]


def test_t1_is_the_only_trigger_that_needs_a_config():
    """T1 is the highest-value trigger precisely because BLIS silently drops unknown
    fields, so an unparsed field is the only alarm for wrong-but-running."""
    report = run([sig(arch="GriffinMoeForCausalLM", org="x", model_ids=("x/griffin",),
                      config=load("novel_arch_large"))])
    assert report.passed[0].triggers == ["T1"]


def test_t1_does_not_fire_on_ignored_boilerplate():
    """Defence in depth: even a surface that leaks IGNORED_CONFIG_KEYS must not make
    T1 fire on ``transformers_version``."""
    leaky = surface(unparsed_override=["transformers_version", "_name_or_path",
                                       "torch_dtype", "architectures"])
    report = run([sig(arch="PlainForCausalLM", org="x", model_ids=("x/plain",),
                      config=load("dense_llama31_70b"))], surf=leaky)
    assert report.passed == []
    assert reasons(report) == ["no_trigger"]


def test_t2_framework_support_pr():
    report = run(
        [
            sig("vllm", arch="KimiK3ForCausalLM", raw_ref="30012",
                display_name="[Model] Add KimiK3ForCausalLM"),
            sig("hf", arch="KimiK3ForCausalLM", org="unknownorg",
                model_ids=("unknownorg/K3",), config=load("dense_llama31_70b")),
        ]
    )
    cand = report.passed[0]
    assert "T2" in cand.triggers
    assert "S3" in cand.significance


def test_t3_frontier_org_from_the_org_field():
    report = run([sig(arch="NewMoonForCausalLM", org="moonshotai",
                      model_ids=("moonshotai/NewMoon",),
                      config=load("dense_llama31_70b"))])
    cand = report.passed[0]
    assert "T3" in cand.triggers
    assert "S2" in cand.significance


def test_t3_frontier_org_inferred_from_the_model_id_prefix():
    """Not every connector sets Signal.org; the owner prefix is the fallback."""
    report = run([sig(arch="NewMoonForCausalLM", org=None,
                      model_ids=("deepseek-ai/DeepSeek-V4-Base",),
                      config=load("dense_llama31_70b"))])
    assert "T3" in report.passed[0].triggers


def test_t3_org_match_is_case_insensitive():
    report = run([sig(arch="NewMoonForCausalLM", org="MoonshotAI",
                      model_ids=("MoonshotAI/NewMoon",),
                      config=load("dense_llama31_70b"))])
    assert "T3" in report.passed[0].triggers


def test_t4_corroboration_from_two_distinct_sources():
    report = run(
        [
            sig("hf", arch="NewThingForCausalLM", org="smallorg",
                model_ids=("smallorg/NewThing",), config=load("dense_llama31_70b")),
            sig("inferencex", arch="NewThingForCausalLM", raw_ref="cafe",
                extra={"perf": {"tokens_per_second": 1234}}),
        ]
    )
    cand = report.passed[0]
    assert "T4" in cand.triggers
    assert cand.sources == ["hf", "inferencex"]


def test_two_signals_from_one_source_do_not_corroborate():
    report = run(
        [
            sig("hf", arch="NewThingForCausalLM", org="smallorg",
                model_ids=("smallorg/NewThing-Base",), config=load("dense_llama31_70b")),
            sig("hf", arch="NewThingForCausalLM", org="smallorg",
                model_ids=("smallorg/NewThing-Chat",)),
        ]
    )
    assert reasons(report) == ["no_trigger"]


def test_no_trigger_is_dropped():
    """A small, unremarkable model on an unremarkable org using only fields BLIS
    already parses is exactly what must not become an issue."""
    plain = {"architectures": ["QuietForCausalLM"], "hidden_size": 2048,
             "num_hidden_layers": 24, "vocab_size": 32000, "num_attention_heads": 16,
             "num_key_value_heads": 4, "intermediate_size": 5632, "hidden_act": "silu",
             "tie_word_embeddings": False}
    report = run([sig(arch="QuietForCausalLM", org="hobbyist",
                      model_ids=("hobbyist/quiet-1b",), config=plain)])
    assert report.passed == []
    assert reasons(report) == ["no_trigger"]
    assert "unparsed_fields=[]" in report.dropped[0].detail


def test_trigger_list_is_ordered_and_keeps_the_alias_marker():
    report = run(
        [
            sig("vllm", display_name="MiniMax-M3", raw_ref="1"),
            sig("inferencex", display_name="MiniMax-M3", raw_ref="2"),
        ]
    )
    cand = report.passed[0]
    assert cand.triggers == ["T2", "T4", "T5", ALIAS_JOIN_MARKER]
    assert [t for t in cand.triggers if t in TRIGGER_IDS] == ["T2", "T4", "T5"]


# ---------------------------------------------------------------------------
# Significance gate
# ---------------------------------------------------------------------------


def test_s1_scale_threshold_is_respected_in_both_directions():
    small = {"architectures": ["TinyNovelForCausalLM"], "hidden_size": 1024,
             "num_hidden_layers": 12, "vocab_size": 32000, "num_attention_heads": 8,
             "intermediate_size": 2816, "hidden_act": "silu",
             "novel_recurrent_gate": 4, "tie_word_embeddings": False}
    report = run([sig(arch="TinyNovelForCausalLM", org="hobbyist",
                      model_ids=("hobbyist/tiny",), config=small)], conf=strict_cfg())
    assert report.passed == []
    assert reasons(report) == ["insignificant"]
    assert "T1" in report.dropped[0].detail

    lenient = cfg(thresholds=Thresholds(min_total_params=100_000_000))
    report2 = run([sig(arch="TinyNovelForCausalLM", org="hobbyist",
                       model_ids=("hobbyist/tiny",), config=small)], conf=lenient)
    assert report2.passed[0].significance == ["S1"]


def test_s2_org_top_downloads_when_the_org_is_not_frontier():
    small_but_popular_org = {
        "architectures": ["UpstartForCausalLM"], "hidden_size": 1024,
        "num_hidden_layers": 12, "vocab_size": 32000, "num_attention_heads": 8,
        "intermediate_size": 2816, "hidden_act": "silu", "novel_gate_rank": 8,
        "tie_word_embeddings": False,
    }
    report = run([sig(arch="UpstartForCausalLM", org="upstart",
                      model_ids=("upstart/model",), config=small_but_popular_org,
                      extra={"org_top_downloads": 5_000_000})])
    assert report.passed[0].significance == ["S2"]


def test_s2_ignores_org_downloads_below_threshold():
    tiny = {"architectures": ["UpstartForCausalLM"], "hidden_size": 1024,
            "num_hidden_layers": 12, "vocab_size": 32000, "num_attention_heads": 8,
            "intermediate_size": 2816, "hidden_act": "silu", "novel_gate_rank": 8,
            "tie_word_embeddings": False}
    report = run([sig(arch="UpstartForCausalLM", org="upstart",
                      model_ids=("upstart/model",), config=tiny,
                      extra={"org_top_downloads": 12})])
    assert reasons(report) == ["insignificant"]


def test_s2_missing_extras_simply_do_not_satisfy_it():
    """PLAN.md: "if unavailable, S2 is simply not satisfied" — never an error."""
    tiny = {"architectures": ["UpstartForCausalLM"], "hidden_size": 1024,
            "num_hidden_layers": 12, "vocab_size": 32000, "num_attention_heads": 8,
            "intermediate_size": 2816, "hidden_act": "silu", "novel_gate_rank": 8,
            "tie_word_embeddings": False}
    report = run([sig(arch="UpstartForCausalLM", org="upstart",
                      model_ids=("upstart/model",), config=tiny)])
    assert reasons(report) == ["insignificant"]


def test_s3_any_curated_source():
    for source in ("vllm", "sglang", "inferencex"):
        report = run(
            [
                sig(source, arch="CuratedForCausalLM", raw_ref="1"),
                sig("hf", arch="CuratedForCausalLM", org="nobody",
                    model_ids=("nobody/curated",)),
            ]
        )
        assert "S3" in report.passed[0].significance, source


@pytest.mark.parametrize(
    "extra",
    [
        {"downloads": 50_000},
        {"likes": 900},
        {"trending": True},
        {"trending_score": 3.5},
        {"is_trending": True},
    ],
)
def test_s4_popularity_and_trending(extra):
    small = {"architectures": ["PopularForCausalLM"], "hidden_size": 1024,
             "num_hidden_layers": 12, "vocab_size": 32000, "num_attention_heads": 8,
             "intermediate_size": 2816, "hidden_act": "silu", "novel_gate": 1,
             "tie_word_embeddings": False}
    report = run([sig(arch="PopularForCausalLM", org="nobody",
                      model_ids=("nobody/popular",), config=small, extra=extra)])
    assert report.passed[0].significance == ["S4"]


def test_s4_ignores_counts_below_threshold():
    small = {"architectures": ["PopularForCausalLM"], "hidden_size": 1024,
             "num_hidden_layers": 12, "vocab_size": 32000, "num_attention_heads": 8,
             "intermediate_size": 2816, "hidden_act": "silu", "novel_gate": 1,
             "tie_word_embeddings": False}
    report = run([sig(arch="PopularForCausalLM", org="nobody",
                      model_ids=("nobody/popular",), config=small,
                      extra={"downloads": 12, "likes": 3})])
    assert reasons(report) == ["insignificant"]


def test_all_recorded_significance_ids_are_known():
    report = run([sig(arch="GriffinMoeForCausalLM", org="moonshotai",
                      model_ids=("moonshotai/griffin",),
                      config=load("novel_arch_large"),
                      extra={"downloads": 1_000_000})])
    cand = report.passed[0]
    assert set(cand.significance) <= set(SIGNIFICANCE_IDS)
    assert cand.significance == ["S1", "S2", "S4"]


# ---------------------------------------------------------------------------
# Findings populated on the Candidate
# ---------------------------------------------------------------------------


def test_findings_are_populated_from_the_surface_and_sizing():
    surf = surface(validator_failures=["hidden_act 'gelu' is not SwiGLU-family "
                                       "(kv_capacity.go:276)"],
                   gaps=("mla_step_time_kv_read",))
    report = run([sig(arch="DeepseekV4ForCausalLM", org="deepseek-ai",
                      model_ids=("deepseek-ai/DeepSeek-V4",),
                      config=load("mla_moe_deepseek_v3"))], surf=surf)
    cand = report.passed[0]
    assert cand.bucket0_failures and "SwiGLU" in cand.bucket0_failures[0]
    assert cand.would_not_run is True
    assert cand.est_total_params == pytest.approx(671_000_000_000, rel=0.02)
    assert cand.est_active_params == pytest.approx(37_000_000_000, rel=0.02)
    assert matched_gaps(cand, surf) == ["mla_step_time_kv_read"]


def test_matched_gaps_is_empty_without_a_config():
    cand = Candidate(arch_id="X", display_name="X", signals=[sig("vllm", arch="X")])
    assert matched_gaps(cand, surface(gaps=("mla_step_time_kv_read",))) == []


def test_rejected_candidates_are_also_annotated():
    """The run log wants findings on the rejects too, so the filter can be tuned."""
    plain = {"architectures": ["QuietForCausalLM"], "hidden_size": 2048,
             "num_hidden_layers": 24, "vocab_size": 32000, "num_attention_heads": 16,
             "intermediate_size": 5632, "hidden_act": "silu",
             "tie_word_embeddings": False}
    cands = join_signals([sig(arch="QuietForCausalLM", org="hobbyist",
                              model_ids=("hobbyist/quiet",), config=plain)])
    evaluate(cands, surface(), cfg(), already_reported=never_reported)
    assert cands[0].est_total_params is not None
    assert cands[0].unparsed_fields == []


# ---------------------------------------------------------------------------
# Ranking and the per-run cap
# ---------------------------------------------------------------------------


def _passing_signals(n: int) -> list[Signal]:
    """``n`` equally-strong T1/S1 candidates, newest first, each with its own arch."""
    out = []
    for i in range(n):
        arch = f"Novel{i}ForCausalLM"
        out.append(
            sig(arch=arch, org="hobbyist", model_ids=(f"hobbyist/novel-{i}",),
                config=dict(load("novel_arch_large"), architectures=[arch]),
                observed_at=NOW - timedelta(days=i))
        )
    return out


def test_per_run_cap_is_honored():
    report = run(_passing_signals(9), conf=cfg(max_issues_per_run=3))
    assert len(report.passed) == 3
    assert reasons(report) == ["over_cap"] * 6
    assert report.counts == {"passed": 3, "over_cap": 6}


def test_cap_of_zero_emits_nothing():
    report = run(_passing_signals(3), conf=cfg(max_issues_per_run=0))
    assert report.passed == []
    assert reasons(report) == ["over_cap"] * 3


def test_ties_are_broken_by_recency():
    """All nine are equally strong, so the newest survive the cap."""
    report = run(_passing_signals(9), conf=cfg(max_issues_per_run=3))
    assert [c.arch_id for c in report.passed] == [
        "Novel0ForCausalLM", "Novel1ForCausalLM", "Novel2ForCausalLM",
    ]


def test_stronger_evidence_outranks_more_recent():
    strong = [
        sig("hf", arch="StrongForCausalLM", org="deepseek-ai",
            model_ids=("deepseek-ai/Strong",), config=load("novel_arch_large"),
            extra={"downloads": 900_000}, observed_at=NOW - timedelta(days=30)),
        sig("vllm", arch="StrongForCausalLM", raw_ref="1",
            observed_at=NOW - timedelta(days=30)),
    ]
    weak = [
        sig("hf", arch="WeakForCausalLM", org="hobbyist", model_ids=("hobbyist/weak",),
            config=load("novel_arch_large"), observed_at=NOW),
    ]
    report = run(strong + weak, conf=cfg(max_issues_per_run=2))
    assert [c.arch_id for c in report.passed] == ["StrongForCausalLM", "WeakForCausalLM"]


def test_bucket0_candidates_rank_above_equally_evidenced_ones():
    """"BLIS would refuse to run this" is the finding with a deadline."""

    class Bucket0ForOne(FakeSurface):
        def check_hard_validators(self, config):
            arch = (config.get("architectures") or [""])[0]
            return ["vocab_size must be > 0"] if arch == "Novel1ForCausalLM" else []

    surf = Bucket0ForOne(known=SEED_SET)
    report = run(_passing_signals(4), surf=surf, conf=cfg(max_issues_per_run=4))
    assert report.passed[0].arch_id == "Novel1ForCausalLM"
    assert report.passed[0].would_not_run


def test_naive_timestamps_are_read_as_utc_not_local():
    """Ranking must not depend on which machine ran the scan."""
    naive_new = sig(arch="AForCausalLM", org="x", model_ids=("x/a",),
                    config=load("novel_arch_large"),
                    observed_at=datetime(2026, 9, 1, 0, 0))
    aware_old = sig(arch="BForCausalLM", org="x", model_ids=("x/b",),
                    config=load("novel_arch_large"),
                    observed_at=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc))
    report = run([aware_old, naive_new], conf=cfg(max_issues_per_run=1))
    assert [c.arch_id for c in report.passed] == ["AForCausalLM"]


# ---------------------------------------------------------------------------
# Report / API surface
# ---------------------------------------------------------------------------


def test_report_counts_tally_every_stage():
    signals = [
        sig(arch="LlamaForCausalLM", model_ids=("org/tune",),
            config=load("dense_llama31_70b")),
        sig(arch="ZForCausalLM", model_ids=("org/z-gguf",),
            config=load("novel_arch_large")),
        sig("hf", arch="LonelyForCausalLM", org="nobody",
            model_ids=("nobody/lonely",)),
        sig(arch="GriffinMoeForCausalLM", org="hobbyist", model_ids=("hobbyist/g",),
            config=load("novel_arch_large")),
    ]
    report = run(signals, conf=strict_cfg())
    assert report.counts == {
        "passed": 1,
        "known_architecture": 1,
        "all_model_ids_derivative": 1,
        "no_config_uncorroborated": 1,
    }


def test_evaluate_matches_evaluate_detailed_passed():
    signals = _passing_signals(4)
    a = evaluate(join_signals(signals), surface(), cfg(), already_reported=never_reported)
    b = evaluate_detailed(join_signals(signals), surface(), cfg(),
                          already_reported=never_reported).passed
    assert [c.arch_id for c in a] == [c.arch_id for c in b]


def test_evaluate_on_an_empty_run_is_a_no_op():
    report = run([])
    assert report.passed == []
    assert report.dropped == []
    assert report.counts == {"passed": 0}


def test_suppression_str_is_loggable():
    report = run([sig(arch="LlamaForCausalLM", model_ids=("org/tune",),
                      config=load("dense_llama31_70b"))])
    text = str(report.dropped[0])
    assert "LlamaForCausalLM" in text and "known_architecture" in text


def test_derivative_patterns_are_read_from_the_frozen_config():
    """No local copy of the pattern list: config.py is the single source."""
    assert "gguf" in DERIVATIVE_PATTERNS


@pytest.mark.parametrize(
    "name,expected",
    [
        ("KimiK3ForCausalLM", "kimik3forcausallm"),
        ("Kimi_K3-ForCausalLM", "kimik3forcausallm"),
        ("  Spaced Out  ", "spacedout"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_arch_key(name, expected):
    assert normalize_arch_key(name) == expected


def test_a_raising_surface_propagates():
    """Deliberate, documented behaviour. A flaky *source* must degrade to a partial
    scan (detector.py wraps each poll), but the support surface is loaded once from
    local YAML: if it raises, the run is built on a broken contract and every verdict
    downstream is meaningless. Failing loudly beats emitting confident nonsense."""

    class Exploding(FakeSurface):
        def unparsed_fields(self, config):
            raise RuntimeError("bad yaml")

    with pytest.raises(RuntimeError):
        run([sig(arch="XForCausalLM", org="x", model_ids=("x/x",),
                 config=load("novel_arch_large"))], surf=Exploding(known=SEED_SET))


# ---------------------------------------------------------------------------
# Real-world config shapes and firehose volume
# ---------------------------------------------------------------------------


def test_frontier_config_under_text_config_still_clears_the_scale_gate():
    """If sizing did not pivot text_config, S1 would never fire for the frontier
    multimodal-wrapper shape and the filter would look like it mysteriously drops
    exactly the biggest models."""
    report = run([sig(arch="KimiK3ForConditionalGeneration", org="moonshotai",
                      model_ids=("moonshotai/Kimi-K3",),
                      config=load("textconfig_only_frontier"))])
    cand = report.passed[0]
    assert "S1" in cand.significance
    assert cand.est_total_params is not None and cand.est_total_params > 500_000_000_000


def test_t1_does_not_fire_on_the_dtype_key():
    """2026 configs spell it ``dtype``; IGNORED_CONFIG_KEYS covers both spellings, so
    neither may be the reason an issue gets filed."""
    plain = {"architectures": ["QuietForCausalLM"], "hidden_size": 2048,
             "num_hidden_layers": 24, "vocab_size": 32000, "num_attention_heads": 16,
             "intermediate_size": 5632, "hidden_act": "silu",
             "tie_word_embeddings": False, "dtype": "bfloat16"}
    report = run([sig(arch="QuietForCausalLM", org="hobbyist",
                      model_ids=("hobbyist/quiet",), config=plain)])
    assert reasons(report) == ["no_trigger"]


def test_architectureless_repos_are_dropped_by_the_weak_evidence_suppressor():
    """~1,690 of a real HF day's repos carry no architectures[] at all (diffusers,
    peft, robotics). Alias-keyed, config-less and uncorroborated is what drops them."""
    noise = [sig("hf", display_name=f"someone/pipeline-{i}", raw_ref=str(i))
             for i in range(50)]
    report = run(noise)
    assert report.passed == []
    assert set(reasons(report)) == {"no_config_uncorroborated"}


def test_suppression_logging_aggregates_instead_of_one_line_per_candidate(caplog):
    """A run log with ~1,700 suppression lines is useless for tuning. Per-candidate
    detail stays on the report; INFO gets one aggregated line."""
    import logging

    noise = [sig("hf", display_name=f"someone/pipeline-{i}", raw_ref=str(i))
             for i in range(60)]
    with caplog.at_level(logging.INFO, logger="archwatch.novelty"):
        report = run(noise)
    info_lines = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert len(info_lines) == 1, "expected exactly one INFO line for 60 drops"
    assert "60 candidates -> 0 passed" in info_lines[0].getMessage()
    assert "no_config_uncorroborated=60" in info_lines[0].getMessage()
    # The unabridged detail is still available for the machine-readable run log.
    assert len(report.dropped) == 60


def test_report_summary_is_a_single_readable_line():
    signals = [
        sig(arch="LlamaForCausalLM", model_ids=("org/tune",),
            config=load("dense_llama31_70b")),
        sig(arch="ZForCausalLM", model_ids=("org/z-gguf",),
            config=load("novel_arch_large")),
        sig(arch="GriffinMoeForCausalLM", org="hobbyist", model_ids=("hobbyist/g",),
            config=load("novel_arch_large")),
    ]
    summary = run(signals, conf=strict_cfg()).summary()
    assert summary.startswith("3 candidates -> 1 passed; dropped: ")
    assert "known_architecture=1" in summary
    assert "all_model_ids_derivative=1" in summary


def test_summary_of_an_empty_run():
    assert run([]).summary() == "0 candidates -> 0 passed"


def test_nonstandard_field_names_are_a_t1_case_without_a_param_estimate():
    """ParallaxOpen/Vela-Lumen-31M: no architectures[], d_model/n_layers/n_heads.
    Every field is unknown to BLIS, so T1 fires — but sizing declines to guess, so S1
    cannot, and at 31M nothing else should carry it either.
    """
    report = run([sig("hf", display_name="ParallaxOpen/Vela-Lumen-31M",
                      model_ids=("ParallaxOpen/Vela-Lumen-31M",),
                      org="parallaxopen", config=load("nonstandard_field_names"))])
    assert report.passed == []
    assert reasons(report) == ["insignificant"]
    cand_detail = report.dropped[0].detail
    assert "T1" in cand_detail
    assert "est_total_params=None" in cand_detail


# ---------------------------------------------------------------------------
# arch_id canonicalization (stable issue filenames across runs)
# ---------------------------------------------------------------------------

_SPELLINGS = ["KimiK3ForCausalLM", "kimik3forcausallm", "Kimi_K3_ForCausalLM"]


@pytest.mark.parametrize(
    "order",
    [
        [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
    ],
)
def test_arch_id_does_not_depend_on_signal_arrival_order(order):
    """The join key ignores case and punctuation but the emitter's filename sanitizer
    does not, so a first-seen arch_id would put the same architecture at
    ``KimiK3ForCausalLM.md`` on one run and ``kimik3forcausallm.md`` on the next —
    silently defeating the stateless dedup.
    """
    cands = join_signals([sig("hf", arch=_SPELLINGS[i]) for i in order])
    assert len(cands) == 1
    assert cands[0].arch_id == "KimiK3ForCausalLM"


def test_arch_id_does_not_depend_on_how_many_signals_carry_each_spelling():
    """Rules out "most common casing", which flips as the repo mix changes."""
    lower_heavy = join_signals(
        [sig("hf", arch="kimik3forcausallm", model_ids=(f"o/k3-{i}",)) for i in range(5)]
        + [sig("hf", arch="KimiK3ForCausalLM", model_ids=("o/k3-canonical",))]
    )
    assert len(lower_heavy) == 1
    assert lower_heavy[0].arch_id == "KimiK3ForCausalLM"


def test_an_hf_architectures_spelling_outranks_a_framework_mined_name():
    """architectures[] is ground truth; a name mined from a patch is a human's reading
    of a diff. The tier must beat alphabetical order, so the HF spelling is chosen here
    even though the mined name sorts first."""
    cands = join_signals(
        [
            sig("vllm", arch="AaaMinedName", model_ids=("org/thing",), raw_ref="1"),
            sig("hf", arch="ZzzRealClassName", model_ids=("org/thing",)),
        ]
    )
    assert len(cands) == 1, "must actually merge for the tier to matter"
    assert cands[0].arch_id == "ZzzRealClassName"
    assert ALIAS_JOIN_MARKER not in cands[0].triggers


def test_a_framework_mined_name_beats_a_display_name():
    cands = join_signals(
        [
            sig("inferencex", display_name="Zzz-Model", model_ids=("org/thing",)),
            sig("vllm", arch="AaaMinedName", model_ids=("org/thing",), raw_ref="1"),
        ]
    )
    assert len(cands) == 1
    assert cands[0].arch_id == "AaaMinedName"
    assert ALIAS_JOIN_MARKER not in cands[0].triggers


def test_canonical_arch_id_is_stable_for_the_alias_path_too():
    a = join_signals([sig("vllm", display_name="MiniMax-M3"),
                      sig("inferencex", display_name="minimax-m3")])
    b = join_signals([sig("inferencex", display_name="minimax-m3"),
                      sig("vllm", display_name="MiniMax-M3")])
    assert a[0].arch_id == b[0].arch_id == "MiniMax-M3"


def test_a_real_arch_string_wins_over_an_alias_in_either_order():
    for signals in (
        [sig("inferencex", display_name="MiniMax-M3"), sig("hf", arch="MiniMax_M3")],
        [sig("hf", arch="MiniMax_M3"), sig("inferencex", display_name="MiniMax-M3")],
    ):
        cands = join_signals(signals)
        assert cands[0].arch_id == "MiniMax_M3"
        assert ALIAS_JOIN_MARKER not in cands[0].triggers


# ---------------------------------------------------------------------------
# recheck_known_architectures
# ---------------------------------------------------------------------------


def test_known_arch_trigger_is_a_trigger_but_not_one_of_the_numbered_ones():
    assert KNOWN_ARCH_TRIGGER in TRIGGER_IDS
    assert KNOWN_ARCH_TRIGGER not in CORE_TRIGGER_IDS
    assert set(CORE_TRIGGER_IDS) == {"T1", "T2", "T3", "T4", "T5"}
    assert ALIAS_JOIN_MARKER not in TRIGGER_IDS


def _known_arch_signal(config: dict, **kw) -> Signal:
    kw.setdefault("org", "meta-llama")
    kw.setdefault("model_ids", ("meta-llama/Llama-3.1-70B-Instruct",))
    return sig(arch="LlamaForCausalLM", config=config, **kw)


def test_known_arch_with_an_inert_config_is_suppressed_under_both_settings():
    """The default must not change, and the re-check must not turn every seeded
    architecture into an issue."""
    signals = [_known_arch_signal(load("dense_llama31_70b"))]

    off = run(signals, conf=cfg(recheck_known_architectures=False))
    assert off.passed == []
    assert reasons(off) == ["known_architecture"]

    on = run(signals, conf=cfg(recheck_known_architectures=True))
    assert on.passed == []
    assert reasons(on) == ["known_architecture_nothing_new"]
    assert "no unparsed field and no silent-misread finding" in on.dropped[0].detail


def test_known_arch_with_novel_fields_surfaces_only_when_rechecking():
    """The recall hole, closed. Llama-3.1-70B's architecture string is seeded, but this
    config has grown a field BLIS cannot read — silent wrong numbers, invisible today.
    """
    grown = dict(load("dense_llama31_70b"), latent_router_rank=96,
                 sparse_attention_window=4096)
    signals = [_known_arch_signal(grown)]

    off = run(signals, conf=cfg(recheck_known_architectures=False))
    assert off.passed == []
    assert reasons(off) == ["known_architecture"], "pre-calibration behaviour"

    on = run(signals, conf=cfg(recheck_known_architectures=True))
    assert [c.arch_id for c in on.passed] == ["LlamaForCausalLM"]
    cand = on.passed[0]
    assert cand.triggers == [KNOWN_ARCH_TRIGGER]
    assert "T1" not in cand.triggers, "must be distinguishable from a new architecture"
    assert cand.unparsed_fields == ["latent_router_rank", "sparse_attention_window"]
    assert "S1" in cand.significance


def test_recheck_withholds_t2_t3_t4():
    """A frontier org republishing a seeded architecture, corroborated by a framework
    PR, is not news. Only the unparsed-field question is asked."""
    signals = [
        _known_arch_signal(load("dense_llama31_70b")),
        sig("vllm", arch="LlamaForCausalLM", raw_ref="30012"),
    ]
    report = run(signals, conf=cfg(recheck_known_architectures=True))
    assert report.passed == []
    assert reasons(report) == ["known_architecture_nothing_new"]


def test_recheck_still_applies_the_derivative_suppressor():
    grown = dict(load("dense_llama31_70b"), latent_router_rank=96)
    report = run(
        [_known_arch_signal(grown, model_ids=("bartowski/Llama-3.1-70B-GGUF",))],
        conf=cfg(recheck_known_architectures=True),
    )
    assert reasons(report) == ["all_model_ids_derivative"]


def test_recheck_still_applies_the_quant_repack_suppressor():
    """A known-family quant repack must not be resurrected by the re-check sweep."""
    report = run(
        [sig(arch="AwqWrappedForCausalLM", org="meta-llama",
             model_ids=("wrapper/llama-70b-w4",),
             config=load("quant_repack_known_family"))],
        conf=cfg(recheck_known_architectures=True),
    )
    assert reasons(report) == ["structurally_identical"]


def test_recheck_still_applies_the_dedup(tmp_path):
    from archwatch.emitter import issue_path

    issue_path("LlamaForCausalLM", tmp_path).write_text("# already reported\n")
    grown = dict(load("dense_llama31_70b"), latent_router_rank=96)
    report = evaluate_detailed(
        join_signals([_known_arch_signal(grown)]),
        surface(), cfg(recheck_known_architectures=True), issues_dir=tmp_path,
    )
    assert reasons(report) == ["already_reported"]


def test_recheck_still_applies_the_significance_gate():
    """A 1B fine-tune with one novel field is not worth a human's attention."""
    tiny_grown = {"architectures": ["LlamaForCausalLM"], "hidden_size": 1024,
                  "num_hidden_layers": 12, "vocab_size": 32000,
                  "num_attention_heads": 8, "intermediate_size": 2816,
                  "hidden_act": "silu", "tie_word_embeddings": False,
                  "latent_router_rank": 96}
    report = run(
        [sig(arch="LlamaForCausalLM", org="hobbyist", model_ids=("hobbyist/tiny",),
             config=tiny_grown)],
        conf=cfg(recheck_known_architectures=True),
    )
    assert report.passed == []
    assert reasons(report) == ["insignificant"]


def test_recheck_leaves_unknown_architectures_untouched():
    """The flag must not perturb the normal path."""
    signals = [sig(arch="GriffinMoeForCausalLM", org="someuni",
                   model_ids=("someuni/griffin-158b",), config=load("novel_arch_large"))]
    off = run(signals)
    on = run(signals, conf=cfg(recheck_known_architectures=True))
    assert [c.triggers for c in off.passed] == [["T1"]]
    assert [c.triggers for c in on.passed] == [["T1"]]


def test_recheck_run_log_separates_the_sweep_cost_from_its_yield():
    """The tally the backtest needs: how many seeded architectures were re-examined for
    nothing, versus how many actually turned something up."""
    grown = dict(load("dense_llama31_70b"), latent_router_rank=96)
    signals = [
        _known_arch_signal(load("dense_llama31_70b"),
                           model_ids=("meta-llama/Llama-3.1-70B",)),
        sig(arch="MixtralForCausalLM", org="mistralai",
            model_ids=("mistralai/Mixtral-8x7B-v0.1",),
            config=load("moe_mixtral_8x7b")),
        sig(arch="LlamaForCausalLM", org="someorg", model_ids=("someorg/grown-llama",),
            config=grown),
    ]
    report = run(signals, conf=cfg(recheck_known_architectures=True))
    assert report.counts == {"passed": 1, "known_architecture_nothing_new": 1}
    assert report.passed[0].triggers == [KNOWN_ARCH_TRIGGER]
    assert "known_architecture_nothing_new=1" in report.summary()


# ---------------------------------------------------------------------------
# Curated sources are exempt from the weak-evidence suppressor
# ---------------------------------------------------------------------------


def test_vllm_only_signal_with_no_config_fires_t2():
    """The regression that matters most. A framework PR structurally has no
    config.json — a PR is not a model repo — so applying no_config_uncorroborated to it
    made T2 unfireable, deleting the purest zero-day signal in the pipeline: reference
    code written before any HF config is public.
    """
    report = run([sig("vllm", arch="NewThingForCausalLM", raw_ref="30012",
                      display_name="[Model] Add NewThingForCausalLM")])
    assert [c.arch_id for c in report.passed] == ["NewThingForCausalLM"]
    cand = report.passed[0]
    assert cand.triggers == ["T2"]
    assert cand.significance == ["S3"]
    assert cand.config is None
    assert cand.est_total_params is None


def test_sglang_only_signal_with_no_config_fires_t2():
    report = run([sig("sglang", arch="NewThingForCausalLM", raw_ref="4411")])
    assert report.passed[0].triggers == ["T2"]


def test_inferencex_only_signal_fires_t5():
    """The genuine zero-day case: SemiAnalysis is benchmarking something before any
    config or PR is public. It survives the weak-evidence suppressor (a benchmark row
    never carries a config.json) and fires T5."""
    report = run([sig("inferencex", arch="NewThingForCausalLM", raw_ref="cafe123")])
    assert [c.arch_id for c in report.passed] == ["NewThingForCausalLM"]
    cand = report.passed[0]
    assert cand.triggers == ["T5"]
    assert cand.significance == ["S3"]


def test_hf_only_archless_no_config_signal_is_still_suppressed():
    """The suppressor's actual purpose, unchanged: archless metadata-free HF junk."""
    report = run([sig("hf", display_name="someone/random-pipeline", raw_ref="1")])
    assert report.passed == []
    assert reasons(report) == ["no_config_uncorroborated"]


def test_hf_only_signal_with_an_arch_but_no_config_is_still_suppressed():
    report = run([sig("hf", arch="MysteryForCausalLM",
                      model_ids=("someone/mystery",), org="someone")])
    assert reasons(report) == ["no_config_uncorroborated"]


def test_framework_signal_with_no_arch_ids_routes_through_the_alias_path():
    """The join step must not crash or treat these as malformed — they key off
    display_name. ``evaluate`` then suppresses them; see
    test_a_framework_pr_yielding_no_architecture_is_suppressed."""
    cands = join_signals([sig("vllm", display_name="[Model] Support Kimi K3 (follow-up)",
                              raw_ref="30099")])
    assert cands[0].arch_id == "[Model] Support Kimi K3 (follow-up)"
    assert ALIAS_JOIN_MARKER in cands[0].triggers


def test_framework_signal_is_still_suppressed_when_it_is_a_known_architecture():
    """Exempting curated sources must not resurrect a seeded architecture. A framework
    PR is not a benchmark, so it gets no known_architecture exemption."""
    report = run([sig("vllm", arch="LlamaForCausalLM", raw_ref="1")],
                 conf=strict_cfg())
    assert reasons(report) == ["known_architecture"]


# ---------------------------------------------------------------------------
# Org case folding at the point of comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "org,model_id",
    [
        ("Qwen", "Qwen/Qwen4-Next"),
        ("qwen", "qwen/qwen4-next"),
        ("QWEN", "QWEN/QWEN4-NEXT"),
        (None, "Qwen/Qwen4-Next"),
        ("  Qwen  ", "someone/mirror"),
        ("MoonshotAI", "MoonshotAI/Kimi-K3"),
        ("DeepSeek-AI", "DeepSeek-AI/DeepSeek-V4"),
    ],
)
def test_frontier_org_matching_folds_case_whatever_the_connector_emits(org, model_id):
    """FRONTIER_ORGS is all lowercase but connectors need not be. Folding at the point
    of comparison means T3/S2 cannot silently miss `Qwen` vs `qwen` for a connector
    written after this module."""
    report = run([sig("hf", arch="NewFrontierForCausalLM", org=org,
                      model_ids=(model_id,), config=load("dense_llama31_70b"))])
    cand = report.passed[0]
    assert "T3" in cand.triggers
    assert "S2" in cand.significance


# ---------------------------------------------------------------------------
# Cross-source join: the merges that MUST happen
# ---------------------------------------------------------------------------


def _kimi_three_sources() -> list[Signal]:
    """One release as the three landed connectors actually describe it.

    Note the disjoint vocabularies: InferenceX has no ``arch_ids`` at all (its data has
    no config.json to read one from), the framework connector has a class name mined
    from a patch and no config, and only HF has both an architecture and a config.
    """
    return [
        sig("hf", arch="KimiK3ForCausalLM", org="moonshotai",
            model_ids=("moonshotai/Kimi-K3-Instruct",),
            # A brand-new hybrid architecture carries fields BLIS cannot read; that is
            # what T1 is for. (The real surface reports nine unparsed fields for this
            # config even without the extra one; the stub's inert list is deliberately
            # more generous, so novelty is added explicitly here.)
            config=dict(load("hybrid_mla_linear_attn"), kda_decay_floor=0.02)),
        sig("vllm", arch="KimiK3ForCausalLM", raw_ref="30012",
            display_name="[Model] Add KimiK3ForCausalLM"),
        sig("inferencex", display_name="Kimi-K3", org="moonshotai",
            model_ids=("moonshotai/Kimi-K3",), raw_ref="cafe123"),
    ]


def test_three_sources_describing_one_release_become_one_candidate():
    """The bug this join exists to fix. Under single-key grouping these filed three
    separate issues and T4 could never fire."""
    cands = join_signals(_kimi_three_sources())
    assert len(cands) == 1
    cand = cands[0]
    assert cand.arch_id == "KimiK3ForCausalLM"
    assert cand.sources == ["hf", "inferencex", "vllm"]
    assert cand.corroborated


def test_the_three_source_merge_fires_every_trigger():
    report = run(_kimi_three_sources())
    cand = report.passed[0]
    assert cand.triggers == ["T1", "T2", "T3", "T4", "T5"]
    assert cand.significance == ["S1", "S2", "S3"]


def test_the_family_edge_alone_bridges_a_class_name_to_a_human_name():
    """No shared arch id, no shared repo id: ``MiniMaxM4ForCausalLM`` -> ``minimaxm4``
    and ``MiniMax-M4`` -> ``minimaxm4`` is the only thing connecting these."""
    cands = join_signals(
        [
            sig("sglang", arch="MiniMaxM4ForCausalLM", raw_ref="1"),
            sig("inferencex", display_name="MiniMax-M4", raw_ref="2"),
        ]
    )
    assert len(cands) == 1
    assert cands[0].join_edges == ["family:minimaxm4"]


def test_variant_repos_of_one_release_join_on_the_repo_edge():
    cands = join_signals(
        [
            sig("hf", model_ids=("moonshotai/Kimi-K3-Base",)),
            sig("hf", model_ids=("moonshotai/Kimi-K3-Instruct",)),
            sig("inferencex", model_ids=("moonshotai/Kimi-K3",), raw_ref="1"),
        ]
    )
    assert len(cands) == 1
    assert cands[0].join_edges == ["repo:moonshotai/kimi-k3"]


def test_a_quantized_repo_joins_its_base_release():
    cands = join_signals(
        [
            sig("hf", model_ids=("moonshotai/Kimi-K3",)),
            sig("hf", model_ids=("moonshotai/Kimi-K3-Instruct-FP8",)),
        ]
    )
    assert len(cands) == 1


def test_the_join_is_transitive():
    """An InferenceX row and a vLLM PR that share no string at all both meet the HF
    signal, and so meet each other. That transitivity is the point of union-find."""
    cands = join_signals(
        [
            sig("inferencex", model_ids=("someorg/Thing-V2",), raw_ref="1"),
            sig("hf", arch="ThingV2ForCausalLM", model_ids=("someorg/Thing-V2-Instruct",)),
            sig("vllm", arch="ThingV2ForCausalLM", raw_ref="2"),
        ]
    )
    assert len(cands) == 1
    assert cands[0].sources == ["hf", "inferencex", "vllm"]


# ---------------------------------------------------------------------------
# Cross-source join: the merges that must NOT happen
# ---------------------------------------------------------------------------
# A false merge is worse than a duplicate issue: two architectures collapsed into one
# stub means the second one is never analysed and never noticed.


def test_dense_and_moe_variants_of_a_family_do_not_merge():
    cands = join_signals(
        [
            sig("hf", arch="Qwen3ForCausalLM", org="qwen", model_ids=("Qwen/Qwen3-8B",)),
            sig("hf", arch="Qwen3MoeForCausalLM", org="qwen",
                model_ids=("Qwen/Qwen3-30B-A3B",)),
        ]
    )
    assert sorted(c.arch_id for c in cands) == ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]
    assert normalize_family_key("Qwen3ForCausalLM") != normalize_family_key("Qwen3MoeForCausalLM")


def test_two_point_releases_of_a_family_do_not_merge():
    cands = join_signals(
        [
            sig("inferencex", display_name="GLM-5.2", raw_ref="1"),
            sig("vllm", arch="Glm5NextForCausalLM", raw_ref="2"),
        ]
    )
    assert len(cands) == 2
    assert normalize_family_key("GLM-5.2") == "glm52"
    assert normalize_family_key("Glm5NextForCausalLM") == "glm5next"


def test_two_orgs_sharing_a_model_basename_do_not_merge():
    """The repo edge keeps its org prefix, and the family edge is never fed from a repo
    basename, precisely so this cannot collapse."""
    cands = join_signals(
        [
            sig("hf", model_ids=("someorg/Mystery-8B",), display_name="someorg/Mystery-8B"),
            sig("hf", model_ids=("otherorg/Mystery-8B",), display_name="otherorg/Mystery-8B"),
        ]
    )
    assert len(cands) == 2
    assert normalize_repo_key("someorg/Mystery-8B") != normalize_repo_key("otherorg/Mystery-8B")


def test_different_sizes_of_one_family_do_not_merge():
    """No key strips a size token — stripping ``-8b`` would merge Qwen3-8B with
    Qwen3-30B, the worst false merge available."""
    cands = join_signals(
        [
            sig("hf", model_ids=("Qwen/Qwen3-8B",)),
            sig("hf", model_ids=("Qwen/Qwen3-30B",)),
        ]
    )
    assert len(cands) == 2


def test_a_generic_family_key_is_not_evidence():
    """Below MIN_FAMILY_KEY_LEN no family edge is emitted, so a two-character release
    name cannot merge unrelated models from different labs."""
    assert normalize_family_key("V3") == ""
    cands = join_signals(
        [
            sig("inferencex", display_name="V3", model_ids=("deepseek-ai/x",), raw_ref="1"),
            sig("inferencex", display_name="V3", model_ids=("someoneelse/y",), raw_ref="2"),
        ]
    )
    assert len(cands) == 2, "'V3' alone must not join two labs' models"


# ---------------------------------------------------------------------------
# Merge auditability
# ---------------------------------------------------------------------------


def test_the_edge_that_caused_each_merge_is_recorded():
    cand = join_signals(_kimi_three_sources())[0]
    edges = cand.join_edges
    assert "arch:kimik3forcausallm" in edges
    assert "repo:moonshotai/kimi-k3" in edges
    assert all(e.split(":", 1)[0] in {"arch", "repo", "family"} for e in edges)
    assert len(edges) == len(set(edges)), "edges must be deduplicated"


def test_a_single_signal_candidate_has_no_join_edges():
    cand = join_signals([sig("hf", arch="XForCausalLM", model_ids=("o/x",))])[0]
    assert cand.join_edges == []


def test_cross_source_merges_are_logged_at_info_for_audit(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="archwatch.novelty"):
        join_signals(_kimi_three_sources())
    merge_lines = [r.getMessage() for r in caplog.records if "joined" in r.getMessage()]
    assert len(merge_lines) == 1
    assert "['hf', 'inferencex', 'vllm']" in merge_lines[0]
    assert "arch:kimik3forcausallm" in merge_lines[0]


def test_same_source_merges_stay_at_debug(caplog):
    """Every Llama fine-tune in a window joins on one arch key. Logging those at INFO
    would bury the cross-source merges that corroboration actually rests on."""
    import logging

    with caplog.at_level(logging.INFO, logger="archwatch.novelty"):
        join_signals([sig("hf", arch="LlamaForCausalLM", model_ids=(f"o/tune-{i}",))
                      for i in range(20)])
    assert [r for r in caplog.records if "joined" in r.getMessage()] == []


# ---------------------------------------------------------------------------
# Repo / family key normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("moonshotai/Kimi-K3", "moonshotai/kimi-k3"),
        ("moonshotai/Kimi-K3-Instruct", "moonshotai/kimi-k3"),
        ("moonshotai/Kimi-K3-Base", "moonshotai/kimi-k3"),
        ("moonshotai/Kimi-K3-Instruct-FP8", "moonshotai/kimi-k3"),
        ("MoonshotAI/Kimi-K3-Thinking", "moonshotai/kimi-k3"),
        ("nvidia/Llama-3.1-70B-Instruct-FP8-dynamic", "nvidia/llama-3.1-70b"),
        ("Qwen/Qwen3-30B-A3B", "qwen/qwen3-30b-a3b"),
        ("Qwen/Qwen3-8B", "qwen/qwen3-8b"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_repo_key(model_id, expected):
    assert normalize_repo_key(model_id) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("KimiK3ForCausalLM", "kimik3"),
        ("Kimi-K3", "kimik3"),
        ("KimiK3MTPModel", "kimik3"),
        ("Llama4ForConditionalGeneration", "llama4"),
        ("Qwen3MoeForCausalLM", "qwen3moe"),
        ("GLM-5.2", "glm52"),
        ("Glm5NextForCausalLM", "glm5next"),
        ("V3", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_family_key(name, expected):
    assert normalize_family_key(name) == expected


# ---------------------------------------------------------------------------
# T5 and the derivative-suppressor scoping
# ---------------------------------------------------------------------------


def test_t5_does_not_fire_for_a_framework_only_candidate():
    """T2 and T5 stay distinct: a PR hands you reference code, a benchmark row hands you
    numbers. Collapsing them would lose that in the report."""
    report = run([sig("vllm", arch="NewThingForCausalLM", raw_ref="1")])
    assert report.passed[0].triggers == ["T2"]


def test_t2_t3_t4_stay_withheld_on_the_recheck_path():
    """A framework PR against a seeded architecture is not news, and neither is a
    frontier org republishing one. Only T1 and T5 may fire there."""
    report = run(
        [
            sig("vllm", arch="LlamaForCausalLM", raw_ref="1"),
            sig("hf", arch="LlamaForCausalLM", org="meta-llama",
                model_ids=("meta-llama/Llama-3.1-70B",), config=load("dense_llama31_70b")),
        ],
        conf=cfg(recheck_known_architectures=True),
    )
    assert reasons(report) == ["known_architecture_nothing_new"]


def test_a_pr_merely_mentioning_a_gguf_repo_does_not_suppress():
    """An id mined from a PR diff is a mention, not an artifact. "[Model] Support
    NewThing-8B-GGUF loading" is work on a loader — suppressing it would delete the very
    architecture the PR adds support for."""
    report = run([sig("vllm", arch="NewThingForCausalLM", raw_ref="1",
                      model_ids=("someone/NewThing-8B-GGUF",),
                      display_name="[Model] Support NewThing-8B-GGUF loading")])
    assert [c.arch_id for c in report.passed] == ["NewThingForCausalLM"]
    assert report.passed[0].triggers == ["T2"]


def test_a_changelog_mentioning_a_quantized_repo_does_not_suppress():
    report = run([sig("inferencex", arch="NewThingForCausalLM", raw_ref="sha1",
                      model_ids=("someone/NewThing-FP8",))])
    assert [c.arch_id for c in report.passed] == ["NewThingForCausalLM"]


def test_an_hf_upload_that_is_only_a_gguf_repack_is_still_suppressed():
    """The suppressor's purpose is unchanged for the source whose ids are artifacts."""
    report = run([sig("hf", arch="NewThingForCausalLM", org="bartowski",
                      model_ids=("bartowski/NewThing-8B-GGUF",),
                      config=load("novel_arch_large"))])
    assert reasons(report) == ["all_model_ids_derivative"]


# ---------------------------------------------------------------------------
# silent_failures — the highest-value finding class
# ---------------------------------------------------------------------------
# BLIS never dispatches on architectures[]; it reads numeric shape fields and drops
# what it does not recognize. So the dangerous outcome is not a crash but a completed
# run with confidently wrong numbers. bucket0_failures says "BLIS refuses to run";
# silent_failures says "BLIS runs and lies". The second is the one nothing else warns
# about, and it must never be conflated with the first.


def _moe_via_unrecognized_spelling(spelling: str) -> dict:
    """A config that signals MoE through a spelling outside BLIS's alias set.

    ``num_experts_per_tok`` is present and recognized, so BLIS knows it is looking at a
    mixture — but no total count resolves, so a trillion-parameter sparse model
    simulates as a dense one behind a single ``logrus.Warnf``.
    """
    return {
        "architectures": ["MysteryMoeForCausalLM"], "hidden_size": 7168,
        "num_hidden_layers": 61, "vocab_size": 129280, "num_attention_heads": 128,
        "num_key_value_heads": 128, "intermediate_size": 18432,
        "moe_intermediate_size": 2048, "hidden_act": "silu",
        "tie_word_embeddings": False, "num_experts_per_tok": 8, spelling: 256,
    }


@pytest.mark.parametrize("spelling", ["num_moe_experts", "expert_count", "n_group_experts"])
def test_silent_failures_are_populated(spelling):
    report = run([sig("hf", arch="MysteryMoeForCausalLM", org="somelab",
                      model_ids=("somelab/mystery-671b",),
                      config=_moe_via_unrecognized_spelling(spelling))])
    cand = report.passed[0]
    assert len(cand.silent_failures) == 2, cand.silent_failures
    assert any("moe_expert_count_resolvable" in m for m in cand.silent_failures)
    assert any("moe_total_required_when_active_present" in m for m in cand.silent_failures)


@pytest.mark.parametrize("spelling", ["moe_num_experts", "n_routed_experts"])
def test_a_recognized_expert_spelling_is_not_a_silent_failure(spelling):
    """Guards the fixtures above: these two spellings ARE in BLIS's alias set, so
    neither can serve as the unrecognized case."""
    assert spelling in RECOGNIZED_EXPERT_COUNT_FIELDS
    cands = join_signals([sig("hf", arch="MysteryMoeForCausalLM", org="somelab",
                              model_ids=("somelab/mystery-671b",),
                              config=_moe_via_unrecognized_spelling(spelling))])
    evaluate(cands, surface(), cfg(), already_reported=never_reported)
    assert cands[0].silent_failures == []
    # And with nothing unparsed either, there is no T1 evidence at all — so a
    # correctly-spelled MoE config is not news, which is the whole point.
    assert cands[0].unparsed_fields == []
    assert cands[0].triggers == []


def test_silent_and_bucket0_failures_never_intersect():
    """Candidate.would_not_run derives from bucket0_failures. A leak would make the
    emitter present "BLIS runs and lies" as "BLIS refuses to run" — the exact confusion
    the split exists to prevent."""
    surf = surface(validator_failures=["hidden_act 'gelu' is not SwiGLU-family"])
    report = run([sig("hf", arch="MysteryMoeForCausalLM", org="somelab",
                      model_ids=("somelab/mystery-671b",),
                      config=_moe_via_unrecognized_spelling("num_moe_experts"))],
                 surf=surf)
    cand = report.passed[0]
    assert cand.bucket0_failures and cand.silent_failures
    assert not set(cand.bucket0_failures) & set(cand.silent_failures)
    assert cand.would_not_run is True


def test_no_finding_appears_in_both_lists_across_every_fixture():
    for name in ("dense_llama31_70b", "moe_qwen3_30b_a3b", "mla_moe_deepseek_v3",
                 "hybrid_mla_linear_attn", "novel_arch_large", "quant_plus_real_novelty",
                 "multimodal_llama4_scout", "textconfig_only_frontier"):
        cands = join_signals([sig("hf", arch="ProbeForCausalLM", org="x",
                                 model_ids=("x/probe",), config=load(name))])
        evaluate(cands, surface(), cfg(), already_reported=never_reported)
        assert not set(cands[0].bucket0_failures) & set(cands[0].silent_failures), name


def test_a_surface_that_double_classifies_is_reported_and_fatal_wins(caplog):
    """A contract violation must not be absorbed silently: would_not_run has to stay
    truthful, so the fatal classification wins and the violation is logged."""
    import logging

    shared = "some finding reported twice"
    surf = surface(validator_failures=[shared], silent_failures=[shared, "a real silent one"])
    with caplog.at_level(logging.WARNING, logger="archwatch.novelty"):
        report = run([sig("hf", arch="ProbeForCausalLM", org="x", model_ids=("x/probe",),
                          config=load("novel_arch_large"))], surf=surf)
    cand = report.passed[0]
    assert cand.bucket0_failures == [shared]
    assert cand.silent_failures == ["a real silent one"]
    assert any("contract violation" in r.getMessage() for r in caplog.records)


def test_silent_failures_are_empty_without_a_config():
    cand = run([sig("vllm", arch="NewThingForCausalLM", raw_ref="1")]).passed[0]
    assert cand.silent_failures == []
    assert cand.bucket0_failures == []


# --- T1 must fire on a silent misread, not just on an unparsed field -------
# A silent validator can fire with NO unparsed field: both configs below use only
# spellings BLIS recognizes, and are misread anyway. Gating T1 on unparsed_fields alone
# would drop these at no_trigger and lose the finding the pipeline exists to catch.


def _silently_misread_recognized_fields() -> dict:
    """num_experts_per_tok exceeds the total: recognized spellings, wrong numbers."""
    return {
        "architectures": ["QuietMoeForCausalLM"], "hidden_size": 7168,
        "num_hidden_layers": 61, "vocab_size": 129280, "num_attention_heads": 128,
        "intermediate_size": 18432, "moe_intermediate_size": 2048,
        "hidden_act": "silu", "tie_word_embeddings": False,
        "n_routed_experts": 8, "num_experts_per_tok": 16,
    }


def test_t1_fires_on_a_silent_misread_with_no_unparsed_field():
    surf = surface(silent_failures=["[moe_active_not_exceeding_total] active 16 > total 8"])
    report = run([sig("hf", arch="QuietMoeForCausalLM", org="somelab",
                      model_ids=("somelab/quiet-671b",),
                      config=_silently_misread_recognized_fields())], surf=surf)
    cand = report.passed[0]
    assert cand.unparsed_fields == [], "premise: nothing unparsed"
    assert cand.silent_failures, "premise: a silent finding exists"
    assert cand.triggers == ["T1"]


def test_a_known_architecture_that_now_silently_misreads_surfaces_under_recheck():
    """The compound case, and the worst one: an architecture BLIS believes it supports,
    whose config now makes it produce wrong numbers with no unparsed field to notice."""
    surf = surface(silent_failures=["[kv_head_count_unreadable] num_key_value_heads='8'"])
    signals = [sig("hf", arch="LlamaForCausalLM", org="meta-llama",
                   model_ids=("meta-llama/Llama-4-Scout",),
                   config=load("dense_llama31_70b"))]
    off = run(signals, surf=surf, conf=cfg(recheck_known_architectures=False))
    assert reasons(off) == ["known_architecture"]

    on = run(signals, surf=surf, conf=cfg(recheck_known_architectures=True))
    assert on.passed[0].triggers == [KNOWN_ARCH_TRIGGER]
    assert on.passed[0].silent_failures


def test_no_trigger_detail_names_both_kinds_of_t1_evidence():
    plain = {"architectures": ["QuietForCausalLM"], "hidden_size": 2048,
             "num_hidden_layers": 24, "vocab_size": 32000, "num_attention_heads": 16,
             "intermediate_size": 5632, "hidden_act": "silu",
             "tie_word_embeddings": False}
    report = run([sig("hf", arch="QuietForCausalLM", org="hobbyist",
                      model_ids=("hobbyist/quiet",), config=plain)])
    detail = report.dropped[0].detail
    assert "unparsed_fields=[]" in detail and "silent_failures=[]" in detail


# ---------------------------------------------------------------------------
# The curated-source exemption, as a class
# ---------------------------------------------------------------------------


def test_a_framework_pr_rescues_an_architecture_whose_only_hf_repos_are_repacks():
    """The bug this exemption fixes. If the only HF uploads for a new architecture are
    community quants while vLLM is adding support, the derivative suppressor dropped the
    candidate before triggers ran, so T2 and S3 never got a say."""
    signals = [
        sig("hf", arch="NewThingForCausalLM", org="bartowski",
            model_ids=("bartowski/NewThing-100B-GGUF",), config=load("novel_arch_large")),
        sig("hf", arch="NewThingForCausalLM", org="someone",
            model_ids=("someone/NewThing-100B-AWQ",)),
        sig("vllm", arch="NewThingForCausalLM", raw_ref="30012"),
    ]
    report = run(signals)
    assert [c.arch_id for c in report.passed] == ["NewThingForCausalLM"]
    cand = report.passed[0]
    assert "T2" in cand.triggers and "T4" in cand.triggers
    assert "S3" in cand.significance


def test_a_benchmark_row_also_grants_the_exemption():
    signals = [
        sig("hf", arch="NewThingForCausalLM", org="bartowski",
            model_ids=("bartowski/NewThing-100B-GGUF",), config=load("novel_arch_large")),
        sig("inferencex", arch="NewThingForCausalLM", raw_ref="cafe"),
    ]
    assert [c.arch_id for c in run(signals).passed] == ["NewThingForCausalLM"]


def test_without_a_curated_source_the_repack_suppressor_still_fires():
    signals = [
        sig("hf", arch="NewThingForCausalLM", org="bartowski",
            model_ids=("bartowski/NewThing-100B-GGUF",), config=load("novel_arch_large")),
        sig("hf", arch="NewThingForCausalLM", org="someone",
            model_ids=("someone/NewThing-100B-AWQ",)),
    ]
    assert reasons(run(signals)) == ["all_model_ids_derivative"]


def test_the_exemption_is_logged(caplog):
    import logging

    signals = [
        sig("hf", arch="NewThingForCausalLM", org="bartowski",
            model_ids=("bartowski/NewThing-100B-GGUF",), config=load("novel_arch_large")),
        sig("vllm", arch="NewThingForCausalLM", raw_ref="30012"),
    ]
    with caplog.at_level(logging.INFO, logger="archwatch.novelty"):
        run(signals)
    lines = [r.getMessage() for r in caplog.records if "exempting" in r.getMessage()]
    assert len(lines) == 1
    assert "all_model_ids_derivative" in lines[0]
    assert "vllm" in lines[0]


def test_the_exemption_does_not_bypass_the_non_hf_noise_suppressors():
    """It is scoped to the HF-noise class. Known-architecture, dedup and quant-repack
    suppressors are about the architecture itself and still apply."""
    known = run([sig("vllm", arch="LlamaForCausalLM", raw_ref="1")], conf=strict_cfg())
    assert reasons(known) == ["known_architecture"]

    repack = run([
        sig("hf", arch="AwqWrappedForCausalLM", org="wrapper",
            model_ids=("wrapper/llama-70b-w4",), config=load("quant_repack_known_family")),
        sig("vllm", arch="AwqWrappedForCausalLM", raw_ref="2"),
    ])
    assert reasons(repack) == ["structurally_identical"]


# ---------------------------------------------------------------------------
# Bucket 0 polarity: "BLIS refuses this" can mean "never a language model"
# ---------------------------------------------------------------------------
# A live HuggingFace run put five non-language models at the top — VibeVoice (ASR),
# MiniMaxMusic3, PiiMasking, Sam3Video, Wav2Vec2 — every one ranked first *because* it
# was bucket 0, consuming the entire per-run cap while twelve other candidates were
# dropped as over_cap. For a speech or video repo, bucket 0 means the config was never
# a transformer LM.

NON_LM_NO_SHAPE_FIXTURES = [
    "nonlm_sam3_video",
    "nonlm_vibevoice_asr",
    "nonlm_music_conditional",
]


@pytest.mark.parametrize("name", NON_LM_NO_SHAPE_FIXTURES)
def test_lm_shape_evidence_finds_no_core_dimensions_in_a_non_lm_config(name):
    ev = lm_shape_evidence(load(name))
    assert ev.missing == [
        "num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size",
    ]
    assert ev.probably_not_lm
    assert not ev.looks_like_lm
    assert ev.vocab_size is None


@pytest.mark.parametrize("name", NON_LM_NO_SHAPE_FIXTURES)
def test_a_non_lm_bucket0_candidate_is_suppressed(name):
    surf = surface(validator_failures=["num_hidden_layers must be > 0",
                                       "hidden_size must be > 0"])
    report = run([sig("hf", arch=load(name)["architectures"][0], org="somelab",
                      model_ids=(f"somelab/{name}",), config=load(name),
                      extra={"downloads": 500_000, "likes": 900})], surf=surf)
    assert report.passed == []
    assert reasons(report) == ["not_a_language_model"]
    assert "not a transformer LM, not because it is novel" in report.dropped[0].detail


def test_vela_lumen_is_a_real_lm_with_nonstandard_names_and_must_not_be_suppressed():
    """The counter-case that constrains the whole rule. ParallaxOpen/Vela-Lumen-31M
    declares ``model_type: small_lm``, no ``architectures[]``, and names its dimensions
    ``d_model``/``n_layers``/``n_heads``/``ffn_dim``. It trips several bucket-0 failures
    and is still a genuine language model — a false suppression here is unrecoverable.
    """
    ev = lm_shape_evidence(load("nonstandard_field_names"))
    assert ev.missing == [], ev.describe()
    assert ev.looks_like_lm and ev.vocab_plausible
    assert ev.present["hidden_size"] == "d_model"
    assert ev.present["num_hidden_layers"] == "n_layers"
    assert ev.present["num_attention_heads"] == "n_heads"
    assert ev.present["intermediate_size"] == "ffn_dim"

    surf = surface(validator_failures=["num_hidden_layers must be > 0",
                                       "hidden_size must be > 0",
                                       "num_attention_heads must be > 0"])
    report = run([sig("hf", display_name="ParallaxOpen/Vela-Lumen-31M",
                      model_ids=("ParallaxOpen/Vela-Lumen-31M",), org="parallaxopen",
                      config=load("nonstandard_field_names"),
                      extra={"downloads": 50_000})], surf=surf)
    assert "not_a_language_model" not in reasons(report)
    assert [c.arch_id for c in report.passed] == ["Vela-Lumen-31M"]


def test_a_real_lm_that_blis_refuses_is_still_suppressed_by_nothing_and_ranked_up():
    """The other half of the split: every core dimension present and well-formed, but a
    non-SwiGLU activation makes BLIS abort. A real LM someone will want to run."""
    cfgj = dict(load("novel_arch_large"), hidden_act="gelu")
    surf = surface(validator_failures=["unsupported activation 'gelu'; only SwiGLU-family"])
    report = run([sig("hf", arch="GriffinMoeForCausalLM", org="someuni",
                      model_ids=("someuni/griffin-158b",), config=cfgj)], surf=surf)
    cand = report.passed[0]
    assert cand.would_not_run
    assert _strength(cand, lm_shape_evidence(cfgj)) > _strength(cand, None)


def test_an_implausible_vocabulary_withholds_the_bucket0_rank_bonus():
    """Wav2Vec2 keeps all four core dimensions, so shape cannot separate it from an LM —
    but a 32-entry "vocabulary" is a phoneme set, not a tokenizer. It is not suppressed
    (a byte-level LM legitimately has ~256 tokens), it just earns no bonus."""
    ev = lm_shape_evidence(load("nonlm_wav2vec2_pretraining"))
    assert ev.looks_like_lm, "shape alone cannot rule this out"
    assert not ev.vocab_plausible
    assert ev.vocab_size == 32

    cand = Candidate(arch_id="Wav2Vec2ForPreTraining", display_name="w2v",
                     signals=[sig("hf", arch="Wav2Vec2ForPreTraining")],
                     triggers=["T1"], significance=["S4"],
                     bucket0_failures=["unsupported activation 'gelu'"])
    assert _strength(cand, ev) == _strength(cand, None), "no bucket-0 bonus"


def test_silently_wrong_outranks_a_plain_would_not_run():
    """The class that justifies this pipeline previously got no bonus while the loud
    class got three. BLIS aborting is visible; BLIS lying is not."""
    base = dict(arch_id="X", display_name="X", signals=[sig("hf", arch="X")],
                triggers=["T1"], significance=["S1"])
    quiet = Candidate(**base, silent_failures=["a sparse MoE simulates as dense"])
    loud = Candidate(**base, bucket0_failures=["unsupported activation 'gelu'"])
    lm = lm_shape_evidence(load("dense_llama31_70b"))
    assert _strength(quiet, lm) > _strength(loud, lm)
    assert quiet.would_not_run is False and loud.would_not_run is True


def test_a_bucket0_finding_alongside_silent_ones_is_ranked_as_bucket0():
    """"Clean bucket 0" is part of the silently_wrong definition: once BLIS aborts, the
    loud problem is the one to fix first."""
    both = Candidate(arch_id="X", display_name="X", signals=[sig("hf", arch="X")],
                     triggers=["T1"], significance=["S1"],
                     bucket0_failures=["vocab_size must be > 0"],
                     silent_failures=["a sparse MoE simulates as dense"])
    lm = lm_shape_evidence(load("dense_llama31_70b"))
    assert _strength(both, lm) < _strength(
        Candidate(arch_id="X", display_name="X", signals=[sig("hf", arch="X")],
                  triggers=["T1"], significance=["S1"],
                  silent_failures=["a sparse MoE simulates as dense"]), lm)


def test_audio_and_video_repos_no_longer_crowd_out_a_frontier_release():
    """The live regression, reproduced. All five non-LM repos are given enough popularity
    to clear the significance gate; the frontier release must still make the cap."""

    class NonLmBucket0(FakeSurface):
        def check_hard_validators(self, config):
            arch = (config.get("architectures") or [""])[0]
            if arch.startswith(("Sam3", "VibeVoice", "MiniMaxMusic", "Wav2Vec2", "PiiMasking")):
                return ["num_hidden_layers must be > 0", "hidden_size must be > 0"]
            return []

    noise = [
        sig("hf", arch=load(n)["architectures"][0], org="somelab",
            model_ids=(f"somelab/{n}",), config=load(n),
            extra={"downloads": 900_000, "likes": 800})
        for n in NON_LM_NO_SHAPE_FIXTURES
        + ["nonlm_wav2vec2_pretraining", "nonlm_pii_masking_encoder"]
    ]
    frontier = sig("hf", arch="GriffinMoeForCausalLM", org="deepseek-ai",
                   model_ids=("deepseek-ai/Griffin-158B",),
                   config=load("novel_arch_large"))
    report = run(noise + [frontier], surf=NonLmBucket0(known=SEED_SET),
                 conf=cfg(max_issues_per_run=3))
    assert "GriffinMoeForCausalLM" in [c.arch_id for c in report.passed]
    assert reasons(report).count("not_a_language_model") == 3


def test_lm_shape_evidence_reads_the_text_tower_of_a_multimodal_config():
    ev = lm_shape_evidence(load("textconfig_only_frontier"))
    assert ev.looks_like_lm and ev.vocab_plausible


def test_lm_shape_evidence_tolerates_a_missing_config():
    ev = lm_shape_evidence(None)
    assert ev.probably_not_lm and ev.vocab_size is None


def test_a_non_lm_config_that_blis_would_run_is_not_touched_by_this_suppressor():
    """The suppressor is bucket-0-gated. Without a bucket-0 failure there is no evidence
    to reinterpret, so the ordinary triggers and gate decide."""
    report = run([sig("hf", arch="Sam3VideoModel", org="somelab",
                      model_ids=("somelab/sam3",), config=load("nonlm_sam3_video"),
                      extra={"downloads": 900_000})])
    assert "not_a_language_model" not in reasons(report)


# ---------------------------------------------------------------------------
# framework_no_architecture: a PR title is not a model
# ---------------------------------------------------------------------------


def _no_arch_pr(title: str, ref: str = "1", source: str = "vllm",
                strength: str = "title_only") -> Signal:
    """A framework PR from which no architecture name could be extracted."""
    return sig(source, display_name=title, raw_ref=ref,
               extra={"signal_strength": strength, "pr_title": title})


@pytest.mark.parametrize(
    "title,strength",
    [
        # The first three became actual stub filenames on a live run. Note the strength:
        # they were new_model_file, NOT title_only, which is exactly why keying on the
        # connector's strength label failed to catch them.
        ("Find attention with a fuser and attach vLLM's layer to it", "new_model_file"),
        ("gfx1250 on ROCM 10", "new_model_file"),
        ("video embeds input", "new_model_file"),
        ("[Model] Support Kimi K3 (follow-up)", "title_only"),
        ("[Model] Add something", "prose"),
    ],
)
def test_a_framework_pr_yielding_no_architecture_is_suppressed(title, strength):
    report = run([_no_arch_pr(title, strength=strength)])
    assert report.passed == []
    assert reasons(report) == ["framework_no_architecture"]
    assert "yielding no architecture id" in report.dropped[0].detail
    assert strength in report.dropped[0].detail, "strength kept for diagnostics"


@pytest.mark.parametrize("strength", ["registry", "new_model_file", "model_class",
                                      "prose", "title_only"])
def test_an_architecture_id_rescues_a_framework_pr_not_its_strength(strength):
    """The corrected rule. A strength label says how hard the connector looked, not
    whether it found anything; only an architecture name earns a stub."""
    report = run([sig("vllm", arch="K2HorizonForCausalLM", raw_ref="1",
                      display_name="[Model] Add K2Horizon",
                      extra={"signal_strength": strength})])
    assert [c.arch_id for c in report.passed] == ["K2HorizonForCausalLM"]
    assert report.passed[0].triggers == ["T2"]


def test_one_arch_bearing_signal_rescues_a_candidate_joined_with_nameless_prs():
    strong = sig("vllm", arch="K2HorizonForCausalLM", raw_ref="2",
                 model_ids=("moonshot/K2-Horizon",), extra={"signal_strength": "registry"})
    weak = _no_arch_pr("K2Horizon", ref="3")
    weak.model_ids = ["moonshot/K2-Horizon"]  # joins on the repo edge
    report = run([weak, strong])
    assert [c.arch_id for c in report.passed] == ["K2HorizonForCausalLM"]


def test_a_nameless_pr_does_not_suppress_a_candidate_with_an_hf_config():
    """Vela-shaped guard: a real model with no ``architectures[]`` also takes the alias
    path, so the rule requires that EVERY signal be a title_only framework signal."""
    weak = _no_arch_pr("Vela-Lumen-31M", ref="9")
    weak.model_ids = ["ParallaxOpen/Vela-Lumen-31M"]  # connector mines ids from PR bodies
    report = run(
        [
            weak,
            sig("hf", display_name="ParallaxOpen/Vela-Lumen-31M",
                model_ids=("ParallaxOpen/Vela-Lumen-31M",), org="parallaxopen",
                config=load("nonstandard_field_names"), extra={"downloads": 50_000}),
        ]
    )
    assert len(report.passed) + len(report.dropped) == 1, "premise: the two must merge"
    assert "framework_no_architecture" not in reasons(report)
    assert [c.arch_id for c in report.passed] == ["Vela-Lumen-31M"]


def test_a_nameless_pr_does_not_suppress_a_candidate_with_a_benchmark_row():
    report = run([_no_arch_pr("Nova-1", ref="9"),
                  sig("inferencex", display_name="Nova-1", raw_ref="beef")])
    assert "framework_no_architecture" not in reasons(report)


def test_the_rule_does_not_depend_on_strength_metadata_being_present():
    """A connector that omits the strength key must behave the same: the test is whether
    an architecture id exists, which needs no metadata at all."""
    with_arch = run([sig("vllm", arch="KimiK3ForCausalLM", raw_ref="1")])
    assert [c.arch_id for c in with_arch.passed] == ["KimiK3ForCausalLM"]

    without_arch = run([sig("vllm", display_name="Kimi-K3", raw_ref="1")])
    assert reasons(without_arch) == ["framework_no_architecture"]


# ---------------------------------------------------------------------------
# One signal naming N architectures must not merge them
# ---------------------------------------------------------------------------
# Union-find membership is all-or-nothing, so a signal offering an arch edge per name
# becomes a member of every one of those groups — which makes the groups one. SGLang PR
# #35634 ("Add DeepEPv2 MoE A2A backend") is a backend change that mines four
# architecture names, and it fused DeepSeek-V3 + DeepSeek-V4 + Qwen3-MoE + Qwen3.5-MoE
# into a single top-ranked candidate whose findings described a model that never existed.

SGLANG_35634_ARCHS = [
    "DeepseekV3ForCausalLM",
    "DeepseekV4ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen35MoeForCausalLM",
]


def test_a_backend_pr_mining_four_architectures_does_not_fuse_them():
    """The exact live false merge, as a regression test."""
    backend_pr = sig("sglang", arch=SGLANG_35634_ARCHS, raw_ref="35634",
                     display_name="Add DeepEPv2 MoE A2A backend")
    real = [
        sig("hf", arch="DeepseekV4ForCausalLM", org="deepseek-ai",
            model_ids=("deepseek-ai/DeepSeek-V4",), config=load("mla_moe_deepseek_v3")),
        sig("hf", arch="Qwen3MoeForCausalLM", org="qwen",
            model_ids=("Qwen/Qwen3-30B-A3B",), config=load("moe_qwen3_30b_a3b")),
        sig("hf", arch="Qwen35MoeForCausalLM", org="qwen",
            model_ids=("Qwen/Qwen3.5-300B-A30B",), config=load("moe_qwen3_30b_a3b")),
    ]
    cands = join_signals([backend_pr] + real)
    by_arch = {c.arch_id: c for c in cands}
    assert set(by_arch) == {
        "DeepseekV3ForCausalLM", "DeepseekV4ForCausalLM",
        "Qwen3MoeForCausalLM", "Qwen35MoeForCausalLM",
    }, "the four architectures must stay four candidates"
    for arch, cand in by_arch.items():
        assert len(cand.signals) <= 2, f"{arch} absorbed unrelated signals"
    assert by_arch["Qwen3MoeForCausalLM"].sources == ["hf"]
    assert by_arch["Qwen35MoeForCausalLM"].sources == ["hf"]


def test_the_join_identity_is_the_primary_architectures_family():
    coherent, ignored = coherent_arch_ids(
        sig("sglang", arch=SGLANG_35634_ARCHS, raw_ref="35634"))
    assert coherent == ["DeepseekV3ForCausalLM"]
    assert ignored == SGLANG_35634_ARCHS[1:]


def test_variant_spellings_within_one_family_are_still_join_evidence():
    """The intent the old code had right: an MTP head ships with its release."""
    coherent, ignored = coherent_arch_ids(
        sig("hf", arch=["KimiK3ForCausalLM", "KimiK3MTPModel"]))
    assert coherent == ["KimiK3ForCausalLM", "KimiK3MTPModel"]
    assert ignored == []
    cands = join_signals(
        [
            sig("hf", arch=["KimiK3ForCausalLM", "KimiK3MTPModel"], model_ids=("m/k3",)),
            sig("vllm", arch=["KimiK3MTPModel"], raw_ref="1"),
        ]
    )
    assert len(cands) == 1


def test_a_signal_whose_primary_family_key_is_unusable_joins_on_the_primary_alone():
    """With no verifiable family key, nothing may bridge on the strength of the name."""
    coherent, ignored = coherent_arch_ids(sig("vllm", arch=["V3", "Qwen3MoeForCausalLM"]))
    assert coherent == ["V3"]
    assert ignored == ["Qwen3MoeForCausalLM"]


def test_an_arch_bearing_signals_title_cannot_bridge_to_another_model():
    """When a signal has an architecture, that is its identity; the display name is a
    title. A PR about DeepEP whose title mentions Kimi must not join Kimi."""
    edges = signal_edges(sig("sglang", arch="DeepseekV3ForCausalLM",
                             display_name="Kimi-K3 kernels for DeepEP", raw_ref="1"))
    assert ("family", "kimik3") not in edges
    assert ("family", "deepseekv3") in edges


def test_a_nameless_signal_still_uses_its_display_name_as_family():
    edges = signal_edges(sig("inferencex", display_name="Kimi-K3", raw_ref="1"))
    assert ("family", "kimik3") in edges


# ---------------------------------------------------------------------------
# The family-key floor
# ---------------------------------------------------------------------------


def test_the_family_key_floor_admits_every_real_frontier_family():
    """The floor is capped by real data, not chosen freely: ``dsv4`` is four characters,
    so five would break a genuine join. Raising MIN_FAMILY_KEY_LEN above 4 must fail
    here rather than silently stop merging DeepSeek-V4."""
    for key in ("kimik3", "glm52", "dsv4", "minimaxm3", "qwen35"):
        assert len(key) >= MIN_FAMILY_KEY_LEN, f"{key} would be excluded by the floor"


def test_a_three_character_family_key_is_no_longer_evidence():
    """``family:asd`` fused MinistralForCausalLM with Qwen3.5-MoE and Bittensor spam."""
    assert MIN_FAMILY_KEY_LEN > 3
    assert normalize_family_key("asd") == ""
    assert normalize_family_key("ASD") == ""
    cands = join_signals(
        [
            sig("hf", display_name="asd", model_ids=("ministral/Ministral-8B",)),
            sig("hf", display_name="ASD", model_ids=("qwen/Qwen3.5-MoE",)),
            sig("hf", display_name="asd", model_ids=("spamlab/bittensor-thing",)),
        ]
    )
    assert len(cands) == 3, "a three-character key must not fuse three unrelated repos"


def test_real_frontier_family_keys_still_join():
    for display, arch in [
        ("Kimi-K3", "KimiK3ForCausalLM"),
        ("GLM-5.2", "Glm52ForCausalLM"),
        ("DSv4", "DSv4ForCausalLM"),
        ("MiniMax-M3", "MiniMaxM3ForCausalLM"),
        ("Qwen3.5", "Qwen35ForCausalLM"),
    ]:
        cands = join_signals([sig("inferencex", display_name=display, raw_ref="1"),
                              sig("vllm", arch=arch, raw_ref="2")])
        assert len(cands) == 1, f"{display} no longer joins {arch}"


# ---------------------------------------------------------------------------
# A successful join must not destroy a signal
# ---------------------------------------------------------------------------


def _kimi_hf_plus_benchmark() -> list[Signal]:
    """InferenceX's ``Kimi-K3`` joined to HF's seeded ``KimiK3ForCausalLM``."""
    return [
        sig("hf", arch="KimiK3ForCausalLM", org="moonshotai",
            model_ids=("moonshotai/Kimi-K3-Instruct",), config=load("dense_llama31_70b")),
        sig("inferencex", display_name="Kimi-K3", org="moonshotai",
            model_ids=("moonshotai/Kimi-K3",), raw_ref="cafe"),
    ]


def test_a_benchmark_signal_survives_being_joined_to_a_seeded_architecture():
    """Corroboration must strengthen a candidate, never suppress it. Alone, the
    InferenceX row passes on T5; joined to HF it adopted the seeded architecture name and
    died at known_architecture — so the join destroyed the signal."""
    surf = surface(known=SEED_SET + ("KimiK3ForCausalLM",))
    alone = run([_kimi_hf_plus_benchmark()[1]], surf=surf, conf=strict_cfg())
    # Keyed "Kimi-K3" on its own, so the seed-set name is not even reached; T3 also
    # fires because moonshotai is a frontier org. What matters is that T5 is there.
    assert "T5" in alone.passed[0].triggers
    assert ALIAS_JOIN_MARKER in alone.passed[0].triggers

    joined = run(_kimi_hf_plus_benchmark(), surf=surf, conf=strict_cfg())
    assert len(joined.passed) == 1, reasons(joined)
    cand = joined.passed[0]
    assert cand.arch_id == "KimiK3ForCausalLM"
    assert cand.triggers == ["T5"]
    assert "S3" in cand.significance
    assert cand.sources == ["hf", "inferencex"]


def test_the_benchmark_exemption_is_logged(caplog):
    import logging

    surf = surface(known=SEED_SET + ("KimiK3ForCausalLM",))
    with caplog.at_level(logging.INFO, logger="archwatch.novelty"):
        run(_kimi_hf_plus_benchmark(), surf=surf, conf=strict_cfg())
    lines = [r.getMessage() for r in caplog.records
             if "exempting" in r.getMessage() and "known_architecture" in r.getMessage()]
    assert len(lines) == 1
    assert "inferencex" in lines[0]


def test_hf_only_and_framework_only_candidates_are_still_dropped_as_known():
    """The exemption is scoped to benchmark sources: a vLLM PR against an architecture
    vLLM already supports is not news, and neither is another HF fine-tune."""
    surf = surface(known=SEED_SET + ("KimiK3ForCausalLM",))
    hf_only = run([_kimi_hf_plus_benchmark()[0]], surf=surf, conf=strict_cfg())
    assert reasons(hf_only) == ["known_architecture"]

    fw_only = run([sig("vllm", arch="KimiK3ForCausalLM", raw_ref="1")],
                  surf=surf, conf=strict_cfg())
    assert reasons(fw_only) == ["known_architecture"]


def test_dedup_still_guards_repeats_for_a_benchmarked_known_architecture(tmp_path):
    """``already_reported`` is what stops a benchmarked release being re-emitted every
    run — the exemption must not bypass it."""
    from archwatch.emitter import issue_path

    issue_path("KimiK3ForCausalLM", tmp_path).write_text("# already reported\n")
    report = evaluate_detailed(
        join_signals(_kimi_hf_plus_benchmark()),
        surface(known=SEED_SET + ("KimiK3ForCausalLM",)),
        strict_cfg(), issues_dir=tmp_path,
    )
    assert reasons(report) == ["already_reported"]


def test_t5_fires_for_a_benchmarked_known_architecture_under_recheck_too():
    surf = surface(known=SEED_SET + ("KimiK3ForCausalLM",))
    report = run(_kimi_hf_plus_benchmark(), surf=surf,
                 conf=cfg(recheck_known_architectures=True))
    assert "T5" in report.passed[0].triggers
