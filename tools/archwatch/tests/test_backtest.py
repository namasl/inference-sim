"""Unit tests for the validation harness's own pure logic.

``tests/backtest.py`` makes live calls and is a script, not a test. What IS testable
offline is the reasoning it does *about* the results: front-matter parsing, the
size-token extractor, the false-merge audit, and the zero-day counterfactual. Those are
what a wrong VALIDATION.md conclusion would come from, so they are pinned here.

No network, per PLAN.md hard rule 3.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from archwatch import emitter
from archwatch.connectors.base import Candidate, Signal
from archwatch.novelty import join_signals
from archwatch.surface import Surface
from tests.backtest import (
    REQUIRED_FM_KEYS,
    SWEEP_PARAMS,
    TARGETS,
    audit_merge,
    front_matter,
    measure_cfg,
    missing_front_matter_keys,
    real_triggers,
    signal_from_dict,
    signal_to_dict,
    silently_wrong,
    size_tokens,
    surface_without,
)

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def hf_signal(repo_id: str, *archs: str, **extra_kw) -> Signal:
    org = repo_id.split("/")[0].lower() if "/" in repo_id else None
    return Signal(
        source="hf",
        observed_at=NOW,
        arch_ids=list(archs),
        model_ids=[repo_id],
        org=org,
        display_name=repo_id.split("/")[-1],
        config={"architectures": list(archs)} if archs else None,
        extra=extra_kw,
    )


# ---------------------------------------------------------------------------
# front matter — addendum 22: keys, never the version string
# ---------------------------------------------------------------------------


def _cand(**kw) -> Candidate:
    base = dict(arch_id="FooForCausalLM", display_name="Foo", signals=[hf_signal("acme/foo", "FooForCausalLM")])
    base.update(kw)
    return Candidate(**base)


def test_front_matter_round_trips_a_real_rendered_stub():
    """The parser must agree with the emitter that actually writes the stubs."""
    cand = _cand(triggers=["T1"], significance=["S1"], unparsed_fields=["mystery_field"])
    fm = front_matter(emitter.render(cand, detected_at=NOW))
    assert fm["arch_id"] == "FooForCausalLM"
    assert fm["triggers"] == ["T1"]
    assert missing_front_matter_keys(fm) == []


def test_front_matter_does_not_care_which_schema_version():
    """Addendum 22. The emitter already ships archwatch/3 where the plan said /2."""
    cand = _cand()
    text = emitter.render(cand, detected_at=NOW)
    bumped = re.sub(r"schema: archwatch/\d+", "schema: archwatch/99", text, count=1)
    assert missing_front_matter_keys(front_matter(bumped)) == []
    assert front_matter(bumped)["schema"] == "archwatch/99"


def test_front_matter_reports_every_missing_required_key():
    assert set(missing_front_matter_keys({})) == set(REQUIRED_FM_KEYS)


def test_front_matter_of_text_without_fences_is_empty():
    assert front_matter("# just a heading\n") == {}
    assert front_matter("---\nno closing fence\n") == {}


@pytest.mark.parametrize(
    "silent,fatal,expected",
    [
        ([], [], False),
        (["kv heads unreadable"], [], True),          # the headline class
        (["kv heads unreadable"], ["dtype"], False),  # loud, so not silently wrong
        ([], ["dtype"], False),
    ],
)
def test_silently_wrong_matches_the_emitters_derivation(silent, fatal, expected):
    """Addendum 21: silently_wrong == silent_failures and not bucket0_failures."""
    cand = _cand(silent_failures=silent, bucket0_failures=fatal)
    assert silently_wrong(cand) is expected
    assert bool(front_matter(emitter.render(cand, detected_at=NOW))["silently_wrong"]) is expected


# ---------------------------------------------------------------------------
# size tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Qwen/Qwen3.5-397B-A17B", {"397b", "17b"}),   # active-param suffix counts
        ("Qwen/Qwen3.5-122B-A10B", {"122b", "10b"}),
        ("mistralai/Mixtral-8x7B-v0.1", {"8x7b"}),
        ("zai-org/GLM-5.2", set()),                    # version number is not a size
        ("moonshotai/Kimi-K3", set()),
        ("acme/model-1.5b-chat", {"1.5b"}),
        ("acme/beta5b-oops", set()),                   # 'b' inside a word is not a token
    ],
)
def test_size_tokens(name, expected):
    assert size_tokens(name) == expected


# ---------------------------------------------------------------------------
# false-merge audit
# ---------------------------------------------------------------------------


def test_unmerged_candidate_is_never_suspicious():
    cand = join_signals([hf_signal("acme/solo", "SoloForCausalLM")])[0]
    audit = audit_merge(cand)
    assert not audit.merged
    assert not audit.suspicious
    assert audit.reasons == []


def test_same_architecture_across_repos_merges_without_suspicion():
    """The routine case: two repos of one release. Nothing to flag."""
    cands = join_signals([
        hf_signal("acme/Model-Instruct", "AcmeForCausalLM"),
        hf_signal("acme/Model-Base", "AcmeForCausalLM"),
    ])
    assert len(cands) == 1
    assert audit_merge(cands[0]).merged
    assert not audit_merge(cands[0]).suspicious


def test_multi_architecture_signal_fuses_unrelated_families_and_is_flagged():
    """The live false merge this audit was written to catch.

    SGLang PR #35634 ("Add DeepEPv2 MoE A2A backend") is a *backend* change whose prose
    names four unrelated architectures. ``signal_edges`` emits an ``arch:`` edge for
    every ``arch_ids`` entry, so that one signal is a clique joining DeepSeek V3,
    DeepSeek V4 and two Qwen families into a single Candidate.
    """
    bridge = Signal(
        source="sglang",
        observed_at=NOW,
        arch_ids=["DeepseekV3ForCausalLM", "DeepseekV4ForCausalLM",
                  "Qwen3MoeForCausalLM", "Qwen3_5MoeForCausalLM"],
        display_name="DeepEPv2 (ElasticBuffer) MoE A2A backend",
        raw_ref="35634",
    )
    cands = join_signals([
        hf_signal("deepseek-ai/DeepSeek-V4-Pro", "DeepseekV4ForCausalLM"),
        hf_signal("Qwen/Qwen3.5-397B-A17B", "Qwen3_5MoeForCausalLM"),
        bridge,
    ])
    assert len(cands) == 1, "the bridging signal fuses DeepSeek and Qwen into one candidate"
    audit = audit_merge(cands[0])
    assert audit.suspicious
    assert any("distinct architecture spellings" in r for r in audit.reasons)


def test_different_size_tokens_fused_is_flagged():
    small = hf_signal("acme/Thing-9B", "ThingForCausalLM")
    big = hf_signal("acme/Thing-397B", "ThingForCausalLM")
    cands = join_signals([small, big])
    assert len(cands) == 1
    audit = audit_merge(cands[0])
    assert audit.suspicious
    assert any("parameter-size tokens" in r for r in audit.reasons)


def test_family_only_merge_across_orgs_is_flagged():
    """Two labs' identically named models can only meet on a bare family edge."""
    a = Signal(source="inferencex", observed_at=NOW, display_name="Falcon-H1", raw_ref="a")
    b = Signal(source="vllm", observed_at=NOW, display_name="Falcon-H1", raw_ref="b",
               arch_ids=[])
    b.org = "tiiuae"
    a.org = "someone-else"
    cands = join_signals([a, b])
    assert len(cands) == 1
    audit = audit_merge(cands[0])
    assert audit.family_only
    assert audit.suspicious
    assert any("orgs fused on a family edge" in r for r in audit.reasons)


def test_audit_records_the_edges_the_emitter_will_publish():
    cands = join_signals([
        hf_signal("acme/Model-Instruct", "AcmeForCausalLM"),
        hf_signal("acme/Model-Base", "AcmeForCausalLM"),
    ])
    audit = audit_merge(cands[0])
    fm = front_matter(emitter.render(cands[0], detected_at=NOW))
    assert audit.join_edges == fm["join_edges"]


# ---------------------------------------------------------------------------
# zero-day counterfactual
# ---------------------------------------------------------------------------


def test_surface_without_forgets_only_the_named_architectures():
    surface = Surface(known_architectures={"fooforcausallm", "barforcausallm"})
    stripped = surface_without(surface, ["FooForCausalLM"])
    assert not stripped.is_known_architecture("FooForCausalLM")
    assert stripped.is_known_architecture("BarForCausalLM")
    # The original must be untouched: the arms are compared against each other.
    assert surface.is_known_architecture("FooForCausalLM")


def test_surface_without_is_case_and_whitespace_insensitive():
    surface = Surface(known_architectures={"fooforcausallm"})
    assert not surface_without(surface, ["  FOOForCausalLM "]).is_known_architecture("FooForCausalLM")


def test_surface_without_ignores_blank_names():
    surface = Surface(known_architectures={"fooforcausallm"})
    assert surface_without(surface, ["", "   ", None]).is_known_architecture("FooForCausalLM")


# ---------------------------------------------------------------------------
# measurement config
# ---------------------------------------------------------------------------


def test_measure_cfg_changes_only_what_is_swept():
    from archwatch.config import DEFAULTS

    cfg = measure_cfg(min_total_params=7_000_000_000, recheck=True)
    assert cfg.thresholds.min_total_params == 7_000_000_000
    assert cfg.recheck_known_architectures is True
    # Everything else stays at the shipped value, or the sweep measures two changes.
    assert cfg.thresholds.min_org_top_downloads == DEFAULTS.thresholds.min_org_top_downloads
    assert cfg.thresholds.min_model_downloads == DEFAULTS.thresholds.min_model_downloads
    assert cfg.thresholds.min_model_likes == DEFAULTS.thresholds.min_model_likes
    assert cfg.max_hf_config_fetches == DEFAULTS.max_hf_config_fetches
    assert cfg.frontier_orgs == DEFAULTS.frontier_orgs


def test_measure_cfg_defaults_match_the_shipped_config():
    from archwatch.config import DEFAULTS

    cfg = measure_cfg()
    assert cfg.thresholds.min_total_params == DEFAULTS.thresholds.min_total_params
    assert cfg.recheck_known_architectures == DEFAULTS.recheck_known_architectures


def test_measure_cap_is_large_enough_not_to_mask_volume():
    """Recall and survivor volume must be measured uncapped, or they measure the cap."""
    from archwatch.config import DEFAULTS

    assert measure_cfg().max_issues_per_run > 20 * DEFAULTS.max_issues_per_run


def test_sweep_brackets_the_shipped_threshold():
    from archwatch.config import DEFAULTS

    shipped = DEFAULTS.thresholds.min_total_params
    assert min(SWEEP_PARAMS) < shipped < max(SWEEP_PARAMS)
    assert shipped in SWEEP_PARAMS


# ---------------------------------------------------------------------------
# targets and trigger bookkeeping
# ---------------------------------------------------------------------------


def test_target_list_covers_every_release_plan_j_names():
    labels = " ".join(t.label for t in TARGETS).lower()
    for wanted in ("kimi k2", "kimi k3", "deepseek v3", "deepseek v4",
                   "glm-5", "minimax m3", "qwen3.5"):
        assert wanted in labels, wanted
    assert sum(1 for t in TARGETS if t.kind == "seeded") >= 3, (
        "PLAN.md J asks for 2-3 older validated models as controls"
    )


def test_target_repo_ids_are_org_qualified_and_unique():
    seen: set[str] = set()
    for target in TARGETS:
        assert target.repo_ids, target.label
        for repo_id in target.repo_ids:
            assert "/" in repo_id, repo_id
            assert repo_id not in seen, f"duplicate target repo {repo_id}"
            seen.add(repo_id)


def test_real_triggers_drops_provenance_markers():
    cand = _cand(triggers=["T1", "alias-join", "T4"])
    assert real_triggers(cand) == ["T1", "T4"]


# ---------------------------------------------------------------------------
# signal caching — one live poll must feed many evaluate() passes unchanged
# ---------------------------------------------------------------------------


def test_signal_round_trip_preserves_everything_the_filter_reads():
    sig = hf_signal("acme/Thing-9B", "ThingForCausalLM", downloads=1234, likes=7,
                    trending=True)
    sig.model_type = "thing"
    sig.urls = {"hf": "https://huggingface.co/acme/Thing-9B"}
    sig.evidence = "new repo"
    sig.raw_ref = "acme/Thing-9B"
    back = signal_from_dict(signal_to_dict(sig))
    for attr in ("source", "arch_ids", "model_type", "model_ids", "org",
                 "display_name", "config", "urls", "evidence", "raw_ref", "extra"):
        assert getattr(back, attr) == getattr(sig, attr), attr
    assert back.observed_at == sig.observed_at


def test_signal_round_trip_makes_naive_timestamps_utc():
    """``observed_at`` is tz-aware UTC (addendum 2); the cache must not lose that."""
    sig = hf_signal("acme/x", "XForCausalLM")
    payload = signal_to_dict(sig)
    payload["observed_at"] = "2026-09-04T12:00:00"
    assert signal_from_dict(payload).observed_at == NOW


def test_signal_round_trip_survives_unjsonable_extras():
    sig = hf_signal("acme/x", "XForCausalLM")
    sig.extra = {"when": NOW, "tags": {"a", "b"}, "n": 3}
    back = signal_from_dict(signal_to_dict(sig))
    assert back.extra["n"] == 3
    assert back.extra["when"] == NOW.isoformat()
