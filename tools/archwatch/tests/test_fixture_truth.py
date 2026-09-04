"""Integration guard: no golden file may claim a finding the real surface does not produce.

Why this exists. The emitter's own suite deliberately does not import ``surface`` — that
decoupling keeps golden tests independent of B's YAML. But it is precisely why the emitter's
fixtures drifted into *fiction*: four separate invented strings ("this validator is fatal", a
line ref, two silent failures that no config actually triggers) sat in goldens while every
component test passed. Two reviewers missed one of them.

Per-component tests cannot catch this class of error by construction: each component is
internally consistent with its own fixtures. Only running the real surface against the real
fixture configs and comparing to what the goldens assert will do it.

If this test fails, the fix is to re-capture the fixture strings from the live surface, never
to edit the expectation here.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from archwatch.surface import load_surface

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "emitter"

# (config fixture, golden stub) pairs. Add a row whenever a new golden is introduced.
PAIRS = [
    ("kimi_k3_config.json", "expected_KimiK3ForCausalLM.md"),
    ("weirdact_config.json", "expected_WeirdActForCausalLM.md"),
    ("qwen3next_config.json", "expected_Qwen3NextForCausalLM.md"),
]


def _front_matter(md: str) -> dict:
    assert md.startswith("---\n"), "golden must open with YAML front matter"
    return yaml.safe_load(md.split("---\n", 2)[1])


@pytest.fixture(scope="module")
def surface():
    return load_surface()


@pytest.mark.parametrize("cfg_name,golden_name", PAIRS)
def test_golden_findings_match_the_real_surface(surface, cfg_name, golden_name):
    """The golden's fatal/silent lists must be exactly what the live surface reports.

    Set equality in both directions. A missing entry means the golden is stale; an extra
    entry means the golden asserts something BLIS does not actually do, which is worse —
    a reader trusts these strings.
    """
    config = json.loads((FIXTURES / cfg_name).read_text())
    golden = (FIXTURES / golden_name).read_text()
    fm = _front_matter(golden)

    live_fatal = set(surface.check_hard_validators(config))
    live_silent = set(surface.check_silent_validators(config))
    claimed_fatal = set(fm.get("bucket0_failures") or [])
    claimed_silent = set(fm.get("silent_failures") or [])

    assert claimed_fatal == live_fatal, (
        f"{golden_name} fatal findings disagree with the live surface.\n"
        f"  invented (in golden, not produced): {sorted(claimed_fatal - live_fatal)}\n"
        f"  missing  (produced, not in golden): {sorted(live_fatal - claimed_fatal)}"
    )
    assert claimed_silent == live_silent, (
        f"{golden_name} silent findings disagree with the live surface.\n"
        f"  invented (in golden, not produced): {sorted(claimed_silent - live_silent)}\n"
        f"  missing  (produced, not in golden): {sorted(live_silent - claimed_silent)}"
    )


@pytest.mark.parametrize("cfg_name,golden_name", PAIRS)
def test_silently_wrong_flag_is_derived_correctly(surface, cfg_name, golden_name):
    """``silently_wrong`` is the backtest's headline metric, so it must not drift.

    It means: BLIS accepts the config and reports wrong numbers with nothing to signal it.
    """
    config = json.loads((FIXTURES / cfg_name).read_text())
    fm = _front_matter((FIXTURES / golden_name).read_text())
    expected = bool(surface.check_silent_validators(config)) and not surface.check_hard_validators(config)
    assert bool(fm.get("silently_wrong")) is expected


def test_the_recognized_expert_count_aliases_are_not_used_as_unrecognized_examples(surface):
    """Guard against a reviewer error that already slipped past two reviewers once.

    ``moe_num_experts`` (Dbrx) and ``n_routed_experts`` (DeepSeek) ARE in BLIS's
    ``moeExpertCountFields``. Neither can serve as an "unrecognized spelling" fixture, so no
    fixture config may use one while its golden claims the expert count failed to resolve.
    """
    for cfg_name, golden_name in PAIRS:
        config = json.loads((FIXTURES / cfg_name).read_text())
        golden = (FIXTURES / golden_name).read_text()
        if "no total expert count resolved" not in golden:
            continue
        for recognized in ("num_experts", "moe_num_experts", "n_routed_experts",
                           "num_local_experts", "num_routed_experts"):
            assert recognized not in config, (
                f"{cfg_name} claims the expert count did not resolve, but it uses the "
                f"RECOGNIZED spelling {recognized!r}. Use num_moe_experts / expert_count / "
                f"n_group_experts instead."
            )
