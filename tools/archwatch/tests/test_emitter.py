"""Tests for component G, the dry-run emitter.

Pure and offline by construction: no fixture here touches the network, and one test
statically proves the emitter module cannot (no network import, no GitHub endpoint).

Golden files live in ``tests/fixtures/emitter/``. To refresh them after an
intentional format change::

    ARCHWATCH_UPDATE_GOLDEN=1 .venv/bin/pytest tests/test_emitter.py
"""

from __future__ import annotations

import ast
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from archwatch import emitter
from archwatch.connectors.base import Candidate, Signal

FIXTURES = Path(__file__).parent / "fixtures" / "emitter"
UPDATE_GOLDEN = os.environ.get("ARCHWATCH_UPDATE_GOLDEN") == "1"

# One pinned instant for every test, so nothing depends on the wall clock.
PINNED = datetime(2026, 2, 14, 9, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixture candidates
# ---------------------------------------------------------------------------


def _load_config(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def kimi_candidate() -> Candidate:
    """The rich case: three sources, a config, params, perf numbers, would-run."""
    config = _load_config("kimi_k3_config.json")
    hf = Signal(
        source="hf",
        observed_at=PINNED - timedelta(hours=6),
        arch_ids=["KimiK3ForCausalLM"],
        model_type="kimi_k3",
        model_ids=["moonshotai/Kimi-K3-Instruct", "moonshotai/Kimi-K3-Base"],
        org="moonshotai",
        display_name="Kimi K3 Instruct",
        config=config,
        urls={
            "hf": "https://huggingface.co/moonshotai/Kimi-K3-Instruct",
            "config": "https://huggingface.co/moonshotai/Kimi-K3-Instruct/blob/main/config.json",
        },
        evidence="New repo from a frontier org; config carries 5 fields BLIS does not parse",
        raw_ref="moonshotai/Kimi-K3-Instruct",
        extra={"downloads": 41823, "likes": 1290, "trending": True},
    )
    vllm = Signal(
        source="vllm",
        observed_at=PINNED - timedelta(hours=30),
        arch_ids=["KimiK3ForCausalLM"],
        model_ids=[],
        display_name="[Model] Add KimiK3ForCausalLM",
        config=None,
        urls={"pr": "https://github.com/vllm-project/vllm/pull/21877"},
        evidence="PR title names the architecture | adds vllm/model_executor/models/kimi_k3.py",
        raw_ref="vllm#21877",
    )
    ix = Signal(
        source="inferencex",
        observed_at=PINNED - timedelta(days=2),
        arch_ids=["KimiK3ForCausalLM"],
        display_name="Kimi K3",
        config=None,
        urls={"commit": "https://github.com/SemiAnalysisAI/InferenceX/commit/9f2c1ab"},
        evidence="MODELS.md diff adds the model with a first perf row",
        raw_ref="9f2c1ab",
        extra={
            "perf": {
                "hardware": "8xH200",
                "output_tok_per_s": 1840.0,
                "ttft_ms_p50": 412,
                "cost_per_mtok_usd": 0.61,
                "notes": "vLLM 0.11.1, tp8, 1k/1k in/out",
            }
        },
    )
    return Candidate(
        arch_id="KimiK3ForCausalLM",
        display_name="Kimi K3",
        signals=[hf, vllm, ix],
        triggers=["T1", "T3", "T4"],
        significance=["S1", "S2", "S3"],
        unparsed_fields=[
            "n_group",
            "topk_group",
            "mtp_num_layers",
            "mtp_loss_weight",
            "router_gate_type",
            "attention_sink_tokens",
        ],
        bucket0_failures=[],
        est_total_params=1_026_000_000_000,
        est_active_params=38_400_000_000,
    )


def bucket0_candidate() -> Candidate:
    """The would-not-run case: hard validators fail, single source, no perf."""
    return Candidate(
        arch_id="WeirdActForCausalLM",
        display_name="WeirdAct 70B",
        signals=[
            Signal(
                source="hf",
                observed_at=PINNED,
                arch_ids=["WeirdActForCausalLM"],
                model_type="weirdact",
                model_ids=["labx/WeirdAct-70B"],
                org="labx",
                display_name="WeirdAct 70B",
                config=_load_config("weirdact_config.json"),
                urls={"hf": "https://huggingface.co/labx/WeirdAct-70B"},
                evidence="Unrecognized torch_dtype and a non-SwiGLU activation",
                raw_ref="labx/WeirdAct-70B",
                extra={"downloads": 12400, "likes": 310},
            )
        ],
        triggers=["T1"],
        significance=["S1", "S4"],
        unparsed_fields=["ssm_state_size", "linear_attn_config"],
        bucket0_failures=[
            'hidden_act "gelu_pytorch_tanh" is not SwiGLU; rejected at '
            "sim/latency/kv_capacity.go:276",
            'torch_dtype "float4_e2m1" is unrecognized -> BytesPerParam=0; rejected at '
            "sim/latency/trained_physics_model.go:615",
            "num_key_value_heads is 0; must be > 0 (sim/latency/config.go:419-431)",
        ],
        est_total_params=71_000_000_000,
        est_active_params=71_000_000_000,
    )


def framework_only_candidate() -> Candidate:
    """The thin case: a framework PR only — no config, so no deterministic verdict."""
    return Candidate(
        arch_id="MysteryNetForCausalLM",
        display_name="MysteryNetForCausalLM",
        signals=[
            Signal(
                source="sglang",
                observed_at=PINNED - timedelta(days=1),
                arch_ids=["MysteryNetForCausalLM"],
                display_name="Support MysteryNet",
                urls={"pr": "https://github.com/sgl-project/sglang/pull/9912"},
                evidence="Adds python/sglang/srt/models/mysterynet.py",
                raw_ref="sglang#9912",
            ),
            Signal(
                source="vllm",
                observed_at=PINNED - timedelta(days=1, hours=4),
                arch_ids=["MysteryNetForCausalLM"],
                display_name="[Model] Support MysteryNet",
                urls={"pr": "https://github.com/vllm-project/vllm/pull/22001"},
                evidence="Adds vllm/model_executor/models/mysterynet.py",
                raw_ref="vllm#22001",
            ),
        ],
        triggers=["T2", "T4"],
        significance=["S3"],
    )


CASES = {
    "KimiK3ForCausalLM": kimi_candidate,
    "WeirdActForCausalLM": bucket0_candidate,
    "MysteryNetForCausalLM": framework_only_candidate,
}


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_render(name: str) -> None:
    rendered = emitter.render(CASES[name](), detected_at=PINNED)
    golden = FIXTURES / f"expected_{name}.md"
    if UPDATE_GOLDEN:
        golden.write_text(rendered, encoding="utf-8")
    assert golden.is_file(), f"missing golden {golden}; rerun with ARCHWATCH_UPDATE_GOLDEN=1"
    assert rendered == golden.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(CASES))
def test_written_file_matches_render(name: str, tmp_path: Path) -> None:
    cand = CASES[name]()
    result = emitter.write_issue(cand, tmp_path, detected_at=PINNED)
    assert result.status == "created"
    assert result.written is True
    assert result.path == tmp_path / f"{name}.md"
    assert result.path.read_text(encoding="utf-8") == emitter.render(cand, detected_at=PINNED)


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_render_is_deterministic() -> None:
    cand = kimi_candidate()
    assert emitter.render(cand, detected_at=PINNED) == emitter.render(cand, detected_at=PINNED)


def test_writing_twice_is_a_noop(tmp_path: Path) -> None:
    cand = kimi_candidate()
    first = emitter.write_issue(cand, tmp_path, detected_at=PINNED)
    before = first.path.read_bytes()
    stat_before = first.path.stat()

    second = emitter.write_issue(cand, tmp_path, detected_at=PINNED)
    assert second.status == "unchanged"
    assert second.written is False
    assert second.path.read_bytes() == before
    # Not merely equal bytes: the file was not touched at all.
    assert first.path.stat().st_mtime_ns == stat_before.st_mtime_ns


def test_default_detected_at_comes_from_signals_not_the_clock() -> None:
    """A Candidate rendered with no explicit stamp still renders identically forever."""
    cand = kimi_candidate()
    text = emitter.render(cand)
    front = _front_matter(text)
    newest = max(s.observed_at for s in cand.signals)
    assert front["detected_at"] == newest.isoformat().replace("+00:00", "Z")
    assert emitter.render(cand) == text


def test_naive_observed_at_is_treated_as_utc() -> None:
    cand = framework_only_candidate()
    cand.signals[0].observed_at = datetime(2026, 2, 13, 9, 30)  # naive
    front = _front_matter(emitter.render(cand))
    assert front["detected_at"] == "2026-02-13T09:30:00Z"


def test_no_timestamp_outside_front_matter() -> None:
    """Volatile values are confined to the front matter, per PLAN.md."""
    for build in CASES.values():
        text = emitter.render(build(), detected_at=PINNED)
        body = text.split("---\n", 2)[2]
        assert not re.search(r"\d{4}-\d{2}-\d{2}", body), body
        assert "09:30" not in body


def test_candidate_with_no_signals_still_renders() -> None:
    text = emitter.render(Candidate(arch_id="EmptyForCausalLM", display_name=""))
    assert "No signals recorded" in text
    assert _front_matter(text)["sources"] == []


# ---------------------------------------------------------------------------
# Front matter (component J parses this)
# ---------------------------------------------------------------------------


def _front_matter(text: str) -> dict:
    assert text.startswith("---\n")
    _, block, _rest = text.split("---\n", 2)
    data = yaml.safe_load(block)
    assert isinstance(data, dict)
    return data


REQUIRED_KEYS = {"arch_id", "sources", "triggers", "significance", "bucket", "detected_at"}


@pytest.mark.parametrize("name", sorted(CASES))
def test_front_matter_is_parseable_and_complete(name: str) -> None:
    front = _front_matter(emitter.render(CASES[name](), detected_at=PINNED))
    assert REQUIRED_KEYS <= set(front)
    cand = CASES[name]()
    assert front["arch_id"] == cand.arch_id
    assert front["sources"] == cand.sources
    assert front["triggers"] == cand.triggers
    assert front["significance"] == cand.significance
    assert front["detected_at"] == "2026-02-14T09:30:00Z"
    assert front["dry_run"] is True
    assert front["stage2"] == "pending"
    assert front["schema"] == emitter.SCHEMA


def test_bucket_is_zero_only_when_hard_validators_failed() -> None:
    assert _front_matter(emitter.render(bucket0_candidate(), detected_at=PINNED))["bucket"] == 0
    # Undetermined until stage 2 runs — never guessed as 1/2/3 by stage 1.
    assert _front_matter(emitter.render(kimi_candidate(), detected_at=PINNED))["bucket"] is None


def test_front_matter_carries_findings_and_perf() -> None:
    front = _front_matter(emitter.render(kimi_candidate(), detected_at=PINNED))
    assert front["est_total_params"] == 1_026_000_000_000
    assert front["est_active_params"] == 38_400_000_000
    assert "n_group" in front["unparsed_fields"]
    assert front["corroborated"] is True
    assert front["has_config"] is True
    assert front["orgs"] == ["moonshotai"]
    assert front["perf"][0]["source"] == "inferencex"
    assert front["perf"][0]["value"]["output_tok_per_s"] == 1840.0
    assert front["urls"]["pr"].endswith("/pull/21877")


def test_colliding_url_kinds_are_qualified_by_source() -> None:
    """Two PRs for one architecture: neither source may silently own the `pr` key."""
    front = _front_matter(emitter.render(framework_only_candidate(), detected_at=PINNED))
    assert front["urls"] == {
        "sglang.pr": "https://github.com/sgl-project/sglang/pull/9912",
        "vllm.pr": "https://github.com/vllm-project/vllm/pull/22001",
    }


def test_front_matter_survives_unrepresentable_extras() -> None:
    """extra[] is source-specific and never validated; front matter must still parse."""

    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    cand = kimi_candidate()
    cand.signals[2].extra["perf"] = {"weird": Opaque(), "when": PINNED, "seq": (1, 2)}
    front = _front_matter(emitter.render(cand, detected_at=PINNED))
    assert front["perf"][0]["value"]["weird"] == "<Opaque>"
    assert front["perf"][0]["value"]["when"] == "2026-02-14T09:30:00Z"
    assert front["perf"][0]["value"]["seq"] == [1, 2]


# ---------------------------------------------------------------------------
# Body content
# ---------------------------------------------------------------------------


def test_body_states_the_required_findings() -> None:
    text = emitter.render(kimi_candidate(), detected_at=PINNED)
    assert "# [archwatch] KimiK3ForCausalLM (Kimi K3) — new architecture detected" in text
    assert "## Sources" in text
    assert "https://github.com/vllm-project/vllm/pull/21877" in text
    assert "## Why it fired" in text
    assert "**T1**" in text and "**T4**" in text
    assert "**S3**" in text
    assert "no hard validator failed" in text
    assert "- `mtp_num_layers`" in text
    assert "1.03T" in text and "1,026,000,000,000" in text
    assert "38.4B" in text
    assert "## Reported performance numbers" in text
    assert "1840.0" in text
    assert "`kv_lora_rank`" in text  # config at a glance


def test_bucket0_failures_are_listed_verbatim() -> None:
    cand = bucket0_candidate()
    text = emitter.render(cand, detected_at=PINNED)
    assert "**No — bucket 0.** 3 hard validator failures" in text
    for failure in cand.bucket0_failures:
        assert failure in text
    assert "Bucket 0 is already established" in text


def test_missing_config_is_reported_as_not_checked() -> None:
    text = emitter.render(framework_only_candidate(), detected_at=PINNED)
    assert "**Not checked**" in text
    assert "no config could be" in text.lower() or "No config available" in text
    assert "_Not estimated" in text
    assert "## Reported performance numbers" not in text
    assert "## Config at a glance" not in text


def test_unknown_trigger_codes_do_not_crash() -> None:
    cand = framework_only_candidate()
    cand.triggers = ["T9", "ALIAS_KEY"]
    cand.significance = []
    text = emitter.render(cand, detected_at=PINNED)
    assert "**T9** — no description on file" in text
    assert "**ALIAS_KEY** — no description on file" in text
    assert "_none recorded_" in text


def test_table_cells_escape_pipes_and_newlines() -> None:
    cand = framework_only_candidate()
    cand.signals[0].evidence = "line one | with a pipe\nand a newline"
    text = emitter.render(cand, detected_at=PINNED)
    row = [ln for ln in text.splitlines() if "line one" in ln][0]
    assert "\\|" in row
    assert row.count("|") - row.count("\\|") == 6  # 5 columns -> 6 delimiters
    assert "line one \\| with a pipe and a newline" in row


def test_perf_list_of_dicts_renders_as_a_table() -> None:
    cand = kimi_candidate()
    cand.signals[2].extra["perf"] = [
        {"hardware": "8xH200", "tok_per_s": 1840},
        {"hardware": "8xB200", "tok_per_s": 3120},
    ]
    text = emitter.render(cand, detected_at=PINNED)
    assert "| `hardware` | `tok_per_s` |" in text
    assert "| 8xB200 | 3120 |" in text


def test_stage2_placeholder_is_present_and_last() -> None:
    text = emitter.render(kimi_candidate(), detected_at=PINNED)
    assert "## Stage 2 — deep dive" in text
    assert "**Status: not yet run.**" in text
    assert emitter.STAGE2_MARKER in text
    assert text.rstrip().endswith(emitter.STAGE2_MARKER)


# ---------------------------------------------------------------------------
# Stage-2 appendix handling
# ---------------------------------------------------------------------------


def test_appended_stage2_analysis_is_not_clobbered(tmp_path: Path) -> None:
    cand = kimi_candidate()
    result = emitter.write_issue(cand, tmp_path, detected_at=PINNED)
    with result.path.open("a", encoding="utf-8") as fh:
        fh.write("\n## Stage 2 analysis\n\nBucket 2: MLA step-time pessimism.\n")
    after_append = result.path.read_text(encoding="utf-8")

    again = emitter.write_issue(cand, tmp_path, detected_at=PINNED)
    assert again.status == "unchanged"
    assert result.path.read_text(encoding="utf-8") == after_append


def test_changed_stub_is_skipped_unless_overwrite(tmp_path: Path) -> None:
    cand = kimi_candidate()
    path = emitter.write_issue(cand, tmp_path, detected_at=PINNED).path
    path.write_text("stale content\n", encoding="utf-8")

    skipped = emitter.write_issue(cand, tmp_path, detected_at=PINNED)
    assert skipped.status == "skipped_existing"
    assert path.read_text(encoding="utf-8") == "stale content\n"

    updated = emitter.write_issue(cand, tmp_path, detected_at=PINNED, overwrite=True)
    assert updated.status == "updated"
    assert path.read_text(encoding="utf-8") == emitter.render(cand, detected_at=PINNED)


def test_overwrite_regenerates_the_stub_and_keeps_the_analysis(tmp_path: Path) -> None:
    cand = kimi_candidate()
    path = emitter.write_issue(cand, tmp_path, detected_at=PINNED).path
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n## Stage 2 analysis\n\nBucket 3: new routing mechanism.\n")

    cand.triggers = ["T1", "T2", "T3", "T4"]
    updated = emitter.write_issue(cand, tmp_path, detected_at=PINNED, overwrite=True)
    assert updated.status == "updated"
    text = path.read_text(encoding="utf-8")
    assert "**T2**" in text  # stub regenerated
    assert "Bucket 3: new routing mechanism." in text  # analysis preserved
    assert text.count(emitter.STAGE2_MARKER) == 1


def test_split_stub_without_marker_keeps_everything_as_stub() -> None:
    stub, tail = emitter.split_stub("no marker here\n")
    assert stub == "no marker here\n"
    assert tail == ""


# ---------------------------------------------------------------------------
# Paths and filenames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch_id, expected",
    [
        ("KimiK3ForCausalLM", "KimiK3ForCausalLM"),
        ("Qwen3.5-MoEForCausalLM", "Qwen3.5-MoEForCausalLM"),
        ("Deep Seek V4", "Deep_Seek_V4-8de3f57d"),
        ("evil/../../etc/passwd", "evil_.._.._etc_passwd-3ba58b19"),
        ("../../../etc/shadow", "etc_shadow-9cfe53c2"),
        (".hidden", "hidden-92f73832"),
        ("--flag-like", "flag-like-4f013f7e"),
        ("", "unnamed-da39a3ee"),
        ("   ", "unnamed-088fb1a4"),
        ("CON", "CON_-7679a072"),
        ("nul", "nul_-d9792951"),
        ("\u6a21\u578bForCausalLM", "ForCausalLM-83c2e797"),
        ("weird\x00name", "weird_name-ea247a1a"),
    ],
)
def test_safe_stem(arch_id: str, expected: str) -> None:
    assert emitter.safe_stem(arch_id) == expected


def test_safe_stem_truncates_but_stays_unique() -> None:
    long_a = "A" * 300
    long_b = "A" * 301
    stem_a = emitter.safe_stem(long_a)
    assert len(stem_a) <= 110
    assert stem_a != emitter.safe_stem(long_b)


def test_distinct_arch_ids_never_share_a_file(tmp_path: Path) -> None:
    ids = ["a/b", "a_b", "a b", "a:b", "A/B", "", "  ", "CON", "con"]
    paths = {emitter.issue_path(i, tmp_path) for i in ids}
    assert len(paths) == len(ids)


@pytest.mark.parametrize("arch_id", ["../../escape", "/abs/olute", "a/b/c", "..", "."])
def test_written_files_stay_inside_out_dir(arch_id: str, tmp_path: Path) -> None:
    cand = Candidate(arch_id=arch_id, display_name=arch_id)
    result = emitter.write_issue(cand, tmp_path, detected_at=PINNED)
    assert result.path.parent.resolve() == tmp_path.resolve()
    assert result.path.suffix == ".md"
    assert result.path.is_file()
    assert list(tmp_path.iterdir()) == [result.path]


def test_out_dir_is_created(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "issues"
    result = emitter.write_issue(kimi_candidate(), target, detected_at=PINNED)
    assert result.path.is_file()
    assert result.path.parent == target


def test_issue_exists_matches_the_written_path(tmp_path: Path) -> None:
    cand = kimi_candidate()
    assert emitter.issue_exists(cand.arch_id, tmp_path) is False
    emitter.write_issue(cand, tmp_path, detected_at=PINNED)
    assert emitter.issue_exists(cand.arch_id, tmp_path) is True
    assert emitter.issue_exists("NotSeenForCausalLM", tmp_path) is False


def test_default_issues_dir_is_the_repo_one() -> None:
    d = emitter.default_issues_dir()
    assert d.name == "issues"
    assert d.parent.name == "archwatch"


def test_write_issues_preserves_order_and_summarizes(tmp_path: Path) -> None:
    cands = [build() for build in (kimi_candidate, bucket0_candidate, framework_only_candidate)]
    results = emitter.write_issues(cands, tmp_path, detected_at=PINNED)
    assert [r.arch_id for r in results] == [c.arch_id for c in cands]
    assert all(r.status == "created" for r in results)
    assert emitter.summarize(results) == {"created": 3}

    again = emitter.write_issues(cands, tmp_path, detected_at=PINNED)
    assert emitter.summarize(again) == {"unchanged": 3}


# ---------------------------------------------------------------------------
# The dry-run guarantee
# ---------------------------------------------------------------------------


def test_emitter_module_cannot_reach_the_network() -> None:
    """PLAN.md hard rule 2: no GitHub call, not even a disabled one.

    Enforced structurally rather than by reading the docstring: the module may only
    import stdlib helpers plus yaml plus the frozen contract.
    """
    source_path = Path(emitter.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
            elif node.module:
                imported.add("." + node.module.split(".")[0])

    allowed = {
        "__future__", "hashlib", "json", "re", "dataclasses", "datetime", "pathlib",
        "typing", "yaml", ".connectors",
    }
    assert imported <= allowed, f"unexpected imports in emitter.py: {imported - allowed}"

    text = source_path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "api.github.com",
        "requests.",
        "urllib",
        "httpx",
        "socket",
        "subprocess",
        "gh api",
        "create_issue",
        "gh_token",
        "github_token",
        "huggingface_hub",
    ):
        assert forbidden not in text, f"emitter.py mentions {forbidden!r}"


def test_render_touches_no_filesystem_state(tmp_path: Path) -> None:
    emitter.render(kimi_candidate(), detected_at=PINNED)
    assert list(tmp_path.iterdir()) == []
