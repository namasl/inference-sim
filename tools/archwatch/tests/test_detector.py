"""Tests for archwatch.detector — the wiring, the budget, and the run log.

No network anywhere. Connectors are injected fakes (including one that raises), the
surface is either a stub or the real YAML read off disk, and an autouse fixture makes
``socket``, ``requests`` and ``huggingface_hub`` blow up if anything reaches for the
live thing. The one test that touches the real support surface reads files, not the
network.

The test this file exists for is
:func:`test_scan_dedup_follows_out_dir_not_the_default`. Threading ``--out`` into
``evaluate(..., issues_dir=...)`` is invisible when it is wrong: the dedup consults
``issues/`` while the writes go elsewhere, and nothing errors. It is caught by asserting
that a stub sitting in the *output* directory suppresses, and a stub sitting in some
*other* directory does not.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from archwatch import detector, emitter
from archwatch.config import IGNORED_CONFIG_KEYS, DetectorConfig, Thresholds
from archwatch.connectors.base import Candidate, Signal
from archwatch.novelty import EvaluationReport, Suppression

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
DETECTOR_SRC = Path(detector.__file__)


# ---------------------------------------------------------------------------
# no network, ever
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any live call from a test is a bug in the test, not a flake."""
    import socket

    def boom(*_args, **_kwargs):  # pragma: no cover - only fires on a regression
        raise AssertionError("test attempted a live network call")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    import requests

    monkeypatch.setattr(requests, "Session", boom)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", boom)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

BIG_CONFIG = {
    "architectures": ["NovelMechForCausalLM"],
    "model_type": "novelmech",
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "vocab_size": 128256,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "intermediate_size": 28672,
    "hidden_act": "silu",
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 131072,
    # The point of the fixture: a field BLIS does not parse -> T1.
    "novel_mechanism_dim": 512,
}

PARSED = frozenset(
    {
        "num_hidden_layers", "hidden_size", "vocab_size", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "hidden_act",
        "max_position_embeddings", "rope_theta", "rms_norm_eps", "tie_word_embeddings",
    }
)


class FakeSurface:
    """The four methods novelty needs, driven by a parsed-field allowlist.

    Config-aware rather than canned, so a candidate with a plain config really does
    fail T1 in these tests instead of being told it passed.
    """

    def __init__(self, known: tuple[str, ...] = (), bucket0: tuple[str, ...] = ()) -> None:
        self.known = {k.lower() for k in known}
        self.bucket0 = list(bucket0)

    def unparsed_fields(self, config: dict) -> list[str]:
        return sorted(
            k for k in config if k not in PARSED and k not in IGNORED_CONFIG_KEYS
        )

    def check_hard_validators(self, config: dict) -> list[str]:
        return list(self.bucket0)

    def check_silent_validators(self, config: dict) -> list[str]:
        return []

    def is_known_architecture(self, arch_id: str) -> bool:
        return (arch_id or "").strip().lower() in self.known

    def match_gaps(self, config: dict) -> list:
        return []


class FakeConnector:
    """A connector that returns canned Signals, or raises on demand."""

    def __init__(self, name: str, signals=(), *, error: Exception | None = None) -> None:
        self.name = name
        self.signals = list(signals)
        self.error = error
        self.poll_calls: list[datetime] = []

    def poll(self, since: datetime):
        self.poll_calls.append(since)
        if self.error is not None:
            raise self.error
        return list(self.signals)


class FakeTrendingConnector(FakeConnector):
    """A connector that also offers the S4 trending sweep."""

    def __init__(self, name: str, signals=(), *, trending=(), trending_error=None, **kw):
        super().__init__(name, signals, **kw)
        self.trending = list(trending)
        self.trending_error = trending_error
        self.trending_calls = 0

    def poll_trending(self):
        self.trending_calls += 1
        if self.trending_error is not None:
            raise self.trending_error
        return list(self.trending)


def hf_signal(
    *,
    arch: str = "NovelMechForCausalLM",
    org: str = "moonshotai",
    model_id: str = "moonshotai/Novel-Mech-70B",
    config: dict | None = BIG_CONFIG,
    observed_at: datetime = NOW,
    extra: dict | None = None,
) -> Signal:
    return Signal(
        source="hf",
        observed_at=observed_at,
        arch_ids=[arch],
        model_type="novelmech",
        model_ids=[model_id],
        org=org,
        display_name=model_id,
        config=dict(config) if config else None,
        urls={"hf": f"https://huggingface.co/{model_id}"},
        evidence="created in window",
        raw_ref=model_id,
        extra=dict(extra or {}),
    )


def pr_signal(
    *,
    arch: str = "OtherNetForCausalLM",
    source: str = "vllm",
    number: int = 4242,
    observed_at: datetime = NOW,
) -> Signal:
    return Signal(
        source=source,
        observed_at=observed_at,
        arch_ids=[arch],
        model_ids=[],
        org=None,
        display_name="Other Net",
        config=None,
        urls={"pr": f"https://github.com/vllm-project/vllm/pull/{number}"},
        evidence="registry line added",
        raw_ref=str(number),
    )


@pytest.fixture
def cfg() -> DetectorConfig:
    """Every knob these tests' assertions depend on, pinned explicitly.

    Not ``DetectorConfig()``. The thresholds are calibration outputs and they move:
    the wave-6 backtest took ``min_total_params`` 30B -> 3B,
    ``recheck_known_architectures`` False -> True and ``max_issues_per_run`` 5 -> 10,
    and three tests here silently changed meaning — one of them had been asserting
    that a known architecture is suppressed, which is no longer what the shipped
    default does at all. A test that inherits a tunable default is asserting the
    calibration, not the code. Tests that are *about* a setting name it themselves;
    both settings of ``recheck_known_architectures`` are covered below.
    """
    return DetectorConfig(
        window_days=1,
        max_issues_per_run=5,
        thresholds=Thresholds(min_total_params=30_000_000_000),
        recheck_known_architectures=False,
    )


def run(**kw):
    """``scan()`` with the boring arguments filled in."""
    kw.setdefault("surface", FakeSurface())
    kw.setdefault("now", NOW)
    kw.setdefault("trending", False)
    return detector.scan(**kw)


# ---------------------------------------------------------------------------
# parse_sources
# ---------------------------------------------------------------------------


def test_parse_sources_defaults_to_every_source():
    assert detector.parse_sources(None) == list(detector.SOURCE_NAMES)


@pytest.mark.parametrize(
    "spec",
    ["hf,vllm", "hf, vllm", " vllm  hf ", ["hf", "vllm"], ["hf,vllm"]],
)
def test_parse_sources_accepts_several_spellings(spec):
    assert detector.parse_sources(spec) == ["hf", "vllm"]


def test_parse_sources_dedupes_and_uses_canonical_order():
    # Order out is always SOURCE_NAMES order so two run logs are comparable.
    assert detector.parse_sources("inferencex,hf,hf,vllm") == ["hf", "vllm", "inferencex"]


def test_parse_sources_expands_aliases():
    assert detector.parse_sources("all") == list(detector.SOURCE_NAMES)
    assert detector.parse_sources("frameworks") == ["vllm", "sglang"]
    assert detector.parse_sources("frameworks,hf") == ["hf", "vllm", "sglang"]


def test_parse_sources_rejects_a_typo_rather_than_scanning_nothing():
    # A silently-ignored typo looks exactly like a quiet day.
    with pytest.raises(ValueError, match="inferncex"):
        detector.parse_sources("hf,inferncex")


def test_parse_sources_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="no sources"):
        detector.parse_sources("  ")


# ---------------------------------------------------------------------------
# the shared GitHub budget
# ---------------------------------------------------------------------------


def test_budget_counts_and_refuses():
    b = detector.GithubBudget(limit=2)
    assert b.take() and b.take()
    assert not b.take()
    assert (b.used, b.refused, b.remaining, b.exhausted) == (2, 1, 0, True)
    assert b.as_dict() == {"budget": 2, "used": 2, "refused": 1, "remaining": 0}


class RecordingSession:
    def __init__(self) -> None:
        self.gets: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url, **kwargs):
        self.gets.append(url)
        return {"url": url, "kwargs": kwargs}


def test_budgeted_session_counts_gets_and_passes_them_through():
    budget = detector.GithubBudget(limit=5)
    inner = RecordingSession()
    session = detector.BudgetedSession(budget, inner)
    session.get("https://api.github.com/x", params={"a": 1})
    assert inner.gets == ["https://api.github.com/x"]
    assert budget.used == 1


def test_budgeted_session_refuses_over_budget_without_touching_the_network():
    budget = detector.GithubBudget(limit=1)
    inner = RecordingSession()
    session = detector.BudgetedSession(budget, inner)
    session.get("https://api.github.com/one")
    refused = session.get("https://api.github.com/two")

    # Nothing was sent, and the caller sees a rate-limit shape it already handles.
    assert inner.gets == ["https://api.github.com/one"]
    assert refused.status_code == 403
    assert refused.headers["X-RateLimit-Remaining"] == "0"
    assert refused.json() == {}
    assert budget.refused == 1


def test_budgeted_session_shared_budget_is_spent_by_whoever_asks_first():
    budget = detector.GithubBudget(limit=3)
    inner = RecordingSession()
    a = detector.BudgetedSession(budget, inner)
    b = detector.BudgetedSession(budget, inner)
    a.get("u1")
    b.get("u2")
    b.get("u3")
    assert a.get("u4").status_code == 403  # a is capped by b's spending
    assert budget.used == 3


@pytest.mark.parametrize("verb", ["post", "put", "patch", "delete"])
def test_budgeted_session_refuses_write_verbs(verb):
    session = detector.BudgetedSession(detector.GithubBudget(10), RecordingSession())
    with pytest.raises(RuntimeError, match="read-only"):
        getattr(session, verb)("https://api.github.com/repos/x/y/issues", json={})


def test_budgeted_session_refuses_non_get_through_request():
    session = detector.BudgetedSession(detector.GithubBudget(10), RecordingSession())
    with pytest.raises(RuntimeError, match="read-only"):
        session.request("POST", "https://api.github.com/x")
    assert session.request("get", "https://api.github.com/x")["url"].endswith("/x")


def test_budgeted_session_delegates_other_attributes():
    inner = RecordingSession()
    inner.headers["User-Agent"] = "test"
    session = detector.BudgetedSession(detector.GithubBudget(1), inner)
    assert session.headers == {"User-Agent": "test"}


# ---------------------------------------------------------------------------
# build_connectors
# ---------------------------------------------------------------------------


def test_build_connectors_hf_only_creates_no_github_budget(cfg):
    conns, budget = detector.build_connectors(["hf"], cfg)
    assert [c.name for c in conns] == ["hf"]
    assert budget is None


def test_build_connectors_shares_one_budget_across_every_github_source(cfg):
    cfg = DetectorConfig(max_github_requests=7)
    inner = RecordingSession()
    conns, budget = detector.build_connectors(
        ["vllm", "sglang", "inferencex"], cfg, github_session=inner
    )
    assert budget is not None and budget.limit == 7

    # One budget object, one session object, three connectors. Two connectors each
    # obeying a 7-request cap would spend 14.
    sessions = [conns[0].http.session, conns[1].http.session, conns[2].session]
    assert sessions[0] is sessions[1] is sessions[2]
    assert all(s.budget is budget for s in sessions)

    for i in range(7):
        sessions[i % 3].get(f"https://api.github.com/{i}")
    assert budget.exhausted
    assert conns[2].session.get("https://api.github.com/last").status_code == 403


def test_build_connectors_rejects_an_unknown_source(cfg):
    with pytest.raises(ValueError, match="unknown source"):
        detector.build_connectors(["nope"], cfg)


# ---------------------------------------------------------------------------
# poll_connectors — isolation and no loops
# ---------------------------------------------------------------------------


def test_poll_connectors_polls_each_source_exactly_once():
    a = FakeConnector("hf", [hf_signal()])
    b = FakeConnector("vllm", [pr_signal()])
    signals, reports = detector.poll_connectors([a, b], NOW - timedelta(days=1))

    # /search/issues allows 30 requests a minute; a retry loop here would burn it.
    assert len(a.poll_calls) == 1
    assert len(b.poll_calls) == 1
    assert len(signals) == 2
    assert [(r.name, r.ok, r.signals) for r in reports] == [
        ("hf", True, 1),
        ("vllm", True, 1),
    ]


def test_poll_connectors_isolates_a_failing_source():
    good = FakeConnector("hf", [hf_signal()])
    bad = FakeConnector("inferencex", error=RuntimeError("repo restructured"))
    signals, reports = detector.poll_connectors([bad, good], NOW)

    assert [s.source for s in signals] == ["hf"]  # the good source survived
    failed = next(r for r in reports if r.name == "inferencex")
    assert failed.ok is False
    assert "RuntimeError: repo restructured" in failed.error
    assert next(r for r in reports if r.name == "hf").ok is True


def test_poll_connectors_runs_the_trending_sweep():
    conn = FakeTrendingConnector("hf", [hf_signal()], trending=[hf_signal(arch="TrendyForCausalLM")])
    signals, reports = detector.poll_connectors([conn], NOW, trending=True)
    assert conn.trending_calls == 1
    assert len(signals) == 2
    assert reports[0].signals == 2
    assert any("trending sweep added 1" in n for n in reports[0].notes)


def test_poll_connectors_can_skip_the_trending_sweep():
    conn = FakeTrendingConnector("hf", [hf_signal()], trending=[hf_signal()])
    signals, _ = detector.poll_connectors([conn], NOW, trending=False)
    assert conn.trending_calls == 0
    assert len(signals) == 1


def test_a_failing_trending_sweep_keeps_the_window_poll():
    conn = FakeTrendingConnector(
        "hf", [hf_signal()], trending_error=ValueError("hub 502")
    )
    signals, reports = detector.poll_connectors([conn], NOW, trending=True)
    assert len(signals) == 1
    assert reports[0].ok is True  # the window poll worked; only the sweep did not
    assert any("trending sweep failed" in n for n in reports[0].notes)


# ---------------------------------------------------------------------------
# scan — the end to end path
# ---------------------------------------------------------------------------


def test_scan_writes_a_stub_and_reports_every_stage(tmp_path, cfg):
    conn = FakeConnector("hf", [hf_signal()])
    summary = run(cfg=cfg, connectors=[conn], out_dir=tmp_path)

    assert summary.signals == 1
    assert summary.signals_by_source == {"hf": 1}
    assert summary.candidates == 1
    assert [c.arch_id for c in summary.passed] == ["NovelMechForCausalLM"]
    assert summary.passed[0].triggers[:1] == ["T1"]
    assert "S1" in summary.passed[0].significance
    assert summary.written == 1
    assert summary.written_by_status == {"created": 1}

    written = tmp_path / "NovelMechForCausalLM.md"
    assert written.is_file()
    assert written.read_text(encoding="utf-8").startswith("---\n")
    assert summary.partial is False and summary.dry_run is True


def test_scan_dedup_follows_out_dir_not_the_default(tmp_path, cfg):
    """The bug this whole component is most likely to have.

    ``evaluate()`` takes ``issues_dir`` keyword-only and defaults it to the emitter's
    ``issues/``. If ``--out`` is not threaded through, the dedup suppressor asks about
    a directory nobody is writing to: existing stubs get re-emitted and new ones get
    suppressed, silently, with no error on either side.
    """
    out = tmp_path / "out"
    elsewhere = tmp_path / "elsewhere"
    signals = [hf_signal()]

    # A stub for this architecture exists, but in a directory that is NOT the output
    # directory. It must not suppress.
    emitter.write_issue(Candidate(arch_id="NovelMechForCausalLM", display_name="x"), elsewhere)
    first = run(cfg=cfg, connectors=[FakeConnector("hf", signals)], out_dir=out)
    assert [c.arch_id for c in first.passed] == ["NovelMechForCausalLM"]
    assert (out / "NovelMechForCausalLM.md").is_file()

    # Now the stub is in the output directory, so the same scan must suppress it —
    # this is the whole of the prototype's stateless dedup.
    again = run(cfg=cfg, connectors=[FakeConnector("hf", signals)], out_dir=out)
    assert again.passed == []
    assert again.suppressed_by_reason == {"already_reported": 1}
    assert [s.stage for s in again.suppressions] == ["suppressor"]


def test_scan_passes_the_resolved_out_dir_to_the_filter(tmp_path, cfg, monkeypatch):
    """Belt and braces on the same wiring, asserted at the call boundary."""
    seen: dict[str, object] = {}

    def spy(cands, surface, config, *, issues_dir=None, already_reported=None):
        seen["issues_dir"] = issues_dir
        return EvaluationReport()

    monkeypatch.setattr(detector, "evaluate_detailed", spy)
    run(cfg=cfg, connectors=[FakeConnector("hf", [hf_signal()])], out_dir=tmp_path / "o")
    assert seen["issues_dir"] == tmp_path / "o"


def test_scan_defaults_out_dir_to_the_emitters_issues_dir(tmp_path, cfg, monkeypatch):
    seen: dict[str, object] = {}

    def spy(cands, surface, config, *, issues_dir=None, already_reported=None):
        seen["issues_dir"] = issues_dir
        return EvaluationReport()

    monkeypatch.setattr(detector, "evaluate_detailed", spy)
    monkeypatch.setattr(emitter, "write_issues", lambda *a, **kw: [])
    summary = run(cfg=cfg, connectors=[FakeConnector("hf", [hf_signal()])])
    assert seen["issues_dir"] == emitter.default_issues_dir()
    assert summary.out_dir == emitter.default_issues_dir()


def test_scan_degrades_to_a_partial_scan_when_one_source_raises(tmp_path, cfg):
    """A source failing must not cost us what the others saw."""
    good = FakeConnector("hf", [hf_signal()])
    bad = FakeConnector("vllm", error=RuntimeError("GitHub 403"))
    summary = run(cfg=cfg, connectors=[good, bad], out_dir=tmp_path)

    assert summary.partial is True
    assert summary.failed_sources == ["vllm"]
    assert summary.ok_sources == ["hf"]
    assert summary.total_failure is False
    # The issue the working source justified was still written.
    assert (tmp_path / "NovelMechForCausalLM.md").is_file()
    assert "PARTIAL SCAN" in summary.text()
    assert summary.as_dict()["sources"]["failed"] == ["vllm"]


def test_scan_reports_total_failure_when_every_source_dies(tmp_path, cfg):
    conns = [
        FakeConnector("hf", error=OSError("dns")),
        FakeConnector("vllm", error=RuntimeError("403")),
    ]
    summary = run(cfg=cfg, connectors=conns, out_dir=tmp_path)
    assert summary.total_failure is True
    assert summary.passed == []
    assert "NOTHING SCANNED" in summary.text()


def test_scan_refuses_a_live_mode(tmp_path, cfg):
    with pytest.raises(ValueError, match="no live mode"):
        run(cfg=cfg, connectors=[], out_dir=tmp_path, dry_run=False)


def test_scan_window_days_sets_since_without_mutating_the_config(tmp_path):
    cfg = DetectorConfig(window_days=7)
    conn = FakeConnector("hf", [])
    summary = run(cfg=cfg, connectors=[conn], out_dir=tmp_path, window_days=3)
    assert cfg.window_days == 7  # the caller's config is theirs
    assert summary.window_days == 3
    assert conn.poll_calls == [NOW - timedelta(days=3)]
    assert summary.since == NOW - timedelta(days=3)


def test_scan_rejects_a_non_positive_window(tmp_path, cfg):
    with pytest.raises(ValueError, match="window_days must be positive"):
        run(cfg=cfg, connectors=[], out_dir=tmp_path, window_days=0)


def test_scan_honours_an_explicit_since(tmp_path, cfg):
    conn = FakeConnector("hf", [])
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    summary = run(cfg=cfg, connectors=[conn], out_dir=tmp_path, since=since)
    assert conn.poll_calls == [since]
    assert summary.since == since


def test_scan_records_the_per_run_cap_as_a_suppression(tmp_path):
    cfg = DetectorConfig(window_days=1, max_issues_per_run=1)
    signals = [
        hf_signal(arch="AlphaNetForCausalLM", model_id="moonshotai/Alpha-70B"),
        hf_signal(arch="BetaNetForCausalLM", model_id="deepseek-ai/Beta-70B", org="deepseek-ai"),
    ]
    summary = run(cfg=cfg, connectors=[FakeConnector("hf", signals)], out_dir=tmp_path)
    assert len(summary.passed) == 1
    assert summary.suppressed_by_reason == {"over_cap": 1}
    assert [s.stage for s in summary.suppressions] == ["cap"]
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_scan_counts_signals_by_source_across_connectors(tmp_path, cfg):
    conns = [
        FakeConnector("hf", [hf_signal(), hf_signal(arch="TwoForCausalLM", model_id="qwen/Two")]),
        FakeConnector("vllm", [pr_signal()]),
    ]
    summary = run(cfg=cfg, connectors=conns, out_dir=tmp_path)
    assert summary.signals_by_source == {"hf": 2, "vllm": 1}
    assert summary.signals == 3


def test_scan_derives_requested_sources_from_injected_connectors(tmp_path, cfg):
    summary = run(
        cfg=cfg,
        connectors=[FakeConnector("hf"), FakeConnector("inferencex")],
        out_dir=tmp_path,
    )
    assert summary.sources_requested == ["hf", "inferencex"]


def test_scan_is_idempotent_on_a_second_run_with_the_same_signals(tmp_path, cfg):
    """Re-running must not rewrite a stub, and must report why nothing was written."""
    signals = [hf_signal()]
    run(cfg=cfg, connectors=[FakeConnector("hf", signals)], out_dir=tmp_path)
    before = (tmp_path / "NovelMechForCausalLM.md").read_bytes()
    second = run(cfg=cfg, connectors=[FakeConnector("hf", signals)], out_dir=tmp_path)
    assert second.written == 0
    assert second.suppressed_by_reason == {"already_reported": 1}
    assert (tmp_path / "NovelMechForCausalLM.md").read_bytes() == before


def test_scan_against_the_real_support_surface(tmp_path):
    """One integration check against the YAML actually shipped, not a stub.

    Reads files; no network. A stub surface can agree with a bug in itself, so at
    least one test has to use the real seed set and the real parsed-field list.
    """
    # The S1 threshold is named here because this test asserts S1 fired: inheriting a
    # calibration output would make the assertion mean "whatever S1 does this week".
    cfg = DetectorConfig(
        window_days=1, thresholds=Thresholds(min_total_params=30_000_000_000)
    )
    config = dict(BIG_CONFIG)
    config["architectures"] = ["ArchwatchTestNetForCausalLM"]
    del config["novel_mechanism_dim"]
    config["archwatch_test_novel_field"] = 512
    signal = hf_signal(arch="ArchwatchTestNetForCausalLM", config=config)

    summary = detector.scan(
        cfg,
        connectors=[FakeConnector("hf", [signal])],
        out_dir=tmp_path,
        now=NOW,
        trending=False,
    )
    cand = summary.passed[0]
    assert cand.arch_id == "ArchwatchTestNetForCausalLM"
    assert cand.unparsed_fields == ["archwatch_test_novel_field"]
    assert "T1" in cand.triggers and "S1" in cand.significance
    assert cand.est_total_params is not None and cand.est_total_params > 30_000_000_000
    assert cand.bucket0_failures == []


def test_scan_suppresses_a_known_architecture_when_the_recheck_is_off(tmp_path):
    """Suppression-of-a-known-architecture, pinned to the setting that produces it.

    ``recheck_known_architectures`` is passed here rather than inherited: the shipped
    default is now True, under which this candidate deliberately does NOT drop (see
    the next test). Inheriting the default would have turned this assertion into
    "whatever the current calibration does", which is the opposite of a regression test.
    """
    cfg = DetectorConfig(window_days=1, recheck_known_architectures=False)
    config = dict(BIG_CONFIG, architectures=["LlamaForCausalLM"])
    signal = hf_signal(arch="LlamaForCausalLM", config=config)
    summary = detector.scan(
        cfg,
        connectors=[FakeConnector("hf", [signal])],
        out_dir=tmp_path,
        now=NOW,
        trending=False,
    )
    assert summary.passed == []
    assert "known_architecture" in summary.suppressed_by_reason
    assert list(tmp_path.glob("*.md")) == []


def test_scan_rechecks_a_known_architecture_when_the_recheck_is_on(tmp_path):
    """The shipped default: a seeded architecture whose config grew a field BLIS
    cannot read still reaches a stub, tagged apart from a genuinely new architecture.

    This is the mode the backtest measured and turned on (frontier recall 0/9 -> 9/9),
    so it needs a wiring test of its own rather than being reachable only by default.
    The trigger code must be the known-arch one: "an architecture BLIS thinks it
    supports has silently grown an unparsed field" is a different finding for a human
    than "an architecture BLIS has never seen".
    """
    cfg = DetectorConfig(window_days=1, recheck_known_architectures=True)
    config = dict(BIG_CONFIG, architectures=["LlamaForCausalLM"])
    summary = detector.scan(
        cfg,
        connectors=[FakeConnector("hf", [hf_signal(arch="LlamaForCausalLM", config=config)])],
        out_dir=tmp_path,
        now=NOW,
        trending=False,
    )
    assert [c.arch_id for c in summary.passed] == ["LlamaForCausalLM"]
    cand = summary.passed[0]
    assert cand.unparsed_fields == ["novel_mechanism_dim"]
    assert "T1-known-arch" in cand.triggers
    assert "T1" not in cand.triggers  # not a new architecture, and must not read as one
    assert (tmp_path / "LlamaForCausalLM.md").is_file()


def test_the_recheck_still_drops_a_seeded_architecture_with_an_inert_config(tmp_path):
    """The recheck must discriminate, not readmit everything it re-examines.

    Its own suppression reason, distinct from ``known_architecture``, is what lets the
    run log price the sweep: how much volume it costs against how much it finds.
    """
    cfg = DetectorConfig(window_days=1, recheck_known_architectures=True)
    inert = {k: v for k, v in BIG_CONFIG.items() if k != "novel_mechanism_dim"}
    inert["architectures"] = ["LlamaForCausalLM"]
    summary = detector.scan(
        cfg,
        connectors=[FakeConnector("hf", [hf_signal(arch="LlamaForCausalLM", config=inert)])],
        out_dir=tmp_path,
        now=NOW,
        trending=False,
    )
    assert summary.passed == []
    assert list(summary.suppressed_by_reason) == ["known_architecture_nothing_new"]
    assert list(tmp_path.glob("*.md")) == []


# ---------------------------------------------------------------------------
# the run log
# ---------------------------------------------------------------------------


def test_run_log_carries_aggregates_and_unabridged_detail(tmp_path, cfg):
    """The exact numbers here are readable only because ``cfg`` pins every knob they
    depend on (see the fixture). ``test_run_log_counts_tally_whatever_the_filter_does``
    covers the same log with assertions that survive re-calibration."""
    conns = [
        FakeConnector("hf", [hf_signal(), hf_signal(arch="LlamaForCausalLM", model_id="a/b")]),
        FakeConnector("vllm", error=RuntimeError("403 rate limited")),
    ]
    summary = run(
        cfg=cfg,
        connectors=conns,
        out_dir=tmp_path / "issues",
        surface=FakeSurface(known=("LlamaForCausalLM",)),
    )
    path = summary.write_run_log(tmp_path / ".runlog")

    assert path.parent == tmp_path / ".runlog"
    assert path.name == f"{summary.run_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema"] == detector.RUNLOG_SCHEMA
    assert data["dry_run"] is True and data["partial"] is True
    assert data["window"]["days"] == 1 and data["window"]["since"].endswith("Z")
    assert data["out_dir"] == str(tmp_path / "issues")

    # per-stage counts a human reads
    counts = data["counts"]
    assert counts["signals"] == 2
    assert counts["signals_by_source"] == {"hf": 2}
    assert counts["candidates"] == 2
    assert counts["passed"] == 1
    assert counts["suppressed"] == 1
    assert counts["suppressed_by_reason"] == {"known_architecture": 1}
    assert counts["suppressed_by_stage"] == {"suppressor": 1}
    assert counts["written"] == {"created": 1}
    assert counts["triggers"]["T1"] == 1
    assert counts["significance"]["S1"] == 1

    # the full detail a script reads
    assert [s["reason"] for s in data["suppressions"]] == ["known_architecture"]
    assert data["suppressions"][0]["arch_id"] == "LlamaForCausalLM"
    assert data["suppressions"][0]["detail"]

    # provenance: which sources worked, and what the run was configured with
    assert data["sources"]["failed"] == ["vllm"]
    assert "403 rate limited" in next(
        r["error"] for r in data["sources"]["reports"] if r["name"] == "vllm"
    )
    assert data["config"]["max_issues_per_run"] == 5
    assert data["config"]["thresholds"]["min_total_params"] == 30_000_000_000

    assert data["config"]["recheck_known_architectures"] is False

    row = data["issues"][0]
    assert row["arch_id"] == "NovelMechForCausalLM"
    assert row["status"] == "created" and row["path"].endswith(".md")
    assert row["rank"] == 1 and row["bucket"] is None


def test_run_log_accounts_for_every_candidate_exactly_once(tmp_path, cfg):
    """Unabridged is the requirement: counts cannot answer "which one did we miss?".

    Asserted as an invariant rather than a headcount — every candidate leaves exactly
    one trace, either in ``issues`` or in ``suppressions``, and nothing is elided or
    double-counted. That property is what makes the log usable for calibration, and it
    has to hold at *any* threshold, which is why this test does not name a number the
    backtest is still moving.
    """
    archs = [f"Cand{i}ForCausalLM" for i in range(40)]
    signals = [
        hf_signal(arch=arch, model_id=f"org{i}/model-{i}")
        for i, arch in enumerate(archs)
    ]
    summary = run(
        cfg=cfg,
        connectors=[FakeConnector("hf", signals)],
        out_dir=tmp_path,
        # Half the corpus is already in the seed set, so both outcomes are exercised.
        surface=FakeSurface(known=tuple(archs[:20])),
    )
    data = json.loads(summary.write_run_log(tmp_path / "log").read_text())
    counts = data["counts"]

    assert counts["candidates"] == len(archs)
    assert counts["passed"] + counts["suppressed"] == counts["candidates"]
    assert len(data["suppressions"]) == counts["suppressed"]  # nothing abridged
    assert len(data["issues"]) == counts["passed"]

    # Every record is attributable and machine-readable.
    for record in data["suppressions"]:
        assert record["reason"] and record["stage"] and record["detail"]
        assert record["arch_id"]
    assert sum(counts["suppressed_by_reason"].values()) == counts["suppressed"]
    assert sum(counts["suppressed_by_stage"].values()) == counts["suppressed"]

    # No candidate is lost, and none is recorded on both sides of the ledger.
    passed_ids = {row["arch_id"] for row in data["issues"]}
    dropped_ids = {record["arch_id"] for record in data["suppressions"]}
    assert passed_ids | dropped_ids == set(archs)
    assert not passed_ids & dropped_ids


def test_run_log_counts_tally_whatever_the_filter_does(tmp_path):
    """The same invariants under the shipped defaults, which no test should pin.

    ``DetectorConfig()`` on purpose: if a future calibration makes the aggregate counts
    stop adding up, this fails without anyone having to remember to re-derive a
    hard-coded number.
    """
    archs = ["LlamaForCausalLM", "NovelMechForCausalLM", "MysteryNetForCausalLM"]
    signals = [
        hf_signal(arch=arch, config=dict(BIG_CONFIG, architectures=[arch]),
                  model_id=f"moonshotai/{arch}-70B")
        for arch in archs
    ]
    # detector.scan directly, not the run() helper: this test wants the real surface
    # and the real defaults, which is exactly what the helper substitutes away.
    summary = detector.scan(
        DetectorConfig(),
        connectors=[FakeConnector("hf", signals)],
        out_dir=tmp_path,
        now=NOW,
        trending=False,
    )
    data = summary.as_dict()
    counts = data["counts"]
    assert counts["candidates"] == len(archs)
    assert counts["passed"] + counts["suppressed"] == counts["candidates"]
    assert len(data["suppressions"]) == counts["suppressed"]
    assert sum(counts["suppressed_by_reason"].values()) == counts["suppressed"]
    assert sum(counts["written"].values()) == counts["passed"]
    assert counts["signals"] == sum(counts["signals_by_source"].values())
    # Whatever fired, it is recorded per code and never exceeds the candidate count.
    for code, n in {**counts["triggers"], **counts["significance"]}.items():
        assert 0 < n <= counts["passed"], code


def test_run_log_does_not_overwrite_a_same_second_run(tmp_path, cfg):
    a = run(cfg=cfg, connectors=[FakeConnector("hf")], out_dir=tmp_path)
    b = run(cfg=cfg, connectors=[FakeConnector("hf")], out_dir=tmp_path)
    first = a.write_run_log(tmp_path / "log")
    second = b.write_run_log(tmp_path / "log")
    assert first != second
    assert second.name.endswith("-2.json")
    assert len(list((tmp_path / "log").glob("*.json"))) == 2


def test_run_log_survives_an_unserializable_extra(tmp_path, cfg):
    """``Signal.extra`` is documented as arbitrary; the run log must still write."""

    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    signal = hf_signal(extra={"perf": Weird()})
    summary = run(cfg=cfg, connectors=[FakeConnector("hf", [signal])], out_dir=tmp_path)
    data = json.loads(summary.write_run_log(tmp_path / "log").read_text())
    assert data["counts"]["passed"] == 1


def test_run_log_records_the_github_budget(tmp_path, cfg):
    budget = detector.GithubBudget(limit=4)
    budget.take()
    summary = run(cfg=cfg, connectors=[FakeConnector("hf")], out_dir=tmp_path)
    summary.github = budget
    data = summary.as_dict()
    assert data["github"] == {"budget": 4, "used": 1, "refused": 0, "remaining": 3}
    assert "1/4 requests used" in summary.text()


def test_default_runlog_dir_is_beside_the_package_not_the_cwd():
    assert detector.default_runlog_dir().name == ".runlog"
    assert detector.default_runlog_dir().parent == Path(emitter.__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# summary rendering
# ---------------------------------------------------------------------------


def test_summary_text_lists_every_stage_and_every_issue(tmp_path, cfg):
    summary = run(
        cfg=cfg, connectors=[FakeConnector("hf", [hf_signal()])], out_dir=tmp_path
    )
    text = summary.text()
    assert "archwatch scan" in text and "(dry-run)" in text
    assert "hf" in text and "signals" in text
    assert "candidates : 1 joined" in text
    assert "passed     : 1" in text
    assert "1. NovelMechForCausalLM" in text
    assert "created" in text
    assert "PARTIAL SCAN" not in text


def test_summary_counts_view_matches_the_report(tmp_path, cfg):
    summary = run(
        cfg=cfg, connectors=[FakeConnector("hf", [hf_signal()])], out_dir=tmp_path
    )
    counts = summary.counts()
    assert counts["passed"] == len(summary.passed) == 1
    assert counts["candidates"] == summary.candidates
    assert summary.issue_rows()[0]["triggers"] == summary.passed[0].triggers


def test_summary_trigger_counts_are_read_from_what_fired_not_a_fixed_list(cfg):
    """A new trigger code must appear in the run log the day it starts firing."""
    cand = Candidate(arch_id="X", display_name="X")
    cand.triggers = ["T9", "alias-join"]
    cand.significance = ["S3"]
    summary = detector.RunSummary(
        run_id="r",
        started_at=NOW,
        finished_at=NOW,
        window_days=1,
        since=NOW,
        out_dir=Path("."),
        passed=[cand],
    )
    assert summary.triggers_by_code == {"T9": 1, "alias-join": 1}
    assert summary.significance_by_code == {"S3": 1}


def test_suppressions_group_by_reason_and_stage():
    summary = detector.RunSummary(
        run_id="r",
        started_at=NOW,
        finished_at=NOW,
        window_days=1,
        since=NOW,
        out_dir=Path("."),
        suppressions=[
            Suppression("A", "A", "suppressor", "known_architecture", "d"),
            Suppression("B", "B", "suppressor", "known_architecture", "d"),
            Suppression("C", "C", "trigger", "no_trigger", "d"),
        ],
    )
    assert summary.suppressed == 3
    assert summary.suppressed_by_reason == {"known_architecture": 2, "no_trigger": 1}
    assert summary.suppressed_by_stage == {"suppressor": 2, "trigger": 1}


# ---------------------------------------------------------------------------
# structural guarantees
# ---------------------------------------------------------------------------


def test_detector_imports_no_network_library_at_module_scope():
    """``requests`` is imported lazily inside the session, and nothing else reaches out.

    A module-scope network import would mean ``--sources hf`` (and every unit test)
    pulls in the GitHub stack, and it is the first symptom of a live call creeping in.
    """
    tree = ast.parse(DETECTOR_SRC.read_text(encoding="utf-8"))
    top: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not top & {"requests", "urllib", "httpx", "socket", "huggingface_hub", "subprocess"}


def test_detector_contains_no_issue_writing_vocabulary():
    """PLAN.md rule 2: there is no code path that files a GitHub issue."""
    src = DETECTOR_SRC.read_text(encoding="utf-8")
    for banned in ("create_issue", "issues/comments", "gh issue create", "session.post"):
        assert banned not in src
