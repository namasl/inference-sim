"""Tests for the vLLM/SGLang framework connector.

Fully offline. Every HTTP response comes from a recorded cassette under
``tests/fixtures/frameworks/`` (see ``capture.py`` there for how they were
recorded and what was trimmed). ``FixtureSession`` is the only thing the
connector is ever given, so nothing here can reach the network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import pytest

from archwatch.connectors import frameworks as fw
from archwatch.connectors.base import Signal

FIXTURES = Path(__file__).parent / "fixtures" / "frameworks"

WINDOW_RECENT = FIXTURES / "cassette_window_2026-08-28.json"
WINDOW_KIMI = FIXTURES / "cassette_window_2026-07-27.json"
SYNTHETIC = FIXTURES / "cassette_synthetic_kimi_k3.json"
TITLES = FIXTURES / "observed_titles.json"
RATE_LIMIT = FIXTURES / "rate_limit_403.json"


# ---------------------------------------------------------------------------
# offline HTTP replay
# ---------------------------------------------------------------------------


def _request_key(url: str, params: dict | None) -> str:
    """MUST match tests/fixtures/frameworks/capture.py::request_key."""
    items = sorted((str(k), str(v)) for k, v in (params or {}).items())
    return f"{url}?{urlencode(items)}" if items else url


class _FakeResponse:
    def __init__(self, status: int, payload, headers: dict | None = None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        if self._payload is _UNPARSEABLE:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


_UNPARSEABLE = object()


class FixtureSession:
    """Replays a recorded cassette. Records misses instead of hiding them.

    ``fail_after``/``fail_status`` let a test flip the source into a rate-limit
    state partway through a poll.
    """

    def __init__(self, *cassettes: Path, fail_after: int | None = None,
                 fail_status: int = 403):
        self.cassette: dict[str, dict] = {}
        self.meta: list[dict] = []
        for path in cassettes:
            doc = json.loads(path.read_text())
            self.meta.append(doc.get("_meta", {}))
            self.cassette.update(doc.get("requests", {}))
        self.fail_after = fail_after
        self.fail_status = fail_status
        self.requests: list[str] = []
        self.misses: list[str] = []

    def get(self, url, params=None, headers=None, timeout=None):
        assert headers and headers.get("Accept") == "application/vnd.github+json"
        key = _request_key(url, params)
        self.requests.append(key)
        if self.fail_after is not None and len(self.requests) > self.fail_after:
            doc = json.loads(RATE_LIMIT.read_text())
            return _FakeResponse(self.fail_status, doc["json"], doc["headers"])
        rec = self.cassette.get(key)
        if rec is None:
            self.misses.append(key)
            return _FakeResponse(404, {"message": "Not Found"})
        return _FakeResponse(rec.get("status", 200), rec.get("json"), rec.get("headers"))


def _connector(*cassettes: Path, repos=None, **kwargs) -> fw.FrameworkConnector:
    session = kwargs.pop("session", None) or FixtureSession(*cassettes)
    conn = fw.FrameworkConnector(
        repos=repos,
        session=session,
        token="fixture-token",
        resolve_token_from_env=False,
        **kwargs,
    )
    conn.session = session  # convenience handle for assertions
    return conn


def _window(cassette: Path) -> tuple[datetime, datetime]:
    meta = json.loads(cassette.read_text())["_meta"]
    return (
        datetime.fromisoformat(meta["since"].replace("Z", "+00:00")),
        datetime.fromisoformat(meta["until"].replace("Z", "+00:00")),
    )


def _poll(cassette: Path, **kwargs) -> list[Signal]:
    since, until = _window(cassette)
    conn = _connector(cassette, until=until, **kwargs)
    signals = conn.poll(since)
    assert conn.session.misses == [], (
        f"cassette is missing responses for {conn.session.misses[:3]} — "
        "re-record with tests/fixtures/frameworks/capture.py"
    )
    return signals


@pytest.fixture(scope="module")
def recent_signals() -> list[Signal]:
    return _poll(WINDOW_RECENT)


@pytest.fixture(scope="module")
def kimi_signals() -> list[Signal]:
    return _poll(WINDOW_KIMI)


def _by_ref(signals) -> dict[tuple[str, str], Signal]:
    return {(s.source, s.raw_ref): s for s in signals}


# ---------------------------------------------------------------------------
# title classification, against the real observed-title corpus
# ---------------------------------------------------------------------------


def _title_cases():
    return json.loads(TITLES.read_text())["cases"]


@pytest.mark.parametrize(
    "case",
    [c for c in _title_cases() if "model_support" in c],
    ids=lambda c: f"{c['repo'].split('/')[-1]}#{c['number']}",
)
def test_title_gate_matches_hand_verified_labels(case):
    assert fw.is_model_support_title(case["title"]) is case["model_support"], (
        f"{case['title']!r} ({case['note']})"
    )


def test_title_gate_is_selective_over_the_raw_corpus():
    """The gate must actually discriminate, not wave everything through.

    The corpus is every merged PR whose title contains "model" or "support" in
    two one-week windows, i.e. deliberately adversarial.
    """
    cases = _title_cases()
    positives = [c for c in cases if fw.is_model_support_title(c["title"])]
    rate = len(positives) / len(cases)
    assert 0.05 < rate < 0.45, f"positive rate {rate:.1%} over {len(cases)} titles"


def test_maintenance_veto_catches_model_tagged_bugfixes():
    assert fw.is_maintenance_title("[Bugfix][Model] Fix FP8 PLE loading in mixed ModelOpt checkpoints")
    assert fw.is_maintenance_title("[rotary] Fix the fused Qwen3.5 RoPE kernel discarding mrope height and width")
    assert fw.is_maintenance_title("[Mypy] Fix typing for M models")
    assert fw.is_maintenance_title("[Model] Remove ten deprecated model architectures")
    assert not fw.is_maintenance_title("[Model] add GLM-5.3-Flash support")
    assert not fw.is_maintenance_title("[New model][Multimodal] Add DeepSeek-V4-Flash-Vision-Exp support")


def test_title_gate_handles_degenerate_input():
    for bad in ("", None, "[]", "[Model]", "   ", "[Model][Model][Model]"):
        assert fw.is_model_support_title(bad) in (True, False)


@pytest.mark.parametrize(
    "title,expected",
    [
        ("[Model] add GLM-5.3-Flash support", "GLM-5.3-Flash"),
        ("[Model] Add K2-Horizon model support", "K2-Horizon"),
        ("[Model] Support Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next"),
        ("[New model][Multimodal] Add DeepSeek-V4-Flash-Vision-Exp support",
         "DeepSeek-V4-Flash-Vision-Exp"),
        ("[Hy4] support Hy4-preview model", "Hy4-preview"),
        ("[Model] Add native IFM K2 Horizon serving support", "IFM K2 Horizon"),
        ("Qwen3.8-27B Model Support", "Qwen3.8-27B"),
        ("Add Spark3 Model", "Spark3"),
        ("[Model] Support Ling-3.0-flash (BailingMoeV3)", "Ling-3.0-flash"),
        ("[New model] Kimi K3", "Kimi K3"),
        ("[Quantization][Autoround][XPU] Support AutoRound MXFP8 MoE models",
         "AutoRound MXFP8 MoE"),
    ],
)
def test_display_name_from_real_titles(title, expected):
    assert fw.display_name_from_title(title) == expected


def test_display_name_never_empty_for_nonempty_title():
    for case in _title_cases():
        assert fw.display_name_from_title(case["title"]).strip(), case["title"]


# ---------------------------------------------------------------------------
# architecture-name shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,ok",
    [
        ("KimiK3ForCausalLM", True),
        ("Glm5NextForConditionalGeneration", True),
        ("Ernie4_5_MoeForCausalLM", True),  # underscores are real in HF names
        ("Qwen4ExpMTP", True),
        ("Glm5NextMTPModel", True),
        ("K2HorizonModel", False),          # internal submodule, not an arch
        ("K2HorizonAttention", False),
        ("nn", False),
        ("MTP", False),
        ("FORCAUSALLM", False),
        ("", False),
        ("moonshotai/Kimi-K3", False),
    ],
)
def test_looks_like_arch_name(name, ok):
    assert fw.looks_like_arch_name(name) is ok


def test_normalize_arch_name_strips_helper_suffixes():
    # vLLM really does define `class Qwen4ExpForCausalLMConfig(...)`.
    assert fw.normalize_arch_name("Qwen4ExpForCausalLMConfig") == "Qwen4ExpForCausalLM"
    assert fw.normalize_arch_name("KimiK3ForConditionalGenerationConfig") == (
        "KimiK3ForConditionalGeneration"
    )
    assert fw.normalize_arch_name("K2HorizonForCausalLM") == "K2HorizonForCausalLM"


def test_arch_extraction_from_the_idealized_plan_title():
    """PLAN.md's stated acceptance case (see observed_titles.json: no real title
    in either repo has ever looked like this)."""
    assert fw.extract_arch_names_from_text("[Model] Add KimiK3ForCausalLM") == [
        "KimiK3ForCausalLM"
    ]


def test_arch_extraction_from_prose_ignores_non_arch_camelcase():
    text = (
        "Adds support for https://huggingface.co/moonshotai/Kimi-K3-Instruct. "
        "Registers KimiK3ForCausalLM and KimiK3MTPModel. Touches FlashAttention "
        "and RMSNorm; see ModelRunner and KimiK3Attention."
    )
    assert fw.extract_arch_names_from_text(text, allow_mtp=True) == [
        "KimiK3ForCausalLM",
        "KimiK3MTPModel",
    ]
    assert fw.extract_model_ids_from_text(text) == ["moonshotai/Kimi-K3-Instruct"]


# ---------------------------------------------------------------------------
# changed-file mining, on real recorded /pulls/{n}/files payloads
# ---------------------------------------------------------------------------


def _recorded_files(cassette: Path, repo: str, number: int) -> list[dict]:
    doc = json.loads(cassette.read_text())
    out: list[dict] = []
    for key, resp in doc["requests"].items():
        if f"/repos/{repo}/pulls/{number}/files?" in key:
            payload = resp.get("json") or []
            if isinstance(payload, list):
                out.extend(payload)
    assert out, f"no recorded files for {repo}#{number}"
    return out


def test_extract_from_files_vllm_registry_addition_raw_patch():
    """vLLM #55063 is recorded with its patches untrimmed (see cassette _meta)."""
    files = _recorded_files(WINDOW_RECENT, "vllm-project/vllm", 55063)
    found = fw.extract_from_files(files, fw.VLLM)

    assert found.registry_archs == ["K2HorizonForCausalLM"]
    assert found.touched_registry is True
    assert "K2HorizonForCausalLM" in found.arch_names
    # the registry key is authoritative, so it must come first
    assert found.arch_names[0] == "K2HorizonForCausalLM"
    assert "k2_horizon" in found.module_stems
    assert "vllm/model_executor/models/k2_horizon.py" in found.added_model_files
    # the HF repo id shows up in tests/models/registry.py's _HfExamplesInfo
    assert "IFM/K2-Horizon-36B" in found.model_ids
    # internal submodule classes must not be mistaken for architectures
    assert not any(n.endswith("Attention") or n == "K2HorizonModel" for n in found.arch_names)


def test_extract_from_files_sglang_entry_class():
    """SGLang has no registry; architectures come from `EntryClass = [...]`."""
    files = _recorded_files(WINDOW_RECENT, "sgl-project/sglang", 37654)
    found = fw.extract_from_files(files, fw.SGLANG)

    assert "XllmForCausalLM" in found.entry_class_archs
    assert "K2HorizonForCausalLM" in found.entry_class_archs
    assert "python/sglang/srt/models/xllm.py" in found.added_model_files
    assert "K2HorizonConfig" in found.config_classes
    assert found.touches_model_code is True


def test_extract_from_files_vllm_package_style_new_model():
    """New vLLM architectures now land in vllm/models/<pkg>/, not
    vllm/model_executor/models/<mod>.py (PLAN.md only names the latter)."""
    files = _recorded_files(WINDOW_RECENT, "vllm-project/vllm", 53906)
    found = fw.extract_from_files(files, fw.VLLM)

    assert found.registry_archs[:2] == [
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
    ]
    assert "glm5next" in found.module_stems
    assert any(f.startswith("vllm/models/glm5next/") for f in found.added_model_files)
    assert "zai-org/GLM-5.3-Flash" in found.model_ids


def test_extract_from_files_tolerates_junk():
    junk = [
        None,
        {},
        {"filename": None},
        {"filename": "vllm/model_executor/models/x.py"},               # no patch key
        {"filename": "vllm/model_executor/models/y.py", "patch": None},
        {"filename": "vllm/model_executor/models/z.py", "patch": 12345},
        {"filename": "README.md", "status": "added", "patch": "+class AForCausalLM(x):"},
        "not-a-dict",
    ]
    found = fw.extract_from_files(junk, fw.VLLM)  # must not raise
    assert found.arch_names == []  # README is outside the model paths


def test_module_stem_ignores_shared_modules():
    assert fw._module_stem("vllm/model_executor/models/k2_horizon.py", fw.VLLM) == "k2_horizon"
    assert fw._module_stem("vllm/models/glm5next/nvidia/model.py", fw.VLLM) == "glm5next"
    assert fw._module_stem("python/sglang/srt/models/xllm.py", fw.SGLANG) == "xllm"
    for shared in (
        "vllm/model_executor/models/registry.py",
        "vllm/model_executor/models/interfaces.py",
        "vllm/model_executor/models/utils.py",
        "vllm/model_executor/models/__init__.py",
        "vllm/attention/backends/flash_attn.py",
    ):
        assert fw._module_stem(shared, fw.VLLM) is None


# ---------------------------------------------------------------------------
# poll() over the recent real window
# ---------------------------------------------------------------------------

# (source, pr number, architecture that MUST be the primary key)
EXPECTED_RECENT = [
    ("vllm", "55063", "K2HorizonForCausalLM"),
    ("vllm", "53906", "Glm5NextForCausalLM"),
    ("vllm", "53896", "Qwen4ExpForCausalLM"),
    ("vllm", "54566", "DeepseekV4ForConditionalGeneration"),
    ("vllm", "54160", "HYV4ForCausalLM"),
    ("sglang", "37654", "XllmForCausalLM"),
]


@pytest.mark.parametrize("source,ref,arch", EXPECTED_RECENT)
def test_recent_window_finds_each_real_model_pr(recent_signals, source, ref, arch):
    sig = _by_ref(recent_signals).get((source, ref))
    assert sig is not None, f"{source} PR #{ref} missing from {[s.raw_ref for s in recent_signals]}"
    assert sig.primary_arch() == arch
    assert sig.urls["pr"].endswith(f"/pull/{ref}")


def test_signal_contract(recent_signals):
    assert recent_signals, "no signals from the recorded window"
    for s in recent_signals:
        assert isinstance(s, Signal)
        assert s.source in ("vllm", "sglang")
        assert s.config is None  # a PR carries no config.json
        assert set(s.urls) == {"pr"}
        assert s.urls["pr"].startswith("https://github.com/")
        assert s.raw_ref.isdigit()  # the PR number
        assert s.urls["pr"].endswith("/pull/" + s.raw_ref)
        assert s.extra["pr_number"] == int(s.raw_ref)
        assert s.display_name
        assert s.evidence
        assert s.model_type is None
        assert all(fw.looks_like_arch_name(a) for a in s.arch_ids)
        if s.org is not None:
            assert s.org == s.org.lower()


def test_observed_at_is_timezone_aware_utc(recent_signals):
    for s in recent_signals:
        assert s.observed_at.tzinfo is not None, f"{s.raw_ref} has a naive datetime"
        assert s.observed_at.utcoffset() == timedelta(0), f"{s.raw_ref} is not UTC"


def test_signals_are_within_the_window_and_newest_first(recent_signals):
    since, until = _window(WINDOW_RECENT)
    for s in recent_signals:
        assert since <= s.observed_at <= until, f"#{s.raw_ref} at {s.observed_at}"
    stamps = [s.observed_at for s in recent_signals]
    assert stamps == sorted(stamps, reverse=True)


def test_one_signal_per_repo_pr(recent_signals):
    keys = [(s.source, s.raw_ref) for s in recent_signals]
    assert len(keys) == len(set(keys))


def test_strong_signals_carry_provenance(recent_signals):
    sig = _by_ref(recent_signals)[("vllm", "53906")]
    assert sig.extra["signal_strength"] == "registry"
    assert sig.extra["repo"] == "vllm-project/vllm"
    assert "registry_commit" in sig.extra["discovery_routes"]
    assert sig.model_ids and sig.org == "zai-org"
    assert "Glm5NextForCausalLM" in sig.evidence
    assert sig.display_name == "GLM-5.3-Flash"


def test_maintenance_prs_are_not_emitted(recent_signals):
    """Bugfixes that touch model code must not become signals."""
    refs = {(s.source, s.raw_ref) for s in recent_signals}
    for source, ref in [
        ("vllm", "54882"),   # [Bugfix][Model] Fix FP8 PLE loading
        ("vllm", "54262"),   # [Mypy] Fix typing for M models
        ("vllm", "54753"),   # [CI] Shard basic model initialization tests
        ("vllm", "54380"),   # [Model] Honor cap_pixels_per_frame ...
        ("sglang", "34446"), # [rotary] Fix the fused Qwen3.5 RoPE kernel ...
        ("sglang", "37750"), # [Docs] Refresh TPU model list
    ]:
        assert (source, ref) not in refs


def test_both_sources_are_represented(recent_signals):
    assert {s.source for s in recent_signals} == {"vllm", "sglang"}


def test_arch_yield_of_the_window(recent_signals):
    """The design assumption: most emitted signals do resolve an arch name."""
    with_arch = [s for s in recent_signals if s.arch_ids]
    assert len(with_arch) / len(recent_signals) >= 0.6
    strengths = {s.extra["signal_strength"] for s in with_arch}
    assert "registry" in strengths


def test_emit_title_only_false_drops_the_weak_tail():
    strict = _poll(WINDOW_RECENT, emit_title_only=False)
    loose = _poll(WINDOW_RECENT, emit_title_only=True)
    assert len(strict) < len(loose)
    assert all(s.extra["signal_strength"] != "title_only" for s in strict)
    # every strong signal survives
    strong = {(s.source, s.raw_ref) for s in loose if s.arch_ids}
    assert strong <= {(s.source, s.raw_ref) for s in strict}


# ---------------------------------------------------------------------------
# poll() over the historical Kimi-K3 window (a second real event)
# ---------------------------------------------------------------------------


def test_kimi_k3_window(kimi_signals):
    by_ref = _by_ref(kimi_signals)
    umbrella = by_ref.get(("vllm", "50000"))
    assert umbrella is not None, "[New model] Kimi K3 not detected"
    assert "KimiK3ForConditionalGeneration" in umbrella.arch_ids
    assert umbrella.display_name == "Kimi K3"

    files_pr = by_ref.get(("vllm", "50089"))
    assert files_pr is not None
    assert "KimiK3ForConditionalGeneration" in files_pr.arch_ids

    all_archs = {a for s in kimi_signals for a in s.arch_ids}
    assert {"Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"} <= all_archs


def test_windows_do_not_leak_into_each_other(kimi_signals, recent_signals):
    assert not ({s.raw_ref for s in kimi_signals} & {s.raw_ref for s in recent_signals})


# ---------------------------------------------------------------------------
# synthetic cassette: PLAN.md's idealized title, end to end
# ---------------------------------------------------------------------------


def test_synthetic_plan_acceptance_case():
    signals = _poll(SYNTHETIC, repos=(fw.VLLM,))
    assert len(signals) == 1
    sig = signals[0]
    assert sig.source == "vllm"
    assert sig.raw_ref == "99999"
    assert sig.primary_arch() == "KimiK3ForCausalLM"
    assert "KimiK3MTPModel" in sig.arch_ids  # from the body's architectures list
    assert sig.extra["arch_from_title"] == ["KimiK3ForCausalLM"]
    assert sig.model_ids == ["moonshotai/Kimi-K3-Instruct"]
    assert sig.org == "moonshotai"
    assert sig.config is None
    assert sig.urls == {"pr": "https://github.com/vllm-project/vllm/pull/99999"}


# ---------------------------------------------------------------------------
# rate limits and other source failures: partial results, never an exception
# ---------------------------------------------------------------------------


def test_403_partway_through_returns_partial_results():
    since, until = _window(WINDOW_RECENT)
    full = _poll(WINDOW_RECENT)
    session = FixtureSession(WINDOW_RECENT, fail_after=12)
    conn = fw.FrameworkConnector(
        session=session, token="t", resolve_token_from_env=False, until=until
    )
    partial = conn.poll(since)  # must not raise
    assert conn.http.rate_limited is True
    assert 0 < len(partial) < len(full)
    assert all(isinstance(s, Signal) for s in partial)
    assert any("rate limited" in e for e in conn.http.errors)


def test_403_on_the_very_first_call_yields_no_signals_and_no_exception():
    since, until = _window(WINDOW_RECENT)
    session = FixtureSession(WINDOW_RECENT, fail_after=0)
    conn = fw.FrameworkConnector(
        session=session, token="t", resolve_token_from_env=False, until=until
    )
    assert conn.poll(since) == []
    assert conn.http.rate_limited is True


def test_429_is_treated_as_a_rate_limit():
    since, until = _window(WINDOW_RECENT)
    session = FixtureSession(WINDOW_RECENT, fail_after=0, fail_status=429)
    conn = fw.FrameworkConnector(
        session=session, token="t", resolve_token_from_env=False, until=until
    )
    assert conn.poll(since) == []
    assert conn.http.rate_limited is True


def test_one_repo_failing_does_not_stop_the_other():
    class HalfBrokenSession(FixtureSession):
        def get(self, url, params=None, headers=None, timeout=None):
            if "sgl-project" in url:
                raise OSError("connection reset by peer")
            return super().get(url, params, headers, timeout)

    since, until = _window(WINDOW_RECENT)
    conn = fw.FrameworkConnector(
        session=HalfBrokenSession(WINDOW_RECENT),
        token="t",
        resolve_token_from_env=False,
        until=until,
    )
    signals = conn.poll(since)
    assert signals
    assert {s.source for s in signals} == {"vllm"}


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, "a string", 42, [None, "x", 3], {"items": "not-a-list"}, _UNPARSEABLE],
)
def test_malformed_payloads_never_raise(payload):
    class WeirdSession:
        def get(self, url, params=None, headers=None, timeout=None):
            return _FakeResponse(200, payload)

    conn = fw.FrameworkConnector(
        session=WeirdSession(), token="t", resolve_token_from_env=False
    )
    assert conn.poll(datetime(2026, 8, 28, tzinfo=timezone.utc)) == []


def test_http_error_statuses_never_raise():
    class ErrSession:
        def __init__(self):
            self.n = 0

        def get(self, url, params=None, headers=None, timeout=None):
            self.n += 1
            return _FakeResponse(500 if self.n % 2 else 404, {"message": "boom"})

    conn = fw.FrameworkConnector(
        session=ErrSession(), token="t", resolve_token_from_env=False
    )
    assert conn.poll(datetime(2026, 8, 28, tzinfo=timezone.utc)) == []
    assert conn.http.rate_limited is False  # 4xx/5xx that are not 403/429


# ---------------------------------------------------------------------------
# window handling, caps, subclasses, token resolution
# ---------------------------------------------------------------------------


def test_naive_since_is_coerced_to_utc():
    since, until = _window(WINDOW_RECENT)
    conn = _connector(WINDOW_RECENT, until=until)
    signals = conn.poll(since.replace(tzinfo=None))  # naive input
    assert signals
    assert all(s.observed_at.tzinfo is not None for s in signals)


def test_until_excludes_later_prs():
    since, until = _window(WINDOW_RECENT)
    conn = _connector(WINDOW_RECENT, until=since + timedelta(days=2))
    early = conn.poll(since)
    assert all(s.observed_at <= since + timedelta(days=2) for s in early)
    assert len(early) < len(_poll(WINDOW_RECENT))


def test_max_prs_per_repo_caps_the_work():
    since, until = _window(WINDOW_RECENT)
    conn = _connector(WINDOW_RECENT, until=until, max_prs_per_repo=1)
    signals = conn.poll(since)
    assert len(signals) <= 2  # at most one examined PR per repo
    assert len(signals) < len(_poll(WINDOW_RECENT))


def test_single_repo_subclasses():
    assert fw.VllmConnector.name == "vllm"
    assert fw.SglangConnector.name == "sglang"
    v = fw.VllmConnector(session=FixtureSession(WINDOW_RECENT), token="t",
                         resolve_token_from_env=False)
    assert [r.source for r in v.repos] == ["vllm"]
    s = fw.SglangConnector(session=FixtureSession(WINDOW_RECENT), token="t",
                           resolve_token_from_env=False)
    assert [r.source for r in s.repos] == ["sglang"]

    since, until = _window(WINDOW_RECENT)
    v = fw.VllmConnector(session=FixtureSession(WINDOW_RECENT), token="t",
                         resolve_token_from_env=False, until=until)
    assert {sig.source for sig in v.poll(since)} == {"vllm"}


def test_repo_specs_cover_the_paths_plan_md_names():
    assert "vllm/model_executor/models/" in fw.VLLM.model_path_prefixes
    assert "python/sglang/srt/models/" in fw.SGLANG.model_path_prefixes
    assert fw.VLLM.source == "vllm" and fw.SGLANG.source == "sglang"
    assert [r.source for r in fw.DEFAULT_REPOS] == ["vllm", "sglang"]


def test_token_resolution_prefers_env(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "from-gh-token")
    monkeypatch.setenv("GITHUB_TOKEN", "from-github-token")
    assert fw.resolve_token() == "from-gh-token"
    monkeypatch.delenv("GH_TOKEN")
    assert fw.resolve_token() == "from-github-token"
    assert fw.resolve_token("explicit") == "explicit"


def test_token_resolution_falls_back_to_gh_cli(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    calls = []

    class Proc:
        returncode = 0
        stdout = "gh-cli-token\n"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Proc()

    monkeypatch.setattr(fw.subprocess, "run", fake_run)
    assert fw.resolve_token() == "gh-cli-token"
    assert calls == [["gh", "auth", "token"]]


def test_token_resolution_returns_none_when_gh_is_missing(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def boom(cmd, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(fw.subprocess, "run", boom)
    assert fw.resolve_token() is None


def test_token_is_sent_as_a_bearer_header_and_nothing_is_written():
    session = FixtureSession(WINDOW_RECENT)
    conn = fw.FrameworkConnector(
        session=session, token="secret", resolve_token_from_env=False
    )
    headers = conn.http._headers()
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Accept"] == "application/vnd.github+json"
    conn.poll(datetime(2026, 8, 28, tzinfo=timezone.utc))
    # read-only: every recorded request is a GET, and only GET exists on the client
    assert not hasattr(conn.http, "post")
    assert all(k.startswith("https://api.github.com/") for k in session.requests)


def test_no_network_helper_is_used_by_these_tests(monkeypatch):
    """A connector built without a session would import requests; every test
    above passes one in, so nothing here can dial out."""
    import requests

    def explode(*a, **k):
        raise AssertionError("tests must not construct a real requests.Session")

    monkeypatch.setattr(requests, "Session", explode)
    session = FixtureSession(WINDOW_RECENT)
    conn = fw.FrameworkConnector(session=session, token="t", resolve_token_from_env=False)
    assert conn.http.session is session


# ---------------------------------------------------------------------------
# fixture hygiene
# ---------------------------------------------------------------------------


def test_cassettes_declare_their_provenance():
    for path in (WINDOW_RECENT, WINDOW_KIMI, SYNTHETIC):
        meta = json.loads(path.read_text())["_meta"]
        assert meta["since"] and meta["until"]
        assert meta.get("synthetic") or meta["patch_trimmed"] is True


def test_recorded_cassettes_only_contain_github_api_gets():
    for path in (WINDOW_RECENT, WINDOW_KIMI, SYNTHETIC):
        for key in json.loads(path.read_text())["requests"]:
            assert key.startswith("https://api.github.com/")
