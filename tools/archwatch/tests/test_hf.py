"""Offline tests for the HuggingFace connector (component C).

Everything here runs from recorded fixtures under ``tests/fixtures/hf/``:

* ``list_window.json`` — 63 records captured from a real
  ``GET https://huggingface.co/api/models?sort=createdAt&expand=...`` over a
  one-day window (2026-09-03/04), curated down from ~3.5k so that every branch
  under test is exercised by real data: quant/merge/LoRA/GGUF repo ids, repeated
  architectures, repos whose architecture the Hub has not indexed, and genuine
  frontier-looking releases.
* ``list_trending.json`` — 20 records from the same endpoint sorted by
  ``trendingScore``; all were created well before the window above, which is the
  point of the S4 sweep.
* ``configs/<org>__<name>.json`` — that repo's real ``config.json``. Absent for
  repos that genuinely have none (a 404 or a gated 403 at capture time), which
  is exactly the "config is unobtainable" path the connector must tolerate.

The only edit made to the recordings is dropping ``config.tokenizer_config``
from the listing records (chat-template noise the connector never reads).

No test touches the network. The connector's ``api`` and ``config_fetcher`` are
injected, and the ``no_network`` autouse fixture below makes
``huggingface_hub.HfApi`` / ``hf_hub_download`` blow up if anything reaches for
the real thing.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from huggingface_hub import ModelInfo
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, HfHubHTTPError

from archwatch.config import DERIVATIVE_PATTERNS, DetectorConfig, Thresholds
from archwatch.connectors.base import Signal
from archwatch.connectors.hf import (
    DEFAULT_TRENDING_LIMIT,
    LIST_EXPAND,
    NON_LM_LIBRARIES,
    NON_LM_PIPELINE_TAGS,
    HFConnector,
    architectures_of,
    is_derivative,
    is_non_lm_artifact,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hf"
CONFIGS = FIXTURES / "configs"

FIXED_NOW = datetime(2026, 9, 4, 21, 40, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# fixture plumbing
# ---------------------------------------------------------------------------


def _records(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())


def _model_info(raw: dict) -> ModelInfo:
    """Rebuild a ModelInfo exactly the way ``HfApi.list_models`` does."""
    raw = dict(raw)
    raw.setdefault("siblings", None)
    return ModelInfo(**raw)


def _http_error(status: int, message: str) -> HfHubHTTPError:
    request = httpx.Request("GET", "https://huggingface.co/x/resolve/main/config.json")
    return HfHubHTTPError(message, response=httpx.Response(status, request=request))


class FakeApi:
    """Stands in for ``HfApi``, replaying recorded listing records in order."""

    def __init__(self, records: list[dict], *, fail_after: int | None = None) -> None:
        self._records = records
        self.fail_after = fail_after
        self.calls: list[dict] = []

    def list_models(self, *, sort=None, limit=None, expand=None, **kwargs):
        self.calls.append({"sort": sort, "limit": limit, "expand": expand, **kwargs})
        for n, raw in enumerate(self._records):
            if limit is not None and n >= limit:
                return
            if self.fail_after is not None and n >= self.fail_after:
                raise _http_error(429, "429 Client Error: Too Many Requests")
            yield _model_info(raw)


class LegacyFakeApi(FakeApi):
    """A huggingface_hub 0.x-shaped API: camelCase sort keys plus ``direction``."""

    def list_models(self, *, sort=None, direction=None, limit=None, expand=None, **kwargs):
        return super().list_models(sort=sort, limit=limit, expand=expand,
                                   direction=direction, **kwargs)


class FixtureFetcher:
    """``config_fetcher`` reading recorded config.json files off disk."""

    def __init__(self, overrides: dict | None = None) -> None:
        self.overrides = overrides or {}
        self.calls: list[str] = []

    def __call__(self, repo_id: str, revision: str | None) -> str | None:
        self.calls.append(repo_id)
        if repo_id in self.overrides:
            value = self.overrides[repo_id]
            if isinstance(value, BaseException):
                raise value
            return value
        path = CONFIGS / (repo_id.replace("/", "__") + ".json")
        if not path.exists():
            raise EntryNotFoundError(f"404: no config.json in {repo_id}")
        return path.read_text()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Hard guarantee: reaching for the live Hub inside a test is an error."""
    import huggingface_hub

    def boom(*args, **kwargs):  # pragma: no cover - only fires on a regression
        raise AssertionError("test attempted a live huggingface_hub call")

    monkeypatch.setattr(huggingface_hub, "HfApi", boom)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)


@pytest.fixture
def cfg() -> DetectorConfig:
    return DetectorConfig(window_days=1, thresholds=Thresholds(), max_hf_config_fetches=200)


def make_connector(records, cfg, *, api_cls=FakeApi, fetcher=None, fail_after=None, **kw):
    fetcher = fetcher if fetcher is not None else FixtureFetcher()
    api = api_cls(records, fail_after=fail_after)
    conn = HFConnector(cfg, api=api, config_fetcher=fetcher, clock=lambda: FIXED_NOW, **kw)
    return conn, api, fetcher


def window_since() -> datetime:
    """A ``since`` that keeps the whole recorded window."""
    return datetime(2026, 9, 3, tzinfo=timezone.utc)


def survivor_ids(records: list[dict]) -> set[str]:
    """The repo ids phase 1 should keep — both drops applied, computed from the
    fixture rather than hard-coded, so the expectation tracks the vocabularies."""
    return {
        r["id"]
        for r in records
        if not is_derivative(r["id"])
        and not (not architectures_of(r.get("config")) and is_non_lm_artifact(_model_info(r)))
    }


# ---------------------------------------------------------------------------
# the installed huggingface_hub API is what the connector assumes
# ---------------------------------------------------------------------------


def test_installed_hub_api_is_v1_shaped():
    """Documents the contract the connector codes against (no network).

    huggingface_hub 1.x renamed the sort keys to snake_case and dropped
    ``direction`` entirely. If this ever fails, the connector's ``_list_kwargs``
    compatibility branch is what needs revisiting.
    """
    # reach past the ``no_network`` monkeypatch to the real class
    from huggingface_hub.hf_api import ExpandModelProperty_T, HfApi, ModelSort_T

    params = inspect.signature(HfApi.list_models).parameters
    assert "sort" in params and "expand" in params and "limit" in params
    assert "direction" not in params
    assert "created_at" in ModelSort_T.__args__
    assert "trending_score" in ModelSort_T.__args__
    # every field the connector expands must be a legal expand property
    assert set(LIST_EXPAND) <= set(ExpandModelProperty_T.__args__)


def test_sort_kwargs_match_the_installed_api(cfg):
    conn, api, _ = make_connector(_records("list_window.json"), cfg)
    conn.poll(window_since())
    call = api.calls[0]
    assert call["sort"] == "created_at"
    assert "direction" not in call
    assert call["expand"] == list(LIST_EXPAND)


def test_sort_kwargs_fall_back_to_legacy_0x_style(cfg):
    """A 0.x-shaped api object gets camelCase sort + direction=-1."""
    conn, api, _ = make_connector(_records("list_window.json"), cfg, api_cls=LegacyFakeApi)
    conn.poll(window_since())
    assert api.calls[0]["sort"] == "createdAt"
    assert api.calls[0]["direction"] == -1

    conn2, api2, _ = make_connector(_records("list_trending.json"), cfg, api_cls=LegacyFakeApi)
    conn2.poll_trending(limit=5)
    assert api2.calls[0]["sort"] == "trendingScore"
    assert api2.calls[0]["direction"] == -1


# ---------------------------------------------------------------------------
# phase 1 — the cheap pre-filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repo_id",
    [
        "Qwen/Qwen3-8B-GGUF",                       # the acceptance example
        "unsloth/Qwen3.8-27B-GGUF",                 # real, from the trending fixture
        "OliviaRossi/QuadQwen-Q5_K_M-GGUF",         # real, from the window fixture
        "420yolomcswaggerpants/nimbus-sft-3b-lora",
        "twokings22/gemma-banking77-merged",
        "Aliados/gemma-3-1b-it-int4-awq",
        "someone/Model-abliterated",
        "mlx-community/Whatever",
    ],
)
def test_prefilter_recognises_derivatives(repo_id):
    assert is_derivative(repo_id)


@pytest.mark.parametrize(
    "repo_id",
    ["moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "zai-org/GLM-5.3", "gpt2"],
)
def test_prefilter_keeps_plain_repos(repo_id):
    assert not is_derivative(repo_id)


def test_derivative_repos_never_reach_the_signal_list(cfg):
    records = _records("list_window.json")
    conn, _, fetcher = make_connector(records, cfg)
    signals = conn.poll(window_since())

    emitted = {s.model_ids[0] for s in signals}
    assert "OliviaRossi/QuadQwen-Q5_K_M-GGUF" not in emitted
    assert "Aliados/gemma-3-1b-it-int4-awq" not in emitted
    assert "twokings22/gemma-banking77-merged" not in emitted
    # ... and the base repo of that GGUF is kept
    assert "OliviaRossi/QuadQwen" in emitted

    dropped = [r["id"] for r in records if is_derivative(r["id"])]
    assert dropped, "fixture must contain derivative repos to make this test meaningful"
    assert emitted.isdisjoint(dropped)
    # a dropped repo never costs a config fetch either
    assert set(fetcher.calls).isdisjoint(dropped)
    for s in signals:
        assert not any(p in s.model_ids[0].lower() for p in DERIVATIVE_PATTERNS)


def test_window_boundary_excludes_older_repos(cfg):
    records = _records("list_window.json")
    # records are createdAt-descending; cut the window at the 20th record
    cut = datetime.fromisoformat(records[20]["createdAt"].replace("Z", "+00:00"))
    conn, _, _ = make_connector(records, cfg)
    signals = conn.poll(cut)

    assert {s.model_ids[0] for s in signals} == survivor_ids(records[:21])
    for s in signals:
        assert datetime.fromisoformat(s.extra["created_at"]) >= cut


def test_naive_since_is_treated_as_utc(cfg):
    records = _records("list_window.json")
    conn, _, _ = make_connector(records, cfg)
    naive = datetime(2026, 9, 3, 0, 0, 0)  # no tzinfo
    assert len(conn.poll(naive)) == len(survivor_ids(records))


# ---------------------------------------------------------------------------
# phase 2 — configs, arch ids, and the fetch budget
# ---------------------------------------------------------------------------


def test_signals_carry_arch_ids_and_real_configs(cfg):
    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    signals = {s.model_ids[0]: s for s in conn.poll(window_since())}

    kimi = signals["foranyone2026/Kimi-K3"]
    assert kimi.source == "hf"
    assert kimi.arch_ids == ["KimiK3ForConditionalGeneration"]
    assert kimi.primary_arch() == "KimiK3ForConditionalGeneration"
    assert kimi.model_type == "kimi_k3"
    assert kimi.org == "foranyone2026"
    assert kimi.display_name == "Kimi-K3"
    assert kimi.raw_ref == "foranyone2026/Kimi-K3"
    assert kimi.urls["hf"] == "https://huggingface.co/foranyone2026/Kimi-K3"
    assert kimi.urls["config"].endswith("/config.json")
    assert kimi.observed_at == FIXED_NOW
    # the real config.json, not the Hub's indexed excerpt
    assert kimi.config is not None
    assert kimi.config["architectures"] == ["KimiK3ForConditionalGeneration"]
    assert kimi.extra["config_source"] == "config.json"
    # a real multimodal config: the shape fields live under text_config, and
    # 2026-era configs spell the precision field ``dtype``, not ``torch_dtype``
    assert kimi.config["text_config"]["hidden_size"] == 7168
    assert kimi.config["dtype"] == "bfloat16"

    glm = signals["foranyone/GLM-5.3-BF16"]
    assert glm.arch_ids == ["GlmMoeDsaForCausalLM"]
    assert glm.config is not None and glm.config.get("num_hidden_layers")


def test_survivors_without_a_fetched_config_still_get_arch_from_the_listing(cfg):
    """The Hub's indexed ``config`` excerpt gives the join key for free."""
    tight = DetectorConfig(max_hf_config_fetches=1)
    conn, _, fetcher = make_connector(_records("list_window.json"), tight)
    signals = [s for s in conn.poll(window_since()) if s.arch_ids]

    assert len(fetcher.calls) == 1
    from_listing = [s for s in signals if s.extra["config_source"] == "hub-listing"]
    assert len(from_listing) >= 20
    for s in from_listing:
        assert s.config is None
        assert s.arch_ids and s.model_type


def test_signals_are_emitted_for_every_survivor(cfg):
    records = _records("list_window.json")
    conn, _, _ = make_connector(records, cfg)
    signals = conn.poll(window_since())
    assert {s.model_ids[0] for s in signals} == survivor_ids(records)
    # including the ones with no architecture anywhere
    headless = [s for s in signals if not s.arch_ids]
    assert headless, "fixture must contain repos whose architecture is unknown"
    for s in headless:
        assert s.extra["config_source"] in (None, "config.json")
        assert s.display_name


@pytest.mark.parametrize("cap", [0, 1, 5, 12])
def test_config_fetch_cap_is_honoured(cap):
    conn, _, fetcher = make_connector(
        _records("list_window.json"), DetectorConfig(max_hf_config_fetches=cap)
    )
    signals = conn.poll(window_since())
    assert len(fetcher.calls) <= cap
    assert len([s for s in signals if s.config is not None]) <= cap
    assert len(signals) > cap  # the cap limits fetches, not signals


def test_config_fetches_are_deduplicated_by_architecture():
    """One fetch per *distinct* architecture — the whole point of the two phases."""
    records = _records("list_window.json")
    conn, _, fetcher = make_connector(records, DetectorConfig(max_hf_config_fetches=200))
    conn.poll(window_since())

    hub_arch = {
        r["id"]: (architectures_of(r.get("config")) or [None])[0]
        for r in records
    }
    fetched_archs = [hub_arch[rid] for rid in fetcher.calls if hub_arch[rid]]
    assert len(fetched_archs) == len(set(fetched_archs)), "same architecture fetched twice"

    survivors = [r for r in records if r["id"] in survivor_ids(records)]
    distinct = {hub_arch[r["id"]] for r in survivors if hub_arch[r["id"]]}
    assert set(fetched_archs) == distinct, "every distinct architecture needs one fetch"

    # e.g. three repos share BertModel in the window; only one is downloaded
    berts = [r["id"] for r in survivors if hub_arch[r["id"]] == "BertModel"]
    assert len(berts) >= 3
    assert len([rid for rid in fetcher.calls if rid in berts]) == 1


def test_frontier_orgs_win_the_fetch_budget():
    records = _records("list_trending.json")
    conn, _, fetcher = make_connector(records, DetectorConfig(max_hf_config_fetches=3))
    conn.poll_trending(limit=DEFAULT_TRENDING_LIMIT)
    orgs = {rid.split("/")[0].lower() for rid in fetcher.calls}
    assert orgs <= DetectorConfig().frontier_orgs, f"budget spent outside frontier orgs: {orgs}"


# ---------------------------------------------------------------------------
# never raise on ordinary source problems
# ---------------------------------------------------------------------------


def test_config_failures_degrade_to_config_none(cfg):
    """404, 403 gated, 429, malformed JSON, and a non-object body are all survivable."""
    overrides = {
        "foranyone2026/Kimi-K3": EntryNotFoundError("404: no config.json"),
        "foranyone/GLM-5.3-BF16": GatedRepoError(
            "403 gated repo",
            response=httpx.Response(403, request=httpx.Request("GET", "https://hf.co/x")),
        ),
        "foranyone/GLM-5.3-Flash-BF16": _http_error(429, "429 Too Many Requests"),
        "OliviaRossi/QuadQwen": '{"architectures": ["Qwen3_5MoeForCausalLM",',  # truncated
        "VERBAREX/LuminoLex-1.5B-think": "[1, 2, 3]",  # valid JSON, wrong shape
        "ananjayram/Hana": None,  # fetcher legitimately has nothing
        "Dexmal/DM05-MEM": RuntimeError("something nobody predicted"),
    }
    conn, _, fetcher = make_connector(
        _records("list_window.json"), cfg, fetcher=FixtureFetcher(overrides)
    )
    signals = {s.model_ids[0]: s for s in conn.poll(window_since())}  # must not raise

    for repo_id in overrides:
        assert repo_id in signals, f"{repo_id} must still yield a Signal"
        assert signals[repo_id].config is None
        assert signals[repo_id].extra["config_source"] == "hub-listing"
        assert "config" not in signals[repo_id].urls
        # the join key survives, because it came from the listing
        assert signals[repo_id].arch_ids

    # unaffected repos still got their real configs
    assert signals["Openintelligent123/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"].config


def test_missing_config_json_is_not_an_error(cfg):
    """Repos with genuinely no config.json (404/403 at capture time) still emit."""
    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    signals = {s.model_ids[0]: s for s in conn.poll(window_since())}
    naked = signals["nur-dev/strata-native-lm"]  # transformers-tagged, no config.json
    assert naked.config is None
    assert naked.arch_ids == []
    assert naked.primary_arch() is None
    assert naked.extra["config_source"] is None
    assert naked.extra["library_name"] == "transformers"


def test_listing_failure_returns_partial_results(cfg):
    """A 429 mid-pagination degrades to a partial scan, per the Connector protocol."""
    records = _records("list_window.json")
    conn, _, _ = make_connector(records, cfg, fail_after=8)
    signals = conn.poll(window_since())  # must not raise
    assert {s.model_ids[0] for s in signals} == survivor_ids(records[:8])


def test_trending_listing_failure_returns_partial_results(cfg):
    conn, _, _ = make_connector(_records("list_trending.json"), cfg, fail_after=4)
    assert 0 < len(conn.poll_trending(limit=20)) <= 4


def test_empty_listing_is_fine(cfg):
    conn, _, fetcher = make_connector([], cfg)
    assert conn.poll(window_since()) == []
    assert conn.poll_trending(limit=10) == []
    assert fetcher.calls == []


# ---------------------------------------------------------------------------
# poll_trending — the S4 sweep
# ---------------------------------------------------------------------------


def test_poll_trending_ignores_creation_date(cfg):
    records = _records("list_trending.json")
    conn, api, _ = make_connector(records, cfg)
    signals = conn.poll_trending(limit=20)

    assert api.calls[0]["sort"] == "trending_score"
    assert api.calls[0]["limit"] == 20
    # every recorded trending repo predates the creation window used above
    cutoff = datetime(2026, 9, 3, tzinfo=timezone.utc)
    assert all(datetime.fromisoformat(s.extra["created_at"]) < cutoff for s in signals)
    assert signals, "trending sweep must return something"

    for s in signals:
        assert s.source == "hf"
        assert s.extra["trending"] is True
        assert s.extra["phase"] == "trending"
        assert "trending" in s.evidence

    by_id = {s.model_ids[0]: s for s in signals}
    qwen = by_id["Qwen/Qwen3.8-Flash-Next"]
    assert qwen.arch_ids == ["Qwen4ExpForConditionalGeneration"]
    assert qwen.org == "Qwen"
    assert qwen.config is not None  # frontier org, so it won the fetch budget
    assert qwen.extra["trending_score"] > 0


def test_poll_trending_applies_the_derivative_prefilter(cfg):
    conn, _, _ = make_connector(_records("list_trending.json"), cfg)
    emitted = {s.model_ids[0] for s in conn.poll_trending(limit=20)}
    assert "unsloth/Qwen3.8-27B-GGUF" not in emitted
    assert "Qwen/Qwen3.8-27B" in emitted


def test_poll_trending_respects_limit(cfg):
    conn, _, _ = make_connector(_records("list_trending.json"), cfg)
    assert len(conn.poll_trending(limit=3)) <= 3


def test_window_signals_are_not_marked_trending(cfg):
    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    for s in conn.poll(window_since()):
        assert s.extra["trending"] is False
        assert s.extra["phase"] == "window"


# ---------------------------------------------------------------------------
# the significance gate's inputs (component F reads these)
# ---------------------------------------------------------------------------


def test_extra_carries_downloads_and_likes(cfg):
    conn, _, _ = make_connector(_records("list_trending.json"), cfg)
    signals = {s.model_ids[0]: s for s in conn.poll_trending(limit=20)}

    raw = {r["id"]: r for r in _records("list_trending.json")}
    for repo_id, s in signals.items():
        assert s.extra["downloads"] == raw[repo_id]["downloads"]
        assert s.extra["likes"] == raw[repo_id]["likes"]
        assert isinstance(s.extra["downloads"], int)
        assert isinstance(s.extra["likes"], int)
        assert s.extra["downloads_all_time"] >= 0
        assert set(s.extra) >= {
            "downloads", "likes", "downloads_all_time", "trending_score", "trending",
            "phase", "config_source", "created_at", "last_modified", "library_name",
            "pipeline_tag", "tags", "gated", "private", "sha",
        }

    big = signals["Qwen/Qwen3.8-27B"]
    assert big.extra["downloads"] > DetectorConfig().thresholds.min_model_downloads
    assert big.extra["likes"] > DetectorConfig().thresholds.min_model_likes


def test_missing_counters_become_zero_not_none(cfg):
    """The gate compares against ints, so absent counters must not be None."""
    kept = survivor_ids(_records("list_window.json"))
    records = [dict(r) for r in _records("list_window.json") if r["id"] in kept][:1]
    for key in ("downloads", "likes", "downloadsAllTime", "trendingScore"):
        records[0].pop(key, None)
    conn, _, _ = make_connector(records, cfg)
    (signal,) = conn.poll(window_since())
    assert signal.extra["downloads"] == 0
    assert signal.extra["likes"] == 0
    assert signal.extra["downloads_all_time"] == 0
    assert signal.extra["trending_score"] == 0


# ---------------------------------------------------------------------------
# Signal shape conformance against the frozen contract
# ---------------------------------------------------------------------------


def test_every_signal_conforms_to_the_frozen_contract(cfg):
    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    trend_conn, _, _ = make_connector(_records("list_trending.json"), cfg)
    signals = conn.poll(window_since()) + trend_conn.poll_trending(limit=20)
    assert signals

    for s in signals:
        assert isinstance(s, Signal)
        assert s.source == "hf"
        assert isinstance(s.observed_at, datetime) and s.observed_at.tzinfo is not None
        assert s.model_ids and s.model_ids == [s.raw_ref]
        assert isinstance(s.arch_ids, list)
        assert all(isinstance(a, str) and a for a in s.arch_ids)
        assert s.primary_arch() == (s.arch_ids[0] if s.arch_ids else None)
        assert s.display_name and "/" not in s.display_name
        assert s.org is None or "/" not in s.org
        assert s.urls["hf"].startswith("https://huggingface.co/")
        assert s.evidence
        assert s.config is None or isinstance(s.config, dict)
        assert s.model_type is None or isinstance(s.model_type, str)


def test_connector_satisfies_the_connector_protocol(cfg):
    conn, _, _ = make_connector([], cfg)
    assert conn.name == "hf"
    assert callable(conn.poll)
    assert list(inspect.signature(HFConnector.poll).parameters) == ["self", "since"]


# ---------------------------------------------------------------------------
# architectures_of
# ---------------------------------------------------------------------------


def test_architectures_of_reads_real_configs():
    deepseek = json.loads((CONFIGS / "deepseek-ai__DeepSeek-V4-Flash-Vision-Exp.json").read_text())
    assert architectures_of(deepseek) == ["DeepseekV4ForCausalLM"]
    qwen = json.loads((CONFIGS / "Qwen__Qwen3.8-27B.json").read_text())
    assert architectures_of(qwen) == ["Qwen3_5ForConditionalGeneration"]


def test_architectures_of_falls_back_to_the_text_tower():
    """Some multimodal configs name the causal LM only under ``text_config``."""
    assert architectures_of({"model_type": "x", "text_config": {"architectures": ["FooForCausalLM"]}}) == [
        "FooForCausalLM"
    ]
    # a top-level list always wins
    assert architectures_of(
        {"architectures": ["TopLevel"], "text_config": {"architectures": ["Nested"]}}
    ) == ["TopLevel"]


@pytest.mark.parametrize(
    "config, expected",
    [
        (None, []),
        ({}, []),
        ({"architectures": []}, []),
        ({"architectures": None}, []),
        ({"architectures": "SingleString"}, ["SingleString"]),
        ({"architectures": ["A", "A", "B"]}, ["A", "B"]),
        ({"architectures": [" Padded "]}, ["Padded"]),
        ({"architectures": ["", None, 7, "Real"]}, ["Real"]),
        ("not a dict", []),
        ({"text_config": "not a dict"}, []),
    ],
)
def test_architectures_of_tolerates_junk(config, expected):
    assert architectures_of(config) == expected


# ---------------------------------------------------------------------------
# timestamps are aware UTC (the emitter derives detected_at from observed_at)
# ---------------------------------------------------------------------------


def test_observed_at_is_aware_utc_by_default(cfg):
    """The default clock must produce an aware UTC instant, not a naive one."""
    records = _records("list_window.json")
    conn = HFConnector(cfg, api=FakeApi(records), config_fetcher=FixtureFetcher())
    before = datetime.now(timezone.utc)
    signals = conn.poll(window_since())
    after = datetime.now(timezone.utc)
    for s in signals:
        assert s.observed_at.tzinfo is not None
        assert s.observed_at.utcoffset() == timedelta(0)
        assert before <= s.observed_at <= after
    # one poll, one instant — so detected_at is stable across a run
    assert len({s.observed_at for s in signals}) == 1


def test_observed_at_is_normalised_even_from_a_bad_clock(cfg):
    """A naive or non-UTC clock must not leak a host-dependent timestamp out."""
    naive_local = datetime(2026, 9, 4, 21, 40)  # no tzinfo
    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    conn._clock = lambda: naive_local
    (signal, *_) = conn.poll(window_since())
    assert signal.observed_at == naive_local.replace(tzinfo=timezone.utc)

    plus_nine = timezone(timedelta(hours=9))
    conn2, _, _ = make_connector(_records("list_window.json"), cfg)
    conn2._clock = lambda: datetime(2026, 9, 5, 6, 40, tzinfo=plus_nine)
    (signal2, *_) = conn2.poll(window_since())
    assert signal2.observed_at.utcoffset() == timedelta(0)
    assert signal2.observed_at == datetime(2026, 9, 4, 21, 40, tzinfo=timezone.utc)


def test_extra_timestamps_are_aware_utc_iso_strings(cfg):
    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    for s in conn.poll(window_since()):
        for key in ("created_at", "last_modified"):
            value = s.extra[key]
            if value is None:
                continue
            parsed = datetime.fromisoformat(value)
            assert parsed.tzinfo is not None, f"{key} is naive: {value}"
            assert parsed.utcoffset() == timedelta(0)


def test_config_fetch_budget_prefers_parsed_configs_in_the_unknown_bucket():
    """An unknown-arch repo whose listing excerpt has a model_type is fetched first.

    Those are repos the Hub *did* parse a config.json for, which just declares
    no ``architectures[]`` — a custom model_type. Live, those have a fetchable
    config.json essentially always; the rest of the unknown bucket rarely does.
    """
    records = _records("list_window.json")
    kept = survivor_ids(records)
    distinct = len({architectures_of(r.get("config"))[0] for r in records
                    if r["id"] in kept and architectures_of(r.get("config"))})
    # cap leaves exactly one slot for the unknown-architecture bucket
    conn, _, fetcher = make_connector(
        records, DetectorConfig(max_hf_config_fetches=distinct + 1)
    )
    conn.poll(window_since())

    by_id = {r["id"]: r for r in records}
    tail = [rid for rid in fetcher.calls if not architectures_of(by_id[rid].get("config"))]
    assert tail == ["ParallaxOpen/Vela-Lumen-31M"]
    assert by_id[tail[0]]["config"] == {"model_type": "small_lm"}


def test_archless_custom_config_still_yields_a_usable_signal(cfg):
    """A real repo with a config.json but no ``architectures[]``.

    ``ParallaxOpen/Vela-Lumen-31M`` declares ``model_type: small_lm`` and names
    its shape fields ``d_model``/``n_layers``/``n_heads`` — nothing BLIS parses.
    The connector must still surface it, config and all, with no architecture;
    the detector's alias path picks it up on ``display_name``.
    """
    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    signals = {s.model_ids[0]: s for s in conn.poll(window_since())}
    s = signals["ParallaxOpen/Vela-Lumen-31M"]
    assert s.arch_ids == []
    assert s.primary_arch() is None
    assert s.model_type == "small_lm"
    assert s.display_name == "Vela-Lumen-31M"
    assert s.config is not None
    assert s.config["d_model"] == 512
    assert "hidden_size" not in s.config
    assert s.extra["config_source"] == "config.json"


# ---------------------------------------------------------------------------
# the non-LM pre-filter — conservative by construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repo_id, why",
    [
        ("lloydchristmas1231/deniaya-claude-35", "library_name=diffusers, pipeline=text-to-image"),
        ("SteveNguyen/sugar_cup_chunkrel_pi05", "library_name=lerobot, pipeline=robotics"),
        ("bunnycore/LMF2.5-2.6B-Hunter", "library_name=peft (an adapter, not an architecture)"),
        ("yizhouzhao-nv/cosmos3-edge-libero-goal-20260904-185434", "pipeline=robotics, no library"),
        ("kunalmiind/10Eros-Max", "pipeline=image-text-to-video, no library"),
    ],
)
def test_non_lm_repos_are_dropped(cfg, repo_id, why):
    """Real archless repos with positive non-LM evidence never become Signals."""
    records = _records("list_window.json")
    assert any(r["id"] == repo_id for r in records), "fixture lost its test subject"
    conn, _, fetcher = make_connector(records, cfg)
    emitted = {s.model_ids[0] for s in conn.poll(window_since())}
    assert repo_id not in emitted, f"should have been dropped: {why}"
    # and it costs no config fetch either
    assert repo_id not in fetcher.calls


@pytest.mark.parametrize(
    "repo_id, library, pipeline",
    [
        # no metadata whatsoever — the zero-day case, must survive
        ("sportsgirl/pic03", None, None),
        ("highlands/ipo02", None, None),
        ("0xgevdhc/0904", None, None),
        # transformers with no pipeline tag: ambiguous, so kept
        ("vera6/tuning_09_05", "transformers", None),
        # an architecture-less custom LM
        ("ParallaxOpen/Vela-Lumen-31M", "transformers", "text-generation"),
    ],
)
def test_archless_repos_without_non_lm_evidence_are_kept(cfg, repo_id, library, pipeline):
    """Absence of evidence is never evidence. When in doubt, keep."""
    records = _records("list_window.json")
    raw = next(r for r in records if r["id"] == repo_id)
    assert raw.get("library_name") == library
    assert raw.get("pipeline_tag") == pipeline
    assert not architectures_of(raw.get("config")), "subject must be architecture-less"

    conn, _, _ = make_connector(records, cfg)
    emitted = {s.model_ids[0] for s in conn.poll(window_since())}
    assert repo_id in emitted


@pytest.mark.parametrize(
    "repo_id, arch, evidence",
    [
        ("27aran/germantris-model", "BertModel", "library_name=sentence-transformers"),
        ("huwenjie333/whisper-v3-ft-af51-0903", "WhisperForConditionalGeneration",
         "pipeline_tag=automatic-speech-recognition"),
    ],
)
def test_repos_with_an_architecture_are_kept_whatever_the_library(cfg, repo_id, arch, evidence):
    """An architecture is the primary key: once we have one, modality is moot.

    Both subjects would be dropped on their metadata alone, so this proves the
    architecture check runs first rather than being incidental.
    """
    records = _records("list_window.json")
    raw = next(r for r in records if r["id"] == repo_id)
    assert is_non_lm_artifact(_model_info(raw)), f"subject must carry {evidence}"

    conn, _, _ = make_connector(records, cfg)
    signals = {s.model_ids[0]: s for s in conn.poll(window_since())}
    assert repo_id in signals, f"kept despite {evidence}, because it names {arch}"
    assert signals[repo_id].arch_ids == [arch]


def test_prefilter_counts_the_two_drops_separately(cfg, caplog):
    """Tuning DERIVATIVE_PATTERNS and the non-LM lists needs distinguishable counts."""
    records = _records("list_window.json")
    with caplog.at_level("INFO", logger="archwatch.connectors.hf"):
        conn, _, _ = make_connector(records, cfg)
        signals = conn.poll(window_since())

    line = next(m for m in caplog.messages if "pre-filter" in m)
    n_derivative = len([r for r in records if is_derivative(r["id"])])
    n_non_lm = len(records) - n_derivative - len(signals)
    assert n_derivative and n_non_lm, "fixture must exercise both drops"
    assert f"{n_derivative} derivative" in line
    assert f"{n_non_lm} non-LM" in line
    assert f"of {len(records)} repos" in line
    assert f"{len(signals)} survive" in line


def test_non_lm_evidence_is_read_from_library_then_pipeline():
    """library_name outranks pipeline_tag: a PEFT adapter is not an architecture."""

    class Info:
        def __init__(self, library=None, pipeline=None):
            self.id, self.library_name, self.pipeline_tag = "x/y", library, pipeline

    assert is_non_lm_artifact(Info(library="peft", pipeline="text-generation"))
    assert is_non_lm_artifact(Info(library="transformers", pipeline="image-classification"))
    assert not is_non_lm_artifact(Info(library="transformers", pipeline="text-generation"))
    assert not is_non_lm_artifact(Info())
    # case and whitespace are not evidence either way
    assert is_non_lm_artifact(Info(library=" Diffusers "))
    assert is_non_lm_artifact(Info(pipeline="Robotics"))


def test_generic_and_unknown_libraries_are_never_treated_as_evidence():
    """Runtime/format labels and one-off vendor libraries must not be drop reasons.

    Quant-packaging runtimes hold a language model (that is
    DERIVATIVE_PATTERNS' business), and an unrecognized library from an
    unexpected lab is the zero-day case itself.
    """
    ambiguous = {
        "transformers", "pytorch", "keras", "onnx", "onnxruntime", "tensorrt",
        "mlx", "gguf", "llama.cpp", "exllamav3", "safetensors", "custom",
        "generic", "nemo", "vllm", "sglang",
        # real one-off library names observed live
        "loom-py-rt", "minimax-h3", "karume", "coreai", "hipfire", "z1t",
        "recurrent-recall-circuits", "voice",
    }
    assert ambiguous.isdisjoint(NON_LM_LIBRARIES)

    lm_tasks = {
        "text-generation", "text2text-generation", "image-text-to-text",
        "any-to-any", "feature-extraction", "sentence-similarity", "fill-mask",
        "text-classification", "token-classification", "question-answering",
        "translation", "summarization", "zero-shot-classification",
        "document-question-answering", "visual-question-answering",
        "image-to-text",
    }
    assert lm_tasks.isdisjoint(NON_LM_PIPELINE_TAGS)


def test_non_lm_filter_applies_to_the_trending_sweep_too(cfg):
    """Same junk problem, same filter — a diffusion checkpoint is not an architecture."""
    records = _records("list_trending.json")
    conn, _, _ = make_connector(records, cfg)
    emitted = {s.model_ids[0] for s in conn.poll_trending(limit=20)}

    assert "Lightricks/LTX-2.5" not in emitted          # image-to-video
    assert "google/timesfm-3.0-pytorch" not in emitted   # time-series-forecasting
    # the language models on the same list are untouched
    assert "Qwen/Qwen3.8-Flash-Next" in emitted
    assert "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp" in emitted
    assert "zai-org/GLM-5.3" in emitted
