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
    LM_PIPELINE_TAGS_ALWAYS_KEEP,
    NON_LM_LIBRARIES,
    NON_LM_PIPELINE_TAGS,
    STRONG_NON_LM_LIBRARIES,
    STRONG_NON_LM_MODEL_TYPES,
    STRONG_NON_LM_PIPELINE_TAGS,
    HFConnector,
    architectures_of,
    is_derivative,
    is_non_lm_artifact,
    strong_non_lm_evidence,
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
    """Stands in for ``HfApi``, replaying recorded listing records.

    Dispatches the way the real endpoint does, because the connector makes three
    different kinds of call: the creation/trending window listing, the
    descending-by-downloads org popularity sweep, and a per-``author`` lookup.
    A fake that ignored ``author`` and ``sort`` would let the org tests pass
    while measuring nothing.
    """

    def __init__(
        self,
        records: list[dict],
        *,
        fail_after: int | None = None,
        top_downloads: list[dict] | None = None,
        org_models: dict[str, list[dict]] | None = None,
        fail_orgs: tuple[str, ...] = (),
    ) -> None:
        self._records = records
        self.fail_after = fail_after
        self._top_downloads = top_downloads
        self._org_models = org_models
        self._fail_orgs = {o.lower() for o in fail_orgs}
        self.calls: list[dict] = []

    @property
    def org_calls(self) -> list[str]:
        return [c["author"] for c in self.calls if c.get("author")]

    @property
    def sweep_calls(self) -> list[dict]:
        return [c for c in self.calls
                if not c.get("author") and c["sort"] in ("downloads", "downloadsAllTime")]

    def list_models(self, *, sort=None, limit=None, expand=None, author=None, **kwargs):
        self.calls.append(
            {"sort": sort, "limit": limit, "expand": expand, "author": author, **kwargs}
        )
        if author is not None:
            if author.lower() in self._fail_orgs:
                raise _http_error(429, "429 Client Error: Too Many Requests")
            source = (self._org_models or {}).get(author.lower(), [])
            fail_after = None
        elif sort == "downloads":
            source = self._top_downloads if self._top_downloads is not None else self._records
            source = sorted(source, key=lambda r: -(r.get("downloads") or 0))
            fail_after = None
        else:
            source = self._records
            fail_after = self.fail_after
        for n, raw in enumerate(source):
            if limit is not None and n >= limit:
                return
            if fail_after is not None and n >= fail_after:
                raise _http_error(429, "429 Client Error: Too Many Requests")
            yield _model_info(raw)


class LegacyFakeApi(FakeApi):
    """A huggingface_hub 0.x-shaped API: camelCase sort keys plus ``direction``."""

    def list_models(self, *, sort=None, direction=None, limit=None, expand=None,
                    author=None, **kwargs):
        return super().list_models(sort=sort, limit=limit, expand=expand,
                                   author=author, direction=direction, **kwargs)


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


def make_connector(records, cfg, *, api_cls=FakeApi, fetcher=None, fail_after=None,
                   top_downloads=None, org_models=None, fail_orgs=(), **kw):
    """Build a connector over recorded records.

    ``org_downloads`` defaults to False here: most tests are about the window
    pipeline, and leaving the S2 lookups on would have them silently exercise
    an org sweep they make no assertions about. The org tests turn it on
    explicitly and pass the org fixtures.
    """
    fetcher = fetcher if fetcher is not None else FixtureFetcher()
    api = api_cls(records, fail_after=fail_after, top_downloads=top_downloads,
                  org_models=org_models, fail_orgs=fail_orgs)
    kw.setdefault("org_downloads", False)
    conn = HFConnector(cfg, api=api, config_fetcher=fetcher, clock=lambda: FIXED_NOW, **kw)
    return conn, api, fetcher


def window_since() -> datetime:
    """A ``since`` that keeps the whole recorded window."""
    return datetime(2026, 9, 3, tzinfo=timezone.utc)


def keeps(raw: dict) -> bool:
    """Whether phase 1 keeps this record — all three drops, mirroring _prefilter.

    Derived from the fixture rather than hard-coded so the expectations track the
    vocabularies instead of freezing today's counts.
    """
    if is_derivative(raw["id"]):
        return False
    info = _model_info(raw)
    strong = strong_non_lm_evidence(info)
    if architectures_of(raw.get("config")):
        return strong is None
    return not (strong or is_non_lm_artifact(info))


def survivor_ids(records: list[dict]) -> set[str]:
    return {r["id"] for r in records if keeps(r)}


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

    survivors = [r for r in records if keeps(r)]
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
        "Akahsizrr/Cyber-Prime-1-2.6B": RuntimeError("something nobody predicted"),
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
    assert qwen.org == "qwen"  # lowercased to match FRONTIER_ORGS
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
        # sentence-transformers hosts text encoders, so it is only weak evidence
        ("27aran/germantris-model", "BertModel", "library_name=sentence-transformers"),
    ],
)
def test_weak_non_lm_evidence_does_not_override_an_architecture(cfg, repo_id, arch, evidence):
    """The broad archless vocabularies do NOT get to drop an arch-bearing repo.

    ``sentence-transformers`` and ``peft`` are in NON_LM_LIBRARIES but not in
    STRONG_NON_LM_LIBRARIES: they host or wrap text models, which is not the
    same as not being a language model.
    """
    records = _records("list_window.json")
    raw = next(r for r in records if r["id"] == repo_id)
    assert is_non_lm_artifact(_model_info(raw)), f"subject must carry {evidence}"
    assert strong_non_lm_evidence(_model_info(raw)) is None, "evidence must be weak"

    conn, _, _ = make_connector(records, cfg)
    signals = {s.model_ids[0]: s for s in conn.poll(window_since())}
    assert repo_id in signals
    assert signals[repo_id].arch_ids == [arch]


def test_prefilter_counts_the_three_drops_separately(cfg, caplog):
    """Each drop is tuned against a different list, so each needs its own count."""
    records = _records("list_window.json") + [
        dict(_records("arch_evidence_records.json")["cubert-gmbh/sam3"],
             createdAt="2026-09-04T12:00:00.000Z"),
    ]
    with caplog.at_level("INFO", logger="archwatch.connectors.hf"):
        conn, _, _ = make_connector(records, cfg)
        signals = conn.poll(window_since())

    line = next(m for m in caplog.messages if "pre-filter" in m)
    n_derivative = len([r for r in records if is_derivative(r["id"])])
    n_arch = len([r for r in records if not is_derivative(r["id"])
                  and architectures_of(r.get("config"))
                  and strong_non_lm_evidence(_model_info(r))])
    n_archless = len(records) - n_derivative - n_arch - len(signals)
    assert n_derivative and n_archless and n_arch, "fixture must exercise all three"
    assert f"{n_derivative} derivative" in line
    assert f"{n_archless} non-LM (no architecture)" in line
    assert f"{n_arch} non-LM (named architecture)" in line
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


# ---------------------------------------------------------------------------
# S2 — org track record (the open-world path)
# ---------------------------------------------------------------------------
#
# Fixtures, all recorded live:
#   top_downloads.json     the Hub's 300 most-downloaded models (the sweep input),
#                          spanning 253.8M .. 1.57M 30-day downloads.
#   org_models.json        {org: its top 5 models by 30-day downloads} — what a
#                          per-org fallback lookup sees.
#   org_probe_records.json {org: one real full-expand record it owns} — used as
#                          synthetic *window* input. Only `createdAt` is
#                          overridden, because these repos were not created in
#                          the recorded window and the point under test is org
#                          attribution, not the window boundary.
#
# SWEEP_THRESHOLD sits inside the recorded sweep's range so the sweep terminates
# within the fixture; that is what makes "absent from the sweep" conclusive.

SWEEP_THRESHOLD = 5_000_000
IN_WINDOW = "2026-09-04T12:00:00.000Z"


def org_cfg(threshold: int = SWEEP_THRESHOLD, **kw) -> DetectorConfig:
    return DetectorConfig(thresholds=Thresholds(min_org_top_downloads=threshold), **kw)


def probe_record(org: str) -> dict:
    """One real repo owned by ``org``, dated into the recorded window."""
    return dict(_records("org_probe_records.json")[org], createdAt=IN_WINDOW)


def make_org_connector(records, *, cfg=None, **kw):
    return make_connector(
        records,
        cfg or org_cfg(),
        top_downloads=_records("top_downloads.json"),
        org_models=_records("org_models.json"),
        org_downloads=True,
        **kw,
    )


def test_org_top_downloads_comes_from_the_bulk_sweep():
    """BAAI is not in FRONTIER_ORGS but owns a 566M-download model — S2 must see it.

    This is the open-world case: without this path, a lab absent from the
    hand-written allowlist can never clear the significance gate on reputation.
    """
    conn, api, _ = make_org_connector([probe_record("baai")])
    (signal,) = conn.poll(window_since())

    assert signal.org == "baai"
    assert signal.org not in DetectorConfig().frontier_orgs
    in_sweep = [r for r in _records("top_downloads.json")
                if (r.get("author") or "").lower() == "baai"
                and (r.get("downloads") or 0) >= SWEEP_THRESHOLD]
    assert len(in_sweep) > 1, "fixture must hold several BAAI models for max() to matter"
    assert signal.extra["org_top_downloads"] == max(r["downloadsAllTime"] for r in in_sweep)
    assert signal.extra["org_top_downloads"] == 566_091_432
    assert signal.extra["org_top_downloads_basis"] == "downloads_all_time"
    assert signal.extra["org_top_downloads"] > DetectorConfig().thresholds.min_org_top_downloads
    # answered by the sweep, so no per-org request was spent
    assert api.org_calls == []


def test_org_lookup_falls_back_per_org_when_the_sweep_misses():
    """A dormant lab still gets a real answer.

    MBZUAI's best model has 198,796 all-time downloads — over the default
    ``min_org_top_downloads`` — but only 76k in the last 30 days, so it never
    reaches a downloads-sorted sweep. This is the only case the fallback exists
    for, and the reason the sweep alone is not the whole answer.
    """
    conn, api, _ = make_org_connector([probe_record("mbzuai")])
    (signal,) = conn.poll(window_since())

    assert signal.org == "mbzuai"
    assert api.org_calls == ["mbzuai"], "exactly one per-org request"
    top5 = _records("org_models.json")["mbzuai"]
    assert signal.extra["org_top_downloads"] == max(r["downloadsAllTime"] for r in top5)
    assert signal.extra["org_top_downloads"] == 198_796
    assert signal.extra["org_top_downloads_basis"] == "downloads_all_time"
    assert signal.extra["org_top_downloads"] >= DetectorConfig().thresholds.min_org_top_downloads
    # ... and its 30-day figure would NOT have cleared the gate
    assert max(r["downloads"] for r in top5) < DetectorConfig().thresholds.min_org_top_downloads


def test_org_lookup_takes_the_best_across_the_orgs_top_models():
    """bunnycore's best all-time model is not its best 30-day model.

    Exercised at the lookup level because bunnycore's own repos do not pass the
    fallback eligibility gate — which is itself the point: an org's *track
    record* spans its whole catalogue, including the three GGUF repos the
    pre-filter would drop as derivatives.
    """
    conn, _, _ = make_org_connector([])
    top5 = _records("org_models.json")["bunnycore"]
    assert conn._lookup_org("bunnycore") == (
        max(r["downloadsAllTime"] for r in top5),
        max(r["downloads"] for r in top5),
    ) == (1723, 1437), "all-time best and 30-day best are different repos"
    assert conn._pick_metric(1723, 1437) == (1723, "downloads_all_time")
    assert len([r for r in top5 if is_derivative(r["id"])]) == 3


def test_org_is_queried_once_per_org_not_once_per_repo():
    """Two real repos from one org in one window cost one request, cached."""
    records = [r for r in _records("list_window.json")
               if (r.get("author") or "").lower() == "nkthebass"]
    assert len(records) == 2, "fixture must hold two repos from one org"
    conn, api, _ = make_org_connector(records)
    signals = conn.poll(window_since())

    assert len(signals) == 2
    assert api.org_calls == ["nkthebass"]
    top5 = _records("org_models.json")["nkthebass"]
    assert {s.extra["org_top_downloads"] for s in signals} == {
        max(r["downloadsAllTime"] for r in top5)
    }


def test_frontier_orgs_are_never_looked_up():
    """They satisfy S2 by membership, so the request would be wasted."""
    conn, api, _ = make_org_connector(_records("list_trending.json"))
    signals = {s.model_ids[0]: s for s in conn.poll_trending(limit=20)}

    frontier = DetectorConfig().frontier_orgs
    assert {o.lower() for o in api.org_calls}.isdisjoint(frontier)
    qwen = signals["Qwen/Qwen3.8-Flash-Next"]
    assert qwen.org in frontier
    assert "org_top_downloads" not in qwen.extra
    assert "org_top_downloads_basis" not in qwen.extra


def test_failed_org_lookup_omits_the_key_rather_than_writing_zero():
    """A missing key means "unknown" and fails S2; a 0 would assert unpopularity."""
    conn, _, _ = make_org_connector([probe_record("mbzuai")], fail_orgs=("mbzuai",))
    (signal,) = conn.poll(window_since())  # a 429 must not raise
    assert "org_top_downloads" not in signal.extra
    assert "org_top_downloads_basis" not in signal.extra


def test_unknown_org_omits_the_key_rather_than_writing_zero():
    """An org with no models returned is unknown, not measured-as-unpopular."""
    conn, api, _ = make_connector(
        [probe_record("verbarex")], org_cfg(), org_downloads=True,
        top_downloads=_records("top_downloads.json"),
        org_models={},  # the Hub knows nothing about anyone
    )
    (signal,) = conn.poll(window_since())
    assert api.org_calls == ["verbarex"]
    assert "org_top_downloads" not in signal.extra


def test_a_measured_zero_is_reported_as_zero():
    """Distinct from the failure cases: this org genuinely has no downloads.

    ``_lookup_org`` returns a real (0, 0), which becomes ``org_top_downloads: 0``
    — a measured absence of popularity, unlike an omitted key.
    """
    conn, _, _ = make_org_connector([])
    top5 = _records("org_models.json")["parallaxopen"]
    assert all((r.get("downloads") or 0) == 0 for r in top5), "fixture org must be at zero"
    assert conn._lookup_org("parallaxopen") == (0, 0)
    assert conn._pick_metric(0, 0) == (0, "downloads")
    assert conn._lookup_org("nobody-at-all") is None


def test_fallback_is_gated_on_shipping_an_engaged_architecture():
    """The gate is a property of the data, so which orgs get answered is stable.

    A cap that truncated an arbitrary tail would make S2 depend on listing
    order, and the backtest calibrates ``min_org_top_downloads`` against these
    numbers.
    """
    # 27aran ships a real architecture but has 0 likes and 0 downloads
    conn, api, _ = make_org_connector([probe_record("_27aran")])
    conn.poll(window_since())
    assert api.org_calls == [], "no engagement, so no request"

    # ParallaxOpen has a like but ships no architectures[] at all
    conn, api, _ = make_org_connector([probe_record("_parallaxopen")])
    conn.poll(window_since())
    assert api.org_calls == [], "no architecture, so no request"

    # VERBAREX ships an architecture with real downloads -> eligible
    conn, api, _ = make_org_connector([probe_record("verbarex")])
    (signal,) = conn.poll(window_since())
    assert api.org_calls == ["verbarex"]
    assert signal.extra["org_top_downloads"] == 1007


def test_org_lookup_fanout_is_capped_and_logged(caplog):
    """A pathological window cannot balloon into thousands of requests."""
    records = _records("list_window.json")
    with caplog.at_level("WARNING", logger="archwatch.connectors.hf"):
        conn, api, _ = make_org_connector(records, max_org_lookups=4)
        conn.poll(window_since())
    assert len(api.org_calls) == 4
    assert any("max_org_lookups=4" in m for m in caplog.messages)


def test_no_cap_warning_when_the_gate_fits_under_the_budget(caplog):
    with caplog.at_level("WARNING", logger="archwatch.connectors.hf"):
        conn, api, _ = make_org_connector([probe_record("verbarex")], max_org_lookups=200)
        conn.poll(window_since())
    assert len(api.org_calls) == 1
    assert not any("max_org_lookups" in m for m in caplog.messages)


def test_org_lookups_can_be_switched_off_entirely():
    conn, api, _ = make_connector(_records("list_window.json"), org_cfg(), org_downloads=False)
    signals = conn.poll(window_since())
    assert api.org_calls == []
    assert api.sweep_calls == []
    assert all("org_top_downloads" not in s.extra for s in signals)


def test_sweep_stops_at_the_threshold_and_says_so(caplog):
    """The sweep's depth is set by the threshold, not by the window's org count."""
    with caplog.at_level("INFO", logger="archwatch.connectors.hf"):
        conn, api, _ = make_org_connector([probe_record("baai")])
        conn.poll(window_since())

    top = sorted(_records("top_downloads.json"), key=lambda r: -(r.get("downloads") or 0))
    over = [r for r in top if (r.get("downloads") or 0) >= SWEEP_THRESHOLD]
    assert 0 < len(over) < len(top), "threshold must fall inside the fixture's range"
    assert any(f"read {len(over)} models" in m and "conclusive=True" in m
               for m in caplog.messages)
    assert len(api.sweep_calls) == 1, "one sweep per poll, not one per org"


def test_sweep_warns_when_it_cannot_reach_the_threshold(caplog):
    """Then absence from the map is inconclusive, and the log must say so."""
    with caplog.at_level("WARNING", logger="archwatch.connectors.hf"):
        # the fixture bottoms out at 1.57M, so a 100k threshold is never reached
        conn, _, _ = make_org_connector([probe_record("baai")],
                                        cfg=org_cfg(threshold=100_000))
        conn.poll(window_since())
    assert any("without reaching min_org_top_downloads" in m for m in caplog.messages)


def test_org_sweep_uses_a_downloads_sort_not_a_creation_sort():
    conn, api, _ = make_org_connector([probe_record("baai")])
    conn.poll(window_since())
    (sweep,) = api.sweep_calls
    assert sweep["sort"] == "downloads"
    assert set(sweep["expand"]) == {"author", "downloads", "downloadsAllTime"}
    assert sweep["author"] is None


class BadSweepApi(FakeApi):
    """An API whose org-popularity sweep always rate-limits."""

    def list_models(self, *, sort=None, author=None, **kw):
        if sort == "downloads" and author is None:
            self.calls.append({"sort": sort, "limit": None, "expand": None, "author": None})
            raise _http_error(429, "429 Client Error: Too Many Requests")
        return super().list_models(sort=sort, author=author, **kw)


def test_sweep_failure_degrades_to_the_per_org_fallback(caplog):
    """A 429 on the sweep must not raise; the fallback still answers what it can."""
    with caplog.at_level("WARNING", logger="archwatch.connectors.hf"):
        conn, api, _ = make_org_connector([probe_record("baai")], api_cls=BadSweepApi)
        (signal,) = conn.poll(window_since())      # must not raise
    assert any("org popularity sweep failed" in m for m in caplog.messages)
    assert api.org_calls == ["baai"], "the sweep's miss becomes a fallback lookup"
    assert signal.extra["org_top_downloads"] == 566_091_432


def test_org_data_is_omitted_when_both_the_sweep_and_the_lookup_fail():
    """No number is better than a wrong one: S2 simply goes unsatisfied."""
    conn, api, _ = make_org_connector([probe_record("baai")], api_cls=BadSweepApi,
                                      fail_orgs=("baai",))
    (signal,) = conn.poll(window_since())          # must not raise
    assert api.org_calls == ["baai"]
    assert "org_top_downloads" not in signal.extra
    assert "org_top_downloads_basis" not in signal.extra


# ---------------------------------------------------------------------------
# S4 — trending flag / score / rank
# ---------------------------------------------------------------------------


def test_trending_signals_carry_flag_score_and_rank(cfg):
    records = _records("list_trending.json")
    conn, _, _ = make_connector(records, cfg)
    signals = conn.poll_trending(limit=20)

    raw = {r["id"]: r for r in records}
    for s in signals:
        assert s.extra["trending"] is True
        assert s.extra["trending_score"] == raw[s.model_ids[0]]["trendingScore"]
        assert isinstance(s.extra["trending_rank"], int) and s.extra["trending_rank"] >= 1

    # rank is the position on the Hub's list, so it survives the pre-filter
    # removing higher-ranked entries and is strictly increasing down the list
    ranks = [s.extra["trending_rank"] for s in signals]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)
    dropped_above = [r["id"] for r in records[:ranks[-1]] if r["id"] not in
                     {s.model_ids[0] for s in signals}]
    assert dropped_above, "fixture must drop something, else rank==index proves nothing"
    assert ranks[-1] > len(signals), "rank must reflect pre-filter position"


def test_window_signals_have_no_trending_rank(cfg):
    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    for s in conn.poll(window_since()):
        assert s.extra["trending"] is False
        assert "trending_rank" not in s.extra


# ---------------------------------------------------------------------------
# org casing
# ---------------------------------------------------------------------------


def test_org_is_lowercased_to_match_frontier_orgs(cfg):
    """FRONTIER_ORGS is all lowercase; T3/S2 must not hinge on Hub capitalisation."""
    conn, _, _ = make_connector(_records("list_trending.json"), cfg)
    signals = {s.model_ids[0]: s for s in conn.poll_trending(limit=20)}

    assert signals["Qwen/Qwen3.8-Flash-Next"].org == "qwen"
    assert signals["Qwen/Qwen3.8-Flash-Next"].org in DetectorConfig().frontier_orgs
    for s in signals.values():
        assert s.org is None or s.org == s.org.lower()
    # the real repo id keeps its casing, so nothing is lost
    assert signals["Qwen/Qwen3.8-Flash-Next"].model_ids == ["Qwen/Qwen3.8-Flash-Next"]
    assert signals["Qwen/Qwen3.8-Flash-Next"].urls["hf"].endswith("/Qwen/Qwen3.8-Flash-Next")


def test_org_falls_back_to_the_repo_namespace_lowercased():
    class Info:
        id, author, config = "MixedCase/Model", None, {}
        created_at = FIXED_NOW
        downloads = likes = trending_score = downloads_all_time = 0
        tags, library_name, pipeline_tag = [], None, None
        sha = last_modified = gated = private = None

    conn = HFConnector(DetectorConfig(), api=FakeApi([]), config_fetcher=FixtureFetcher(),
                       clock=lambda: FIXED_NOW, org_downloads=False)
    signal = conn._to_signal(Info(), None, phase="window", observed_at=FIXED_NOW)
    assert signal.org == "mixedcase"


# ---------------------------------------------------------------------------
# strong non-LM evidence overrides an architecture name
# ---------------------------------------------------------------------------
#
# arch_evidence_records.json holds real full-expand records for the five
# architectures that consumed a live run's entire issue cap, plus two frontier
# multimodal LLMs that must survive. Only `createdAt` is overridden, to place
# them in the recorded window; the metadata under test is untouched.


def arch_evidence(repo_id: str) -> dict:
    return dict(_records("arch_evidence_records.json")[repo_id], createdAt=IN_WINDOW)


@pytest.mark.parametrize(
    "repo_id, arch, reason",
    [
        # ASR: names an architecture, tagged automatic-speech-recognition
        ("microsoft/VibeVoice-ASR-Streaming-7B", "VibeVoiceForASRStreamingTraining",
         "pipeline_tag=automatic-speech-recognition"),
        # music generation: diffusers + text-to-audio
        ("MiniMaxAI/MiniMax-Music3", "MiniMaxMusic3ForConditionalGeneration",
         "pipeline_tag=text-to-audio"),
        # video segmentation with NO library_name and NO pipeline_tag at all —
        # model_type is the only evidence that exists
        ("cubert-gmbh/sam3", "Sam3VideoModel", "model_type=sam3_video"),
        ("hf-tiny-v2/tiny-random-Wav2Vec2ForPreTraining", "Wav2Vec2ForPreTraining",
         "model_type=wav2vec2"),
    ],
)
def test_arch_bearing_non_lm_repos_are_dropped(cfg, repo_id, arch, reason):
    """The regression: these five ate a whole run's candidate cap."""
    raw = arch_evidence(repo_id)
    assert architectures_of(raw["config"]) == [arch], "subject must name an architecture"
    assert strong_non_lm_evidence(_model_info(raw)) == reason

    conn, _, fetcher = make_connector([raw], cfg)
    assert conn.poll(window_since()) == []
    assert fetcher.calls == [], "a dropped repo costs no config fetch"


@pytest.mark.parametrize(
    "repo_id, arch",
    [
        ("Qwen/Qwen3-VL-8B-Instruct", "Qwen3VLForConditionalGeneration"),
        ("Qwen/Qwen3.5-9B", "Qwen3_5ForConditionalGeneration"),
    ],
)
def test_multimodal_llms_are_never_dropped(cfg, repo_id, arch):
    """A vision-language model IS a language model for BLIS's purposes.

    Both subjects are tagged ``image-text-to-text``, which is in
    LM_PIPELINE_TAGS_ALWAYS_KEEP and therefore short-circuits every drop rule.
    These are the frontier releases the whole pipeline exists to catch.
    """
    raw = arch_evidence(repo_id)
    assert raw["pipeline_tag"] == "image-text-to-text"
    assert strong_non_lm_evidence(_model_info(raw)) is None

    conn, _, _ = make_connector([raw], cfg)
    (signal,) = conn.poll(window_since())
    assert signal.arch_ids == [arch]


def test_a_language_task_tag_beats_every_other_signal():
    """The keep-set is checked first, so no vision/audio evidence can override it."""

    class Info:
        def __init__(self, pipeline=None, library=None, model_type=None):
            self.id = "x/y"
            self.pipeline_tag, self.library_name = pipeline, library
            self.config = {"architectures": ["FooForCausalLM"]}
            if model_type:
                self.config["model_type"] = model_type

    for tag in sorted(LM_PIPELINE_TAGS_ALWAYS_KEEP):
        assert strong_non_lm_evidence(
            Info(pipeline=tag, library="diffusers", model_type="wav2vec2")
        ) is None, f"{tag} must be immune"

    # without the language tag, each of the three sources drops it on its own
    assert strong_non_lm_evidence(Info(pipeline="text-to-image")) == "pipeline_tag=text-to-image"
    assert strong_non_lm_evidence(Info(library="diffusers")) == "library_name=diffusers"
    assert strong_non_lm_evidence(Info(model_type="wav2vec2")) == "model_type=wav2vec2"
    assert strong_non_lm_evidence(Info()) is None


def test_strong_evidence_required_tags_and_libraries_are_present():
    """The set the coordinator specified, asserted explicitly so it cannot drift."""
    required_tags = {
        "text-to-image", "image-to-image", "automatic-speech-recognition",
        "audio-classification", "text-to-audio", "text-to-video",
        "video-classification", "image-classification", "object-detection",
        "depth-estimation", "robotics", "reinforcement-learning",
    }
    assert required_tags <= STRONG_NON_LM_PIPELINE_TAGS
    required_libs = {"diffusers", "lerobot", "stable-baselines3", "espnet",
                     "speechbrain", "timm"}
    assert required_libs <= STRONG_NON_LM_LIBRARIES


def test_strong_sets_are_narrower_than_the_archless_sets():
    """Overriding an architecture name demands more evidence than filling a blank."""
    assert STRONG_NON_LM_PIPELINE_TAGS < NON_LM_PIPELINE_TAGS
    # these stay out of the strong set: they host or wrap text models
    for weak in ("peft", "adapter-transformers", "sentence-transformers",
                 "sklearn", "spacy", "flair", "setfit", "bertopic"):
        assert weak in NON_LM_LIBRARIES
        assert weak not in STRONG_NON_LM_LIBRARIES
    # ... and these tags a text-capable model could plausibly carry
    for weak in ("image-feature-extraction", "image-text-to-video",
                 "tabular-classification", "time-series-forecasting", "graph-ml"):
        assert weak in NON_LM_PIPELINE_TAGS
        assert weak not in STRONG_NON_LM_PIPELINE_TAGS


def test_no_drop_vocabulary_ever_intersects_the_keep_set():
    """The single invariant that protects every frontier multimodal release."""
    for name, vocab in (("strong pipeline", STRONG_NON_LM_PIPELINE_TAGS),
                        ("broad pipeline", NON_LM_PIPELINE_TAGS)):
        overlap = vocab & LM_PIPELINE_TAGS_ALWAYS_KEEP
        assert not overlap, f"{name} tags would drop a language model: {overlap}"


def test_model_type_evidence_covers_only_vision_and_audio_families():
    """No LM family may appear in the model_type drop list."""
    lm_families = {
        "llama", "qwen2", "qwen3", "qwen3_5", "qwen3_vl", "gemma3", "gemma4",
        "mistral", "mixtral", "deepseek_v3", "deepseek_v4", "glm4", "glm5_next",
        "phi3", "falcon", "gpt2", "gpt_neox", "bert", "roberta", "t5",
        "kimi_k3", "olmoe", "nemotron_h", "minimax_h3", "small_lm",
    }
    assert lm_families.isdisjoint(STRONG_NON_LM_MODEL_TYPES)
    # every model_type present in the recorded fixtures that we keep must be safe
    for fixture in ("list_window.json", "list_trending.json"):
        for raw in _records(fixture):
            if keeps(raw) and architectures_of(raw.get("config")):
                info = _model_info(raw)
                assert strong_non_lm_evidence(info) is None


def test_musicgen_in_the_window_fixture_is_dropped_on_model_type_alone(cfg):
    """A real repo with an architecture and no library or pipeline metadata."""
    raw = next(r for r in _records("list_window.json")
               if r["id"] == "Renarovich12/musicgen-melody")
    assert raw.get("library_name") is None and raw.get("pipeline_tag") is None
    assert architectures_of(raw["config"]) == ["MusicgenMelodyForConditionalGeneration"]
    assert strong_non_lm_evidence(_model_info(raw)) == "model_type=musicgen_melody"

    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    assert "Renarovich12/musicgen-melody" not in {
        s.model_ids[0] for s in conn.poll(window_since())
    }


def test_robotics_and_asr_repos_in_the_window_fixture_are_dropped(cfg):
    """Real arch-bearing repos the old rule let through."""
    conn, _, _ = make_connector(_records("list_window.json"), cfg)
    emitted = {s.model_ids[0] for s in conn.poll(window_since())}
    for repo_id in ("huwenjie333/whisper-v3-ft-af51-0903",
                    "Dexmal/DM05-MEM",
                    "twanghcmut/GR00T-N1.7-ALOHA-RightArm-Multitask"):
        assert repo_id not in emitted
    # the language models around them are untouched
    for repo_id in ("foranyone2026/Kimi-K3", "foranyone/GLM-5.3-BF16",
                    "Openintelligent123/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"):
        assert repo_id in emitted


def test_strong_filter_applies_to_the_trending_sweep(cfg):
    conn, _, _ = make_connector(
        _records("list_trending.json") + [arch_evidence("cubert-gmbh/sam3")], cfg
    )
    emitted = {s.model_ids[0] for s in conn.poll_trending(limit=25)}
    assert "cubert-gmbh/sam3" not in emitted
    assert "Qwen/Qwen3.8-Flash-Next" in emitted
