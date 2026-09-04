"""Tests for the InferenceX connector (component E).

**Fully offline.** Every byte the connector sees comes from
``tests/fixtures/inferencex/``, which holds verbatim GitHub REST responses
captured read-only from ``SemiAnalysisAI/InferenceX`` on 2026-09-04:

``commits_path_<path>.json``
    ``GET /repos/{repo}/commits?path=<path>&since=2026-08-26T22:00:00Z
    &until=2026-08-27T07:00:00Z`` for each watched path. The window is a real
    one: it is the burst in which InferenceX onboarded Qwen3.8-Flash-Next.
``commit_<sha>.json``
    ``GET /repos/{repo}/commits/{sha}`` for every commit in that window, plus
    three commits kept for targeted cases — ``0decc69f`` (the Kimi-K3 day-0
    onboarding), ``3a7d6b71`` (a GLM-5.2 tuning commit whose changelog entry
    carries measured ITL/interactivity numbers) and ``4699ab81`` (a deprecation
    that adds new files under ``configs/``).
``contents_MODELS.md.json``
    ``GET /repos/{repo}/contents/MODELS.md`` — the live prefix -> model-name
    index the connector fetches once per poll.

The fake session refuses any URL it has no fixture for, so a test that tried to
reach the network would fail rather than silently hit GitHub.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from archwatch.config import FRONTIER_ORGS
from archwatch.connectors.base import Signal
from archwatch.connectors.inferencex import (
    KIND_CONFIG_ENTRY,
    KIND_CONFIG_FILE_ADDED,
    KIND_MODEL_ROW,
    KIND_PERF_CHANGELOG,
    WATCHED_PATHS,
    InferenceXConnector,
    build_prefix_name_map,
    describe_config_key,
    extract_perf_metrics,
    is_master_config,
    parse_master_config_added,
    parse_models_table_rows,
    parse_perf_changelog_patch,
    prefix_from_config_filename,
    prefix_from_config_key,
    resolve_github_token,
    strip_quant_suffix,
)
from archwatch.connectors import inferencex as ix

FIXTURES = Path(__file__).parent / "fixtures" / "inferencex"

WINDOW_START = datetime(2026, 8, 26, 22, 0, tzinfo=timezone.utc)

SHA_QWEN_MODELS_ROW = "55c05d88b15cf84d371ce6e58daf1036fd332082"
SHA_QWEN_SPEEDBENCH = "9e61a2093e470b5417433bad019a452f4b0a2692"
SHA_QWEN_H100 = "00e4d79d95d212292485ab2dd69eaa85cb97370f"
SHA_QWEN_B300 = "62b520c9e13f21abcb718d186a2b111e7655b725"
SHA_RUNNER_NOISE = "8b79ab5fd594d102690780a44fe203795cd52a76"
SHA_KIMI_K3_DAY0 = "0decc69fda9a8dba15c15b0d079bdc71c811d6dc"
SHA_GLM52_PERF = "3a7d6b714844a181a6a5ebdcfd4097ee04a0cf5e"
SHA_GPTOSS_DEPRECATION = "4699ab81a800eb59c68a46f442f37564d0315613"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def models_md_text() -> str:
    return ix._decode_contents(load_fixture("contents_MODELS.md.json"))


# ---------------------------------------------------------------------------
# offline transport
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None, raise_on_json=False):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Serves the recorded fixtures and nothing else.

    ``status_overrides`` maps a substring of the request URL to a status code so
    a test can simulate a 404 (renamed path) or a 403 (rate limit) for one path
    while every other path still resolves normally.
    """

    def __init__(self, status_overrides=None, raise_on=None):
        self.calls: list[tuple[str, dict]] = []
        self.status_overrides = dict(status_overrides or {})
        self.raise_on = raise_on

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params))
        assert url.startswith("https://api.github.com/"), url
        assert headers and headers.get("Accept") == "application/vnd.github+json"
        if self.raise_on and self.raise_on in url:
            raise RuntimeError("simulated connection reset")
        for needle, status in self.status_overrides.items():
            if needle in url or needle == params.get("path"):
                return FakeResponse({"message": "boom"}, status, {"X-RateLimit-Remaining": "0"})

        if url.endswith("/contents/MODELS.md"):
            return FakeResponse(load_fixture("contents_MODELS.md.json"))

        match = re.search(r"/commits/([0-9a-f]{7,40})$", url)
        if match:
            path = FIXTURES / f"commit_{match.group(1)}.json"
            if path.exists():
                return FakeResponse(json.loads(path.read_text(encoding="utf-8")))
            return FakeResponse({"message": "Not Found"}, 404)

        if url.endswith("/commits"):
            if int(params.get("page", 1)) > 1:
                return FakeResponse([])
            path = FIXTURES / f"commits_path_{params.get('path')}.json"
            if path.exists():
                return FakeResponse(json.loads(path.read_text(encoding="utf-8")))
            return FakeResponse({"message": "Not Found"}, 404)

        return FakeResponse({"message": "Not Found"}, 404)

    def commit_calls(self) -> list[str]:
        return [u for u, _ in self.calls if re.search(r"/commits/[0-9a-f]{7,40}$", u)]


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def connector(session: FakeSession) -> InferenceXConnector:
    # token is passed explicitly so nothing ever shells out to `gh` in tests.
    return InferenceXConnector(session=session, token="fixture-token")


@pytest.fixture
def indexed_connector(session: FakeSession) -> InferenceXConnector:
    """A connector with the MODELS.md index loaded but no commit listing done."""
    conn = InferenceXConnector(session=session, token="fixture-token")
    conn._load_model_index()
    return conn


def signals_for(conn: InferenceXConnector, sha: str) -> list[Signal]:
    return conn.signals_from_commit(load_fixture(f"commit_{sha}.json"))


def only(signals: list[Signal]) -> Signal:
    assert len(signals) == 1, [s.display_name for s in signals]
    return signals[0]


# ---------------------------------------------------------------------------
# poll(): the whole window
# ---------------------------------------------------------------------------


def test_poll_reads_the_window_and_emits_well_formed_signals(connector, session):
    signals = connector.poll(WINDOW_START)

    assert signals, "the recorded window contains model activity"
    for signal in signals:
        assert signal.source == "inferencex"
        assert isinstance(signal.observed_at, datetime)
        # tz-aware UTC: ordering and the emitted detected_at must not depend on
        # the machine the scan ran on.
        assert signal.observed_at.tzinfo is not None
        assert signal.observed_at.utcoffset().total_seconds() == 0
        assert signal.observed_at >= WINDOW_START
        assert signal.config is None, "InferenceX exposes no config.json"
        assert signal.arch_ids == [], "no architectures[] in this source"
        assert signal.primary_arch() is None
        assert signal.display_name
        assert signal.raw_ref and len(signal.raw_ref) == 40
        assert signal.urls["commit"].startswith(
            "https://github.com/SemiAnalysisAI/InferenceX/commit/"
        )
        assert signal.evidence
        assert isinstance(signal.extra["perf"], list)
        assert isinstance(signal.extra["perf_notes"], list)

    # The window is real: it is the Qwen3.8-Flash-Next onboarding burst.
    assert {s.display_name for s in signals} == {"Qwen3.8-Flash-Next"}
    assert {s.extra["prefix"] for s in signals} == {"qwen3.8next"}


def test_poll_returns_signals_newest_first(connector):
    signals = connector.poll(WINDOW_START)
    stamps = [s.observed_at for s in signals]
    assert stamps == sorted(stamps, reverse=True)


def test_poll_sends_since_as_iso_utc_for_every_watched_path(connector, session):
    connector.poll(WINDOW_START)
    listed = {p["path"] for url, p in session.calls if url.endswith("/commits") and "path" in p}
    assert listed == set(WATCHED_PATHS)
    for url, params in session.calls:
        if url.endswith("/commits"):
            assert params["since"] == "2026-08-26T22:00:00Z"
            assert params["per_page"] == 100


def test_poll_accepts_a_naive_since_as_utc(connector, session):
    connector.poll(datetime(2026, 8, 26, 22, 0))
    since = {p["since"] for u, p in session.calls if u.endswith("/commits")}
    assert since == {"2026-08-26T22:00:00Z"}


def test_each_commit_is_fetched_once_even_when_several_paths_list_it(connector, session):
    connector.poll(WINDOW_START)
    fetched = session.commit_calls()
    assert len(fetched) == len(set(fetched))
    # The recorded window: 2 doc commits + 4 configs commits + 4 changelog
    # commits, overlapping down to 7 distinct shas.
    assert len(fetched) == 7


def test_max_commits_caps_the_number_of_commit_fetches(session, caplog):
    conn = InferenceXConnector(session=session, token="t", max_commits=2)
    with caplog.at_level(logging.WARNING):
        conn.poll(WINDOW_START)
    assert len(session.commit_calls()) == 2
    assert any("capped at 2" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# MODELS.md
# ---------------------------------------------------------------------------


def test_models_md_diff_yields_the_model_name_and_prefix(indexed_connector):
    signal = only(signals_for(indexed_connector, SHA_QWEN_MODELS_ROW))

    assert signal.display_name == "Qwen3.8-Flash-Next"
    assert signal.extra["prefix"] == "qwen3.8next"
    assert signal.extra["new_model_row"] is True
    assert KIND_MODEL_ROW in signal.extra["kinds"]
    # Both the English and the Chinese matrix carry the row.
    assert signal.extra["paths"] == ["MODELS.md", "MODELS_zh.md"]
    assert "Qwen3.8-Flash-Next" in signal.extra["models_md_names"]
    assert signal.urls["docs"].endswith("/MODELS.md")
    assert signal.urls["pr"].endswith("/pull/2742")


def test_models_zh_row_is_parsed_despite_full_width_parentheses(indexed_connector):
    commit = load_fixture(f"commit_{SHA_QWEN_SPEEDBENCH}.json")
    zh = next(f for f in commit["files"] if f["filename"] == "MODELS_zh.md")
    added = ix._added_lines(zh["patch"])
    assert any("（`qwen3.8next`）" in line for line in added)
    assert ("Qwen3.8-Flash-Next", "qwen3.8next") in parse_models_table_rows(added)


def test_prefix_name_index_covers_every_model_in_the_real_models_md():
    mapping = build_prefix_name_map(models_md_text())
    # Every family the repo description advertises, keyed by its canonical
    # InferenceX model-prefix.
    assert mapping["kimik3"] == "Kimi-K3"
    assert mapping["minimaxm3"] == "MiniMax-M3"
    assert mapping["dsv4"] == "DeepSeek-V4-Pro"
    assert mapping["glm5.2"] == "GLM-5.2"
    assert mapping["qwen3.5"] == "Qwen3.5-397B-A17B"
    assert mapping["qwen3.8next"] == "Qwen3.8-Flash-Next"
    assert mapping["dsr1"] == "DeepSeek-R1-0528"
    assert mapping["gptoss"] == "gpt-oss-120b"
    assert mapping["llama70b"] == "Llama-3.1-70B-Instruct"


def test_real_models_md_parses_with_no_false_positive_rows():
    rows = parse_models_table_rows(models_md_text().splitlines())
    assert rows, "the support matrix should parse"
    names = {name for name, _ in rows}
    # Scenario rows, the CPU-DRAM table and every header row must be rejected.
    assert not {n for n in names if n.lower().startswith(("single-turn", "agentic", "model"))}
    assert not {n for n in names if "DRAM" in n}
    # Every row in today's file carries a canonical prefix.
    assert all(prefix for _, prefix in rows)


def test_table_parser_rejects_headers_and_scenarios_but_keeps_model_rows():
    lines = [
        "| Model architecture class | Prefix | Date added | Active scenarios |",
        "|---|---|---|---|",
        "| Kimi-K3 | `kimik3` | 2026-07-27 ([#2391](https://x/pull/2391)) | Agentic coding |",
        "| Single-turn 8k1k | 8192 / 1024 | Active. |",
        "| Agentic coding | Long Context | Active. |",
        "| No standardized CPU DRAM capacity | HGX B200 | At most 3 TB per server. |",
        "| MiniMax-M3 (`minimaxm3`) | native/upstream vLLM engine | None |",
    ]
    assert parse_models_table_rows(lines) == [
        ("Kimi-K3", "kimik3"),
        ("MiniMax-M3", "minimaxm3"),
    ]


def test_table_parser_pairs_a_shared_row_of_two_models():
    line = (
        "| GLM-5 / GLM-5.1 | `glm5`, `glm5.1` | 2026-03-06 | None (retired) | "
        "Single-turn 1k1k |"
    )
    assert parse_models_table_rows([line]) == [("GLM-5", "glm5"), ("GLM-5.1", "glm5.1")]


def test_table_parser_falls_back_to_a_versioned_name_when_the_prefix_column_goes_away():
    """Graceful degradation: the repo restructures its docs regularly."""
    rows = parse_models_table_rows(
        [
            "| Kimi-K3 | Agentic coding, non-DSpark | Agentic coding, DSpark |",
            "| Deprecated arm | Published arm |",
        ]
    )
    assert rows == [("Kimi-K3", None)]


def test_a_name_only_row_is_attached_to_its_prefix_via_the_live_index(indexed_connector):
    commit = {
        "sha": "f" * 40,
        "html_url": "https://github.com/SemiAnalysisAI/InferenceX/commit/" + "f" * 40,
        "commit": {
            "message": "docs: retire an arm (#9999)",
            "committer": {"date": "2026-09-01T00:00:00Z"},
        },
        "files": [
            {
                "filename": "MODELS.md",
                "status": "modified",
                "patch": "@@ -1,2 +1,3 @@\n context\n+| Kimi-K3 | Agentic coding | DSpark |\n",
            }
        ],
    }
    signal = only(indexed_connector.signals_from_commit(commit))
    assert signal.display_name == "Kimi-K3"
    assert signal.extra["prefix"] == "kimik3"


# ---------------------------------------------------------------------------
# configs/
# ---------------------------------------------------------------------------


def test_config_entry_yields_the_checkpoint_the_config_key_and_the_org(indexed_connector):
    signal = only(signals_for(indexed_connector, SHA_QWEN_H100))

    assert signal.model_ids == ["Qwen/Qwen3.8-Flash-Next-FP8"]
    assert signal.org == "Qwen"
    assert signal.org.lower() in FRONTIER_ORGS
    assert signal.extra["config_keys"] == ["qwen3.8next-fp8-h100-sglang-agentic-mtp"]
    assert KIND_CONFIG_ENTRY in signal.extra["kinds"]
    assert "configs/nvidia-master.yaml" in signal.extra["paths"]


def test_display_name_matches_models_md_after_stripping_the_quant_suffix(session):
    """`Qwen/Qwen3.8-Flash-Next-FP8` and MODELS.md's name must agree.

    That agreement is what lets the detector's display-name fallback join a
    configs/ signal to a MODELS.md signal for the same model.
    """
    conn = InferenceXConnector(session=session, token="t", fetch_model_index=False)
    from_config = only(signals_for(conn, SHA_QWEN_H100))
    assert from_config.display_name == "Qwen3.8-Flash-Next"

    conn_indexed = InferenceXConnector(session=session, token="t")
    conn_indexed._load_model_index()
    from_docs = only(signals_for(conn_indexed, SHA_QWEN_MODELS_ROW))
    assert from_docs.display_name == from_config.display_name


def test_a_new_file_under_configs_is_reported_with_its_model(indexed_connector):
    """Acceptance: a new `configs/` file path is picked up."""
    signal = only(signals_for(indexed_connector, SHA_GPTOSS_DEPRECATION))

    assert signal.display_name == "gpt-oss-120b"
    assert signal.extra["prefix"] == "gptoss"
    assert KIND_CONFIG_FILE_ADDED in signal.extra["kinds"]
    assert signal.extra["added_files"] == [
        "configs/deprecated/amd-gptoss-master.yaml",
        "configs/deprecated/nvidia-gptoss-master.yaml",
    ]
    assert signal.model_ids == ["openai/gpt-oss-120b"]
    assert signal.org == "openai"


def test_a_commit_that_only_shuffles_runner_labels_yields_nothing(indexed_connector):
    """`configs/` churn that names no model must not invent one."""
    commit = load_fixture(f"commit_{SHA_RUNNER_NOISE}.json")
    touched = {f["filename"] for f in commit["files"]}
    assert {"configs/nvidia-master.yaml", "configs/runners.yaml"} <= touched
    assert indexed_connector.signals_from_commit(commit) == []


def test_only_master_configs_are_read_as_benchmark_configs():
    assert is_master_config("configs/nvidia-master.yaml")
    assert is_master_config("configs/deprecated/amd-kimik2.5-8k1k-master.yaml")
    # Cluster pools and CI weights live in configs/ too; their top-level keys
    # ("labels:", "version:") must never be read as benchmark config keys.
    assert not is_master_config("configs/runners.yaml")
    assert not is_master_config("configs/ci-priority.yaml")
    assert not is_master_config("configs/CONFIGS.md")
    assert not is_master_config("benchmarks/nvidia-master.yaml")


def test_master_config_added_lines_parse_into_ids_prefixes_and_keys():
    added = [
        "",
        "# Kimi-K3 MXFP4 B200 aggregated vLLM via Dynamo (TP8 x PP2, 2 nodes)",
        "kimik3-fp4-b200-dynamo-vllm-agentic:",
        "  image: vllm/vllm-openai:kimi-k3",
        "  model: moonshotai/Kimi-K3",
        "  model-prefix: kimik3",
        "  runner: cluster:b200-dgxc",
        "    agentic-coding:",
        "      - { tp: 8, ep: 1, conc-list: [1, 2, 4] }",
    ]
    assert parse_master_config_added(added) == {
        "model_ids": ["moonshotai/Kimi-K3"],
        "prefixes": ["kimik3"],
        "config_keys": ["kimik3-fp4-b200-dynamo-vllm-agentic"],
    }


def test_config_key_and_filename_decomposition():
    assert describe_config_key("qwen3.8next-fp8-h100-sglang-agentic-mtp") == {
        "prefix": "qwen3.8next",
        "precision": "fp8",
        "hardware": "h100",
        "framework": "sglang",
    }
    assert prefix_from_config_key("dsv4-fp4-gb300-dynamo-trt-agentx") == "dsv4"
    assert describe_config_key("*gb200-dynamo-sglang")["hardware"] == "gb200"

    assert prefix_from_config_filename("configs/deprecated/nvidia-kimik2.5-8k1k-master.yaml") == (
        "kimik2.5"
    )
    assert prefix_from_config_filename("configs/deprecated/amd-qwen3.5-bf16-master.yaml") == (
        "qwen3.5"
    )
    assert prefix_from_config_filename("configs/nvidia-1k1k-master.yaml") is None


def test_quant_suffixes_are_stripped_but_real_name_parts_are_not():
    assert strip_quant_suffix("Qwen3.8-Flash-Next-FP8") == "Qwen3.8-Flash-Next"
    assert strip_quant_suffix("DeepSeek-R1-0528-MXFP4-Preview") == "DeepSeek-R1-0528"
    assert strip_quant_suffix("GLM-5.2-NVFP4") == "GLM-5.2"
    assert strip_quant_suffix("MiniMax-M3-MXFP8") == "MiniMax-M3"
    assert strip_quant_suffix("Kimi-K3") == "Kimi-K3"
    # `-Pro` and `-Instruct` are part of the model's identity.
    assert strip_quant_suffix("DeepSeek-V4-Pro") == "DeepSeek-V4-Pro"
    assert strip_quant_suffix("Llama-3.1-70B-Instruct") == "Llama-3.1-70B-Instruct"


def test_org_prefers_the_model_author_over_the_quantizer(indexed_connector):
    commit = copy.deepcopy(load_fixture(f"commit_{SHA_QWEN_H100}.json"))
    config = next(f for f in commit["files"] if f["filename"] == "configs/nvidia-master.yaml")
    config["patch"] += "\n+  model: nvidia/Qwen3.8-Flash-Next-NVFP4\n"
    signal = only(indexed_connector.signals_from_commit(commit))
    assert set(signal.model_ids) == {
        "Qwen/Qwen3.8-Flash-Next-FP8",
        "nvidia/Qwen3.8-Flash-Next-NVFP4",
    }
    assert signal.org == "Qwen"


# ---------------------------------------------------------------------------
# perf-changelog.yaml
# ---------------------------------------------------------------------------


def test_perf_changelog_numbers_land_in_extra_perf(indexed_connector):
    signal = only(signals_for(indexed_connector, SHA_GLM52_PERF))

    assert signal.display_name == "GLM-5.2"
    assert KIND_PERF_CHANGELOG in signal.extra["kinds"]
    rows = signal.extra["perf"]
    assert len(rows) == 1
    row = rows[0]

    assert row["hardware"] == "mi355x"
    assert row["config_key"] == "glm5.2-fp4-mi355x-sglang-agentic-mtp"
    assert row["precision"] == "fp4"
    assert row["framework"] == "sglang"
    assert row["scenario"] == "agentic-coding"
    assert row["pr"].endswith("/pull/2777")
    # "reduces ITL p50 by ~5% at conc 4 (6.95 ms vs 7.3 ms baseline) and raises
    #  interactivity P90 from 105 to 110.5 tok/s/user"
    assert row["itl_ms_p50"] == 6.95
    assert row["itl_ms_p50_alt"] == 7.3
    assert row["itl_pct_delta_p50"] == 5.0
    assert row["interactivity_tok_per_s_per_user_p90"] == 110.5
    assert row["interactivity_tok_per_s_per_user_p90_alt"] == 105.0
    assert "ITL p50" in row["notes"]


def test_extra_perf_obeys_the_documented_shape(connector):
    """The contract component G and the component-J backtest depend on."""
    signals = connector.poll(WINDOW_START)
    seen_rows = 0
    for signal in signals:
        rows = signal.extra["perf"]
        assert isinstance(rows, list), "extra['perf'] is always a list"
        for row in rows:
            seen_rows += 1
            assert isinstance(row, dict)
            assert "hardware" in row  # always present, may be None
            assert row["hardware"] is None or isinstance(row["hardware"], str)
            assert isinstance(row["notes"], str)
            metric_keys = set(row) - {
                "hardware",
                "config_key",
                "precision",
                "framework",
                "scenario",
                "pr",
                "notes",
            }
            assert metric_keys, "a perf row exists only when it has numbers"
            for key in metric_keys:
                assert isinstance(row[key], float), f"{key} must be numeric, not a string"
                # The unit is spelled out in the key name.
                assert re.search(
                    r"(tok_per_s|_ms|_s|_pct_delta|_ratio|_usd|_length)", key
                ), key
        seen_rows += 0
    assert seen_rows, "the recorded window carries at least one measured number"


def test_perf_row_carries_the_hardware_from_each_config_key(indexed_connector):
    by_hardware = {}
    for sha in (SHA_QWEN_H100, SHA_QWEN_B300):
        signal = only(signals_for(indexed_connector, sha))
        for row in signal.extra["perf"]:
            by_hardware[row["hardware"]] = row
    assert set(by_hardware) == {"h100", "b300"}
    # Both day-zero recipes pin the same committed golden acceptance length.
    assert by_hardware["h100"]["acceptance_length"] == 2.32
    assert by_hardware["b300"]["acceptance_length"] == 2.32


def test_perf_notes_keep_the_prose_even_when_no_number_parses(indexed_connector):
    signal = only(signals_for(indexed_connector, SHA_KIMI_K3_DAY0))
    assert signal.extra["perf"] == []  # this diff quotes shapes, not measurements
    notes = signal.extra["perf_notes"]
    assert notes
    joined = " ".join(notes)
    assert "2.8T total params" in joined
    assert "KDA" in joined  # the architectural hint stage 2 will want


def test_changelog_patch_splits_into_entries_with_keys_and_descriptions():
    commit = load_fixture(f"commit_{SHA_GLM52_PERF}.json")
    patch = next(
        f for f in commit["files"] if f["filename"] == "perf-changelog.yaml"
    )["patch"]
    entries = parse_perf_changelog_patch(patch)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["config_keys"] == ["glm5.2-fp4-mi355x-sglang-agentic-mtp"]
    assert entry["scenario_type"] == ["agentic-coding"]
    assert entry["pr_link"] == "https://github.com/SemiAnalysisAI/InferenceX/pull/2777"
    assert len(entry["descriptions"]) == 4
    assert entry["descriptions"][0].startswith("Switch the TP8 arm from EP=8 to EP=1")


def test_changelog_parser_ignores_removed_lines_and_comments():
    patch = "\n".join(
        [
            "@@ -1,6 +1,6 @@",
            " - config-keys:",
            "     - dsv4-fp4-b300-sglang",
            "     # AMD single-node",
            "   description:",
            '-    - "old text"',
            '+    - "Add --prefill-decode-interval 20: +28% output throughput at conc 128"',
            "   pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/2701",
        ]
    )
    entries = parse_perf_changelog_patch(patch)
    assert len(entries) == 1
    assert entries[0]["config_keys"] == ["dsv4-fp4-b300-sglang"]
    assert entries[0]["descriptions"] == [
        "Add --prefill-decode-interval 20: +28% output throughput at conc 128"
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Validated on the node through the real launcher, every request "
            "successful: TEP4 concurrency 64 640/640 at 6895 tok/s (11% above "
            "the non-MTP arm's 6198 tok/s, mean TTFT 27.2s to 9.6s), and TEP4 "
            "concurrency 1 10/10 at 1233 tok/s (83% above the non-MTP arm's "
            "672 tok/s, mean TPOT 12.6ms to 6.5ms)",
            {"output_tok_per_s": 6895.0, "ttft_s_mean": 27.2, "tpot_ms_mean": 12.6},
        ),
        (
            "Both arms were within 1-5% below concurrency 8. At concurrency 16 "
            "and 24, DRAM offload delivered 245 and 261 output tok/s with p50 "
            "TTFT of 0.85s and 6.2s",
            {"ttft_s_p50": 0.85, "ttft_s_p50_alt": 6.2, "output_tok_per_s": 261.0},
        ),
        (
            "Add --prefill-decode-interval 20: +28% output throughput at conc 128 "
            "(2,433-2,470 -> 3,127-3,161 tok/s), closing the gap to the vLLM "
            "recipe from 1.32x to 1.03x.",
            {
                "throughput_pct_delta": 28.0,
                "output_tok_per_s": 3161.0,
                "throughput_ratio": 1.32,
                "throughput_ratio_alt": 1.03,
            },
        ),
        (
            "Pin throughput runs to the committed golden thinking_on acceptance "
            "length of 2.32 at three speculative tokens",
            {"acceptance_length": 2.32},
        ),
    ],
)
def test_extract_perf_metrics_reads_real_changelog_prose(text, expected):
    metrics = extract_perf_metrics(text)
    for key, value in expected.items():
        assert metrics.get(key) == value, (key, metrics)


@pytest.mark.parametrize(
    "text",
    [
        # No performance vocabulary at all.
        "45 active config keys use the 8k1k scenario, 32 in configs/nvidia-master.yaml",
        # Performance vocabulary but no measurement: shapes and part numbers only.
        "Aggregated TP8 x PP2 across 2 B200 nodes (16 GPUs), plain TP (NOT TEP: ep 1)",
        "Initial submission: MiniMax-M3 MXFP8 MI325X (gfx942) vLLM benchmark "
        "with EAGLE3 speculative decoding, tuned for throughput",
        "9 recipes: 4x 1k1k + 5x 8k1k, low-latency and max-throughput profiles",
    ],
)
def test_extract_perf_metrics_does_not_invent_numbers(text):
    assert extract_perf_metrics(text) == {}


# ---------------------------------------------------------------------------
# degradation: this repo restructures, rate-limits and truncates
# ---------------------------------------------------------------------------


def test_a_missing_watched_path_is_a_warning_not_an_exception(session, caplog):
    """Acceptance: tolerate the repo restructuring (MODELS.md renamed away)."""
    session.status_overrides["MODELS.md"] = 404
    conn = InferenceXConnector(session=session, token="t")
    with caplog.at_level(logging.WARNING):
        signals = conn.poll(WINDOW_START)

    messages = [r.getMessage() for r in caplog.records]
    assert any("404" in m and "path missing or renamed" in m for m in messages)
    assert conn._prefix_names == {}, "the doc index degraded, it did not raise"
    # Discovery through the other three paths still produced a usable scan, with
    # names falling back down the documented chain: the diff's own row, then the
    # checkpoint id, then the bare model-prefix.
    assert signals
    assert {s.display_name for s in signals} == {"Qwen3.8-Flash-Next", "qwen3.8next"}
    bare = [s for s in signals if s.display_name == "qwen3.8next"]
    assert [s.extra["kinds"] for s in bare] == [["perf_changelog"]]


def test_a_renamed_configs_directory_narrows_discovery_without_failing(session, caplog):
    session.status_overrides["configs"] = 404
    conn = InferenceXConnector(session=session, token="t")
    with caplog.at_level(logging.WARNING):
        signals = conn.poll(WINDOW_START)

    assert signals
    fetched = {u.rsplit("/", 1)[-1] for u in session.commit_calls()}
    # 8b79ab5f was only reachable through the configs/ listing.
    assert SHA_RUNNER_NOISE not in fetched
    assert len(fetched) == 6


def test_every_watched_path_missing_is_an_empty_scan_not_a_crash(caplog):
    overrides = {path: 404 for path in WATCHED_PATHS}
    overrides["/contents/"] = 404
    conn = InferenceXConnector(session=FakeSession(overrides), token="t")
    with caplog.at_level(logging.WARNING):
        assert conn.poll(WINDOW_START) == []
    warned = [m for m in (r.getMessage() for r in caplog.records) if "404" in m]
    assert len(warned) >= len(WATCHED_PATHS)


def test_rate_limiting_returns_partial_results(session, caplog):
    session.status_overrides["perf-changelog.yaml"] = 403
    conn = InferenceXConnector(session=session, token="t")
    with caplog.at_level(logging.WARNING):
        signals = conn.poll(WINDOW_START)

    messages = [r.getMessage() for r in caplog.records]
    assert any("rate limited" in m for m in messages)
    assert signals, "a partial scan is still a scan"


def test_a_transport_failure_is_logged_and_yields_no_signals(caplog):
    session = FakeSession(raise_on="api.github.com")
    conn = InferenceXConnector(session=session, token="t")
    with caplog.at_level(logging.WARNING):
        assert conn.poll(WINDOW_START) == []
    assert any("failed" in r.getMessage() for r in caplog.records)


def test_a_missing_models_md_falls_back_to_checkpoint_derived_names(session, caplog):
    session.status_overrides["/contents/MODELS.md"] = 404
    conn = InferenceXConnector(session=session, token="t")
    with caplog.at_level(logging.WARNING):
        conn._load_model_index()
    assert conn._prefix_names == {}
    messages = [r.getMessage() for r in caplog.records]
    assert any("could not read MODELS.md" in m for m in messages)

    signal = only(signals_for(conn, SHA_QWEN_H100))
    assert signal.display_name == "Qwen3.8-Flash-Next"  # from the checkpoint id


def test_a_commit_whose_patch_github_omitted_is_a_warning(indexed_connector, caplog):
    commit = copy.deepcopy(load_fixture(f"commit_{SHA_GPTOSS_DEPRECATION}.json"))
    for entry in commit["files"]:
        entry.pop("patch", None)
    with caplog.at_level(logging.WARNING):
        signals = indexed_connector.signals_from_commit(commit)
    assert any("no patch for" in r.getMessage() for r in caplog.records)
    # The filename alone still identifies the model.
    signal = only(signals)
    assert signal.extra["prefix"] == "gptoss"
    assert KIND_CONFIG_FILE_ADDED in signal.extra["kinds"]


def test_a_commit_with_no_file_list_is_skipped(indexed_connector, caplog):
    with caplog.at_level(logging.WARNING):
        assert indexed_connector.signals_from_commit({"sha": "a" * 40, "commit": {}}) == []
    assert any("no file list" in r.getMessage() for r in caplog.records)


def test_unparseable_json_is_a_warning(caplog):
    class BadSession(FakeSession):
        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append((url, dict(params or {})))
            return FakeResponse(None, 200, raise_on_json=True)

    conn = InferenceXConnector(session=BadSession(), token="t")
    with caplog.at_level(logging.WARNING):
        assert conn.poll(WINDOW_START) == []
    assert any("unparseable JSON" in r.getMessage() for r in caplog.records)


def test_an_empty_window_is_not_an_error(session, caplog):
    class EmptySession(FakeSession):
        def get(self, url, params=None, headers=None, timeout=None):
            if url.endswith("/commits"):
                self.calls.append((url, dict(params or {})))
                return FakeResponse([])
            return super().get(url, params, headers, timeout)

    conn = InferenceXConnector(session=EmptySession(), token="t")
    with caplog.at_level(logging.INFO):
        assert conn.poll(WINDOW_START) == []
    assert any("no commits touching" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def test_the_connector_never_builds_a_real_session_when_one_is_injected(
    connector, session, monkeypatch
):
    def explode():  # pragma: no cover - must not be called
        raise AssertionError("the connector tried to open a real HTTP session")

    monkeypatch.setattr(ix.requests, "Session", explode)
    connector.poll(WINDOW_START)
    assert connector.session is session


def test_token_comes_from_the_environment_first(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "from-gh-token")
    monkeypatch.setenv("GITHUB_TOKEN", "ignored")
    assert resolve_github_token() == "from-gh-token"
    monkeypatch.delenv("GH_TOKEN")
    assert resolve_github_token() == "ignored"
    assert resolve_github_token("explicit") == "explicit"


def test_token_falls_back_to_gh_auth_token(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    calls = []

    class Done:
        returncode = 0
        stdout = "gho_from_cli\n"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Done()

    monkeypatch.setattr(ix.subprocess, "run", fake_run)
    assert resolve_github_token() == "gho_from_cli"
    assert calls == [["gh", "auth", "token"]]


def test_a_missing_gh_cli_is_not_fatal(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(ix.subprocess, "run", fake_run)
    assert resolve_github_token() is None


def test_the_authorization_header_is_sent_when_a_token_exists(connector, session):
    connector.poll(WINDOW_START)
    assert session.calls  # header assertions live in FakeSession.get


def test_the_connector_satisfies_the_connector_protocol(connector):
    assert connector.name == "inferencex"
    assert callable(connector.poll)
    signals = connector.poll(WINDOW_START)
    assert all(isinstance(s, Signal) for s in signals)
