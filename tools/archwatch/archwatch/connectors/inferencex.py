"""InferenceX connector — SemiAnalysis's open continuous inference benchmark.

`SemiAnalysisAI/InferenceX` is a benchmark *harness* repo, not a model zoo: it
onboards a frontier model within days of release because it has to benchmark it.
That makes it an unusually early and unusually low-noise signal — a handful of
model families, each landing with hardware recipes and measured numbers.

Watched artifacts at the repo root:

- ``MODELS.md`` / ``MODELS_zh.md`` — the model support matrix. A new row
  (``| Kimi-K3 | `kimik3` | 2026-07-27 (#2391) | Agentic coding | |``) is the
  single highest-value event this source produces.
- ``configs/`` — ``*-master.yaml`` benchmark configs. Each entry carries
  ``model: <hf-repo-id>`` and ``model-prefix: <prefix>``, so an added entry
  names the exact HuggingFace checkpoint being benchmarked.
- ``perf-changelog.yaml`` — an append-mostly log of ``config-keys`` /
  ``description`` / ``pr-link`` entries. Descriptions carry measured throughput,
  TTFT, TPOT/ITL and interactivity numbers.

**Stateless / window-based.** We never diff against a stored snapshot. `poll()`
lists *commits* touching each watched path since the window start
(``GET /repos/{owner}/{repo}/commits?path=…&since=…``), fetches each commit for
its patch, and parses only the **added** lines. A missing or renamed path is a
logged warning, never an exception — this repo restructures (``configs/`` moved
to the repo root in #1992).

Identity
--------
InferenceX exposes no ``config.json`` and therefore no ``architectures[]``
array, so **``Signal.arch_ids`` is always empty** for this source. That is
explicitly sanctioned by ``base.Signal`` ("May be empty when a source discusses
a model without exposing a config") and the detector's alias fallback keys such
Signals on a normalized ``display_name``. To make that fallback join usable we
canonicalize the display name to the name InferenceX itself uses in
``MODELS.md`` (``Kimi-K3``, ``GLM-5.2``, ``Qwen3.8-Flash-Next``): the connector
fetches ``MODELS.md`` at HEAD once per poll to build a ``prefix -> name`` map,
and otherwise derives the name from the HF repo id with quantization suffixes
stripped (``Qwen/Qwen3.8-Flash-Next-FP8`` -> ``Qwen3.8-Flash-Next``). Those two
paths agree on every model currently in the repo. ``config`` is always ``None``.

``extra["perf"]`` contract
--------------------------
``extra["perf"]`` is **always a list of dicts** (empty when the window carried no
numbers), one dict per benchmark config key that a changelog entry attributed
numbers to. Downstream (the emitter and the component-J backtest) treats these
as BLIS validation ground truth, so values are **numbers, never formatted
strings**, and every metric key spells out its unit::

    {
      "hardware": "mi355x",            # always present; None when unknown
      "config_key": "glm5.2-fp4-mi355x-sglang-agentic-mtp",
      "precision": "fp4",              # None when unknown
      "framework": "sglang",           # None when unknown
      "scenario": "agentic-coding",    # None when unknown
      "pr": "https://github.com/SemiAnalysisAI/InferenceX/pull/2777",
      "itl_ms_p50": 6.95,
      "itl_ms_p50_alt": 7.3,
      "interactivity_tok_per_s_per_user_p90": 110.5,
      "throughput_tok_per_s_per_gpu_pct_delta": 12.0,
      "notes": "Switch the TP8 arm from EP=8 to EP=1: …",
    }

Metric-key vocabulary (all floats): ``output_tok_per_s``, ``draft_tok_per_s``,
``throughput_tok_per_s_per_gpu``, ``interactivity_tok_per_s_per_user``,
``ttft_s`` / ``ttft_ms``, ``tpot_s`` / ``tpot_ms``, ``itl_s`` / ``itl_ms``,
``e2e_latency_s`` / ``e2e_latency_ms``, ``acceptance_length``,
``cost_per_mtok_usd``, plus ``<metric>_pct_delta`` and ``<metric>_ratio`` for
relative numbers. A ``_p50`` / ``_p90`` / ``_p95`` / ``_p99`` / ``_mean`` suffix
is appended when the text names a percentile. A ``_alt`` suffix holds the
comparison value when the same sentence gives a baseline ("from 105 to 110.5",
"6.95 ms vs 7.3 ms baseline").

The extractor is deliberately conservative — a number needs both a performance
unit and an adjacent performance keyword — but it reads English prose, so treat
the numbers as a strong hint and ``notes`` as the truth. Specifically: an
unsigned ``_pct_delta`` is a magnitude (only a literal ``+``/``-`` in the text
becomes a sign), a range like "8-10%" is recorded as its upper bound, and a bare
percentage is filed under ``_pct_delta`` even where the prose means an absolute
rate. ``notes`` is the verbatim changelog prose the numbers came from —
deliberately kept, because the prose is more trustworthy than any parser. Full
prose for the window, including entries that carried no numbers at all, is in
``extra["perf_notes"]``.

Everything here is read-only. No writes, no issue filing, no GitHub mutations.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import requests

from ..config import FRONTIER_ORGS
from .base import Signal

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "SemiAnalysisAI/InferenceX"

#: Repo-root artifacts we watch. ``configs`` is a directory prefix; the GitHub
#: commits API accepts it and returns commits touching anything beneath it.
WATCHED_PATHS: tuple[str, ...] = (
    "MODELS.md",
    "MODELS_zh.md",
    "configs",
    "perf-changelog.yaml",
)

# Signal.extra["kinds"] vocabulary.
KIND_MODEL_ROW = "models_md_row"
KIND_CONFIG_ENTRY = "config_entry"
KIND_CONFIG_FILE_ADDED = "config_file_added"
KIND_PERF_CHANGELOG = "perf_changelog"

# Orgs that repackage/quantize someone else's model. Used only to pick the most
# informative `org` when a commit names several checkpoints of one model.
PACKAGER_ORGS: frozenset[str] = frozenset(
    {
        "amd",
        "nvidia",
        "intel",
        "neuralmagic",
        "redhatai",
        "unsloth",
        "nscale",
        "inferact",
        "lmsysorg",
        "radixark",
        "modelcloud",
        "tensorblock",
    }
)

# Tokens that appear in InferenceX config keys / config filenames.
_HARDWARE_TOKENS: frozenset[str] = frozenset(
    {
        "a100", "l40s", "gh200", "h100", "h200", "h20",
        "b200", "b300", "gb200", "gb300", "vr200", "rubin",
        "mi300x", "mi325x", "mi355x", "mi355", "mi455x",
        "rtx6000pro", "rtx5090", "rtx4090",
        "tpuv5e", "tpuv6e", "tpuv7", "tpuv8t", "tpuv8i",
        "trainium2", "trainium3", "trn2",
    }
)
_PRECISION_TOKENS: frozenset[str] = frozenset(
    {
        "fp4", "fp8", "fp16", "bf16", "int4", "int8",
        "mxfp4", "mxfp8", "nvfp4", "w4a16", "w8a8", "fp6",
    }
)
_FRAMEWORK_TOKENS: frozenset[str] = frozenset(
    {
        "vllm", "sglang", "trt", "trtllm", "atom", "dynamo",
        "llmd", "tokenspeed", "tilert", "tensorrt",
    }
)
# Non-model tokens in configs/ filenames (``nvidia-kimik2.5-8k1k-master.yaml``).
_FILENAME_STOP_TOKENS: frozenset[str] = frozenset(
    {
        "nvidia", "amd", "intel", "google", "aws", "master", "deprecated",
        "configs", "config", "runners", "runner", "ci", "priority",
        "1k1k", "8k1k", "1k8k", "agentic", "speedbench", "al", "sweep",
    }
) | _PRECISION_TOKENS | _HARDWARE_TOKENS | _FRAMEWORK_TOKENS

# Quantization / release suffixes to strip off an HF repo name so that
# ``Qwen/Qwen3.8-Flash-Next-FP8`` and MODELS.md's ``Qwen3.8-Flash-Next`` agree.
_QUANT_SUFFIXES: frozenset[str] = frozenset(
    {
        "fp4", "fp8", "fp16", "bf16", "int4", "int8",
        "mxfp4", "mxfp8", "nvfp4", "w4a16", "w8a8",
        "awq", "gptq", "gguf", "quantized", "quant",
        "preview", "v2", "v3", "v4preview",
    }
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def _utc(dt: datetime) -> datetime:
    """Coerce to timezone-aware UTC. A naive datetime is assumed to be UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_gh_time(value: Any) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp into an aware UTC datetime."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def resolve_github_token(explicit: str | None = None) -> str | None:
    """Read-only token from the explicit argument, the env, or ``gh auth token``.

    Returns ``None`` when nothing is available; unauthenticated requests still
    work against a public repo, just with a much smaller rate limit.
    """
    if explicit:
        return explicit
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value.strip()
    try:
        done = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        log.debug("inferencex: `gh auth token` unavailable: %s", exc)
        return None
    if done.returncode == 0 and done.stdout.strip():
        return done.stdout.strip()
    log.debug("inferencex: no GitHub token found; using unauthenticated requests")
    return None


def _norm_key(value: str) -> str:
    """Normalize a model name/prefix for comparison: lowercase alphanumerics."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def strip_quant_suffix(name: str) -> str:
    """``DeepSeek-R1-0528-MXFP4-Preview`` -> ``DeepSeek-R1-0528``."""
    out = (name or "").strip()
    for _ in range(4):
        head, sep, tail = out.rpartition("-")
        if not sep or not head:
            break
        if tail.lower().replace("_", "") in _QUANT_SUFFIXES:
            out = head
            continue
        break
    return out


def _classify_token(token: str) -> str | None:
    token = token.lower()
    if token in _HARDWARE_TOKENS:
        return "hardware"
    if token in _PRECISION_TOKENS:
        return "precision"
    if token in _FRAMEWORK_TOKENS:
        return "framework"
    return None


def describe_config_key(key: str) -> dict[str, str | None]:
    """Split an InferenceX config key into its named parts.

    ``qwen3.8next-fp8-h100-sglang-agentic-mtp`` ->
    ``{"prefix": "qwen3.8next", "precision": "fp8", "hardware": "h100",
    "framework": "sglang"}``. Glob keys (``*gb200-dynamo-sglang``) work too;
    unknown parts come back ``None``.
    """
    out: dict[str, str | None] = {
        "prefix": None,
        "precision": None,
        "hardware": None,
        "framework": None,
    }
    tokens = [t for t in (key or "").split("-") if t]
    for index, raw in enumerate(tokens):
        token = raw.strip("*")
        if not token:
            continue
        kind = _classify_token(token)
        if kind and out[kind] is None:
            out[kind] = token.lower()
        elif index == 0 and not kind and "*" not in raw:
            out["prefix"] = token.lower()
    return out


def prefix_from_config_key(key: str) -> str | None:
    """The InferenceX ``model-prefix`` embedded in a benchmark config key."""
    return describe_config_key(key)["prefix"]


def prefix_from_config_filename(filename: str) -> str | None:
    """``configs/deprecated/nvidia-kimik2.5-8k1k-master.yaml`` -> ``kimik2.5``."""
    base = (filename or "").rsplit("/", 1)[-1]
    base = re.sub(r"\.(ya?ml|md|json)$", "", base, flags=re.IGNORECASE)
    for token in base.split("-"):
        token = token.strip().lower()
        if not token or token in _FILENAME_STOP_TOKENS:
            continue
        if not re.fullmatch(r"[a-z][a-z0-9.]*", token):
            continue
        return token
    return None


# ---------------------------------------------------------------------------
# MODELS.md table parsing
# ---------------------------------------------------------------------------

# Leading words that mark a table row as a header or a non-model row
# ("Single-turn 8k1k" is a scenario, not a model).
_NON_MODEL_ROW = re.compile(
    r"^(model|models|prefix|scenario|scenarios|date|deprecated|remains|published"
    r"|single-turn|agentic|cpu dram|no standardized|standardized|example|total"
    r"|primary|agreed|proposed|additional|模型|场景|前缀|加入日期|已弃用|启用)",
    re.IGNORECASE,
)
_MODEL_NAME_OK = re.compile(r"^[A-Za-z][A-Za-z0-9.+]*(?:[-_ /][A-Za-z0-9.+]+)*$")
_PREFIX_OK = re.compile(r"^[a-z][a-z0-9.]*$")


def _clean_cell(cell: str) -> str:
    return cell.replace("**", "").strip()


def _extract_prefix_tokens(cell: str) -> list[str]:
    out: list[str] = []
    for token in re.findall(r"`([^`]+)`", cell):
        token = token.strip()
        if re.search(r"\.(ya?ml|md|sh|py|json)$", token, re.IGNORECASE):
            continue
        if "/" in token or " " in token:
            continue
        if _PREFIX_OK.match(token) and len(token) <= 24:
            out.append(token)
    return out


def parse_models_table_rows(lines: Iterable[str]) -> list[tuple[str, str | None]]:
    """Pull ``(model_name, prefix|None)`` pairs out of markdown table rows.

    Handles both shapes MODELS.md uses — ``| Kimi-K3 | `kimik3` | … |`` (the
    support matrix) and ``| MiniMax-M3 (`minimaxm3`) | … |`` (the engine and
    deprecation tables) — plus the ``（`prefix`）`` full-width variant in
    ``MODELS_zh.md``. Header rows, separator rows and scenario rows are dropped.
    """
    found: list[tuple[str, str | None]] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [_clean_cell(c) for c in line.strip("|").split("|")]
        if not cells:
            continue
        if all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells if c):
            continue

        prefixes = _extract_prefix_tokens(cells[0])
        if not prefixes and len(cells) > 1:
            prefixes = _extract_prefix_tokens(cells[1])

        name = re.sub(r"[(（][^)）]*[)）]", "", cells[0])
        name = name.replace("`", "").strip().rstrip(",;").strip()
        if not name or len(name) > 60:
            continue
        if _NON_MODEL_ROW.match(name):
            continue
        if not prefixes:
            # No canonical prefix on the row: only accept something that really
            # looks like a versioned model name (they all carry a digit).
            if not _MODEL_NAME_OK.match(name) or not re.search(r"\d", name):
                continue
            found.append((name, None))
            continue

        parts = [p.strip() for p in name.split("/") if p.strip()]
        if len(parts) == len(prefixes) and len(prefixes) > 1:
            found.extend(zip(parts, prefixes))
        else:
            for prefix in prefixes:
                found.append((name, prefix))
    return found


def build_prefix_name_map(models_md: str) -> dict[str, str]:
    """``{prefix: canonical model name}`` from a whole ``MODELS.md``.

    When a prefix appears in several tables the shortest name wins, which picks
    the support matrix's bare class name (``DeepSeek-V4-Pro``) over the engine
    table's decorated one (``DeepSeek-V4-Pro 1.6T``).
    """
    out: dict[str, str] = {}
    for name, prefix in parse_models_table_rows(models_md.splitlines()):
        if not prefix:
            continue
        current = out.get(prefix)
        if current is None or len(name) < len(current):
            out[prefix] = name
    return out


# ---------------------------------------------------------------------------
# configs/*-master.yaml parsing
# ---------------------------------------------------------------------------

_MODEL_LINE = re.compile(r"^\s*model:\s*(?P<value>[^\s#]+)")
_PREFIX_LINE = re.compile(r"^\s*model-prefix:\s*(?P<value>[^\s#]+)")
_CONFIG_KEY_LINE = re.compile(r"^(?P<key>[a-z0-9][a-z0-9._-]*):\s*(?:#.*)?$")
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


def is_master_config(filename: str) -> bool:
    """True for the ``configs/**/…master.yaml`` files that describe benchmarks.

    ``configs/runners.yaml`` and ``configs/ci-priority.yaml`` live in the same
    directory but describe cluster pools and CI weights, so they are excluded:
    their top-level keys would otherwise be mistaken for benchmark config keys.
    """
    if not filename.startswith("configs/"):
        return False
    base = filename.rsplit("/", 1)[-1].lower()
    return "master" in base and base.endswith((".yaml", ".yml"))


def parse_master_config_added(added: Iterable[str]) -> dict[str, list[str]]:
    """Model ids, ``model-prefix`` values and new config keys from added lines."""
    model_ids: list[str] = []
    prefixes: list[str] = []
    config_keys: list[str] = []
    for line in added:
        match = _MODEL_LINE.match(line)
        if match:
            value = match.group("value").strip().strip("\"'")
            if _HF_REPO_ID.match(value):
                model_ids.append(value)
            continue
        match = _PREFIX_LINE.match(line)
        if match:
            value = match.group("value").strip().strip("\"'").lower()
            if _PREFIX_OK.match(value):
                prefixes.append(value)
            continue
        match = _CONFIG_KEY_LINE.match(line)
        if match:
            key = match.group("key")
            # Real benchmark keys are `<prefix>-<precision>-<gpu>-<framework>`.
            if "-" in key and len(key) >= 6:
                config_keys.append(key)
    return {
        "model_ids": _dedupe(model_ids),
        "prefixes": _dedupe(prefixes),
        "config_keys": _dedupe(config_keys),
    }


# ---------------------------------------------------------------------------
# perf-changelog.yaml parsing
# ---------------------------------------------------------------------------

_SECTION_LINE = re.compile(
    r"^\s*(?:-\s*)?(?P<name>config-keys|description|pr-link|scenario-type"
    r"|append-only|evals-only):\s*(?P<rest>.*)$"
)
_LIST_ITEM = re.compile(r"^\s*-\s*(?P<value>.*)$")
_ENTRY_START = re.compile(r"^\s*-\s*config-keys:")
_CONFIG_KEY_VALUE = re.compile(r"^[A-Za-z0-9*][A-Za-z0-9.*_-]*$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        if value[0] == "'":
            return inner.replace("''", "'")
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    return value


def parse_perf_changelog_patch(patch: str) -> list[dict[str, Any]]:
    """Split a ``perf-changelog.yaml`` patch into changelog entries.

    Returns ``[{"config_keys": [...], "descriptions": [...],
    "scenario_type": [...], "pr_link": str|None, "added": bool}, ...]``.

    Structure is read from *all* patch lines (added and context alike) so an
    entry whose ``config-keys`` sit in the diff context is still identified,
    while ``descriptions`` — the new information — come only from added lines.
    """
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None

    def start_entry() -> dict[str, Any]:
        entry: dict[str, Any] = {
            "config_keys": [],
            "descriptions": [],
            "scenario_type": [],
            "pr_link": None,
            "added": False,
        }
        entries.append(entry)
        return entry

    for raw in (patch or "").splitlines():
        if raw.startswith(("@@", "---", "+++", "diff ")):
            current, section = None, None
            continue
        if not raw:
            continue
        marker, body = raw[0], raw[1:]
        if marker not in "+- ":
            marker, body = " ", raw
        added = marker == "+"
        if marker == "-":
            # Removed lines still delimit entries but contribute nothing.
            if _ENTRY_START.match(body):
                current, section = None, None
            continue

        if _ENTRY_START.match(body):
            current = start_entry()
            current["added"] = added
            section = "config-keys"
            continue

        match = _SECTION_LINE.match(body)
        if match:
            if current is None:
                current = start_entry()
            section = match.group("name")
            rest = match.group("rest").strip()
            if section == "pr-link" and rest:
                current["pr_link"] = _unquote(rest)
                if added:
                    current["added"] = True
            continue

        match = _LIST_ITEM.match(body)
        if match and current is not None and section:
            value = _unquote(match.group("value"))
            if not value or value.startswith("#"):
                continue
            if section == "config-keys":
                if _CONFIG_KEY_VALUE.match(value):
                    current["config_keys"].append(value)
                    if added:
                        current["added"] = True
            elif section == "scenario-type":
                current["scenario_type"].append(value)
            elif section == "description" and added:
                current["descriptions"].append(value)
                current["added"] = True

    return [e for e in entries if e["added"] and (e["config_keys"] or e["descriptions"])]


# ---------------------------------------------------------------------------
# perf number extraction
# ---------------------------------------------------------------------------

_PERF_KEYWORDS = re.compile(
    r"throughput|tput|tok/s|tokens?/s|ttft|tpot|\bitl\b|latency|interactivity"
    r"|goodput|\bqps\b|uplift|speedup|regress|acceptance|draft|\be2el?\b|cost",
    re.IGNORECASE,
)

_UNIT_ALTERNATION = (
    r"tok/s/user|tokens?/s/user|tok/s/gpu|tokens?/s/gpu"
    r"|tok/sec|tokens?/sec|tok/s|tokens?/s|ms|s|%|x"
)
# One qualifier word may sit between the number and its unit: the changelog
# writes "261 output tok/s" and "163 tok/s/gpu out" as often as "6198 tok/s".
_QUALIFIER = r"(?:\s+(?:output|input|total|decode|prefill|generation|out|avg|mean))?"
_MEASUREMENT = re.compile(
    # The leading lookbehind keeps us out of identifiers and ranges: the "325X"
    # of `MI325X`, the "200" of `8xB200`, the "8 x" of `TP8 x PP2`, and the
    # leading `-` of a range like `489-515 tok/s` (which is not a minus sign).
    r"(?<![A-Za-z0-9.])(?P<sign>[+-]|~)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    + _QUALIFIER
    + r"\s*"
    r"(?P<unit>" + _UNIT_ALTERNATION + r")(?![A-Za-z0-9/])",
    re.IGNORECASE,
)
_BASELINE_BEFORE = re.compile(r"from\s+(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s+to\s*$")
_BASELINE_AFTER = re.compile(
    r"^\s*(?:vs\.?|versus|rather than|compared (?:to|with)|down from|up from|instead of)\s+"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_PERCENTILE = re.compile(r"\b(p50|p90|p95|p99|p999|median|mean|avg|average)\b", re.IGNORECASE)
_COST = re.compile(
    r"\$\s*(?P<num>\d+(?:\.\d+)?)\s*(?:/|per\s+)\s*(?:m|mtok|million)\b", re.IGNORECASE
)
_ACCEPTANCE_LENGTH = re.compile(
    r"acceptance length[^.\d]{0,30}(?P<num>\d+(?:\.\d+)?)", re.IGNORECASE
)

# nearest-keyword -> metric stem
_KEYWORD_STEMS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"ttft", re.I), "ttft"),
    (re.compile(r"tpot", re.I), "tpot"),
    (re.compile(r"\bitl\b|inter-?token", re.I), "itl"),
    (re.compile(r"interactivity", re.I), "interactivity"),
    (re.compile(r"tok/s/user|tokens?/s/user", re.I), "interactivity_tok_per_s_per_user"),
    (re.compile(r"tok/s/gpu|tokens?/s/gpu", re.I), "throughput_tok_per_s_per_gpu"),
    (re.compile(r"acceptance", re.I), "acceptance"),
    (re.compile(r"\bdraft(?:ing)?\b", re.I), "draft"),
    (re.compile(r"\be2el?\b|end-to-end", re.I), "e2e_latency"),
    (re.compile(r"latency", re.I), "latency"),
    (re.compile(r"throughput|tput|uplift|speedup|tok/s|tokens?/s", re.I), "throughput"),
)

_LOOKBEHIND = 110
_LOOKAHEAD = 40


def _nearest_stem(text: str, start: int, end: int) -> tuple[str | None, bool]:
    """Metric stem implied by the nearest perf keyword around a measurement.

    Returns ``(stem, keyword_was_before_the_number)``; the flag lets the caller
    be stricter about units that are easy to misread out of context.
    """
    before = text[max(0, start - _LOOKBEHIND) : start]
    after = text[end : end + _LOOKAHEAD]
    best: tuple[int, str] | None = None
    for pattern, stem in _KEYWORD_STEMS:
        pos = -1
        for match in pattern.finditer(before):
            pos = match.start()
        if pos >= 0 and (best is None or pos > best[0]):
            best = (pos, stem)  # later in `before` == closer to the number
    if best is not None:
        return best[1], True
    for pattern, stem in _KEYWORD_STEMS:
        if pattern.search(after):
            return stem, False
    return None, False


def _percentile_suffix(text: str, start: int, end: int) -> str:
    # The corpus always names the percentile *before* the number ("TTFT p50 86s",
    # "interactivity P90 from 105 to 110.5 tok/s/user"), so only look behind:
    # looking ahead would tag "261 output tok/s with p50 TTFT of 0.85s" as p50.
    del end
    matches = _PERCENTILE.findall(text[max(0, start - 70) : start])
    if not matches:
        return ""
    token = matches[-1].lower()
    if token in ("median",):
        return "_p50"
    if token in ("mean", "avg", "average"):
        return "_mean"
    return "_" + token


def _metric_name(unit: str, stem: str | None) -> str | None:
    unit = unit.lower()
    if unit in ("tok/s/user", "tokens/s/user", "token/s/user"):
        return "interactivity_tok_per_s_per_user"
    if unit in ("tok/s/gpu", "tokens/s/gpu", "token/s/gpu"):
        return "throughput_tok_per_s_per_gpu"
    if unit in ("tok/s", "tokens/s", "token/s", "tok/sec", "tokens/sec", "token/sec"):
        if stem == "draft":
            return "draft_tok_per_s"
        if stem in ("interactivity", "interactivity_tok_per_s_per_user"):
            return "interactivity_tok_per_s_per_user"
        if stem == "throughput_tok_per_s_per_gpu":
            return stem
        return "output_tok_per_s"
    if unit in ("ms", "s"):
        if stem in ("ttft", "tpot", "itl", "e2e_latency"):
            return f"{stem}_{unit}"
        if stem == "latency":
            return f"e2e_latency_{unit}"
        return None
    if unit == "%":
        if stem is None:
            return None
        return f"{stem}_pct_delta"
    if unit == "x":
        if stem is None:
            return None
        return f"{stem}_ratio"
    return None


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:  # pragma: no cover - regex guarantees a number
        return None


def extract_perf_metrics(text: str) -> dict[str, float]:
    """Numeric performance metrics from one changelog description sentence.

    Conservative by design: a number is only recorded when it carries a
    performance unit *and* a performance keyword sits next to it, so prose like
    "45 active config keys" or "TP8 x PP2" contributes nothing. The first value
    for a metric wins; a comparison value from the same sentence lands in
    ``<metric>_alt``.
    """
    metrics: dict[str, float] = {}
    if not text or not _PERF_KEYWORDS.search(text):
        return metrics

    def put(name: str, value: float) -> None:
        if name not in metrics:
            metrics[name] = value
        elif metrics[name] != value and f"{name}_alt" not in metrics:
            metrics[f"{name}_alt"] = value

    for match in _MEASUREMENT.finditer(text):
        value = _to_float(match.group("num"))
        if value is None:
            continue
        unit = match.group("unit").lower()
        stem, stem_before = _nearest_stem(text, match.start("num"), match.end())
        if unit == "x" and "." not in match.group("num") and not stem_before:
            # "4x 1k1k + 5x 8k1k" is a count, not a speedup. A real ratio either
            # carries a decimal ("2.9x") or follows the metric it scales.
            continue
        name = _metric_name(unit, stem)
        if not name:
            continue
        name += _percentile_suffix(text, match.start("num"), match.end())
        if match.group("sign") == "-":
            value = -value
        put(name, value)

        baseline = _BASELINE_BEFORE.search(text[: match.start()])
        if baseline is None:
            after = _BASELINE_AFTER.match(text[match.end() :])
            baseline = after
        if baseline is not None:
            alt = _to_float(baseline.group(1))
            if alt is not None and alt != value:
                metrics.setdefault(f"{name}_alt", alt)

    cost = _COST.search(text)
    if cost is not None:
        value = _to_float(cost.group("num"))
        if value is not None:
            metrics.setdefault("cost_per_mtok_usd", value)

    accept = _ACCEPTANCE_LENGTH.search(text)
    if accept is not None:
        value = _to_float(accept.group("num"))
        if value is not None:
            metrics.setdefault("acceptance_length", value)

    return metrics


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


# ---------------------------------------------------------------------------
# per-model accumulator
# ---------------------------------------------------------------------------


class _Bucket:
    """Everything one commit said about one InferenceX model."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.prefix: str | None = None
        self.row_names: list[str] = []
        self.model_ids: list[str] = []
        self.config_keys: list[str] = []
        self.paths: list[str] = []
        self.kinds: list[str] = []
        self.perf: list[dict[str, Any]] = []
        self.perf_notes: list[str] = []
        self.added_files: list[str] = []
        self.new_model_row = False

    def note(self, path: str, kind: str) -> None:
        if path not in self.paths:
            self.paths.append(path)
        if kind not in self.kinds:
            self.kinds.append(kind)


# ---------------------------------------------------------------------------
# the connector
# ---------------------------------------------------------------------------


class InferenceXConnector:
    """Window-based reader of ``SemiAnalysisAI/InferenceX`` commit diffs."""

    name = "inferencex"
    source = "inferencex"

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        *,
        paths: Sequence[str] = WATCHED_PATHS,
        token: str | None = None,
        session: Any | None = None,
        api_base: str = GITHUB_API,
        timeout: float = 20.0,
        max_commits: int = 80,
        max_pages: int = 3,
        max_signals: int = 200,
        max_perf_rows: int = 20,
        fetch_model_index: bool = True,
    ) -> None:
        self.repo = repo
        self.paths = tuple(paths)
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_commits = max_commits
        self.max_pages = max_pages
        self.max_signals = max_signals
        self.max_perf_rows = max_perf_rows
        self.fetch_model_index = fetch_model_index
        self._session = session
        self._token = token
        self._token_resolved = token is not None
        self._prefix_names: dict[str, str] = {}
        self._name_prefixes: dict[str, str] = {}

    # -- plumbing ---------------------------------------------------------

    @property
    def session(self) -> Any:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "archwatch/0.1 (+prototype, read-only)",
        }
        if not self._token_resolved:
            self._token = resolve_github_token()
            self._token_resolved = True
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any | None:
        """GET returning parsed JSON, or ``None`` on any problem. Never raises."""
        try:
            response = self.session.get(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
        except Exception as exc:  # network, DNS, TLS, injected stub failures
            log.warning("inferencex: request to %s failed: %s", url, exc)
            return None
        status = getattr(response, "status_code", None)
        if status == 404:
            log.warning(
                "inferencex: %s returned 404 — path missing or renamed in %s "
                "(this repo restructures); continuing without it",
                url,
                self.repo,
            )
            return None
        if status in (403, 429):
            remaining = ""
            try:
                remaining = str(response.headers.get("X-RateLimit-Remaining", ""))
            except Exception:  # pragma: no cover - defensive
                remaining = ""
            log.warning(
                "inferencex: %s returned %s (rate limited; remaining=%r) — "
                "returning partial results",
                url,
                status,
                remaining,
            )
            return None
        if status != 200:
            log.warning("inferencex: %s returned unexpected status %s", url, status)
            return None
        try:
            return response.json()
        except Exception as exc:
            log.warning("inferencex: %s returned unparseable JSON: %s", url, exc)
            return None

    # -- MODELS.md index --------------------------------------------------

    def _load_model_index(self) -> None:
        """Fetch ``MODELS.md`` at HEAD to canonicalize prefix -> model name.

        Not stored state: it is re-fetched every poll and only used to name
        things. A 404 (renamed doc) degrades to name-from-checkpoint.
        """
        self._prefix_names, self._name_prefixes = {}, {}
        payload = self._get(f"{self.api_base}/repos/{self.repo}/contents/MODELS.md")
        text = _decode_contents(payload)
        if not text:
            log.warning(
                "inferencex: could not read MODELS.md from %s; falling back to "
                "checkpoint-derived model names",
                self.repo,
            )
            return
        self._prefix_names = build_prefix_name_map(text)
        for prefix, name in self._prefix_names.items():
            self._name_prefixes.setdefault(_norm_key(name), prefix)
        log.debug("inferencex: MODELS.md index holds %d prefixes", len(self._prefix_names))

    # -- commit listing ---------------------------------------------------

    def _list_commits(self, path: str, since: datetime) -> list[dict[str, Any]]:
        url = f"{self.api_base}/repos/{self.repo}/commits"
        out: list[dict[str, Any]] = []
        for page in range(1, self.max_pages + 1):
            params = {
                "path": path,
                "since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "per_page": 100,
                "page": page,
            }
            payload = self._get(url, params=params)
            if not isinstance(payload, list):
                break
            out.extend(c for c in payload if isinstance(c, dict))
            if len(payload) < 100:
                break
        else:
            log.warning(
                "inferencex: hit the %d-page cap listing commits for %s; the "
                "window may be truncated",
                self.max_pages,
                path,
            )
        return out

    def _fetch_commit(self, sha: str) -> dict[str, Any] | None:
        payload = self._get(f"{self.api_base}/repos/{self.repo}/commits/{sha}")
        if isinstance(payload, dict):
            return payload
        return None

    # -- Connector protocol ----------------------------------------------

    def poll(self, since: datetime) -> list[Signal]:
        """Signals for everything the watched paths saw since ``since``."""
        since = _utc(since)
        if self.fetch_model_index:
            self._load_model_index()

        order: list[str] = []
        seen: set[str] = set()
        for path in self.paths:
            for commit in self._list_commits(path, since):
                sha = commit.get("sha")
                if isinstance(sha, str) and sha and sha not in seen:
                    seen.add(sha)
                    order.append(sha)

        if not order:
            log.info(
                "inferencex: no commits touching %s in %s since %s",
                ", ".join(self.paths),
                self.repo,
                since.isoformat(),
            )
            return []
        if len(order) > self.max_commits:
            log.warning(
                "inferencex: %d commits in window, capped at %d (widen "
                "max_commits or shorten the window)",
                len(order),
                self.max_commits,
            )

        signals: list[Signal] = []
        for sha in order[: self.max_commits]:
            commit = self._fetch_commit(sha)
            if commit is None:
                continue
            signals.extend(self.signals_from_commit(commit))

        signals.sort(key=lambda s: s.observed_at, reverse=True)
        return signals[: self.max_signals]

    # -- commit -> Signals -----------------------------------------------

    def signals_from_commit(self, commit: dict[str, Any]) -> list[Signal]:
        """Parse one ``/commits/{sha}`` payload into Signals (one per model)."""
        sha = str(commit.get("sha") or "")
        commit_meta = commit.get("commit") or {}
        message = str(commit_meta.get("message") or "")
        subject = message.splitlines()[0] if message else ""
        observed_at = (
            _parse_gh_time((commit_meta.get("committer") or {}).get("date"))
            or _parse_gh_time((commit_meta.get("author") or {}).get("date"))
            or datetime.now(timezone.utc)
        )
        html_url = str(commit.get("html_url") or "")
        pr_number = _pr_number_from_subject(subject)

        files = commit.get("files")
        if not isinstance(files, list):
            log.warning(
                "inferencex: commit %s carries no file list; skipping", sha[:12] or "?"
            )
            return []

        buckets: dict[str, _Bucket] = {}
        pending_names: list[tuple[str, str, str]] = []  # (name, path, kind)

        def bucket_for(key: str) -> _Bucket:
            return buckets.setdefault(key, _Bucket(key))

        for entry in files:
            if not isinstance(entry, dict):
                continue
            filename = str(entry.get("filename") or "")
            kind = self._classify_path(filename)
            if kind is None:
                continue
            patch = entry.get("patch")
            if not isinstance(patch, str) or not patch:
                log.warning(
                    "inferencex: commit %s has no patch for %s (too large or "
                    "binary); recording the path only",
                    sha[:12] or "?",
                    filename,
                )
                patch = ""
            added = _added_lines(patch)
            status = str(entry.get("status") or "")

            if kind == "models_md":
                for name, prefix in parse_models_table_rows(added):
                    if prefix:
                        bucket = bucket_for(prefix)
                        bucket.prefix = prefix
                        if name not in bucket.row_names:
                            bucket.row_names.append(name)
                        bucket.note(filename, KIND_MODEL_ROW)
                        bucket.new_model_row = True
                    else:
                        pending_names.append((name, filename, KIND_MODEL_ROW))

            elif kind == "master_config":
                parsed = parse_master_config_added(added)
                prefixes = parsed["prefixes"] or _dedupe(
                    p for p in (prefix_from_config_key(k) for k in parsed["config_keys"]) if p
                )
                if not prefixes and status in ("added", "renamed"):
                    guess = prefix_from_config_filename(filename)
                    prefixes = [guess] if guess else []
                for prefix in prefixes:
                    bucket = bucket_for(prefix)
                    bucket.prefix = prefix
                    bucket.model_ids.extend(parsed["model_ids"])
                    bucket.config_keys.extend(
                        k
                        for k in parsed["config_keys"]
                        if prefix_from_config_key(k) in (None, prefix)
                    )
                    bucket.note(filename, KIND_CONFIG_ENTRY)
                    if status in ("added", "renamed"):
                        bucket.note(filename, KIND_CONFIG_FILE_ADDED)
                        if filename not in bucket.added_files:
                            bucket.added_files.append(filename)

            elif kind == "config_other":
                if status not in ("added", "renamed"):
                    continue
                prefix = prefix_from_config_filename(filename)
                if not prefix:
                    continue
                bucket = bucket_for(prefix)
                bucket.prefix = prefix
                bucket.note(filename, KIND_CONFIG_FILE_ADDED)
                if filename not in bucket.added_files:
                    bucket.added_files.append(filename)

            elif kind == "perf_changelog":
                for changelog in parse_perf_changelog_patch(patch):
                    self._absorb_changelog(
                        changelog, filename, bucket_for, pr_number, html_url
                    )

        # MODELS.md rows without an inline prefix: attach via the live index.
        for name, filename, kind in pending_names:
            prefix = self._name_prefixes.get(_norm_key(name))
            bucket = bucket_for(prefix or f"name:{_norm_key(name)}")
            if prefix:
                bucket.prefix = prefix
            if name not in bucket.row_names:
                bucket.row_names.append(name)
            bucket.note(filename, kind)
            bucket.new_model_row = True

        signals: list[Signal] = []
        for bucket in buckets.values():
            signal = self._build_signal(
                bucket,
                sha=sha,
                subject=subject,
                observed_at=observed_at,
                html_url=html_url,
                pr_number=pr_number,
            )
            if signal is not None:
                signals.append(signal)
        return signals

    # -- helpers ----------------------------------------------------------

    def _classify_path(self, filename: str) -> str | None:
        """Which watched artifact a changed file belongs to (``None`` = ignore)."""
        for path in self.paths:
            if path.endswith(".md") and filename == path:
                return "models_md"
            if path.endswith((".yaml", ".yml")) and filename == path:
                return "perf_changelog"
            if not path.endswith((".md", ".yaml", ".yml")) and (
                filename == path or filename.startswith(path.rstrip("/") + "/")
            ):
                return "master_config" if is_master_config(filename) else "config_other"
        return None

    def _absorb_changelog(
        self,
        changelog: dict[str, Any],
        filename: str,
        bucket_for: Any,
        pr_number: str | None,
        html_url: str,
    ) -> None:
        descriptions = list(changelog["descriptions"])
        metrics: dict[str, float] = {}
        note_parts: list[str] = []
        for text in descriptions:
            found = extract_perf_metrics(text)
            if found:
                note_parts.append(text)
                for key, value in found.items():
                    metrics.setdefault(key, value)

        config_keys = changelog["config_keys"] or []
        scenario = (changelog["scenario_type"] or [None])[0]
        pr_link = changelog["pr_link"] or (
            f"https://github.com/{self.repo}/pull/{pr_number}" if pr_number else None
        )

        by_prefix: dict[str, list[str]] = {}
        for key in config_keys:
            prefix = prefix_from_config_key(key)
            if prefix:
                by_prefix.setdefault(prefix, []).append(key)
        if not by_prefix:
            return

        notes = " ".join(note_parts)[:1500]
        for prefix, keys in by_prefix.items():
            bucket = bucket_for(prefix)
            bucket.prefix = prefix
            bucket.config_keys.extend(keys)
            bucket.note(filename, KIND_PERF_CHANGELOG)
            for text in descriptions:
                if text not in bucket.perf_notes:
                    bucket.perf_notes.append(text)
            if not metrics:
                continue
            for key in keys:
                parts = describe_config_key(key)
                row: dict[str, Any] = {
                    "hardware": parts["hardware"],
                    "config_key": key,
                    "precision": parts["precision"],
                    "framework": parts["framework"],
                    "scenario": scenario,
                }
                if pr_link:
                    row["pr"] = pr_link
                row.update(metrics)
                row["notes"] = notes
                bucket.perf.append(row)

    def _display_name(self, bucket: _Bucket) -> str:
        if bucket.prefix and bucket.prefix in self._prefix_names:
            return self._prefix_names[bucket.prefix]
        if bucket.row_names:
            return min(bucket.row_names, key=len)
        for model_id in bucket.model_ids:
            stripped = strip_quant_suffix(model_id.split("/", 1)[-1])
            if stripped:
                return stripped
        if bucket.prefix:
            return bucket.prefix
        return bucket.key.removeprefix("name:")

    def _pick_org(self, model_ids: Sequence[str]) -> str | None:
        best: tuple[int, str] | None = None
        for model_id in model_ids:
            owner = model_id.split("/", 1)[0]
            lowered = owner.lower()
            if lowered in FRONTIER_ORGS and lowered not in PACKAGER_ORGS:
                rank = 3
            elif lowered not in PACKAGER_ORGS:
                rank = 2
            else:
                rank = 1
            if best is None or rank > best[0]:
                best = (rank, owner)
        return best[1] if best else None

    def _build_signal(
        self,
        bucket: _Bucket,
        *,
        sha: str,
        subject: str,
        observed_at: datetime,
        html_url: str,
        pr_number: str | None,
    ) -> Signal | None:
        if not bucket.kinds:
            return None
        model_ids = _dedupe(bucket.model_ids)
        bucket.config_keys = config_keys = _dedupe(bucket.config_keys)
        display_name = self._display_name(bucket)
        if not display_name:
            return None

        urls: dict[str, str] = {}
        if html_url:
            urls["commit"] = html_url
        if pr_number:
            urls["pr"] = f"https://github.com/{self.repo}/pull/{pr_number}"
        else:
            for row in bucket.perf:
                if row.get("pr"):
                    urls["pr"] = str(row["pr"])
                    break
        doc_path = next((p for p in bucket.paths if p.endswith(".md")), None)
        if doc_path and sha:
            urls["docs"] = f"https://github.com/{self.repo}/blob/{sha}/{doc_path}"

        perf_rows = bucket.perf[: self.max_perf_rows]
        extra: dict[str, Any] = {
            "repo": self.repo,
            "prefix": bucket.prefix,
            "paths": list(bucket.paths),
            "kinds": list(bucket.kinds),
            "config_keys": config_keys,
            "new_model_row": bucket.new_model_row,
            "added_files": list(bucket.added_files),
            "commit_subject": subject,
            "perf": perf_rows,
            "perf_notes": bucket.perf_notes[:20],
        }
        if bucket.row_names:
            extra["models_md_names"] = list(bucket.row_names)

        return Signal(
            source=self.source,
            observed_at=observed_at,
            arch_ids=[],  # InferenceX exposes no architectures[]; see module docstring
            model_type=None,
            model_ids=model_ids,
            org=self._pick_org(model_ids),
            display_name=display_name,
            config=None,
            urls=urls,
            evidence=_evidence(display_name, bucket, sha),
            raw_ref=sha,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# module-level helpers used by the connector
# ---------------------------------------------------------------------------


def _added_lines(patch: str) -> list[str]:
    """The ``+`` lines of a unified diff, without the marker."""
    return [
        line[1:]
        for line in (patch or "").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def _pr_number_from_subject(subject: str) -> str | None:
    match = re.search(r"\(#(\d+)\)\s*$", subject or "")
    return match.group(1) if match else None


def _decode_contents(payload: Any) -> str | None:
    """Decode a ``/contents/{path}`` response body into text."""
    if not isinstance(payload, dict):
        return None
    content = payload.get("content")
    encoding = payload.get("encoding")
    if isinstance(content, str) and encoding == "base64":
        import base64

        try:
            return base64.b64decode(content).decode("utf-8", "replace")
        except Exception:  # pragma: no cover - defensive
            return None
    if isinstance(content, str) and not encoding:
        return content
    return None


_KIND_PHRASES = {
    KIND_MODEL_ROW: "model support-matrix row added/changed in {paths}",
    KIND_CONFIG_ENTRY: "benchmark config entry in {paths}",
    KIND_CONFIG_FILE_ADDED: "new benchmark config file {paths}",
    KIND_PERF_CHANGELOG: "perf-changelog entry in {paths}",
}


def _evidence(display_name: str, bucket: _Bucket, sha: str) -> str:
    bits: list[str] = []
    for kind in bucket.kinds:
        phrase = _KIND_PHRASES.get(kind)
        if phrase:
            relevant = [p for p in bucket.paths if _relevant_path(kind, p)]
            bits.append(phrase.format(paths=", ".join(relevant) or "configs/"))
    detail = "; ".join(bits) or "touched a watched InferenceX path"
    keys = ", ".join(bucket.config_keys[:3])
    if keys:
        detail += f" (keys: {keys})"
    if bucket.perf:
        detail += f"; {len(bucket.perf)} perf row(s)"
    return f"InferenceX benchmarks {display_name}: {detail} [commit {sha[:9]}]"


def _relevant_path(kind: str, path: str) -> bool:
    if kind == KIND_MODEL_ROW:
        return path.endswith(".md")
    if kind == KIND_PERF_CHANGELOG:
        return path.endswith("perf-changelog.yaml")
    return path.startswith("configs/")


__all__ = [
    "InferenceXConnector",
    "WATCHED_PATHS",
    "build_prefix_name_map",
    "describe_config_key",
    "extract_perf_metrics",
    "is_master_config",
    "parse_master_config_added",
    "parse_models_table_rows",
    "parse_perf_changelog_patch",
    "prefix_from_config_filename",
    "prefix_from_config_key",
    "resolve_github_token",
    "strip_quant_suffix",
]
