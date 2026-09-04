"""Framework connector: watch vLLM and SGLang for model-support PRs.

Why this source matters: a serving framework merging a new model architecture is
the earliest *public, structured* signal that a new architecture exists. It often
lands on the same day as the weights (vLLM calls it "day-0 support"), and unlike a
HuggingFace repo it comes with a diff that names the architecture class exactly as
it appears in ``config.json``'s ``architectures[]``.

Two discovery routes, unioned per repo:

1. **Registry-path commits** (high precision). ``GET /repos/{repo}/commits?path=...``
   over the model-registry path. In vLLM every new architecture must add a key to
   ``vllm/model_executor/models/registry.py``, so this path is almost a pure feed of
   architecture additions. The PR number is recovered from the squash-commit
   subject's trailing ``(#12345)``.
2. **Title search** (recall). ``GET /search/issues`` for merged PRs in the window
   whose title contains "model" or "support". GitHub tokenizes ``[Model]`` down to
   ``model``, so one query covers the ``[Model]`` / ``[New Model]`` tag convention
   and the untagged ``Add Foo Model`` style at once. Titles are then re-filtered
   locally, because the search side is fuzzy.

The architecture names themselves come from the PR's **changed files**, not the
title. Empirically (see ``tests/fixtures/frameworks/``) no merged vLLM or SGLang PR
has ever put ``XxxForCausalLM`` in its title -- titles carry marketing names
("GLM-5.3-Flash", "Kimi K3"). The reliable extraction points are:

* added lines in ``**/registry.py`` -- ``"K2HorizonForCausalLM": ("k2_horizon", ...)``
* ``class XxxForCausalLM(`` in an added/modified model module
* SGLang's ``EntryClass = [XllmForCausalLM, K2HorizonForCausalLM]`` footer

Titles are still scanned (cheap, and occasionally a PR does name the class), and
they supply ``display_name`` for the alias-fallback join path.

Read-only. No write scopes, no mutations, never raises for source problems.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from .base import Signal

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
_API_VERSION = "2022-11-28"


# ---------------------------------------------------------------------------
# Repo descriptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoSpec:
    """Everything that differs between the two frameworks."""

    source: str  # Signal.source: "vllm" | "sglang"
    repo: str  # "owner/name"

    # Paths handed to GET /commits?path= — the narrow, high-signal feed.
    commit_paths: tuple[str, ...]

    # A changed file under one of these prefixes makes the PR "model code".
    model_path_prefixes: tuple[str, ...]

    # Files whose *added* lines are an architecture registry (dict keys are
    # architecture strings). Matched as filename suffixes.
    registry_file_suffixes: tuple[str, ...]


VLLM = RepoSpec(
    source="vllm",
    repo="vllm-project/vllm",
    commit_paths=("vllm/model_executor/models/registry.py",),
    # NOTE: PLAN.md names only vllm/model_executor/models/. Current vLLM has
    # migrated new architectures into per-model packages under vllm/models/
    # (e.g. vllm/models/glm5next/, vllm/models/kimi_k3/) while keeping the
    # registry where it was. Both prefixes are needed.
    model_path_prefixes=(
        "vllm/model_executor/models/",
        "vllm/models/",
        "vllm/transformers_utils/configs/",
    ),
    registry_file_suffixes=(
        "vllm/model_executor/models/registry.py",
        "tests/models/registry.py",
    ),
)

SGLANG = RepoSpec(
    source="sglang",
    repo="sgl-project/sglang",
    commit_paths=("python/sglang/srt/models",),
    model_path_prefixes=(
        "python/sglang/srt/models/",
        "python/sglang/srt/configs/",
    ),
    # SGLang has no central registry; it discovers architectures from each
    # module's EntryClass. Kept for symmetry / future-proofing.
    registry_file_suffixes=("python/sglang/srt/models/registry.py",),
)

DEFAULT_REPOS: tuple[RepoSpec, ...] = (VLLM, SGLANG)


# ---------------------------------------------------------------------------
# Title classification
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"\[([^\]]*)\]")
_LEADING_TAGS_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)+")

# A bracketed tag that marks maintenance rather than a new architecture.
_NEG_TAG_RE = re.compile(
    r"^(?:bug\s*fix|bugfix|fix|fixes|hotfix|doc|docs|ci|ci/build|ci-build|mypy"
    r"|mypy\s*fix|test|tests|refactor|revert|chore|typo|lint|build|misc|perf"
    r"|performance|kernel|benchmark|bench|release|deps|dependabot)$",
    re.I,
)

# A word anywhere in the tag-stripped title that marks maintenance.
_NEG_WORD_RE = re.compile(
    r"\b(?:fix|fixes|fixed|fixing|bugfix|revert|reverts|reverted|refactor"
    r"|refactors|refactoring|deprecate|deprecates|deprecated|remove|removes"
    r"|removed|removing|rename|renames|renamed|cleanup|typo|mypy|optimize"
    r"|optimizes|optimization|optimizations|disable|disables|disabled"
    r"|document|documents|documentation|doc|docs)\b",
    re.I,
)

_MODEL_TAG_RE = re.compile(r"^\s*(?:new\s+)?models?(?:\s+support)?\s*$", re.I)
_NEW_MODEL_TAG_RE = re.compile(r"^\s*new\s+models?(?:\s+support)?\s*$", re.I)

_SUPPORT_RE = re.compile(r"\bsupport(?:s|ed|ing)?\b", re.I)
_ADD_RE = re.compile(r"\b(?:add|adds|adding|introduce|introduces|land|lands)\b", re.I)
_MODEL_WORD_RE = re.compile(r"\bmodels?\b", re.I)


def strip_tags(title: str) -> str:
    """Drop the leading ``[Foo][Bar]`` tag run."""
    return _LEADING_TAGS_RE.sub("", title or "").strip()


def title_tags(title: str) -> list[str]:
    return [t.strip() for t in _TAG_RE.findall(title or "")]


def is_maintenance_title(title: str) -> bool:
    """A bracketed maintenance tag or a maintenance verb -> not a new model.

    ``[Bugfix][Model] Fix FP8 PLE loading in mixed ModelOpt checkpoints`` and
    ``[rotary] Fix the fused Qwen3.5 RoPE kernel ...`` are maintenance;
    ``[Model] add GLM-5.3-Flash support`` is not. This is the veto that keeps
    the registry-path commit route (which otherwise bypasses the positive title
    gate) from dragging in every bugfix that touches model code.
    """
    if not title:
        return False
    if any(_NEG_TAG_RE.match(t) for t in title_tags(title)):
        return True
    return bool(_NEG_WORD_RE.search(strip_tags(title)))


def is_model_support_title(title: str) -> bool:
    """True when the title reads like "this PR adds support for a model".

    Tuned against the real merged-title corpus recorded in
    ``tests/fixtures/frameworks/observed_titles.json``.
    """
    if not title:
        return False
    if is_maintenance_title(title):
        return False
    tags = title_tags(title)
    body = strip_tags(title)

    has_model_tag = any(_MODEL_TAG_RE.match(t) for t in tags)
    has_new_model_tag = any(_NEW_MODEL_TAG_RE.match(t) for t in tags)
    has_support = bool(_SUPPORT_RE.search(body))
    has_add = bool(_ADD_RE.search(body))
    has_model_word = bool(_MODEL_WORD_RE.search(body))

    if has_new_model_tag:
        return True
    if has_model_tag and (has_support or has_add):
        return True
    if has_support and (has_model_word or has_add):
        return True
    if has_add and has_model_word:
        return True
    return False


_DISPLAY_LEAD_RE = re.compile(
    r"^(?:(?:day-?0|native|initial|basic|preliminary|full)\s+)*"
    r"(?:add|adds|adding|support|supports|supporting|introduce|introduces|enable|enables|land|lands)\b"
    r"(?:\s+(?:support|native|day-?0|initial|for|the|a|an|new)\b)*\s*",
    re.I,
)
_DISPLAY_TAIL_RE = re.compile(
    r"(?:\s*[:,-]?\s*(?:day-?0|native|serving|inference|initial|basic|full)?"
    r"\s*(?:models?)?\s*(?:support(?:ing)?|enablement)?\s*)$",
    re.I,
)
_DISPLAY_TRAILING_PAREN_RE = re.compile(r"\s*\(([A-Z][A-Za-z0-9_]*)\)\s*$")


def display_name_from_title(title: str) -> str:
    """Best-effort human model name out of a PR title.

    ``[Model] add GLM-5.3-Flash support`` -> ``GLM-5.3-Flash``.
    Falls back to the tag-stripped title, and finally to the raw title, so this
    never returns "" for a non-empty input.
    """
    body = strip_tags(title)
    if not body:
        return (title or "").strip()

    name = body
    # "model: support FastH3 ..." style prefix
    name = re.sub(r"^\s*models?\s*:\s*", "", name, flags=re.I)
    name = _DISPLAY_LEAD_RE.sub("", name, count=1).strip()
    prev = None
    while prev != name:
        prev = name
        name = _DISPLAY_TAIL_RE.sub("", name, count=1).strip()
        name = name.rstrip(":,-").strip()
    # "Ling-3.0-flash (BailingMoeV3)" -> keep the marketing half
    m = _DISPLAY_TRAILING_PAREN_RE.search(name)
    if m:
        name = name[: m.start()].strip()
    return name or body


# ---------------------------------------------------------------------------
# Architecture-name extraction
# ---------------------------------------------------------------------------

# HuggingFace architecture strings are class names: CamelCase with a
# ``For<Task>`` tail. Underscores occur (Ernie4_5_ForCausalLM).
_ARCH_FOR_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*For[A-Z][A-Za-z0-9_]*)\b")
# Multi-token-prediction heads register as architectures too
# (Glm5NextMTPModel, Qwen4ExpMTP, DeepSeekMTPModel).
_ARCH_MTP_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*MTP(?:Model)?)\b")

_CLASS_DEF_RE = re.compile(r"^\+\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(:]")
_CONFIG_CLASS_RE = re.compile(
    r"^\+\s*class\s+([A-Za-z_][A-Za-z0-9_]*Config)\s*[(:]"
)
_ENTRY_CLASS_RE = re.compile(r"^\+\s*EntryClass\s*=")
_REGISTRY_KEY_RE = re.compile(r'^\+\s*"([A-Za-z][A-Za-z0-9_]*)"\s*:')
_HF_URL_RE = re.compile(
    r"huggingface\.co/(?:models/)?([A-Za-z0-9][\w.-]*)/([\w.-]+)"
)
_HF_QUOTED_REPO_RE = re.compile(r'"([A-Za-z0-9][\w.-]*/[\w.-]+)"')

_NON_MODEL_OWNERS = {"docs", "blog", "spaces", "datasets", "papers", "collections"}


def _clean_repo_id(owner: str, name: str) -> str:
    """A repo id lifted out of prose keeps the sentence's punctuation."""
    return f"{owner.strip('.-_')}/{name.strip('.-_')}"

# Suffixes that turn an architecture name into a *helper* class name. vLLM has
# `class Qwen4ExpForCausalLMConfig` (a per-arch config patcher) and
# `class KimiK3ForConditionalGenerationConfig`; the architecture is the stem.
_HELPER_SUFFIXES = ("Config", "Mixin", "Base", "Info", "Impl", "Wrapper", "Test")


def normalize_arch_name(name: str) -> str:
    """Strip a helper-class suffix so ``XForCausalLMConfig`` -> ``XForCausalLM``."""
    changed = True
    while changed:
        changed = False
        for sfx in _HELPER_SUFFIXES:
            if len(name) > len(sfx) + 3 and name.endswith(sfx):
                name = name[: -len(sfx)]
                changed = True
    return name


def looks_like_arch_name(name: str) -> bool:
    """Does this token look like an ``architectures[]`` entry?"""
    if not name or len(name) < 4 or "/" in name:
        return False
    if not name[0].isupper():
        return False
    if name.isupper():  # ACRONYM, not a class name
        return False
    if not any(c.islower() for c in name):
        return False
    return bool(_ARCH_FOR_RE.fullmatch(name) or _ARCH_MTP_RE.fullmatch(name))


def _is_registry_file(filename: str, spec: RepoSpec) -> bool:
    return any(filename.endswith(sfx) for sfx in spec.registry_file_suffixes)


def _is_model_path(filename: str, spec: RepoSpec) -> bool:
    return any(filename.startswith(p) for p in spec.model_path_prefixes)


def _added_lines(patch: Any) -> list[str]:
    if not patch or not isinstance(patch, str):
        return []
    return [
        ln
        for ln in patch.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]


def extract_arch_names_from_text(text: str, *, allow_mtp: bool = False) -> list[str]:
    """Architecture-shaped tokens in free text (a title or PR body)."""
    found: list[str] = []
    pats = [_ARCH_FOR_RE] + ([_ARCH_MTP_RE] if allow_mtp else [])
    for pat in pats:
        for m in pat.finditer(text or ""):
            name = normalize_arch_name(m.group(1))
            if looks_like_arch_name(name):
                found.append(name)
    return _dedup(found)


def _dedup(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


@dataclass
class FileFindings:
    """What the changed-file list of one PR yielded."""

    registry_archs: list[str] = field(default_factory=list)
    class_archs: list[str] = field(default_factory=list)
    entry_class_archs: list[str] = field(default_factory=list)
    config_classes: list[str] = field(default_factory=list)
    module_stems: list[str] = field(default_factory=list)  # e.g. "k2_horizon"
    added_model_files: list[str] = field(default_factory=list)
    touched_model_files: list[str] = field(default_factory=list)
    touched_registry: bool = False
    model_ids: list[str] = field(default_factory=list)

    @property
    def arch_names(self) -> list[str]:
        """Registry keys first: they are the authoritative architecture strings."""
        return _dedup(self.registry_archs + self.entry_class_archs + self.class_archs)

    @property
    def touches_model_code(self) -> bool:
        return bool(self.touched_model_files or self.touched_registry)


def _module_stem(filename: str, spec: RepoSpec) -> str | None:
    """The model module name a changed path implies.

    ``vllm/model_executor/models/k2_horizon.py`` -> ``k2_horizon``
    ``vllm/models/glm5next/nvidia/model.py``     -> ``glm5next``
    ``python/sglang/srt/models/xllm.py``         -> ``xllm``
    """
    for prefix in spec.model_path_prefixes:
        if not filename.startswith(prefix):
            continue
        rest = filename[len(prefix) :]
        if not rest:
            return None
        head = rest.split("/", 1)[0]
        if head.endswith(".py"):
            head = head[:-3]
        if head in ("", "__init__", "registry", "config", "interfaces", "utils"):
            return None
        return head
    return None


def extract_from_files(files: Sequence[dict[str, Any]], spec: RepoSpec) -> FileFindings:
    """Mine a PR's changed-file list (with patches) for architecture names."""
    out = FileFindings()
    for f in files or []:
        if not isinstance(f, dict):
            continue
        filename = f.get("filename") or ""
        status = (f.get("status") or "").lower()
        if not filename:
            continue

        is_reg = _is_registry_file(filename, spec)
        is_model = _is_model_path(filename, spec)
        if is_reg:
            out.touched_registry = True
        if is_model:
            out.touched_model_files.append(filename)
            if status == "added":
                out.added_model_files.append(filename)
            stem = _module_stem(filename, spec)
            if stem:
                out.module_stems.append(stem)

        if not (is_reg or is_model):
            continue

        added = _added_lines(f.get("patch"))

        if is_reg:
            for ln in added:
                m = _REGISTRY_KEY_RE.match(ln)
                if m:
                    key = normalize_arch_name(m.group(1))
                    if looks_like_arch_name(key):
                        out.registry_archs.append(key)
                for repo_id in _HF_QUOTED_REPO_RE.findall(ln):
                    owner, _, name = repo_id.partition("/")
                    cleaned = _clean_repo_id(owner, name)
                    if _plausible_hf_repo(cleaned):
                        out.model_ids.append(cleaned)

        if is_model:
            for i, ln in enumerate(added):
                cm = _CLASS_DEF_RE.match(ln)
                if cm:
                    cls = normalize_arch_name(cm.group(1))
                    if looks_like_arch_name(cls):
                        out.class_archs.append(cls)
                cc = _CONFIG_CLASS_RE.match(ln)
                if cc:
                    out.config_classes.append(cc.group(1))
                if _ENTRY_CLASS_RE.match(ln):
                    # EntryClass may be a one-liner or a multi-line list.
                    window = "\n".join(added[i : i + 12])
                    window = window.split("]", 1)[0] if "[" in window else window
                    for name in extract_arch_names_from_text(window, allow_mtp=True):
                        out.entry_class_archs.append(name)

        for ln in added:
            for owner, name in _HF_URL_RE.findall(ln):
                repo_id = _clean_repo_id(owner, name)
                if _plausible_hf_repo(repo_id):
                    out.model_ids.append(repo_id)

    out.registry_archs = _dedup(out.registry_archs)
    out.class_archs = _dedup(out.class_archs)
    out.entry_class_archs = _dedup(out.entry_class_archs)
    out.config_classes = _dedup(out.config_classes)
    out.module_stems = _dedup(out.module_stems)
    out.model_ids = _dedup(out.model_ids)
    return out


def _plausible_hf_repo(repo_id: str) -> bool:
    if repo_id.count("/") != 1:
        return False
    owner, name = repo_id.split("/")
    if not owner or not name:
        return False
    if owner.lower() in _NON_MODEL_OWNERS:
        return False
    if any(name.endswith(ext) for ext in (".py", ".yaml", ".yml", ".json", ".md", ".txt")):
        return False
    if owner.startswith(".") or name.startswith("."):
        return False
    return True


def extract_model_ids_from_text(text: str) -> list[str]:
    out = []
    for owner, name in _HF_URL_RE.findall(text or ""):
        repo_id = _clean_repo_id(owner, name)
        if _plausible_hf_repo(repo_id):
            out.append(repo_id)
    return _dedup(out)


# ---------------------------------------------------------------------------
# GitHub token + HTTP
# ---------------------------------------------------------------------------


def resolve_token(explicit: str | None = None) -> str | None:
    """``GH_TOKEN`` / ``GITHUB_TOKEN``, else ``gh auth token``, else None.

    Read-only usage only. Returns None rather than raising when nothing is
    available; the caller degrades to unauthenticated (and will very likely be
    rate limited, which is handled as a partial scan).
    """
    if explicit:
        return explicit
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        tok = os.environ.get(var)
        if tok and tok.strip():
            return tok.strip()
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env
        log.debug("gh auth token unavailable: %s", exc)
        return None
    if proc.returncode == 0:
        tok = (proc.stdout or "").strip()
        if tok:
            return tok
    log.debug("gh auth token returned %s", proc.returncode)
    return None


@dataclass
class _Resp:
    status: int
    data: Any
    headers: dict[str, str] = field(default_factory=dict)


class _Http:
    """Thin, forgiving GET wrapper. Never raises; records rate-limit state."""

    def __init__(
        self,
        session: Any = None,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if session is None:
            import requests  # local import keeps the module importable offline

            session = requests.Session()
        self.session = session
        self.token = token
        self.timeout = timeout
        self.rate_limited = False
        self.calls = 0
        self.errors: list[str] = []

    def _headers(self) -> dict[str, str]:
        h = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": _API_VERSION,
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get(self, url: str, params: dict[str, Any] | None = None) -> _Resp | None:
        """Return a successful response, or None (and log) for anything else."""
        if self.rate_limited:
            return None
        self.calls += 1
        try:
            raw = self.session.get(
                url, params=params or {}, headers=self._headers(), timeout=self.timeout
            )
        except Exception as exc:  # network/DNS/TLS/timeout - a partial scan is fine
            msg = f"GET {url} failed: {exc!r}"
            log.warning("%s", msg)
            self.errors.append(msg)
            return None

        status = int(getattr(raw, "status_code", 0) or 0)
        headers = {str(k).lower(): str(v) for k, v in dict(getattr(raw, "headers", {}) or {}).items()}
        if status in (403, 429):
            self.rate_limited = True
            msg = (
                f"GET {url} -> {status} (rate limited; remaining="
                f"{headers.get('x-ratelimit-remaining')}, reset={headers.get('x-ratelimit-reset')}); "
                "returning partial results"
            )
            log.warning("%s", msg)
            self.errors.append(msg)
            return None
        if status != 200:
            msg = f"GET {url} -> {status}"
            log.warning("%s", msg)
            self.errors.append(msg)
            return None
        try:
            data = raw.json()
        except Exception as exc:
            msg = f"GET {url} returned unparseable JSON: {exc!r}"
            log.warning("%s", msg)
            self.errors.append(msg)
            return None
        return _Resp(status=status, data=data, headers=headers)

    def paginate(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        max_pages: int = 3,
        per_page: int = 100,
    ) -> list[Any]:
        out: list[Any] = []
        for page in range(1, max_pages + 1):
            p = dict(params or {})
            p.update({"per_page": per_page, "page": page})
            resp = self.get(url, p)
            if resp is None:
                break
            items = resp.data
            if isinstance(items, dict):  # search endpoints
                items = items.get("items") or []
            if not isinstance(items, list):
                break
            out.extend(items)
            if len(items) < per_page:
                break
        return out


# ---------------------------------------------------------------------------
# Candidate PRs
# ---------------------------------------------------------------------------

_PR_NUM_IN_SUBJECT_RE = re.compile(r"\(#(\d+)\)\s*$")


@dataclass
class _Candidate:
    number: int
    title: str = ""
    body: str = ""
    html_url: str = ""
    when: datetime | None = None
    author: str | None = None
    routes: set[str] = field(default_factory=set)  # {"registry_commit", "title_search"}
    needs_detail: bool = True


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    txt = value.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# The connector
# ---------------------------------------------------------------------------


class FrameworkConnector:
    """Poll vLLM + SGLang for model-support PRs. Stateless and window-based."""

    name = "frameworks"
    repos: tuple[RepoSpec, ...] = DEFAULT_REPOS

    def __init__(
        self,
        repos: Sequence[RepoSpec] | None = None,
        *,
        token: str | None = None,
        session: Any = None,
        api_base: str = GITHUB_API,
        max_prs_per_repo: int = 40,
        max_file_pages: int = 3,
        max_search_pages: int = 2,
        timeout: float = 30.0,
        resolve_token_from_env: bool = True,
        until: datetime | None = None,
        emit_title_only: bool = True,
    ) -> None:
        if repos is not None:
            self.repos = tuple(repos)
        self.api_base = api_base.rstrip("/")
        # Optional upper bound on the window. poll(since) alone is what the
        # Connector protocol requires; `until` exists so a historical window can
        # be replayed reproducibly (PLAN.md component J, the backtest) and so
        # recorded fixtures do not drift as the repos move on.
        self.until = _as_utc(until) if until is not None else None
        self.max_prs_per_repo = max_prs_per_repo
        self.max_file_pages = max_file_pages
        self.max_search_pages = max_search_pages
        # A PR that only *looks* like a model PR by its title, with no new arch
        # name and no new model file, is a very weak signal. Kept by default so
        # the run log can be used to tune the filter (PLAN.md component H);
        # set False for a high-precision scan.
        self.emit_title_only = emit_title_only
        tok = token
        if tok is None and resolve_token_from_env:
            tok = resolve_token()
        self.http = _Http(session=session, token=tok, timeout=timeout)

    # -- public ------------------------------------------------------------

    def poll(self, since: datetime) -> list[Signal]:
        """Signals for model-support PRs merged since ``since``.

        Never raises: a failing repo, a rate limit or a shape surprise degrades
        to fewer Signals.
        """
        since_utc = _as_utc(since)
        signals: list[Signal] = []
        for spec in self.repos:
            try:
                signals.extend(self._poll_repo(spec, since_utc))
            except Exception as exc:  # defensive: one repo must not kill the poll
                log.warning("frameworks: %s poll failed: %r", spec.repo, exc)
        signals.sort(
            key=lambda s: (s.observed_at, s.raw_ref),
            reverse=True,
        )
        return signals

    # -- per repo ----------------------------------------------------------

    def _poll_repo(self, spec: RepoSpec, since: datetime) -> list[Signal]:
        cands = self._discover(spec, since)
        ordered = sorted(
            cands.values(),
            key=lambda c: (
                "registry_commit" in c.routes,
                c.when or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )

        signals: list[Signal] = []
        examined = 0
        for cand in ordered:
            if examined >= self.max_prs_per_repo:
                log.info(
                    "frameworks: %s hit max_prs_per_repo=%d; stopping early",
                    spec.repo,
                    self.max_prs_per_repo,
                )
                break
            if self.http.rate_limited:
                log.warning(
                    "frameworks: %s rate limited after %d PRs; partial results",
                    spec.repo,
                    examined,
                )
                break

            # Only pay for /files when the PR is worth inspecting.
            # The registry-path route waives the *positive* title requirement
            # (a title need not advertise the model), but not the maintenance
            # veto: SGLang's registry "path" is its whole models/ directory, so
            # without the veto every bugfix in there would be examined.
            from_registry = "registry_commit" in cand.routes
            title_ok = is_model_support_title(cand.title)
            worth_it = title_ok or (from_registry and not is_maintenance_title(cand.title))
            if not worth_it:
                continue

            if cand.needs_detail:
                self._fill_detail(spec, cand)
                if cand.when is not None and not self._in_window(cand.when, since):
                    continue
                title_ok = is_model_support_title(cand.title)
                if not (title_ok or (from_registry and not is_maintenance_title(cand.title))):
                    continue

            examined += 1
            files = self._fetch_files(spec, cand.number)
            findings = extract_from_files(files, spec)

            sig = self._build_signal(spec, cand, findings, title_ok, files)
            if sig is not None:
                signals.append(sig)
        return signals

    def _in_window(self, when: datetime, since: datetime) -> bool:
        if when < since:
            return False
        if self.until is not None and when > self.until:
            return False
        return True

    def _discover(self, spec: RepoSpec, since: datetime) -> dict[int, _Candidate]:
        cands: dict[int, _Candidate] = {}
        for c in self._from_registry_commits(spec, since):
            cands.setdefault(c.number, c).routes.update(c.routes)
        for c in self._from_title_search(spec, since):
            existing = cands.get(c.number)
            if existing is None:
                cands[c.number] = c
            else:
                existing.routes.update(c.routes)
                existing.title = existing.title or c.title
                existing.body = existing.body or c.body
                existing.html_url = existing.html_url or c.html_url
                existing.when = existing.when or c.when
                existing.author = existing.author or c.author
                if c.title and c.body is not None:
                    existing.needs_detail = False
        return cands

    def _from_registry_commits(
        self, spec: RepoSpec, since: datetime
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
        url = f"{self.api_base}/repos/{spec.repo}/commits"
        for path in spec.commit_paths:
            params = {"path": path, "since": since.strftime("%Y-%m-%dT%H:%M:%SZ")}
            if self.until is not None:
                params["until"] = self.until.strftime("%Y-%m-%dT%H:%M:%SZ")
            commits = self.http.paginate(url, params, max_pages=2)
            for c in commits:
                if not isinstance(c, dict):
                    continue
                commit = c.get("commit") or {}
                message = commit.get("message") or ""
                subject = message.splitlines()[0] if message else ""
                m = _PR_NUM_IN_SUBJECT_RE.search(subject)
                if not m:
                    continue
                number = int(m.group(1))
                when = _parse_dt(
                    (commit.get("committer") or {}).get("date")
                    or (commit.get("author") or {}).get("date")
                )
                title = _PR_NUM_IN_SUBJECT_RE.sub("", subject).strip()
                out.append(
                    _Candidate(
                        number=number,
                        title=title,
                        html_url=f"https://github.com/{spec.repo}/pull/{number}",
                        when=when,
                        routes={"registry_commit"},
                        needs_detail=True,
                    )
                )
        return out

    def _from_title_search(self, spec: RepoSpec, since: datetime) -> list[_Candidate]:
        out: list[_Candidate] = []
        url = f"{self.api_base}/search/issues"
        day = since.strftime("%Y-%m-%d")
        if self.until is None:
            merged_q = f"merged:>={day}"
        else:
            merged_q = f"merged:{day}..{self.until.strftime('%Y-%m-%d')}"
        for term in ("model in:title", "support in:title"):
            q = f"repo:{spec.repo} is:pr is:merged {merged_q} {term}"
            items = self.http.paginate(
                url,
                {"q": q, "sort": "updated", "order": "desc"},
                max_pages=self.max_search_pages,
            )
            for it in items:
                if not isinstance(it, dict):
                    continue
                number = it.get("number")
                if not isinstance(number, int):
                    continue
                when = _parse_dt(it.get("closed_at")) or _parse_dt(it.get("updated_at"))
                if when is not None and not self._in_window(when, since):
                    continue
                out.append(
                    _Candidate(
                        number=number,
                        title=it.get("title") or "",
                        body=it.get("body") or "",
                        html_url=it.get("html_url")
                        or f"https://github.com/{spec.repo}/pull/{number}",
                        when=when,
                        author=((it.get("user") or {}).get("login")),
                        routes={"title_search"},
                        needs_detail=False,
                    )
                )
        return out

    def _fill_detail(self, spec: RepoSpec, cand: _Candidate) -> None:
        resp = self.http.get(f"{self.api_base}/repos/{spec.repo}/pulls/{cand.number}")
        cand.needs_detail = False
        if resp is None or not isinstance(resp.data, dict):
            return
        pr = resp.data
        cand.title = pr.get("title") or cand.title
        cand.body = pr.get("body") or cand.body or ""
        cand.html_url = pr.get("html_url") or cand.html_url
        cand.when = (
            _parse_dt(pr.get("merged_at"))
            or _parse_dt(pr.get("closed_at"))
            or cand.when
        )
        cand.author = ((pr.get("user") or {}).get("login")) or cand.author

    def _fetch_files(self, spec: RepoSpec, number: int) -> list[dict[str, Any]]:
        items = self.http.paginate(
            f"{self.api_base}/repos/{spec.repo}/pulls/{number}/files",
            max_pages=self.max_file_pages,
        )
        return [i for i in items if isinstance(i, dict)]

    # -- signal ------------------------------------------------------------

    def _build_signal(
        self,
        spec: RepoSpec,
        cand: _Candidate,
        findings: FileFindings,
        title_ok: bool,
        files: Sequence[dict[str, Any]],
    ) -> Signal | None:
        arch_ids = list(findings.arch_names)
        # MTP heads register as architectures in their own right, and a PR body
        # that lists architectures usually lists them.
        from_title = extract_arch_names_from_text(cand.title, allow_mtp=True)
        from_body = extract_arch_names_from_text(cand.body, allow_mtp=True)
        arch_ids = _dedup(arch_ids + from_title + from_body)

        added_model_file = bool(findings.added_model_files)
        if not (arch_ids or added_model_file or (title_ok and findings.touches_model_code)):
            return None

        if findings.registry_archs:
            strength = "registry"
        elif findings.entry_class_archs or findings.class_archs:
            strength = "model_class"
        elif from_title or from_body:
            strength = "prose"
        elif added_model_file:
            strength = "new_model_file"
        else:
            strength = "title_only"
            if not self.emit_title_only:
                return None

        model_ids = _dedup(
            findings.model_ids + extract_model_ids_from_text(cand.body)
        )
        org = model_ids[0].split("/")[0].lower() if model_ids else None

        evidence_bits = [f"{spec.source} PR #{cand.number} {cand.title!r}"]
        if findings.registry_archs:
            evidence_bits.append(
                "adds registry key(s) " + ", ".join(findings.registry_archs)
            )
        elif findings.entry_class_archs:
            evidence_bits.append(
                "EntryClass exports " + ", ".join(findings.entry_class_archs)
            )
        elif findings.class_archs:
            evidence_bits.append("defines " + ", ".join(findings.class_archs))
        if findings.added_model_files:
            evidence_bits.append(
                f"{len(findings.added_model_files)} new file(s) under model registry paths"
            )
        elif findings.touched_model_files:
            evidence_bits.append(
                f"touches {len(findings.touched_model_files)} model-registry file(s)"
            )
        evidence = "; ".join(evidence_bits)

        observed_at = cand.when or datetime.now(timezone.utc)
        urls = {"pr": cand.html_url or f"https://github.com/{spec.repo}/pull/{cand.number}"}

        return Signal(
            source=spec.source,
            observed_at=observed_at,
            arch_ids=arch_ids,
            model_type=None,
            model_ids=model_ids,
            org=org,
            display_name=display_name_from_title(cand.title),
            config=None,  # a PR carries no config.json
            urls=urls,
            evidence=evidence,
            raw_ref=str(cand.number),
            extra={
                "repo": spec.repo,
                "pr_number": cand.number,
                "pr_title": cand.title,
                "pr_author": cand.author,
                "discovery_routes": sorted(cand.routes),
                "signal_strength": strength,
                "title_matched": title_ok,
                "registry_archs": findings.registry_archs,
                "entry_class_archs": findings.entry_class_archs,
                "class_archs": findings.class_archs,
                "config_classes": findings.config_classes,
                "module_stems": findings.module_stems,
                "added_model_files": findings.added_model_files,
                "touched_model_files": findings.touched_model_files[:50],
                "changed_file_count": len(files),
                "arch_from_title": from_title,
                "arch_from_body": from_body,
            },
        )


class VllmConnector(FrameworkConnector):
    """vLLM only — lets ``--sources vllm`` select one framework."""

    name = "vllm"
    repos = (VLLM,)

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("repos", (VLLM,))
        super().__init__(**kwargs)


class SglangConnector(FrameworkConnector):
    """SGLang only — lets ``--sources sglang`` select one framework."""

    name = "sglang"
    repos = (SGLANG,)

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("repos", (SGLANG,))
        super().__init__(**kwargs)


__all__ = [
    "DEFAULT_REPOS",
    "FileFindings",
    "FrameworkConnector",
    "RepoSpec",
    "SGLANG",
    "SglangConnector",
    "VLLM",
    "VllmConnector",
    "display_name_from_title",
    "extract_arch_names_from_text",
    "extract_from_files",
    "extract_model_ids_from_text",
    "is_maintenance_title",
    "is_model_support_title",
    "looks_like_arch_name",
    "normalize_arch_name",
    "resolve_token",
    "strip_tags",
]
