"""Render a Candidate to a markdown stub issue and write it to ``issues/<arch_id>.md``.

**Dry-run only, permanently.** This module contains no network client, no GitHub
API call, and no code path that files an issue anywhere. Its entire externally
visible behaviour is: format a string, and write that string to a local file.
``tests/test_emitter.py`` enforces that with an AST check on this module's imports.

The markdown file *is* the stub issue. A human (or the stage-2 deep-dive skill)
reads it; if anyone ever wants it on the tracker they paste it there themselves.

Layout of a rendered file::

    ---
    ... YAML front matter (machine-readable; the backtest parses this) ...
    ---
    # [archwatch] <arch_id> — ...
    ## Sources                               (+ the join-edge audit trail beneath it)
    ## Why it fired
    ## Silently wrong today                  (only when silent_failures is non-empty)
    ## Deterministic findings (stage 1)
    ## Config at a glance
    ## Reported performance numbers          (only when a source carried some)
    ## Third-party mechanism notes           (only when a source carried perf_notes)
    ## Stage 2 — deep dive
    <!-- archwatch:stage2:append-below -->

Two failure classes, kept strictly apart because they need opposite reactions:

``bucket0_failures``
    Fatal. BLIS refuses the config. Loud and self-announcing, so it is *not* the
    dangerous case — someone will notice.
``silent_failures``
    BLIS runs, logs a warning at most, and reports confidently wrong numbers. This is
    the reason the pipeline exists, so it is rendered above the bucket-0 verdict and
    a clean bucket 0 next to a non-empty ``silent_failures`` is called out explicitly
    as the dangerous combination.

Everything above the append marker is the stage-1 stub and is regenerated from the
Candidate. Stage 2 appends *below* the marker; :func:`write_issue` preserves
anything it finds there, so re-emitting never destroys an analysis.

Idempotence: :func:`render` is a pure function of the Candidate plus the injected
``detected_at``. No wall-clock value is read unless the Candidate carries no
timestamp at all, and no timestamp appears anywhere outside the front matter, so
re-rendering the same Candidate yields byte-identical output.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from .connectors.base import Candidate, Signal

__all__ = [
    "STAGE2_MARKER",
    "SCHEMA",
    "EmitResult",
    "default_issues_dir",
    "safe_stem",
    "issue_path",
    "issue_exists",
    "render",
    "write_issue",
    "write_issues",
    "split_stub",
    "summarize",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Everything after this line belongs to stage 2 and is never regenerated.
STAGE2_MARKER = "<!-- archwatch:stage2:append-below -->"

#: Bumped when the front-matter shape changes, so the backtest can adapt. Changes are
#: additive: /2 added silent_failures, silently_wrong, known_arch_drift, markers and
#: has_perf_notes to /1; /3 added join_edges. Readers should test for keys, not pin the
#: version.
SCHEMA = "archwatch/3"

TRIGGER_DESCRIPTIONS: dict[str, str] = {
    "T1": "new architecture whose config carries fields BLIS does not parse "
    "(unknown keys are silently dropped, so the risk is wrong numbers, not a crash)",
    "T2": "a framework support PR (vLLM / SGLang) for an architecture we had not seen",
    "T3": "a frontier org published a new architecture",
    "T4": "corroboration: two or more independent sources named the same architecture",
    "T5": "a curated benchmark / analyst entry names a model we had not seen",
    "T1-known-arch": "an architecture BLIS already knows, re-checked: its config has grown "
    "fields BLIS does not parse (config drift, not a new architecture)",
}

#: Codes that describe *how the candidate was assembled*, not why it is interesting.
#: They arrive mixed into Candidate.triggers; the front matter keeps the list verbatim
#: and also splits them out under "markers" so the backtest need not know the taxonomy.
MARKER_DESCRIPTIONS: dict[str, str] = {
    "alias-join": "no signal carried an `architectures[]` entry, so these signals were "
    "joined on a normalized display name — the `arch_id` above is that alias key, not a "
    "string read out of a config",
}

#: Triggers that assert the architecture itself is new. "T1-known-arch" contradicts
#: them, and T4 (corroboration) asserts nothing about novelty on its own.
NOVEL_ARCH_TRIGGERS: frozenset[str] = frozenset({"T1", "T2", "T3", "T5"})

#: What each join-edge prefix means. Signals are merged by union-find over these keys,
#: so an edge a human disagrees with is a false merge waiting to be split.
JOIN_EDGE_KINDS: dict[str, str] = {
    "arch": "architecture id",
    "repo": "normalized repo id",
    "family": "normalized family name",
    "alias": "normalized display-name alias",
}

KNOWN_ARCH_TRIGGER = "T1-known-arch"

SIGNIFICANCE_DESCRIPTIONS: dict[str, str] = {
    "S1": "scale: estimated total parameters at or above the threshold",
    "S2": "org track record: a frontier org, or an org whose top model clears the "
    "download threshold",
    "S3": "framework / analyst attention: a vLLM, SGLang or InferenceX signal",
    "S4": "popularity: downloads or likes over threshold, or seen on the trending sweep",
}

SOURCE_LABELS: dict[str, str] = {
    "hf": "HuggingFace",
    "vllm": "vLLM",
    "sglang": "SGLang",
    "inferencex": "InferenceX",
}

#: Config keys worth showing a human, in reading order. Only those present are shown.
CONFIG_HIGHLIGHT_KEYS: tuple[str, ...] = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "hidden_act",
    "torch_dtype",
    "tie_word_embeddings",
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "n_shared_experts",
    "first_k_dense_replace",
    "interleave_moe_layer_step",
    "kv_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
    "rope_theta",
)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_MAX_STEM = 100


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def default_issues_dir() -> Path:
    """``tools/archwatch/issues`` — resolved from this file, not from the cwd."""
    return Path(__file__).resolve().parent.parent / "issues"


def safe_stem(arch_id: str) -> str:
    """A filesystem-safe stem for ``arch_id``.

    Architecture ids are normally clean identifiers (``KimiK3ForCausalLM``) and pass
    through untouched. Anything else is sanitized: characters outside
    ``[A-Za-z0-9._-]`` collapse to ``_``; leading dots, dashes and underscores are
    stripped (no hidden files, no names that look like CLI flags); Windows device
    names get a suffix; over-long stems are truncated.

    Sanitization is lossy, so whenever it changes the string a short digest of the
    *original* arch_id is appended. Distinct architectures therefore always get
    distinct filenames, even when they sanitize to the same text.
    """
    original = arch_id if isinstance(arch_id, str) else str(arch_id)
    cleaned = _UNSAFE_CHARS.sub("_", original.strip())
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("_")
    cleaned = cleaned.lstrip("._-")
    if cleaned.rstrip(".") != cleaned:  # trailing dots break on Windows
        cleaned = cleaned.rstrip(".")
    if cleaned.lower() in _WINDOWS_RESERVED:
        cleaned = cleaned + "_"
    if not cleaned:
        cleaned = "unnamed"
    if len(cleaned) > _MAX_STEM:
        cleaned = cleaned[:_MAX_STEM].rstrip("._-")
    if cleaned != original:
        digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned}-{digest}"
    return cleaned


def issue_path(arch_id: str, out_dir: Path | str | None = None) -> Path:
    """Where the stub for ``arch_id`` lives. Also the stateless dedup key."""
    base = Path(out_dir) if out_dir is not None else default_issues_dir()
    return base / f"{safe_stem(arch_id)}.md"


def issue_exists(arch_id: str, out_dir: Path | str | None = None) -> bool:
    """True when a stub for this architecture is already on disk.

    The novelty filter's "an issue already exists" suppressor should call this
    rather than building the path itself, so both sides agree on sanitization.
    """
    return issue_path(arch_id, out_dir).is_file()


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _as_utc(dt: datetime) -> datetime:
    """Normalize to UTC. A naive datetime is *assumed* UTC — never local time,
    which would make output depend on the machine that rendered it."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return _as_utc(dt).isoformat().replace("+00:00", "Z")


def _human_count(n: int | None) -> str:
    """1_010_000_000_000 -> '1.01T'. Exact digits are kept alongside by callers."""
    if n is None:
        return "unknown"
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return str(n)
    n = int(n)
    sign = "-" if n < 0 else ""
    v = abs(n)
    units = ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for div, suffix in units:
        if v >= div:
            scaled = v / div
            text = f"{scaled:.2f}".rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"
    return f"{sign}{v}"


def _defang(text: str) -> str:
    """Neutralize a stage-2 append marker embedded in third-party text.

    Source prose is rendered verbatim. If it ever contained the literal marker, the
    stub/appendix split would cut in the wrong place and a later write could drop real
    analysis, so the colons are rewritten and the marker stops matching.
    """
    return text.replace("archwatch:stage2:", "archwatch_stage2_")


def _blockquote(text: str) -> list[str]:
    """Quote third-party prose verbatim, one ``>`` line per source line."""
    out: list[str] = []
    for line in _defang(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.rstrip()
        out.append("> " + stripped if stripped else ">")
    while out and out[-1] == ">":
        out.pop()
    return out or [">"]


def _cell(value: Any) -> str:
    """Make a value safe to drop inside a markdown table cell."""
    if value is None:
        return "—"
    text = value if isinstance(value, str) else _scalar(value)
    text = _defang(text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace("|", "\\|")
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text or "—"


def _scalar(value: Any) -> str:
    """Render a config value compactly (dicts/lists become one-line JSON)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, sort_keys=True, default=str, separators=(", ", ": "))
        except (TypeError, ValueError):  # pragma: no cover - json is tolerant with default=str
            return str(value)
    return str(value)


def _yaml_safe(value: Any, _depth: int = 0) -> Any:
    """Coerce arbitrary connector payloads into YAML-representable primitives.

    ``Signal.extra`` is explicitly source-specific and never validated, so it can
    contain anything at all. Front matter must still be parseable, so unknown types
    degrade to ``str`` rather than raising a RepresenterError at dump time.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return _iso(value)
    if _depth >= 6:
        return _scalar(value)
    if isinstance(value, dict):
        return {str(k): _yaml_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else list(value)
        return [_yaml_safe(v, _depth + 1) for v in items]
    return _scalar(value)


def _dump_front_matter(data: dict[str, Any]) -> str:
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=10_000,  # never wrap: a wrapped URL is still valid YAML but reads badly
    )
    return text if text.endswith("\n") else text + "\n"


# ---------------------------------------------------------------------------
# Candidate accessors (defensive: a Candidate may be half-populated)
# ---------------------------------------------------------------------------


def _signals(cand: Candidate) -> list[Signal]:
    return list(cand.signals or [])


def _detected_at(cand: Candidate, detected_at: datetime | None) -> datetime:
    """Injected value wins; otherwise the newest signal's ``observed_at``.

    Falling back to the newest signal (rather than to ``now()``) is what makes a
    re-render of the same Candidate byte-identical on a later day. ``now()`` is used
    only for the degenerate case of a Candidate with no signals at all.
    """
    if detected_at is not None:
        return _as_utc(detected_at)
    stamps = [
        _as_utc(s.observed_at)
        for s in _signals(cand)
        if isinstance(s.observed_at, datetime)
    ]
    if stamps:
        return max(stamps)
    return datetime.now(timezone.utc)


def _orgs(cand: Candidate) -> list[str]:
    return sorted({s.org for s in _signals(cand) if s.org})


def _model_ids(cand: Candidate) -> list[str]:
    seen: dict[str, None] = {}
    for s in _signals(cand):
        for mid in s.model_ids or []:
            if mid:
                seen.setdefault(str(mid), None)
    return list(seen)


def _model_types(cand: Candidate) -> list[str]:
    return sorted({s.model_type for s in _signals(cand) if s.model_type})


def _all_urls(cand: Candidate) -> dict[str, str]:
    """Union of every Signal's urls.

    A link kind that only one source supplied keeps its bare name (``pr``); when two
    sources supply different urls for the same kind, *both* are qualified
    (``vllm.pr``, ``sglang.pr``) so no source silently wins the short key.
    """
    by_kind: dict[str, list[tuple[str, str]]] = {}
    for s in _signals(cand):
        for kind, url in (s.urls or {}).items():
            if not url:
                continue
            entries = by_kind.setdefault(str(kind), [])
            if (s.source, str(url)) not in entries:
                entries.append((s.source, str(url)))

    out: dict[str, str] = {}
    for kind, entries in by_kind.items():
        distinct = {url for _, url in entries}
        if len(distinct) == 1:
            out[kind] = entries[0][1]
            continue
        for source, url in entries:
            key = f"{source}.{kind}"
            n = 2
            while key in out and out[key] != url:
                key = f"{source}.{kind}.{n}"
                n += 1
            out[key] = url
    return dict(sorted(out.items()))


def _extras(cand: Candidate, key: str) -> list[tuple[Signal, Any]]:
    """Every ``signal.extra[key]`` that carries something, in signal order."""
    out: list[tuple[Signal, Any]] = []
    for s in _signals(cand):
        extra = s.extra or {}
        if not isinstance(extra, dict):
            continue
        value = extra.get(key)
        if value is None or (isinstance(value, (str, list, tuple, dict, set)) and not value):
            continue
        out.append((s, value))
    return out


def _perf_entries(cand: Candidate) -> list[tuple[Signal, Any]]:
    """``extra["perf"]``: documented as a list of dicts, each with ``hardware`` (possibly
    None), unit-bearing float metrics and free-text ``notes``. Older/odd shapes still
    render — see :func:`_perf_rows`."""
    return _extras(cand, "perf")


def _perf_note_entries(cand: Candidate) -> list[tuple[Signal, list[str]]]:
    """``extra["perf_notes"]``: verbatim third-party prose, one or more blocks."""
    out: list[tuple[Signal, list[str]]] = []
    for signal, value in _extras(cand, "perf_notes"):
        if isinstance(value, str):
            blocks = [value]
        elif isinstance(value, (list, tuple)):
            blocks = [v if isinstance(v, str) else _scalar(v) for v in value]
        else:
            blocks = [_scalar(value)]
        blocks = [b for b in (b.strip() for b in blocks) if b]
        if blocks:
            out.append((signal, blocks))
    return out


def _join_edges(cand: Candidate) -> list[str]:
    """``Candidate.join_edges``: the union-find keys that merged this candidate's signals.

    Read through ``getattr`` so a Candidate built before the field existed still renders.
    """
    return [str(e) for e in (getattr(cand, "join_edges", None) or [])]


def _silent_failures(cand: Candidate) -> list[str]:
    """``Candidate.silent_failures``: BLIS runs, warns at most, and is wrong.

    Read through ``getattr`` so a Candidate built before the field existed still
    renders instead of raising.
    """
    return [str(f) for f in (getattr(cand, "silent_failures", None) or [])]


def _silently_wrong(cand: Candidate) -> bool:
    """The dangerous combination: nothing fatal, so nothing announces the problem."""
    return bool(_silent_failures(cand)) and not cand.bucket0_failures


def _trigger_split(cand: Candidate) -> tuple[list[str], list[str]]:
    """(triggers, markers) — markers describe how the candidate was assembled."""
    triggers: list[str] = []
    markers: list[str] = []
    for code in cand.triggers or []:
        (markers if code in MARKER_DESCRIPTIONS else triggers).append(str(code))
    return triggers, markers


def _is_known_arch_drift(cand: Candidate) -> bool:
    """True when this is config drift in an architecture BLIS already knows."""
    triggers = set(cand.triggers or [])
    return KNOWN_ARCH_TRIGGER in triggers and not (triggers & NOVEL_ARCH_TRIGGERS)


def _bucket(cand: Candidate) -> int | None:
    """0 when the deterministic checks already prove it would not run; else unknown.

    Buckets 1-3 are a stage-2 judgement, so stage 1 must not guess one.
    """
    return 0 if cand.bucket0_failures else None


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def _front_matter(cand: Candidate, detected_at: datetime) -> str:
    perf: list[dict[str, Any]] = []
    for signal, payload in _perf_entries(cand):
        entry: dict[str, Any] = {"source": signal.source, "value": _yaml_safe(payload)}
        if signal.raw_ref:
            entry["raw_ref"] = str(signal.raw_ref)
        perf.append(entry)

    data: dict[str, Any] = {
        "schema": SCHEMA,
        "arch_id": cand.arch_id,
        "display_name": cand.display_name or cand.arch_id,
        "detected_at": _iso(detected_at),
        "sources": list(cand.sources),
        "triggers": list(cand.triggers or []),
        "markers": _trigger_split(cand)[1],
        # Audit trail for the union-find join. A wrong edge means two distinct
        # architectures were fused into one report; the backtest reads these to
        # detect over-merging.
        "join_edges": _join_edges(cand),
        "significance": list(cand.significance or []),
        # 0 = proven not to run by the deterministic validators.
        # null = undetermined; stage 2 assigns 1, 2 or 3.
        "bucket": _bucket(cand),
        "bucket0_failures": [str(f) for f in (cand.bucket0_failures or [])],
        # BLIS runs and is wrong. Counted separately from bucket0_failures on purpose.
        "silent_failures": _silent_failures(cand),
        # True = silent failures with a clean bucket 0: wrong numbers, nothing announced.
        "silently_wrong": _silently_wrong(cand),
        "known_arch_drift": _is_known_arch_drift(cand),
        "unparsed_fields": [str(f) for f in (cand.unparsed_fields or [])],
        "est_total_params": cand.est_total_params,
        "est_active_params": cand.est_active_params,
        "corroborated": cand.corroborated,
        "model_types": _model_types(cand),
        "model_ids": _model_ids(cand),
        "orgs": _orgs(cand),
        "urls": _all_urls(cand),
        "has_config": cand.config is not None,
        "has_perf_notes": bool(_perf_note_entries(cand)),
        "perf": perf,
        "stage2": "pending",
        "dry_run": True,
    }
    return _dump_front_matter(data)


# ---------------------------------------------------------------------------
# Body sections
# ---------------------------------------------------------------------------


def _verdict_line(cand: Candidate) -> str:
    """One-line deterministic verdict for the skim-reader."""
    silent = _silent_failures(cand)
    fatal = list(cand.bucket0_failures or [])
    if fatal and silent:
        return (
            f"- **Verdict:** BLIS would **not run** this config "
            f"({len(fatal)} fatal {_plural(len(fatal), 'failure')}), and it also carries "
            f"{len(silent)} **silent** {_plural(len(silent), 'failure')}"
        )
    if fatal:
        return (
            f"- **Verdict:** BLIS would **not run** this config — {len(fatal)} fatal "
            f"validator {_plural(len(fatal), 'failure')} (bucket 0)"
        )
    if silent:
        return (
            f"- **Verdict:** BLIS **runs this and reports wrong numbers** — "
            f"{len(silent)} silent {_plural(len(silent), 'failure')}, no error raised"
        )
    if cand.config is None:
        return "- **Verdict:** not checked — no `config.json` available yet"
    return (
        "- **Verdict:** BLIS would run this config; no fatal or silent validator failure "
        "found (stage 2 still judges fidelity)"
    )


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def _header(cand: Candidate) -> list[str]:
    name = cand.display_name or cand.arch_id
    title = f"# [archwatch] {cand.arch_id}"
    if name and name != cand.arch_id:
        title += f" ({name})"
    if _is_known_arch_drift(cand):
        title += " — known architecture, config grew fields BLIS does not parse"
    else:
        title += " — new architecture detected"

    sources = ", ".join(SOURCE_LABELS.get(s, s) for s in cand.sources) or "none recorded"
    lines = [
        title,
        "",
        "Stage-1 stub, generated by `archwatch` from deterministic checks only — no LLM ran, "
        "and nothing here was posted anywhere. Tracking only.",
        "",
        _verdict_line(cand),
        f"- **Architecture:** `{cand.arch_id}` (the pipeline's primary key)",
    ]
    if name and name != cand.arch_id:
        lines.append(f"- **Model family:** {name}")
    types = _model_types(cand)
    if types:
        lines.append(f"- **`model_type`:** {', '.join(f'`{t}`' for t in types)}")
    orgs = _orgs(cand)
    if orgs:
        lines.append(f"- **Org(s):** {', '.join(orgs)}")
    lines.append(
        f"- **Seen by:** {sources}"
        + (" — corroborated by 2+ sources" if cand.corroborated else " — single source")
    )
    models = _model_ids(cand)
    if models:
        shown = ", ".join(f"`{m}`" for m in models[:6])
        if len(models) > 6:
            shown += f", … (+{len(models) - 6} more)"
        lines.append(f"- **Model repos:** {shown}")
    lines.append("")
    return lines


def _sources_section(cand: Candidate) -> list[str]:
    lines = ["## Sources", ""]
    signals = _signals(cand)
    if not signals:
        lines += ["_No signals recorded on this candidate._", ""]
        return lines

    lines += [
        "| Source | Ref | Org | Evidence | Links |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in sorted(signals, key=lambda x: (x.source, x.raw_ref, x.display_name)):
        links = " ".join(
            f"[{_cell(kind)}]({url})" for kind, url in sorted((s.urls or {}).items()) if url
        )
        ref = s.raw_ref or (s.model_ids[0] if s.model_ids else "") or s.display_name
        lines.append(
            "| {src} | {ref} | {org} | {ev} | {links} |".format(
                src=_cell(SOURCE_LABELS.get(s.source, s.source)),
                ref=f"`{_cell(ref)}`" if ref else "—",
                org=_cell(s.org),
                ev=_cell(s.evidence),
                links=links or "—",
            )
        )
    lines.append("")
    return lines


def _join_block(cand: Candidate) -> list[str]:
    """Show *why* these signals were considered one model, so a reader can disagree.

    Rendered right under the sources table: that is where a false merge is visible —
    two rows that do not belong to the same model.
    """
    edges = _join_edges(cand)
    if not edges:
        return []
    multi = len(_signals(cand)) > 1
    lead = (
        "**How these signals were joined** — union-find over architecture id, normalized "
        "repo id and normalized family name. Every source row above was merged into this "
        "one report because of these edges:"
        if multi
        else "**Join key** — the union-find "
        f"{_plural(len(edges), 'edge')} this single signal was filed under:"
    )
    lines = [lead, ""]
    for edge in edges:
        prefix = edge.split(":", 1)[0] if ":" in edge else ""
        kind = JOIN_EDGE_KINDS.get(prefix)
        lines.append(f"- `{edge}`" + (f" — {kind}" if kind else ""))
    lines.append("")
    if multi:
        lines += [
            "> If any edge above is wrong, this stub has **fused two distinct "
            "architectures into one report** — worse than emitting two stubs, because the "
            "findings below then describe a model that does not exist. Check that every "
            "source row really is the same model before trusting the rest of this file.",
            "",
        ]
    return lines


def _why_section(cand: Candidate) -> list[str]:
    lines = ["## Why it fired", ""]
    triggers, markers = _trigger_split(cand)
    if KNOWN_ARCH_TRIGGER in triggers:
        lines += [
            f"> **This is not a new architecture.** `{KNOWN_ARCH_TRIGGER}` fired: "
            f"`{cand.arch_id}` is already in BLIS's known set. What changed is the "
            "*config* — it now carries fields BLIS does not parse. Read this as drift in "
            "a family we thought we supported, not as a new model class.",
            "",
        ]
    lines += ["**Triggers** (any one is enough):", ""]
    if triggers:
        for code in triggers:
            desc = TRIGGER_DESCRIPTIONS.get(code, "no description on file")
            lines.append(f"- **{code}** — {desc}")
    else:
        lines.append("- _none recorded_ (the candidate was emitted without a trigger code)")
    lines += ["", "**Significance** (at least one must hold):", ""]
    sig = list(cand.significance or [])
    if sig:
        for code in sig:
            desc = SIGNIFICANCE_DESCRIPTIONS.get(code, "no description on file")
            lines.append(f"- **{code}** — {desc}")
    else:
        lines.append("- _none recorded_ (the candidate was emitted without a significance code)")
    if markers:
        lines += ["", "**Markers** (how this candidate was assembled, not why it matters):", ""]
        for code in markers:
            desc = MARKER_DESCRIPTIONS.get(code, "no description on file")
            lines.append(f"- **`{code}`** — {desc}")
    lines.append("")
    return lines


def _silent_failures_section(cand: Candidate) -> list[str]:
    """The loudest section in the stub, and deliberately above the bucket-0 verdict.

    Fatal failures announce themselves; these do not. Rendered only when there is
    something to report, so it never cries wolf.
    """
    silent = _silent_failures(cand)
    if not silent:
        return []
    n = len(silent)
    lines = [
        "## Silently wrong today — BLIS runs this and reports confident nonsense",
        "",
        f"**{n} silent validator {_plural(n, 'failure')}.** BLIS does not crash, does not "
        "refuse the config and does not raise an error. It logs a warning at most, then "
        "reports numbers that are wrong. This is the failure class archwatch exists to "
        "catch: a wrong answer nobody is told about.",
        "",
    ]
    lines += [f"- {f}" for f in silent]
    lines.append("")
    if cand.bucket0_failures:
        lines += [
            "> This candidate **also** fails hard validators (bucket 0, below). The fatal "
            "failures will be noticed on their own; the silent ones above will not.",
            "",
        ]
    else:
        lines += [
            "> **The bucket-0 verdict below is clean, and that is exactly what makes this "
            "dangerous.** The config sails through validation, so nothing downstream ever "
            "signals a problem — BLIS will happily produce numbers, and they will be wrong. "
            "Treat any BLIS output for this architecture as unusable until stage 2 says "
            "otherwise.",
            "",
        ]
    return lines


def _findings_section(cand: Candidate) -> list[str]:
    lines = [
        "## Deterministic findings (stage 1)",
        "",
        "Computed by reading BLIS as text against the harvested support surface. No LLM, "
        "no simulation run — these are facts about the config, not a fidelity judgement.",
        "",
        "### Bucket 0 — would BLIS accept this config?",
        "",
    ]
    silent = _silent_failures(cand)
    if cand.bucket0_failures:
        n = len(cand.bucket0_failures)
        lines += [
            f"**No — bucket 0.** {n} fatal validator "
            f"{_plural(n, 'failure')}; BLIS would refuse this config before any simulation:",
            "",
        ]
        lines += [f"- {f}" for f in cand.bucket0_failures]
        lines.append("")
        if silent:
            lines += [
                f"Separately, {len(silent)} **silent** "
                f"{_plural(len(silent), 'failure')} — see the section above.",
                "",
            ]
    elif cand.config is None:
        lines += [
            "**Not checked** — no `config.json` was obtainable for this architecture, so the "
            "validators could not run. Bucket is left undetermined.",
            "",
        ]
    elif silent:
        lines += [
            "**Yes, BLIS would run it — and this is the dangerous case, not the safe one.** "
            f"No *fatal* validator failed, but {len(silent)} **silent** "
            f"{_plural(len(silent), 'failure')} were found above: BLIS accepts the config, "
            "runs, and reports wrong numbers with no error to show for it. A clean bucket 0 "
            "here means nothing will ever tell you.",
            "",
        ]
    else:
        lines += [
            "**Yes — no fatal validator failed, and no silent failure was found either.** "
            "BLIS would load this config and produce numbers. Whether those numbers are "
            "*right* is still stage 2's call: unknown fields are dropped silently, so a "
            "clean pass here is not a clean bill of health.",
            "",
        ]

    lines += ["### Config fields BLIS does not parse", ""]
    if cand.unparsed_fields:
        n = len(cand.unparsed_fields)
        lines += [
            f"{n} architecture-relevant "
            f"{'key' if n == 1 else 'keys'} present in the config that BLIS never reads. "
            "BLIS drops unknown keys without warning, so each one is a candidate for a "
            "silently wrong number rather than an error:",
            "",
        ]
        lines += [f"- `{f}`" for f in cand.unparsed_fields]
        lines.append("")
    elif cand.config is None:
        lines += ["_No config available, so no field diff could be computed._", ""]
    else:
        lines += [
            "None — every architecture-relevant key in this config maps to something BLIS "
            "parses (boilerplate keys are excluded by `IGNORED_CONFIG_KEYS`).",
            "",
        ]

    lines += ["### Parameter estimates", ""]
    if cand.est_total_params is None and cand.est_active_params is None:
        lines += [
            "_Not estimated — the config lacked the fields the estimator needs._",
            "",
        ]
    else:
        total_exact = "—" if cand.est_total_params is None else f"{cand.est_total_params:,d}"
        active_exact = "—" if cand.est_active_params is None else f"{cand.est_active_params:,d}"
        lines += [
            "| Estimate | Approx. | Exact |",
            "| --- | --- | --- |",
            f"| Total parameters | {_human_count(cand.est_total_params)} | {total_exact} |",
            f"| Active per token | {_human_count(cand.est_active_params)} | {active_exact} |",
            "",
            "Estimated with standard transformer arithmetic from the config; approximate, and "
            "not a substitute for the published count.",
            "",
        ]
    return lines


def _config_section(cand: Candidate) -> list[str]:
    config = cand.config
    if not config:
        return []
    rows = [(k, config[k]) for k in CONFIG_HIGHLIGHT_KEYS if k in config]
    if not rows:
        return []
    lines = [
        "## Config at a glance",
        "",
        f"Selected fields from the richest `config.json` seen ({len(config)} top-level keys "
        "in total).",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    lines += [f"| `{k}` | {_cell(v)} |" for k, v in rows]
    lines.append("")
    return lines


def _perf_section(cand: Candidate) -> list[str]:
    entries = _perf_entries(cand)
    if not entries:
        return []
    lines = [
        "## Reported performance numbers",
        "",
        "Carried in a signal's `extra[\"perf\"]`. Recorded verbatim — these are future "
        "validation ground truth for BLIS, not archwatch's own measurements.",
        "",
    ]
    for signal, payload in entries:
        label = SOURCE_LABELS.get(signal.source, signal.source)
        ref = f" (`{signal.raw_ref}`)" if signal.raw_ref else ""
        lines.append(f"**{label}**{ref}")
        lines.append("")
        lines += _perf_rows(payload)
        lines.append("")
    return lines


def _perf_rows(payload: Any) -> list[str]:
    """Render an arbitrary perf payload as a table when it is dict-shaped, else verbatim."""
    if isinstance(payload, dict):
        lines = ["| Metric | Value |", "| --- | --- |"]
        lines += [f"| `{_cell(k)}` | {_cell(v)} |" for k, v in payload.items()]
        return lines
    if isinstance(payload, (list, tuple)):
        items = list(payload)
        if items and all(isinstance(i, dict) for i in items):
            keys: list[str] = []
            for item in items:
                for k in item:
                    if k not in keys:
                        keys.append(str(k))
            # Documented shape: hardware identifies the row, notes is free text.
            order = {k: i for i, k in enumerate(keys)}
            keys.sort(key=lambda k: (k == "notes", k != "hardware", order[k]))
            lines = [
                "| " + " | ".join(f"`{_cell(k)}`" for k in keys) + " |",
                "| " + " | ".join("---" for _ in keys) + " |",
            ]
            lines += [
                "| " + " | ".join(_cell(item.get(k)) for k in keys) + " |" for item in items
            ]
            return lines
        return [f"- {_cell(i)}" for i in items]
    return [f"- {_cell(payload)}"]


def _perf_notes_section(cand: Candidate) -> list[str]:
    """Verbatim third-party prose. Often names the mechanism before any config is public,
    which is exactly what stage 2 needs — so it is quoted, attributed, and never
    paraphrased."""
    entries = _perf_note_entries(cand)
    if not entries:
        return []
    lines = [
        "## Third-party mechanism notes (verbatim)",
        "",
        "Quoted from the source's own changelog or write-up — **their words, unverified, "
        "not archwatch's analysis.** Notes like these routinely name a mechanism days "
        "before a config is public, so read them as the earliest available hint about what "
        "stage 2 will have to model.",
        "",
    ]
    for signal, blocks in entries:
        label = SOURCE_LABELS.get(signal.source, signal.source)
        ref = f" (`{signal.raw_ref}`)" if signal.raw_ref else ""
        lines.append(f"**{label}**{ref}:")
        lines.append("")
        for block in blocks:
            lines += _blockquote(block)
            lines.append("")
    return lines


def _stage2_section(cand: Candidate) -> list[str]:
    lines = [
        "## Stage 2 — deep dive",
        "",
        "**Status: not yet run.**",
        "",
        "Stage 2 is the `archwatch-deep-dive` skill: it names the *mechanism* this "
        "architecture introduces, assigns a bucket, and estimates how wrong BLIS is today.",
        "",
        "| Bucket | Meaning |",
        "| --- | --- |",
        "| 0 | would not run — already decided above, deterministically |",
        "| 1 | runs as-is; only fields BLIS already parses |",
        "| 2 | known-gap mechanism; cite the `known-gaps.yaml` entry |",
        "| 3 | new mechanism with no seam; name the functions needing a branch |",
        "",
    ]
    if cand.bucket0_failures:
        lines += [
            "Bucket 0 is already established for this candidate — stage 2 only needs to say "
            "what the failing fields mean and how much it matters.",
            "",
        ]
    lines += [
        "Everything above this marker is regenerated from the candidate on every run; the "
        "analysis is appended below it and is never overwritten.",
        "",
        STAGE2_MARKER,
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(cand: Candidate, *, detected_at: datetime | None = None) -> str:
    """Render ``cand`` to the full markdown stub.

    Pure: no filesystem, no network, no clock (unless the Candidate carries no
    signal timestamp at all — see :func:`_detected_at`). Pass ``detected_at`` to pin
    the only volatile value; it appears solely in the front matter.
    """
    stamp = _detected_at(cand, detected_at)
    parts: list[str] = ["---", _front_matter(cand, stamp).rstrip("\n"), "---", ""]
    parts += _header(cand)
    parts += _sources_section(cand)
    parts += _join_block(cand)
    parts += _why_section(cand)
    parts += _silent_failures_section(cand)
    parts += _findings_section(cand)
    parts += _config_section(cand)
    parts += _perf_section(cand)
    parts += _perf_notes_section(cand)
    parts += _stage2_section(cand)
    text = "\n".join(parts)
    return text if text.endswith("\n") else text + "\n"


def split_stub(text: str) -> tuple[str, str]:
    """Split rendered markdown into ``(stub_through_marker, stage2_appendix)``.

    When the marker is absent the whole text is treated as stub and the appendix is
    empty — that keeps a hand-edited file from being silently truncated.
    """
    idx = text.find(STAGE2_MARKER)
    if idx == -1:
        return text, ""
    cut = idx + len(STAGE2_MARKER)
    return text[:cut], text[cut:]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmitResult:
    """Outcome of one write. ``status`` is one of:

    ``created``
        The file did not exist and was written.
    ``unchanged``
        The stub on disk already matches byte for byte (any stage-2 appendix
        included in that comparison). Nothing was written — re-running is a no-op.
    ``updated``
        The stub changed and was rewritten; an existing stage-2 appendix was carried
        over verbatim. Only happens with ``overwrite=True``.
    ``skipped_existing``
        A different file is already there and ``overwrite`` was False, so it was left
        alone. This is also the stateless dedup path.
    """

    arch_id: str
    path: Path
    status: str
    markdown: str

    @property
    def written(self) -> bool:
        return self.status in ("created", "updated")


def write_issue(
    cand: Candidate,
    out_dir: Path | str | None = None,
    *,
    detected_at: datetime | None = None,
    overwrite: bool = False,
) -> EmitResult:
    """Render ``cand`` and write ``<out_dir>/<arch_id>.md``.

    Writing twice is a no-op: an identical file on disk is left untouched (same
    bytes, same mtime). A file whose stage-1 stub matches but which has a stage-2
    analysis appended also counts as identical — the appendix is not stub drift.

    Creates ``out_dir`` if needed. Never calls out to anything.
    """
    path = issue_path(cand.arch_id, out_dir)
    markdown = render(cand, detected_at=detected_at)

    existing: str | None = None
    if path.is_file():
        existing = path.read_text(encoding="utf-8")

    if existing is not None:
        new_stub, _ = split_stub(markdown)
        old_stub, old_tail = split_stub(existing)
        if existing == markdown or (old_stub == new_stub and old_tail.strip()):
            return EmitResult(cand.arch_id, path, "unchanged", existing)
        if not overwrite:
            return EmitResult(cand.arch_id, path, "skipped_existing", existing)
        merged = new_stub + old_tail if old_tail.strip() else markdown
        path.write_text(merged, encoding="utf-8")
        return EmitResult(cand.arch_id, path, "updated", merged)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return EmitResult(cand.arch_id, path, "created", markdown)


def write_issues(
    cands: Iterable[Candidate],
    out_dir: Path | str | None = None,
    *,
    detected_at: datetime | None = None,
    overwrite: bool = False,
) -> list[EmitResult]:
    """Write one stub per candidate, in the order given. Order is preserved so the
    detector's ranking survives into the run log."""
    return [
        write_issue(cand, out_dir, detected_at=detected_at, overwrite=overwrite)
        for cand in cands
    ]


def summarize(results: Sequence[EmitResult]) -> dict[str, int]:
    """Counts per status, for the run log."""
    out: dict[str, int] = {}
    for r in results:
        out[r.status] = out.get(r.status, 0) + 1
    return out
