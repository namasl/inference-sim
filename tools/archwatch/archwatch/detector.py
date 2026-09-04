"""The wiring: connectors -> join -> evaluate -> emit, plus the run log.

This module owns one run of the pipeline and nothing else. Every decision about
*what* is interesting lives in :mod:`archwatch.novelty`; every decision about how a
stub reads lives in :mod:`archwatch.emitter`. What is here is the plumbing, and the
plumbing has four properties worth stating outright because each one was learned from
a way this could fail quietly.

**1. The output directory is threaded into the dedup.**
``evaluate_detailed(..., issues_dir=out_dir)`` is not optional. The
``already_reported`` suppressor asks the emitter whether a stub is on disk, and the
emitter looks in whatever directory it is given — defaulting to ``issues/``. Run with
``--out /tmp/scan`` and forget to pass it through, and the dedup consults ``issues/``
while the writes go to ``/tmp/scan``: every architecture already reported in the real
directory is silently re-emitted, and every architecture in the scratch directory is
silently suppressed. No error either way. :func:`scan` resolves ``out_dir`` once and
passes the same value to both sides.

**2. Every poll is isolated.**
One source failing is the normal case, not the exceptional one: HuggingFace has
outages, GitHub returns 403 when the token is missing, and InferenceX restructures its
repo without warning. Each ``poll()`` runs in its own ``try``/``except`` and a failure
becomes a :class:`SourceReport` with ``ok=False``. The run continues, joins what it
has, evaluates it, writes the issues those sources justify, and reports itself as
``partial``. A scan that produced nothing because one connector raised would be the
worst outcome available: the sources that *did* work saw a real architecture and we
threw it away.

**3. Nothing is ever polled in a loop.**
Each connector is polled exactly once per run. GitHub's ``/search/issues`` allows 30
requests per minute against 5,000/hour for the core API, so the search-backed paths are
the binding constraint and a retry loop would burn the budget in seconds.
``cfg.max_github_requests`` is enforced as a *shared* budget across the framework and
InferenceX connectors by handing them one :class:`BudgetedSession`: when the budget is
spent the session returns a synthetic 403 without touching the network, which both
connectors already interpret as "rate limited, return partial results".

**4. There is no live mode.**
:class:`BudgetedSession` refuses every HTTP verb except GET, so the dry-run guarantee
holds for the session this module hands out even if a connector is later changed to
try a write. ``dry_run=False`` raises. The only thing this pipeline writes is markdown
under ``out_dir`` and JSON under ``.runlog/``.

The run log is the point of the whole module. It carries the aggregate counts a human
reads while tuning the filter *and* the unabridged per-candidate suppression records a
script reads, because "which suppressor ate the release we missed?" is not answerable
from counts.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from archwatch import emitter
from archwatch.config import DEFAULTS, DetectorConfig
from archwatch.connectors.base import Candidate, Signal
from archwatch.novelty import Suppression, evaluate_detailed, join_signals

log = logging.getLogger("archwatch.detector")

__all__ = [
    "RUNLOG_SCHEMA",
    "SOURCE_NAMES",
    "SOURCE_ALIASES",
    "GITHUB_SOURCES",
    "GithubBudget",
    "BudgetedSession",
    "SourceReport",
    "RunSummary",
    "default_runlog_dir",
    "parse_sources",
    "build_connectors",
    "poll_connectors",
    "scan",
]

#: Bumped when the run-log shape changes, so component J can adapt.
RUNLOG_SCHEMA = "archwatch-runlog/1"

#: Canonical source order. ``--sources`` output is always sorted into this order so
#: two runs of the same set produce comparable run logs.
SOURCE_NAMES: tuple[str, ...] = ("hf", "vllm", "sglang", "inferencex")

#: Shorthands accepted by ``--sources``.
SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "all": SOURCE_NAMES,
    "frameworks": ("vllm", "sglang"),
}

#: Sources that spend the shared GitHub REST budget.
GITHUB_SOURCES: frozenset[str] = frozenset({"vllm", "sglang", "inferencex"})

_SPLIT = re.compile(r"[,\s]+")


def default_runlog_dir() -> Path:
    """``tools/archwatch/.runlog`` — resolved from this file, not from the cwd.

    Mirrors :func:`archwatch.emitter.default_issues_dir` deliberately: a run started
    from a different working directory must not scatter run logs. The directory is
    gitignored.
    """
    return Path(__file__).resolve().parent.parent / ".runlog"


# ---------------------------------------------------------------------------
# The shared GitHub budget
# ---------------------------------------------------------------------------


@dataclass
class GithubBudget:
    """One request allowance, shared by every GitHub-backed connector in a run.

    ``cfg.max_github_requests`` is documented as a budget *across* the framework and
    InferenceX connectors, so it cannot live inside either of them: two connectors each
    obeying a 400-request cap spend 800. Counting in one place also makes the number
    observable, which is what lets the cap be calibrated instead of guessed.
    """

    limit: int
    used: int = 0
    refused: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def take(self) -> bool:
        """Claim one request. False when the budget is spent."""
        if self.used >= self.limit:
            self.refused += 1
            return False
        self.used += 1
        return True

    def as_dict(self) -> dict[str, int]:
        return {
            "budget": self.limit,
            "used": self.used,
            "refused": self.refused,
            "remaining": self.remaining,
        }


class _BudgetRefusal:
    """A synthetic 403, returned instead of making a request over budget.

    Both GitHub connectors treat 403/429 as "rate limited: log it and return what I
    have", which is exactly the degradation wanted here — so refusing in the shape of
    a rate limit needs no cooperation from them and cannot raise through code that was
    written to expect a response object.
    """

    status_code = 403
    reason = "archwatch budget exhausted"
    text = "archwatch: shared GitHub request budget exhausted; not sent"

    def __init__(self, budget: GithubBudget) -> None:
        self.headers = {
            "X-RateLimit-Remaining": "0",
            "X-Archwatch-Budget": f"exhausted after {budget.used} requests",
        }

    def json(self) -> dict[str, Any]:
        return {}

    def raise_for_status(self) -> None:  # pragma: no cover - defensive
        raise RuntimeError(self.text)


class BudgetedSession:
    """A ``requests.Session`` wrapper that is read-only and hard-capped.

    Two jobs, both structural rather than by convention:

    * **Cap.** Every GET claims one unit from the shared :class:`GithubBudget`. Past
      the cap nothing is sent; the caller gets a synthetic 403 and degrades to partial
      results on its own.
    * **Read-only.** POST/PATCH/PUT/DELETE raise. PLAN.md's rule 2 is that nothing is
      ever posted to GitHub; a wrapper that refuses the verb keeps that true even if a
      connector is later changed to try one, instead of relying on every future edit to
      remember.
    """

    _WRITE_VERBS = ("post", "put", "patch", "delete", "head", "options")

    def __init__(self, budget: GithubBudget, session: Any | None = None) -> None:
        self.budget = budget
        self._session = session
        self._warned = False

    @property
    def session(self) -> Any:
        if self._session is None:
            import requests  # local import keeps this module importable offline

            self._session = requests.Session()
        return self._session

    def get(self, url: str, **kwargs: Any) -> Any:
        if not self.budget.take():
            if not self._warned:
                log.warning(
                    "shared GitHub budget of %d requests exhausted; remaining calls "
                    "are refused locally and connectors will return partial results",
                    self.budget.limit,
                )
                self._warned = True
            log.debug("budget refusal: GET %s not sent", url)
            return _BudgetRefusal(self.budget)
        return self.session.get(url, **kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        if str(method).lower() != "get":
            raise RuntimeError(
                f"archwatch is read-only: refusing {str(method).upper()} {url}"
            )
        return self.get(url, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this class does not define. Write verbs are
        # refused loudly; everything else (close, headers, mount, ...) delegates.
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._WRITE_VERBS:
            def _refuse(*_a: Any, **_kw: Any) -> Any:
                raise RuntimeError(
                    f"archwatch is read-only: refusing {name.upper()} request"
                )

            return _refuse
        return getattr(self.session, name)


# ---------------------------------------------------------------------------
# Per-run reporting
# ---------------------------------------------------------------------------


@dataclass
class SourceReport:
    """What one connector did — including the case where it blew up.

    ``ok=False`` with a populated ``error`` is a first-class outcome, not an
    exception path: it is how a partial scan records itself.
    """

    name: str
    ok: bool = True
    signals: int = 0
    duration_s: float = 0.0
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "signals": self.signals,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "notes": list(self.notes),
        }


@dataclass
class RunSummary:
    """Everything one scan did, in the shape the run log and the CLI both need.

    Holds the live objects (``passed`` candidates, ``suppressions``, ``emit_results``)
    rather than pre-rendered text, so a caller — component J's backtest especially —
    can ask questions the CLI does not print. :meth:`as_dict` is the serializable
    projection; :meth:`text` is the human one.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime
    window_days: int
    since: datetime
    out_dir: Path
    sources_requested: list[str] = field(default_factory=list)
    source_reports: list[SourceReport] = field(default_factory=list)
    signals: int = 0
    signals_by_source: dict[str, int] = field(default_factory=dict)
    candidates: int = 0
    passed: list[Candidate] = field(default_factory=list)
    suppressions: list[Suppression] = field(default_factory=list)
    emit_results: list[Any] = field(default_factory=list)
    github: GithubBudget | None = None
    dry_run: bool = True
    cfg: DetectorConfig = field(default_factory=lambda: DEFAULTS)
    runlog_path: Path | None = None

    # -- derived views -----------------------------------------------------

    @property
    def duration_s(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def failed_sources(self) -> list[str]:
        return [r.name for r in self.source_reports if not r.ok]

    @property
    def ok_sources(self) -> list[str]:
        return [r.name for r in self.source_reports if r.ok]

    @property
    def partial(self) -> bool:
        """True when at least one selected source failed."""
        return bool(self.failed_sources)

    @property
    def total_failure(self) -> bool:
        """True when *every* selected source failed: nothing was scanned at all."""
        return bool(self.source_reports) and not self.ok_sources

    @property
    def suppressed(self) -> int:
        return len(self.suppressions)

    @property
    def suppressed_by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.suppressions:
            out[s.reason] = out.get(s.reason, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def suppressed_by_stage(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.suppressions:
            out[s.stage] = out.get(s.stage, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def written_by_status(self) -> dict[str, int]:
        return emitter.summarize(self.emit_results)

    @property
    def written(self) -> int:
        return sum(1 for r in self.emit_results if getattr(r, "written", False))

    @property
    def triggers_by_code(self) -> dict[str, int]:
        """Trigger codes among the candidates that passed.

        Counts whatever the filter recorded rather than a fixed list, so a new
        trigger code shows up in the run log the day it starts firing.
        """
        out: dict[str, int] = {}
        for cand in self.passed:
            for code in cand.triggers or []:
                out[code] = out.get(code, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def significance_by_code(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for cand in self.passed:
            for code in cand.significance or []:
                out[code] = out.get(code, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def issue_rows(self) -> list[dict[str, Any]]:
        """One row per emitted stub, in the filter's ranking order."""
        status = {r.arch_id: r for r in self.emit_results}
        rows: list[dict[str, Any]] = []
        for rank, cand in enumerate(self.passed, start=1):
            result = status.get(cand.arch_id)
            rows.append(
                {
                    "rank": rank,
                    "arch_id": cand.arch_id,
                    "display_name": cand.display_name,
                    "sources": list(cand.sources),
                    "triggers": list(cand.triggers or []),
                    "significance": list(cand.significance or []),
                    "bucket": 0 if cand.would_not_run else None,
                    "bucket0_failures": list(cand.bucket0_failures or []),
                    "silent_failures": list(getattr(cand, "silent_failures", []) or []),
                    "unparsed_fields": list(cand.unparsed_fields or []),
                    "est_total_params": cand.est_total_params,
                    "est_active_params": cand.est_active_params,
                    "model_ids": sorted(
                        {m for s in cand.signals for m in (s.model_ids or [])}
                    ),
                    "status": getattr(result, "status", "not_written"),
                    "path": str(getattr(result, "path", "")),
                }
            )
        return rows

    # -- serialization -----------------------------------------------------

    def counts(self) -> dict[str, Any]:
        """The aggregate tally, which is what a human tuning the filter reads."""
        return {
            "signals": self.signals,
            "signals_by_source": dict(self.signals_by_source),
            "candidates": self.candidates,
            "passed": len(self.passed),
            "suppressed": self.suppressed,
            "suppressed_by_reason": self.suppressed_by_reason,
            "suppressed_by_stage": self.suppressed_by_stage,
            "written": self.written_by_status,
            "triggers": self.triggers_by_code,
            "significance": self.significance_by_code,
        }

    def as_dict(self) -> dict[str, Any]:
        """The full run log: aggregates for a human, unabridged detail for a script."""
        return {
            "schema": RUNLOG_SCHEMA,
            "run_id": self.run_id,
            "dry_run": self.dry_run,
            "partial": self.partial,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "duration_s": round(self.duration_s, 3),
            "window": {
                "days": self.window_days,
                "since": _iso(self.since),
                "until": _iso(self.finished_at),
            },
            "out_dir": str(self.out_dir),
            "config": _config_dict(self.cfg),
            "sources": {
                "requested": list(self.sources_requested),
                "ok": self.ok_sources,
                "failed": self.failed_sources,
                "reports": [r.as_dict() for r in self.source_reports],
            },
            "github": self.github.as_dict() if self.github else None,
            "counts": self.counts(),
            "issues": self.issue_rows(),
            # Unabridged on purpose. A real HuggingFace day drops thousands of
            # candidates, and the one question worth asking of a scan that missed a
            # release is "which suppressor ate it?" — unanswerable from counts.
            "suppressions": [
                {
                    "arch_id": s.arch_id,
                    "display_name": s.display_name,
                    "stage": s.stage,
                    "reason": s.reason,
                    "detail": s.detail,
                }
                for s in self.suppressions
            ],
        }

    def write_run_log(self, dir: Path | str | None = None) -> Path:  # noqa: A002
        """Write ``<dir>/<run_id>.json`` and remember the path.

        ``.runlog/`` is gitignored: this is a working artifact for calibration, not
        something to commit. A same-second collision gets a ``-2`` suffix rather than
        overwriting a previous run.
        """
        base = Path(dir) if dir is not None else default_runlog_dir()
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{self.run_id}.json"
        n = 2
        while path.exists():
            path = base / f"{self.run_id}-{n}.json"
            n += 1
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=False, default=str) + "\n",
            encoding="utf-8",
        )
        self.runlog_path = path
        return path

    # -- human output ------------------------------------------------------

    def text(self) -> str:
        """The scan summary a human reads at the end of a run."""
        lines: list[str] = []
        mode = "dry-run" if self.dry_run else "LIVE (impossible)"
        lines.append(f"archwatch scan {self.run_id} ({mode})")
        lines.append(
            f"  window     : {self.window_days}d, since {_iso(self.since)} "
            f"(took {self.duration_s:.1f}s)"
        )
        lines.append("  sources    :")
        for r in self.source_reports:
            state = "ok    " if r.ok else "FAILED"
            line = (
                f"    {r.name:<11} {state} {r.signals:>6} signals  {r.duration_s:>6.1f}s"
            )
            if r.error:
                line += f"  {r.error}"
            lines.append(line)
            for note in r.notes:
                lines.append(f"      note: {note}")
        by_source = ", ".join(
            f"{k}={v}" for k, v in sorted(self.signals_by_source.items())
        )
        lines.append(f"  signals    : {self.signals}" + (f" ({by_source})" if by_source else ""))
        lines.append(f"  candidates : {self.candidates} joined")
        drops = ", ".join(f"{k}={v}" for k, v in self.suppressed_by_reason.items())
        lines.append(f"  suppressed : {self.suppressed}" + (f" ({drops})" if drops else ""))
        lines.append(f"  passed     : {len(self.passed)}")
        wrote = ", ".join(f"{k}={v}" for k, v in sorted(self.written_by_status.items()))
        lines.append(f"  issues     : {self.out_dir}" + (f" ({wrote})" if wrote else " (none)"))
        for row in self.issue_rows():
            trig = ",".join(row["triggers"]) or "-"
            sig = ",".join(row["significance"]) or "-"
            bucket = "0" if row["bucket"] == 0 else "?"
            lines.append(
                f"    {row['rank']}. {row['arch_id']}  [{trig}] [{sig}] "
                f"bucket={bucket}  {row['status']}"
            )
        if self.github:
            g = self.github
            lines.append(
                f"  github     : {g.used}/{g.limit} requests used"
                + (f", {g.refused} refused over budget" if g.refused else "")
            )
        if self.runlog_path:
            lines.append(f"  run log    : {self.runlog_path}")
        if self.total_failure:
            lines.append(
                "  NOTHING SCANNED: every selected source failed — "
                f"{', '.join(self.failed_sources)}"
            )
        elif self.partial:
            lines.append(
                "  PARTIAL SCAN: "
                f"{', '.join(self.failed_sources)} failed; the next overlapping "
                "window recovers what was missed"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    """UTC ISO-8601 with a ``Z``. Naive input is assumed UTC, never local."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id(started_at: datetime) -> str:
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    return started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _config_dict(cfg: DetectorConfig) -> dict[str, Any]:
    """The knobs that shaped this run, so a run log explains its own numbers.

    Thresholds are placeholders being calibrated, so a run log that does not record
    which ones were in force is not comparable with the next one.
    """
    t = cfg.thresholds
    return {
        "window_days": cfg.window_days,
        "max_issues_per_run": cfg.max_issues_per_run,
        "max_hf_config_fetches": cfg.max_hf_config_fetches,
        "max_github_requests": cfg.max_github_requests,
        "recheck_known_architectures": cfg.recheck_known_architectures,
        "frontier_orgs": len(cfg.frontier_orgs),
        "thresholds": {
            "min_total_params": t.min_total_params,
            "min_org_top_downloads": t.min_org_top_downloads,
            "min_model_downloads": t.min_model_downloads,
            "min_model_likes": t.min_model_likes,
        },
    }


def parse_sources(spec: str | Iterable[str] | None) -> list[str]:
    """``"hf,vllm"`` / ``["hf", "vllm"]`` / ``None`` -> canonical source list.

    Unknown names raise :class:`ValueError` rather than being ignored: a typo in
    ``--sources inferncex`` that silently scanned nothing would look exactly like a
    quiet day, and the whole point of this tool is not to miss things.
    """
    if spec is None:
        return list(SOURCE_NAMES)
    raw: list[str] = []
    chunks = [spec] if isinstance(spec, str) else [str(c) for c in spec]
    for chunk in chunks:
        raw.extend(p for p in _SPLIT.split(chunk.strip()) if p)
    if not raw:
        raise ValueError(
            "no sources selected; choose from " + ", ".join(SOURCE_NAMES) + " or 'all'"
        )
    wanted: set[str] = set()
    for name in raw:
        key = name.strip().lower()
        if key in SOURCE_ALIASES:
            wanted.update(SOURCE_ALIASES[key])
        elif key in SOURCE_NAMES:
            wanted.add(key)
        else:
            raise ValueError(
                f"unknown source {name!r}; choose from "
                + ", ".join(SOURCE_NAMES)
                + " or "
                + ", ".join(sorted(SOURCE_ALIASES))
            )
    return [s for s in SOURCE_NAMES if s in wanted]


def build_connectors(
    sources: Sequence[str],
    cfg: DetectorConfig | None = None,
    *,
    github_session: Any | None = None,
    budget: GithubBudget | None = None,
    hf_kwargs: dict[str, Any] | None = None,
) -> tuple[list[Any], GithubBudget | None]:
    """Instantiate one connector per selected source.

    Returns ``(connectors, budget)``. The budget is ``None`` when no GitHub-backed
    source was selected, and otherwise is shared by every one of them — that sharing
    is the whole point (see :class:`GithubBudget`).

    Imports are local so that ``--sources hf`` does not import the GitHub connectors
    and a broken connector module cannot stop an unrelated scan.
    """
    cfg = cfg or DEFAULTS
    conns: list[Any] = []
    needs_github = bool(set(sources) & GITHUB_SOURCES)
    shared_budget = budget
    session: Any | None = None
    if needs_github:
        if shared_budget is None:
            shared_budget = GithubBudget(limit=max(0, cfg.max_github_requests))
        session = BudgetedSession(shared_budget, github_session)

    for name in sources:
        if name == "hf":
            from archwatch.connectors.hf import HFConnector

            conns.append(HFConnector(cfg, **(hf_kwargs or {})))
        elif name == "vllm":
            from archwatch.connectors.frameworks import VllmConnector

            conns.append(VllmConnector(session=session))
        elif name == "sglang":
            from archwatch.connectors.frameworks import SglangConnector

            conns.append(SglangConnector(session=session))
        elif name == "inferencex":
            from archwatch.connectors.inferencex import InferenceXConnector

            conns.append(InferenceXConnector(session=session))
        else:  # pragma: no cover - parse_sources already rejected it
            raise ValueError(f"unknown source {name!r}")
    return conns, (shared_budget if needs_github else None)


def poll_connectors(
    connectors: Sequence[Any],
    since: datetime,
    *,
    trending: bool = True,
) -> tuple[list[Signal], list[SourceReport]]:
    """Poll each connector exactly once, isolating failures.

    Each connector gets its own ``try``/``except`` around ``poll()`` and another
    around the optional ``poll_trending()`` sweep, because a trending sweep that
    fails must not discard the window poll that already succeeded. Nothing here
    retries: see the module docstring on the 30 req/min search limit.
    """
    signals: list[Signal] = []
    reports: list[SourceReport] = []

    for conn in connectors:
        name = str(getattr(conn, "name", conn.__class__.__name__))
        report = SourceReport(name=name)
        started = time.monotonic()
        try:
            got = list(conn.poll(since) or [])
            signals.extend(got)
            report.signals += len(got)
        except Exception as exc:  # a source failing is normal; the scan continues
            report.ok = False
            report.error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "source %s failed: %s — degrading to a partial scan",
                name,
                report.error,
                exc_info=log.isEnabledFor(logging.DEBUG),
            )
        else:
            log.info("source %s: %d signals", name, report.signals)

        # The trending sweep feeds S4 ("seen via the trending sweep") and is
        # deliberately window-independent: a model published last month that is
        # trending today is exactly the late-blooming case S4 exists for.
        sweep = getattr(conn, "poll_trending", None)
        if trending and callable(sweep):
            try:
                got = list(sweep() or [])
                signals.extend(got)
                report.signals += len(got)
                report.notes.append(f"trending sweep added {len(got)} signals")
            except Exception as exc:
                report.notes.append(f"trending sweep failed: {type(exc).__name__}: {exc}")
                log.warning("source %s trending sweep failed: %r", name, exc)

        report.duration_s = time.monotonic() - started
        reports.append(report)

    return signals, reports


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def scan(
    cfg: DetectorConfig | None = None,
    *,
    sources: str | Iterable[str] | None = None,
    window_days: int | None = None,
    since: datetime | None = None,
    now: datetime | None = None,
    out_dir: Path | str | None = None,
    surface: Any | None = None,
    surface_dir: Path | str | None = None,
    connectors: Sequence[Any] | None = None,
    github_session: Any | None = None,
    hf_kwargs: dict[str, Any] | None = None,
    trending: bool = True,
    overwrite: bool = False,
    dry_run: bool = True,
) -> RunSummary:
    """Run the whole pipeline once and return the summary.

    Steps, in order: build connectors, poll each (isolated), join signals onto
    architectures, evaluate against the support surface, write a stub per survivor.

    ``connectors`` overrides construction entirely — that is how the tests inject
    fakes, including one that raises, with no network anywhere.

    ``dry_run`` must be True. It is a parameter rather than an assumption so that the
    CLI flag has something to pass and so a caller reading this signature sees there
    is no other mode.
    """
    if not dry_run:
        raise ValueError(
            "archwatch has no live mode: it never posts to GitHub. The only output is "
            "markdown under out_dir (PLAN.md rule 2)."
        )

    cfg = cfg or DEFAULTS
    if window_days is not None:
        # Copy rather than mutate: a caller's config object is theirs.
        cfg = _with_window(cfg, window_days)
    days = cfg.window_days

    started_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    window_since = (
        since.astimezone(timezone.utc)
        if since is not None
        else started_at - timedelta(days=days)
    )

    # Resolved exactly once. Both the dedup suppressor and the writer must see the
    # same directory or dedup silently breaks under --out (see module docstring).
    out = Path(out_dir) if out_dir is not None else emitter.default_issues_dir()

    if connectors is None:
        wanted = parse_sources(sources)
        conns, budget = build_connectors(
            wanted, cfg, github_session=github_session, hf_kwargs=hf_kwargs
        )
    else:
        conns = list(connectors)
        budget = None
        wanted = [str(getattr(c, "name", c.__class__.__name__)) for c in conns]

    log.info(
        "scan: sources=%s window=%dd since=%s out=%s",
        ",".join(wanted),
        days,
        _iso(window_since),
        out,
    )

    raw_signals, reports = poll_connectors(conns, window_since, trending=trending)

    by_source: dict[str, int] = {}
    for sig in raw_signals:
        key = str(sig.source or "?")
        by_source[key] = by_source.get(key, 0) + 1

    cands = join_signals(raw_signals)

    if surface is None:
        from archwatch.surface import load_surface

        surface = (
            load_surface(surface_dir) if surface_dir is not None else load_surface()
        )

    # issues_dir=out is load-bearing: without it the dedup consults the default
    # issues/ directory no matter where --out points.
    report = evaluate_detailed(cands, surface, cfg, issues_dir=out)

    results = emitter.write_issues(report.passed, out, overwrite=overwrite)

    finished_at = datetime.now(timezone.utc) if now is None else started_at
    summary = RunSummary(
        run_id=_run_id(started_at),
        started_at=started_at,
        finished_at=finished_at,
        window_days=days,
        since=window_since,
        out_dir=out,
        sources_requested=list(wanted),
        source_reports=reports,
        signals=len(raw_signals),
        signals_by_source=dict(sorted(by_source.items())),
        candidates=len(cands),
        passed=list(report.passed),
        suppressions=list(report.dropped),
        emit_results=list(results),
        github=budget,
        dry_run=True,
        cfg=cfg,
    )
    log.info(
        "scan done: %d signals -> %d candidates -> %d passed -> %d written",
        summary.signals,
        summary.candidates,
        len(summary.passed),
        summary.written,
    )
    return summary


def _with_window(cfg: DetectorConfig, window_days: int) -> DetectorConfig:
    """A copy of ``cfg`` with a different window. Never mutates the caller's config."""
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}")
    return replace(cfg, window_days=window_days, frontier_orgs=set(cfg.frontier_orgs))
