#!/usr/bin/env python3
"""archwatch validation harness — component J.

**This is a runnable script, not a pytest test.** It makes live, read-only calls to
the HuggingFace Hub and the GitHub REST API, so it must never be collected by
pytest (PLAN.md hard rule 3: no network in tests). The filename has no ``test_``
prefix for exactly that reason. The harness's own pure logic is unit-tested in
``tests/test_backtest.py``.

Everything it writes goes to ``--out`` (a temp directory by default). It never
writes to the repo's ``issues/`` — PLAN.md hard rule 2, and also a measurement
requirement: a stub already on disk fires the ``already_reported`` suppressor and
would silently zero the recall number the harness is trying to measure.

Steps
-----
``recall``     (A) Per named frontier release: fetch its real ``config.json`` and Hub
               metadata by repo id, build a Signal through the HF connector's own
               ``_to_signals`` (so the real pre-filter, config fetch and org sweep all
               run), then join + evaluate against the real support surface.
``join``       (E) Cross-source join + false-merge audit: the recall targets' HF
               signals joined against a live poll of the curated sources.
``replay``     (C) Historical replay of the GitHub-backed sources over a fixed
               ``[since, until]`` window that contains a known model-support PR.
``sweep``      (D) Threshold calibration: recall against the target list and survivor
               volume at each ``min_total_params`` and each
               ``recheck_known_architectures`` setting.
``precision``  (B) Real ``detector.scan()`` per source over a recent window; dumps
               every survivor for hand categorization.
``all``        recall, join, replay, sweep, precision — in that order. Precision runs
               last on purpose (see PROGRESS/VALIDATION: two components were being
               fixed concurrently for a non-LM precision bug).

Why the recall test does not replay a historical HuggingFace window
------------------------------------------------------------------
PLAN.md section J says "replay a historical window". For HuggingFace that is
infeasible: the Hub exposes **no server-side date filter**, so
``list_models(sort="created_at")`` must be walked from the newest repo backwards.
Reaching a 30-day-old window means paging past ~100,000 repos (a live day is ~2,600
after the derivative pre-filter). The recall question — "would the filter have
flagged this release?" — does not need the listing at all: it needs the release's
real config and metadata, which cost one request each by repo id. So the harness
fetches those directly and runs the identical downstream pipeline. The listing walk
is exercised instead by the ``precision`` step, where it is cheap because the window
is recent.

The zero-day counterfactual
---------------------------
``support-surface/known-architectures.yaml`` was seeded from vLLM's registry on
2026-09-04. vLLM already supports every frontier architecture on the target list, so
**every target is in the seed set** and the ``known_architecture`` suppressor drops
all of them. That is correct behaviour and a real measurement, but it measures a
counterfactual nobody cares about: "would archwatch flag a release that vLLM shipped
support for months ago?" — no, by design.

So each target is evaluated in three arms:

``as_shipped``  the config as it ships today (``recheck_known_architectures=False``).
``recheck``     ``recheck_known_architectures=True`` — known architectures survive to
                a T1-only re-examination.
``zero_day``    the target's own architecture name removed from the seed set,
                simulating the day before vLLM added support. This is the arm that
                answers PLAN.md's actual question.

The counterfactual is explicit and narrow: only the exact lowercased architecture
strings the target publishes are removed (``Surface.is_known_architecture`` is exact
membership, so nothing else changes), and the removal is reported per target.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:  # runnable straight from a checkout
    sys.path.insert(0, str(ROOT))

from archwatch import detector, emitter  # noqa: E402
from archwatch.config import DEFAULTS, DetectorConfig, Thresholds  # noqa: E402
from archwatch.connectors.base import Candidate, Signal  # noqa: E402
from archwatch.novelty import (  # noqa: E402
    ALIAS_JOIN_MARKER,
    TRIGGER_IDS,
    EvaluationReport,
    evaluate_detailed,
    join_signals,
    normalize_arch_key,
    normalize_family_key,
    normalize_repo_key,
)
from archwatch.sizing import estimate_params, format_params  # noqa: E402
from archwatch.surface import Surface, load_surface  # noqa: E402

log = logging.getLogger("archwatch.backtest")


# ===========================================================================
# Targets
# ===========================================================================


@dataclass(frozen=True)
class Target:
    """One named release the filter is required to notice.

    ``repo_ids`` is a tuple because a release ships as several repos (DeepSeek V4 as
    Pro and Flash, Qwen3.5 across five sizes) and the recall answer can differ between
    them — only the biggest carries an MoE config, only some are multimodal. Every id
    is measured and reported; the target counts as hit when **any** of its repos does.
    """

    label: str
    repo_ids: tuple[str, ...]
    #: "frontier" — must be flagged. "seeded" — an older architecture BLIS already
    #: supports, where suppression as *known* is the correct answer, not a miss.
    kind: str = "frontier"
    note: str = ""


#: Resolved live against the Hub on 2026-09-04 by searching each family name and
#: taking the lab's own (non-quantized, non-community) repos. Recorded verbatim so a
#: re-run measures the same artifacts.
TARGETS: tuple[Target, ...] = (
    Target("Kimi K2", ("moonshotai/Kimi-K2-Instruct",),
           note="ships architectures=[DeepseekV3ForCausalLM] — it reuses DeepSeek V3's class"),
    Target("Kimi K3", ("moonshotai/Kimi-K3",)),
    Target("DeepSeek V3", ("deepseek-ai/DeepSeek-V3",)),
    Target("DeepSeek V4", ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash")),
    Target("GLM-5", ("zai-org/GLM-5",)),
    Target("GLM-5.2", ("zai-org/GLM-5.2",)),
    Target("GLM-5.3", ("zai-org/GLM-5.3",), note="newest zai-org release; not named in PLAN.md"),
    Target("MiniMax M3", ("MiniMaxAI/MiniMax-M3",)),
    Target("Qwen3.5", ("Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.5-122B-A10B", "Qwen/Qwen3.5-9B")),
    Target("Qwen3-14B", ("Qwen/Qwen3-14B",), kind="seeded",
           note="BLIS-validated dense architecture"),
    Target("Llama-3.1-70B", ("meta-llama/Llama-3.1-70B",), kind="seeded",
           note="BLIS-validated dense architecture"),
    Target("Mixtral-8x7B", ("mistralai/Mixtral-8x7B-v0.1",), kind="seeded",
           note="BLIS-validated MoE architecture"),
)

#: ``min_total_params`` values the sweep walks. Spans "any LM at all" to
#: "frontier-scale only", so the knee is inside the range rather than at an endpoint.
SWEEP_PARAMS: tuple[int, ...] = (
    1_000_000_000, 3_000_000_000, 7_000_000_000, 15_000_000_000,
    30_000_000_000, 70_000_000_000, 150_000_000_000, 400_000_000_000,
)

#: A large cap for measurement runs. The per-run cap is a *reporting* budget, not part
#: of the filter, so measuring recall or survivor volume through it would conflate the
#: two. Every step reports the uncapped survivor list and, separately, what the
#: shipped cap of 5 would have shown.
MEASURE_CAP = 500


# ===========================================================================
# Pure helpers (unit-tested in tests/test_backtest.py)
# ===========================================================================

_FM_FENCE = "---"

#: Front-matter keys component J depends on. Addendum 22: **test for keys, not for the
#: schema version string** — the schema grows additively, so pinning ``archwatch/2``
#: would turn every future field addition into a J failure. (It has already grown:
#: the emitter ships ``archwatch/3``.)
REQUIRED_FM_KEYS: tuple[str, ...] = (
    "arch_id", "sources", "triggers", "significance", "bucket",
    "bucket0_failures", "silent_failures", "silently_wrong", "join_edges",
    "unparsed_fields", "est_total_params", "detected_at", "stage2", "dry_run",
)


def front_matter(text: str) -> dict[str, Any]:
    """Parse a stub's YAML front matter. ``{}`` when there is none.

    Deliberately does not care what ``schema:`` says.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FM_FENCE:
        return {}
    for idx in range(1, len(lines)):
        if lines[idx].strip() == _FM_FENCE:
            import yaml

            parsed = yaml.safe_load("\n".join(lines[1:idx]))
            return parsed if isinstance(parsed, dict) else {}
    return {}


def missing_front_matter_keys(fm: dict[str, Any]) -> list[str]:
    """Which of :data:`REQUIRED_FM_KEYS` this front matter does not carry."""
    return [k for k in REQUIRED_FM_KEYS if k not in fm]


# ``a?`` catches the active-parameter suffix: ``Qwen3.5-397B-A17B`` must yield BOTH
# ``397b`` and ``17b``, since a repack that kept one and dropped the other is a
# different artifact. The lookbehind sits before the optional ``a`` so ``beta5b`` (an
# ``a`` inside a word) cannot match.
_SIZE_TOKEN = re.compile(
    r"(?<![a-z0-9])a?(\d+(?:\.\d+)?x\d+(?:\.\d+)?b|\d+(?:\.\d+)?b)(?![a-z0-9])"
)


def size_tokens(text: str) -> set[str]:
    """Parameter-count tokens in a name: ``{"8x7b"}``, ``{"397b", "17b"}``.

    Two model ids carrying *different* size tokens are different models, so a join that
    merged them is the false merge this audit exists to find. ``A17B``-style active-param
    suffixes count too: ``Qwen3.5-397B-A17B`` yields both.
    """
    return set(_SIZE_TOKEN.findall(text.lower()))


@dataclass
class MergeAudit:
    """One joined Candidate's audit: what was fused, on which edge, and is it wrong?"""

    arch_id: str
    n_signals: int
    sources: list[str]
    join_edges: list[str]
    arch_keys: list[str]
    repo_keys: list[str]
    family_keys: list[str]
    orgs: list[str]
    size_token_sets: list[list[str]]
    reasons: list[str] = field(default_factory=list)

    @property
    def merged(self) -> bool:
        return bool(self.join_edges)

    @property
    def family_only(self) -> bool:
        """Every merge edge is a ``family:`` edge — the weakest and most error-prone."""
        return self.merged and all(e.startswith("family:") for e in self.join_edges)

    @property
    def suspicious(self) -> bool:
        return bool(self.reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "n_signals": self.n_signals,
            "sources": self.sources,
            "join_edges": self.join_edges,
            "arch_keys": self.arch_keys,
            "repo_keys": self.repo_keys,
            "family_keys": self.family_keys,
            "orgs": self.orgs,
            "size_token_sets": self.size_token_sets,
            "family_only": self.family_only,
            "suspicious": self.suspicious,
            "reasons": self.reasons,
        }


def audit_merge(cand: Candidate) -> MergeAudit:
    """Flag a joined Candidate whose merge looks wrong.

    Pure and deliberately over-eager: it raises a hand for anything a human should look
    at, because a false merge silently fuses two architectures into one report and no
    other check would catch it. The three heuristics, weakest evidence last:

    1. **Two distinct architecture spellings fused.** More than one normalized
       ``arch_id`` key in the group. Sometimes legitimate (a release shipping both a
       causal and a conditional-generation head, or an MTP twin), always worth a look.
    2. **Different size tokens fused.** ``-9b`` and ``-397b`` are different models.
       ``normalize_repo_key`` deliberately strips no size token, so this should be
       impossible via a ``repo:`` edge and can only arrive on a ``family:`` edge.
    3. **Different orgs fused on a family edge alone.** Two labs' identically named
       models are a real risk (``*/Falcon-H1``); the family edge keeps an org-qualified
       display name org-qualified, but a bare display name can meet another bare one.
    """
    arch_keys: list[str] = []
    repo_keys: list[str] = []
    family_keys: list[str] = []
    orgs: list[str] = []
    size_sets: list[set[str]] = []
    for sig in cand.signals:
        for arch in sig.arch_ids:
            key = normalize_arch_key(arch)
            if key and key not in arch_keys:
                arch_keys.append(key)
        for mid in sig.model_ids:
            key = normalize_repo_key(mid)
            if key and key not in repo_keys:
                repo_keys.append(key)
            toks = size_tokens(mid)
            if toks and toks not in size_sets:
                size_sets.append(toks)
            if "/" in mid:
                org = mid.split("/", 1)[0].strip().lower()
                if org and org not in orgs:
                    orgs.append(org)
        if sig.org and sig.org.strip().lower() not in orgs:
            orgs.append(sig.org.strip().lower())
        for name in list(sig.arch_ids) + ([sig.display_name] if sig.display_name else []):
            key = normalize_family_key(name)
            if key and key not in family_keys:
                family_keys.append(key)

    audit = MergeAudit(
        arch_id=cand.arch_id,
        n_signals=len(cand.signals),
        sources=list(cand.sources),
        join_edges=list(getattr(cand, "join_edges", None) or []),
        arch_keys=arch_keys,
        repo_keys=repo_keys,
        family_keys=family_keys,
        orgs=orgs,
        size_token_sets=[sorted(s) for s in size_sets],
    )
    if not audit.merged:
        return audit
    if len(arch_keys) > 1:
        audit.reasons.append(f"{len(arch_keys)} distinct architecture spellings fused: {arch_keys}")
    if len(size_sets) > 1:
        audit.reasons.append(
            "different parameter-size tokens fused: "
            + " vs ".join("/".join(sorted(s)) for s in size_sets)
        )
    if audit.family_only and len(orgs) > 1:
        audit.reasons.append(f"{len(orgs)} orgs fused on a family edge alone: {orgs}")
    return audit


def surface_without(surface: Surface, arch_names: Iterable[str]) -> Surface:
    """A copy of ``surface`` that has never heard of ``arch_names``.

    The zero-day counterfactual. ``Surface.is_known_architecture`` is exact lowercased
    membership, so removing the exact strings is sufficient and cannot perturb any other
    architecture's verdict. Every other part of the surface — parsed fields, validators,
    gaps — is shared, not copied: those are what is being measured.
    """
    drop = {n.strip().lower() for n in arch_names if n and n.strip()}
    return replace(surface, known_architectures=set(surface.known_architectures) - drop)


def measure_cfg(
    *,
    min_total_params: int | None = None,
    recheck: bool | None = None,
    cap: int = MEASURE_CAP,
    window_days: int | None = None,
) -> DetectorConfig:
    """A DetectorConfig for measurement: shipped defaults except what is being swept."""
    thresholds = Thresholds(
        min_total_params=(
            DEFAULTS.thresholds.min_total_params if min_total_params is None else min_total_params
        ),
        min_org_top_downloads=DEFAULTS.thresholds.min_org_top_downloads,
        min_model_downloads=DEFAULTS.thresholds.min_model_downloads,
        min_model_likes=DEFAULTS.thresholds.min_model_likes,
    )
    return DetectorConfig(
        window_days=DEFAULTS.window_days if window_days is None else window_days,
        max_issues_per_run=cap,
        thresholds=thresholds,
        frontier_orgs=set(DEFAULTS.frontier_orgs),
        max_hf_config_fetches=DEFAULTS.max_hf_config_fetches,
        max_github_requests=DEFAULTS.max_github_requests,
        recheck_known_architectures=(
            DEFAULTS.recheck_known_architectures if recheck is None else recheck
        ),
    )


def real_triggers(cand: Candidate) -> list[str]:
    """``Candidate.triggers`` minus provenance markers such as ``alias-join``."""
    return [t for t in (cand.triggers or []) if t in TRIGGER_IDS]


def silently_wrong(cand: Candidate) -> bool:
    """The emitter's derivation (addendum 21), computed straight off the Candidate."""
    return bool(getattr(cand, "silent_failures", None)) and not cand.bucket0_failures


# ===========================================================================
# Signal (de)serialization — so one live poll can feed many evaluate() passes
# ===========================================================================


def signal_to_dict(sig: Signal) -> dict[str, Any]:
    return {
        "source": sig.source,
        "observed_at": sig.observed_at.isoformat() if isinstance(sig.observed_at, datetime) else None,
        "arch_ids": list(sig.arch_ids),
        "model_type": sig.model_type,
        "model_ids": list(sig.model_ids),
        "org": sig.org,
        "display_name": sig.display_name,
        "config": sig.config,
        "urls": dict(sig.urls),
        "evidence": sig.evidence,
        "raw_ref": sig.raw_ref,
        "extra": _jsonable(sig.extra),
    }


def signal_from_dict(d: dict[str, Any]) -> Signal:
    stamp = d.get("observed_at")
    when = (
        datetime.fromisoformat(stamp)
        if isinstance(stamp, str)
        else datetime.now(timezone.utc)
    )
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return Signal(
        source=d.get("source", "?"),
        observed_at=when,
        arch_ids=list(d.get("arch_ids") or []),
        model_type=d.get("model_type"),
        model_ids=list(d.get("model_ids") or []),
        org=d.get("org"),
        display_name=d.get("display_name") or "",
        config=d.get("config"),
        urls=dict(d.get("urls") or {}),
        evidence=d.get("evidence") or "",
        raw_ref=d.get("raw_ref") or "",
        extra=dict(d.get("extra") or {}),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_signals(path: Path, signals: Sequence[Signal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([signal_to_dict(s) for s in signals], indent=1, default=str),
        encoding="utf-8",
    )
    log.info("cached %d signals -> %s", len(signals), path)


def load_signals(path: Path) -> list[Signal]:
    return [signal_from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


# ===========================================================================
# Shared reporting
# ===========================================================================


def cand_row(cand: Candidate) -> dict[str, Any]:
    """Everything a human needs to categorize one candidate by hand."""
    return {
        "arch_id": cand.arch_id,
        "display_name": cand.display_name,
        "sources": list(cand.sources),
        "n_signals": len(cand.signals),
        "triggers": list(cand.triggers or []),
        "significance": list(cand.significance or []),
        "join_edges": list(getattr(cand, "join_edges", None) or []),
        "est_total_params": cand.est_total_params,
        "est_total_h": format_params(cand.est_total_params),
        "est_active_h": format_params(cand.est_active_params),
        "n_unparsed": len(cand.unparsed_fields or []),
        "unparsed_fields": list(cand.unparsed_fields or [])[:12],
        "bucket0_failures": list(cand.bucket0_failures or []),
        "silent_failures": list(getattr(cand, "silent_failures", None) or []),
        "silently_wrong": silently_wrong(cand),
        "would_not_run": cand.would_not_run,
        "model_ids": sorted({m for s in cand.signals for m in s.model_ids})[:8],
        "orgs": sorted({s.org for s in cand.signals if s.org}),
        "model_types": sorted({s.model_type for s in cand.signals if s.model_type}),
        "urls": {k: v for s in cand.signals for k, v in list(s.urls.items())[:2]},
        "has_config": cand.config is not None,
    }


def write_json(out: Path, name: str, payload: Any) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"  wrote {path}")
    return path


def stub_front_matter_report(issues_dir: Path) -> dict[str, Any]:
    """Parse every stub in ``issues_dir`` and tally the headline metric.

    Addendum 21: ``silently_wrong`` is the headline, not ``bucket == 0``. Addendum 22:
    key presence is checked, the ``schema`` string is not.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(issues_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        fm = front_matter(text)
        stub, tail = emitter.split_stub(text)
        rows.append({
            "file": path.name,
            "arch_id": fm.get("arch_id"),
            "schema": fm.get("schema"),
            "missing_keys": missing_front_matter_keys(fm),
            "silently_wrong": bool(fm.get("silently_wrong")),
            "bucket": fm.get("bucket"),
            "bucket0_failures": len(fm.get("bucket0_failures") or []),
            "silent_failures": list(fm.get("silent_failures") or []),
            "triggers": fm.get("triggers"),
            "significance": fm.get("significance"),
            "join_edges": fm.get("join_edges"),
            "known_arch_drift": fm.get("known_arch_drift"),
            "stage2_appendix_nonempty": bool(tail.strip()),
        })
    return {
        "issues_dir": str(issues_dir),
        "n_stubs": len(rows),
        "n_silently_wrong": sum(1 for r in rows if r["silently_wrong"]),
        "silently_wrong": [r["arch_id"] for r in rows if r["silently_wrong"]],
        "n_bucket0": sum(1 for r in rows if r["bucket"] == 0),
        "n_missing_keys": sum(1 for r in rows if r["missing_keys"]),
        "schemas_seen": sorted({str(r["schema"]) for r in rows}),
        "stubs": rows,
    }


# ===========================================================================
# (A) Recall
# ===========================================================================


def fetch_target_signals(
    cfg: DetectorConfig, targets: Sequence[Target] = TARGETS
) -> tuple[dict[str, list[Signal]], list[dict[str, Any]]]:
    """One Signal per target repo id, built by the HF connector's own code path.

    Per repo id, not batched, on purpose: ``_plan_config_fetches`` dedups by
    architecture, so a batched call would fetch one config for all three GLM releases
    (they share ``GlmMoeDsaForCausalLM``) and leave the others config-less — measuring
    the fetch planner instead of the filter.

    Cost: one ``model_info`` plus one ``config.json`` per repo id. Every target org is
    in ``FRONTIER_ORGS``, so the connector's org-download sweep does not fire.
    """
    from huggingface_hub import HfApi

    from archwatch.connectors.hf import LIST_EXPAND, HFConnector

    api = HfApi()
    conn = HFConnector(cfg, api=api)
    by_repo: dict[str, list[Signal]] = {}
    problems: list[dict[str, Any]] = []
    for target in targets:
        for repo_id in target.repo_ids:
            try:
                info = api.model_info(repo_id, expand=list(LIST_EXPAND))
            except Exception as exc:
                problems.append({"repo_id": repo_id, "stage": "model_info",
                                 "error": f"{type(exc).__name__}: {exc}"})
                by_repo[repo_id] = []
                continue
            try:
                # The connector's real assembly: pre-filter, budgeted config fetch,
                # org sweep, Signal construction. If the pre-filter drops the repo this
                # returns [] — a genuine recall failure, recorded as such.
                by_repo[repo_id] = list(conn._to_signals([info], phase="window"))
            except Exception as exc:
                problems.append({"repo_id": repo_id, "stage": "_to_signals",
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc(limit=4)})
                by_repo[repo_id] = []
    return by_repo, problems


def _arm(
    signals: Sequence[Signal],
    surface: Surface,
    cfg: DetectorConfig,
    issues_dir: Path,
) -> tuple[EvaluationReport, list[Candidate]]:
    cands = join_signals(list(signals))
    report = evaluate_detailed(cands, surface, cfg, issues_dir=issues_dir)
    return report, cands


def step_recall(out: Path, *, targets: Sequence[Target] = TARGETS) -> dict[str, Any]:
    print("\n=== (A) RECALL — named frontier releases, real configs, real surface ===")
    surface = load_surface()
    issues_dir = out / "recall-issues"          # empty: dedup must not mask recall
    issues_dir.mkdir(parents=True, exist_ok=True)

    cfg_asis = measure_cfg()
    cfg_recheck = measure_cfg(recheck=True)

    by_repo, problems = fetch_target_signals(cfg_asis, targets)
    save_signals(out / "target-signals.json", [s for v in by_repo.values() for s in v])

    rows: list[dict[str, Any]] = []
    for target in targets:
        for repo_id in target.repo_ids:
            signals = by_repo.get(repo_id) or []
            row: dict[str, Any] = {
                "target": target.label,
                "kind": target.kind,
                "repo_id": repo_id,
                "note": target.note,
                "n_signals": len(signals),
            }
            if not signals:
                row["outcome"] = "no_signal"
                row["detail"] = "HF connector produced no Signal (pre-filter drop or fetch error)"
                rows.append(row)
                continue
            sig = signals[0]
            arch_ids = list(sig.arch_ids)
            row.update({
                "arch_ids": arch_ids,
                "model_type": sig.model_type,
                "org": sig.org,
                "has_config": sig.config is not None,
                "n_config_keys": len(sig.config or {}),
                "downloads": sig.extra.get("downloads"),
                "likes": sig.extra.get("likes"),
                "known_at_seed": [a for a in arch_ids if surface.is_known_architecture(a)],
            })
            est = estimate_params(sig.config)
            row["est_total_params"] = est.total
            row["est_total_h"] = format_params(est.total)
            row["est_active_h"] = format_params(est.active)

            zero_day_surface = surface_without(surface, arch_ids)
            for arm_name, arm_surface, arm_cfg in (
                ("as_shipped", surface, cfg_asis),
                ("recheck", surface, cfg_recheck),
                ("zero_day", zero_day_surface, cfg_asis),
            ):
                report, _ = _arm(signals, arm_surface, arm_cfg, issues_dir)
                if report.passed:
                    cand = report.passed[0]
                    row[arm_name] = {
                        "passed": True,
                        "arch_id": cand.arch_id,
                        "triggers": real_triggers(cand),
                        "markers": [t for t in (cand.triggers or []) if t not in TRIGGER_IDS],
                        "significance": list(cand.significance or []),
                        "n_unparsed": len(cand.unparsed_fields or []),
                        "unparsed_fields": list(cand.unparsed_fields or []),
                        "bucket0_failures": list(cand.bucket0_failures or []),
                        "silent_failures": list(getattr(cand, "silent_failures", None) or []),
                        "silently_wrong": silently_wrong(cand),
                        "would_not_run": cand.would_not_run,
                    }
                else:
                    drop = report.dropped[0] if report.dropped else None
                    row[arm_name] = {
                        "passed": False,
                        "stage": drop.stage if drop else "?",
                        "reason": drop.reason if drop else "no_candidate",
                        "detail": (drop.detail if drop else "")[:300],
                    }
            rows.append(row)

    # Per-target roll-up: a target is a hit when ANY of its repos passes.
    per_target: list[dict[str, Any]] = []
    for target in targets:
        mine = [r for r in rows if r["target"] == target.label]
        entry: dict[str, Any] = {"target": target.label, "kind": target.kind,
                                 "repos": [r["repo_id"] for r in mine]}
        for arm in ("as_shipped", "recheck", "zero_day"):
            hits = [r for r in mine if isinstance(r.get(arm), dict) and r[arm].get("passed")]
            entry[arm] = {
                "hit": bool(hits),
                "repos_hit": [r["repo_id"] for r in hits],
                "triggers": sorted({t for r in hits for t in r[arm]["triggers"]}),
                "significance": sorted({s for r in hits for s in r[arm]["significance"]}),
                "miss_reasons": sorted({
                    r[arm].get("reason", "?") for r in mine
                    if isinstance(r.get(arm), dict) and not r[arm].get("passed")
                }),
            }
        per_target.append(entry)

    frontier = [e for e in per_target if e["kind"] == "frontier"]
    seeded = [e for e in per_target if e["kind"] == "seeded"]
    summary = {
        "n_frontier_targets": len(frontier),
        "n_seeded_targets": len(seeded),
        "recall": {
            arm: {
                "frontier_hits": sum(1 for e in frontier if e[arm]["hit"]),
                "frontier_total": len(frontier),
                "seeded_hits": sum(1 for e in seeded if e[arm]["hit"]),
                "seeded_total": len(seeded),
            }
            for arm in ("as_shipped", "recheck", "zero_day")
        },
        "n_silently_wrong": sum(
            1 for r in rows for arm in ("as_shipped", "recheck", "zero_day")
            if isinstance(r.get(arm), dict) and r[arm].get("silently_wrong")
        ),
        "problems": problems,
    }

    _print_recall_table(rows, per_target, summary)
    payload = {"step": "recall", "summary": summary, "per_target": per_target, "per_repo": rows}
    write_json(out, "recall.json", payload)
    return payload


def _print_recall_table(
    rows: list[dict[str, Any]], per_target: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    print(f"\n{'target':16s} {'repo':34s} {'arch_id':40s} {'params':>8s}  "
          f"{'as-shipped':<24s} {'recheck':<24s} {'zero-day':<24s}")
    print("-" * 180)
    for r in rows:
        def cell(arm: str) -> str:
            v = r.get(arm)
            if not isinstance(v, dict):
                return "-"
            if v.get("passed"):
                return "PASS " + "+".join(v["triggers"]) + "/" + "+".join(v["significance"])
            return "drop:" + str(v.get("reason", "?"))
        arch = ",".join(r.get("arch_ids") or []) or "(none)"
        print(f"{r['target'][:16]:16s} {r['repo_id'][:34]:34s} {arch[:40]:40s} "
              f"{r.get('est_total_h', '-') or '-':>8s}  "
              f"{cell('as_shipped')[:24]:<24s} {cell('recheck')[:24]:<24s} {cell('zero_day')[:24]:<24s}")
    print("\nper-target recall (a target hits when any of its repos passes):")
    for arm in ("as_shipped", "recheck", "zero_day"):
        rec = summary["recall"][arm]
        print(f"  {arm:11s} frontier {rec['frontier_hits']}/{rec['frontier_total']}   "
              f"seeded {rec['seeded_hits']}/{rec['seeded_total']}")
    misses = [
        (e["target"], arm, e[arm]["miss_reasons"])
        for e in per_target for arm in ("zero_day",) if not e[arm]["hit"]
    ]
    if misses:
        print("  zero-day misses:")
        for label, _arm, reasons in misses:
            print(f"    {label}: {reasons}")


# ===========================================================================
# (E) Cross-source join + false-merge audit
# ===========================================================================


def poll_curated(
    window_days: int,
    *,
    now: datetime | None = None,
    sources: Sequence[str] = ("vllm", "sglang", "inferencex"),
    cfg: DetectorConfig | None = None,
) -> tuple[list[Signal], list[dict[str, Any]]]:
    """Live poll of the curated (GitHub-backed) sources, budget-shared. Never raises."""
    cfg = cfg or measure_cfg()
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    since = end - timedelta(days=window_days)
    conns, budget = detector.build_connectors(list(sources), cfg)
    signals, reports = detector.poll_connectors(conns, since, trending=True)
    rows = [r.as_dict() for r in reports]
    if budget is not None:
        rows.append({"github_budget": budget.as_dict()})
    return signals, rows


def step_join(out: Path, *, window_days: int = 14) -> dict[str, Any]:
    print(f"\n=== (E) CROSS-SOURCE JOIN + FALSE-MERGE AUDIT (curated window {window_days}d) ===")
    cached = out / "target-signals.json"
    hf_signals = load_signals(cached) if cached.is_file() else []
    if not hf_signals:
        by_repo, _ = fetch_target_signals(measure_cfg())
        hf_signals = [s for v in by_repo.values() for s in v]
        save_signals(cached, hf_signals)

    curated, source_reports = poll_curated(window_days)
    save_signals(out / "curated-signals.json", curated)
    print(f"  hf target signals: {len(hf_signals)}   curated signals: {len(curated)}")
    for row in source_reports:
        print(f"    {row}")

    audits: list[MergeAudit] = []
    joined_multi: list[dict[str, Any]] = []
    for label, signals in (
        ("curated_only", curated),
        ("hf_targets_plus_curated", list(hf_signals) + list(curated)),
    ):
        cands = join_signals(signals)
        print(f"\n  [{label}] {len(signals)} signals -> {len(cands)} candidates")
        for cand in cands:
            audit = audit_merge(cand)
            audit.arch_id = f"{label}:{cand.arch_id}"
            audits.append(audit)
            if len(cand.sources) > 1:
                joined_multi.append({
                    "set": label,
                    "arch_id": cand.arch_id,
                    "sources": list(cand.sources),
                    "join_edges": list(getattr(cand, "join_edges", None) or []),
                    "model_ids": sorted({m for s in cand.signals for m in s.model_ids})[:10],
                })
        multi = [c for c in cands if len(c.sources) > 1]
        print(f"    cross-source candidates: {len(multi)}")
        for cand in multi:
            print(f"      {cand.arch_id!r} <- {cand.sources} via "
                  f"{getattr(cand, 'join_edges', None) or []}")

    suspicious = [a for a in audits if a.suspicious]
    family_only = [a for a in audits if a.family_only]
    print(f"\n  merges audited: {sum(1 for a in audits if a.merged)}   "
          f"family-edge-only: {len(family_only)}   flagged suspicious: {len(suspicious)}")
    for a in suspicious:
        print(f"    SUSPECT {a.arch_id!r} edges={a.join_edges} :: {'; '.join(a.reasons)}")

    payload = {
        "step": "join",
        "window_days": window_days,
        "n_hf_target_signals": len(hf_signals),
        "n_curated_signals": len(curated),
        "source_reports": source_reports,
        "cross_source_candidates": joined_multi,
        "n_merged": sum(1 for a in audits if a.merged),
        "n_family_only": len(family_only),
        "n_suspicious": len(suspicious),
        "audits": [a.as_dict() for a in audits if a.merged],
    }
    write_json(out, "join.json", payload)
    return payload


# ===========================================================================
# (C) GitHub-source historical replay
# ===========================================================================

#: A window containing known model-support PRs, found with
#:   gh api search/issues -f q='repo:vllm-project/vllm is:pr is:merged "[Model]" merged:...'
#: Recorded so the replay is reproducible rather than "whatever is recent".
REPLAY_SINCE = datetime(2026, 8, 10, tzinfo=timezone.utc)
REPLAY_UNTIL = datetime(2026, 8, 21, tzinfo=timezone.utc)

#: Merged in the replay window; the replay is a pass only if the pipeline surfaces one.
REPLAY_EXPECTED_PRS: tuple[str, ...] = ("51655", "51255", "52114", "52706")


def step_replay(
    out: Path,
    *,
    since: datetime = REPLAY_SINCE,
    until: datetime = REPLAY_UNTIL,
) -> dict[str, Any]:
    print(f"\n=== (C) HISTORICAL REPLAY — GitHub sources, {since.date()}..{until.date()} ===")
    from archwatch.connectors.frameworks import SglangConnector, VllmConnector
    from archwatch.connectors.inferencex import InferenceXConnector

    cfg = measure_cfg()
    budget = detector.GithubBudget(limit=cfg.max_github_requests)
    session = detector.BudgetedSession(budget)

    signals: list[Signal] = []
    reports: list[dict[str, Any]] = []

    # The framework connector takes `until` (addendum 12), so its window is bounded
    # server-side-ish: it filters each PR's merge time against [since, until].
    for conn in (VllmConnector(session=session, until=until),
                 SglangConnector(session=session, until=until)):
        started = time.monotonic()
        try:
            got = list(conn.poll(since))
        except Exception as exc:
            got = []
            reports.append({"source": conn.name, "ok": False,
                            "error": f"{type(exc).__name__}: {exc}"})
        else:
            reports.append({"source": conn.name, "ok": True, "signals": len(got),
                            "duration_s": round(time.monotonic() - started, 2)})
        signals.extend(got)

    # InferenceXConnector has NO `until` parameter (addendum 12 is unimplemented there),
    # so the upper bound is applied client-side on observed_at. Volume is low tens per
    # week, so nothing is lost by over-fetching and filtering.
    ix = InferenceXConnector(session=session)
    started = time.monotonic()
    try:
        raw = list(ix.poll(since))
    except Exception as exc:
        raw = []
        reports.append({"source": ix.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    else:
        kept = [s for s in raw if s.observed_at <= until]
        reports.append({
            "source": ix.name, "ok": True, "signals_fetched": len(raw),
            "signals_in_window": len(kept),
            "note": "no `until=` on InferenceXConnector; bounded client-side on observed_at",
            "duration_s": round(time.monotonic() - started, 2),
        })
        raw = kept
    signals.extend(raw)

    save_signals(out / "replay-signals.json", signals)
    print(f"  {len(signals)} signals in window; github requests used: {budget.used}/{budget.limit}")
    for row in reports:
        print(f"    {row}")

    refs = {str(s.raw_ref) for s in signals}
    expected_seen = {pr: any(pr in r for r in refs) for pr in REPLAY_EXPECTED_PRS}

    surface = load_surface()
    # Cleared, so the measurement does not depend on whether this step ran before:
    # stubs left by a previous run fire the `already_reported` suppressor and would
    # silently report 0 survivors on the second invocation.
    issues_dir = out / "replay-issues"
    if issues_dir.is_dir():
        for stale in issues_dir.glob("*.md"):
            stale.unlink()
    issues_dir.mkdir(parents=True, exist_ok=True)
    cands = join_signals(signals)
    report = evaluate_detailed(cands, surface, cfg, issues_dir=issues_dir)
    results = emitter.write_issues(report.passed, issues_dir)

    # Idempotence + the stateless dedup, measured rather than assumed: writing the same
    # candidates again must change no bytes, and a re-evaluation against the now-populated
    # directory must suppress every one of them as already_reported.
    rewrite = emitter.write_issues(report.passed, issues_dir)
    dedup_report = evaluate_detailed(join_signals(signals), surface, cfg, issues_dir=issues_dir)
    idempotent = all(r.status == "unchanged" for r in rewrite)
    dedup_suppressed = dedup_report.counts.get("already_reported", 0)
    print(f"  idempotent re-write: {idempotent}; "
          f"re-scan suppressed {dedup_suppressed} as already_reported "
          f"({len(dedup_report.passed)} still passed)")

    print(f"  {len(cands)} candidates -> {len(report.passed)} passed "
          f"-> {sum(1 for r in results if r.written)} stubs")
    print(f"  {report.summary()}")
    for cand in report.passed:
        print(f"    PASS {cand.arch_id!r} {real_triggers(cand)}/{cand.significance} "
              f"sources={cand.sources} refs="
              f"{sorted({s.raw_ref for s in cand.signals})[:4]}")
    print(f"  expected PRs present in signal set: {expected_seen}")

    # --- zero-day arm ---------------------------------------------------
    # Same seed-set-from-the-future problem as the recall step: the surface was
    # harvested from vLLM's registry AFTER this window closed, so every architecture
    # the window's PRs add is already in it and drops at `known_architecture`. The arm
    # forgets exactly the architectures the window's framework signals name — a
    # deliberately generous counterfactual (it also forgets incidental mentions such as
    # LlamaForCausalLM), reported as such.
    framework_archs = sorted({
        a for s in signals if s.source in ("vllm", "sglang") for a in s.arch_ids
    })
    zero_day = surface_without(surface, framework_archs)
    zd_report = evaluate_detailed(
        join_signals(signals), zero_day, cfg, issues_dir=out / "replay-issues-zeroday"
    )
    zd_framework = [c for c in zd_report.passed if set(c.sources) & {"vllm", "sglang"}]
    print(f"\n  [zero-day] forgot {len(framework_archs)} architectures the window's PRs name")
    print(f"  [zero-day] {zd_report.summary()}")
    print(f"  [zero-day] framework-backed survivors: {len(zd_framework)}")
    for cand in zd_framework:
        print(f"    PASS {cand.arch_id!r} {real_triggers(cand)}/{cand.significance} "
              f"refs={sorted({s.raw_ref for s in cand.signals})}")

    payload = {
        "step": "replay",
        "since": since.isoformat(),
        "until": until.isoformat(),
        "github_budget": budget.as_dict(),
        "source_reports": reports,
        "n_signals": len(signals),
        "n_candidates": len(cands),
        "expected_prs_seen": expected_seen,
        "signal_refs": sorted(refs),
        "passed": [cand_row(c) for c in report.passed],
        "drop_counts": report.counts,
        "dropped": [
            {"arch_id": d.arch_id, "stage": d.stage, "reason": d.reason, "detail": d.detail[:200]}
            for d in report.dropped
        ],
        "framework_signals": [
            {
                "source": s.source,
                "pr": s.raw_ref,
                "signal_strength": s.extra.get("signal_strength"),
                "arch_ids": list(s.arch_ids),
                "display_name": s.display_name,
            }
            for s in signals if s.source in ("vllm", "sglang")
        ],
        "zero_day": {
            "forgotten_architectures": framework_archs,
            "drop_counts": zd_report.counts,
            "n_passed": len(zd_report.passed),
            "framework_backed_survivors": [cand_row(c) for c in zd_framework],
        },
        "idempotent_rewrite": idempotent,
        "dedup_suppressed_on_rescan": dedup_suppressed,
        "merge_audit": [a.as_dict() for a in (audit_merge(c) for c in cands) if a.suspicious],
        "stub_front_matter": stub_front_matter_report(issues_dir),
    }
    write_json(out, "replay.json", payload)
    return payload


# ===========================================================================
# (D) Threshold calibration sweep
# ===========================================================================


def step_sweep(out: Path, *, window_days: int = 1) -> dict[str, Any]:
    print(f"\n=== (D) THRESHOLD SWEEP (HF volume window {window_days}d) ===")
    surface = load_surface()
    issues_dir = out / "sweep-issues"
    issues_dir.mkdir(parents=True, exist_ok=True)

    # --- one live HF poll, reused for every sweep point ---------------------
    volume_cache = out / "sweep-hf-signals.json"
    if volume_cache.is_file():
        hf_window = load_signals(volume_cache)
        print(f"  reusing cached HF window poll: {len(hf_window)} signals")
    else:
        from archwatch.connectors.hf import HFConnector

        end = datetime.now(timezone.utc)
        conn = HFConnector(measure_cfg(window_days=window_days))
        started = time.monotonic()
        hf_window = list(conn.poll(end - timedelta(days=window_days)))
        hf_window += list(conn.poll_trending())
        print(f"  live HF poll: {len(hf_window)} signals in {time.monotonic() - started:.1f}s")
        save_signals(volume_cache, hf_window)

    target_cache = out / "target-signals.json"
    if target_cache.is_file():
        target_signals = load_signals(target_cache)
    else:
        by_repo, _ = fetch_target_signals(measure_cfg())
        target_signals = [s for v in by_repo.values() for s in v]
        save_signals(target_cache, target_signals)

    # Zero-day surface for the recall half: every target architecture removed at once,
    # so one surface serves all of them. (Per-target removal in step_recall is finer
    # grained; here the question is only "how many targets clear the gate".)
    target_archs = {a for s in target_signals for a in s.arch_ids}
    zero_day = surface_without(surface, target_archs)
    frontier_labels = {t.label for t in TARGETS if t.kind == "frontier"}
    repo_to_label = {r: t.label for t in TARGETS for r in t.repo_ids}

    def recall_at(cfg: DetectorConfig, surf: Surface) -> tuple[int, list[str]]:
        hit_labels: set[str] = set()
        for sig in target_signals:
            report = evaluate_detailed(join_signals([sig]), surf, cfg, issues_dir=issues_dir)
            if report.passed:
                label = repo_to_label.get(sig.model_ids[0] if sig.model_ids else "", "?")
                if label in frontier_labels:
                    hit_labels.add(label)
        return len(hit_labels), sorted(hit_labels)

    joined_window = join_signals(hf_window)
    print(f"  HF window: {len(hf_window)} signals -> {len(joined_window)} candidates")

    rows: list[dict[str, Any]] = []
    for recheck in (False, True):
        for min_total in SWEEP_PARAMS:
            cfg = measure_cfg(min_total_params=min_total, recheck=recheck)
            # join_signals is deterministic and cheap; re-run per point so a candidate
            # mutated in place by a previous evaluate() cannot leak across points.
            report = evaluate_detailed(
                join_signals(hf_window), surface, cfg, issues_dir=issues_dir
            )
            n_hits, hits = recall_at(cfg, zero_day)
            n_live, live_hits = recall_at(cfg, surface)
            row = {
                "recheck_known_architectures": recheck,
                "min_total_params": min_total,
                "min_total_h": format_params(min_total),
                "zero_day_recall": n_hits,
                "zero_day_recall_of": len(frontier_labels),
                "zero_day_hits": hits,
                # The same recall question against the surface as it actually ships,
                # i.e. with every target already in the seed set. This is the column
                # that shows what recheck_known_architectures buys.
                "as_shipped_recall": n_live,
                "as_shipped_hits": live_hits,
                "hf_survivors": len(report.passed),
                "hf_survivors_silently_wrong": sum(1 for c in report.passed if silently_wrong(c)),
                "hf_survivors_bucket0": sum(1 for c in report.passed if c.would_not_run),
                "drop_counts": report.counts,
                "survivor_arch_ids": [c.arch_id for c in report.passed][:40],
            }
            rows.append(row)
            print(f"  recheck={str(recheck):5s} min_total={format_params(min_total):>7s}  "
                  f"recall zero-day {n_hits}/{len(frontier_labels)} "
                  f"as-shipped {n_live}/{len(frontier_labels)}  "
                  f"HF survivors {len(report.passed):4d}  "
                  f"(silently_wrong {row['hf_survivors_silently_wrong']}, "
                  f"bucket0 {row['hf_survivors_bucket0']})")

    payload = {
        "step": "sweep",
        "window_days": window_days,
        "n_hf_signals": len(hf_window),
        "n_hf_candidates": len(joined_window),
        "n_target_signals": len(target_signals),
        "frontier_targets": sorted(frontier_labels),
        "rows": rows,
    }
    write_json(out, "sweep.json", payload)
    return payload


# ===========================================================================
# (B) Precision
# ===========================================================================


def step_precision(
    out: Path,
    *,
    window_days: int = 1,
    sources: Sequence[str] = ("hf", "vllm", "sglang", "inferencex"),
    recheck: bool = False,
) -> dict[str, Any]:
    print(f"\n=== (B) PRECISION — live scans, window {window_days}d, "
          f"recheck_known_architectures={recheck} ===")
    runs: list[dict[str, Any]] = []
    cfg = measure_cfg(window_days=window_days, recheck=recheck)
    tag = "recheck" if recheck else "asis"
    for source in list(sources) + ["all"]:
        spec = list(sources) if source == "all" else [source]
        issues_dir = out / f"precision-{tag}-{source}"
        issues_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  --- scan sources={','.join(spec)} ---")
        try:
            summary = detector.scan(
                cfg, sources=spec, window_days=window_days, out_dir=issues_dir, dry_run=True
            )
        except Exception as exc:
            print(f"    SCAN FAILED: {type(exc).__name__}: {exc}")
            runs.append({"sources": spec, "ok": False,
                         "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(limit=6)})
            continue
        rows = [cand_row(c) for c in summary.passed]
        shipped_cap = DEFAULTS.max_issues_per_run
        print(f"    {summary.signals} signals -> {summary.candidates} candidates -> "
              f"{len(summary.passed)} survivors (shipped cap {shipped_cap} would show "
              f"{min(shipped_cap, len(summary.passed))})")
        print(f"    drops: {summary.suppressed_by_reason}")
        for i, row in enumerate(rows, start=1):
            flag = " *SILENTLY-WRONG*" if row["silently_wrong"] else ""
            flag += " *BUCKET0*" if row["would_not_run"] else ""
            print(f"    {i:3d}. {row['arch_id'][:44]:44s} {row['sources']} "
                  f"{row['triggers']}/{row['significance']} "
                  f"{row['est_total_h']:>7s} unparsed={row['n_unparsed']}"
                  f" mt={row['model_types']}{flag}")
        runs.append({
            "sources": spec,
            "ok": True,
            "run_id": summary.run_id,
            "signals": summary.signals,
            "signals_by_source": summary.signals_by_source,
            "candidates": summary.candidates,
            "n_survivors": len(summary.passed),
            "n_would_show_at_shipped_cap": min(shipped_cap, len(summary.passed)),
            "drop_counts": summary.suppressed_by_reason,
            "triggers_by_code": summary.triggers_by_code,
            "significance_by_code": summary.significance_by_code,
            "source_reports": [r.as_dict() for r in summary.source_reports],
            "github": summary.github.as_dict() if summary.github else None,
            "survivors": rows,
            "stub_front_matter": stub_front_matter_report(issues_dir),
        })

    payload = {"step": "precision", "window_days": window_days,
               "recheck_known_architectures": recheck, "runs": runs}
    write_json(out, f"precision-{tag}-{window_days}d.json", payload)
    return payload


# ===========================================================================
# main
# ===========================================================================


def _provenance() -> dict[str, Any]:
    import subprocess

    def sh(*args: str) -> str:
        try:
            return subprocess.run(args, cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return "?"

    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "git_head": sh("git", "rev-parse", "HEAD"),
        "git_dirty_files": [l for l in sh("git", "status", "--porcelain").splitlines() if l],
        "python": sys.version.split()[0],
        "hf_token": bool(os.environ.get("HF_TOKEN")),
        "surface_known_architectures": len(load_surface().known_architectures),
    }


STEPS = ("recall", "join", "replay", "sweep", "precision")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("step", choices=(*STEPS, "all"))
    parser.add_argument("--out", default=None,
                        help="artifact directory (default: a fresh temp dir; NEVER issues/)")
    parser.add_argument("--window-days", type=int, default=1,
                        help="window for the HF volume poll and the precision scans")
    parser.add_argument("--curated-window-days", type=int, default=14,
                        help="window for the curated-source poll in the join step")
    parser.add_argument("--sources", default="hf,vllm,sglang,inferencex")
    parser.add_argument("--recheck", action="store_true",
                        help="precision step: set recheck_known_architectures=True")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)],
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.out:
        out = Path(args.out).resolve()
    else:
        import tempfile

        out = Path(tempfile.mkdtemp(prefix="archwatch-backtest-"))
    # Belt and braces on PLAN.md rule 2 and on the recall measurement itself.
    if out.resolve() == emitter.default_issues_dir().resolve():
        parser.error("refusing to write into the repo's issues/ directory")
    out.mkdir(parents=True, exist_ok=True)

    prov = _provenance()
    print(f"archwatch backtest — artifacts in {out}")
    print(f"  HEAD {prov['git_head'][:12]} dirty={len(prov['git_dirty_files'])} "
          f"seed-set={prov['surface_known_architectures']} archs")
    write_json(out, "provenance.json", prov)

    steps = STEPS if args.step == "all" else (args.step,)
    results: dict[str, Any] = {"provenance": prov}
    for step in steps:
        started = time.monotonic()
        if step == "recall":
            results[step] = step_recall(out)
        elif step == "join":
            results[step] = step_join(out, window_days=args.curated_window_days)
        elif step == "replay":
            results[step] = step_replay(out)
        elif step == "sweep":
            results[step] = step_sweep(out, window_days=args.window_days)
        elif step == "precision":
            results[step] = step_precision(
                out, window_days=args.window_days,
                sources=tuple(detector.parse_sources(args.sources)),
                recheck=args.recheck,
            )
        print(f"  [{step} took {time.monotonic() - started:.1f}s]")

    write_json(out, "backtest-summary.json", results)
    print(f"\ndone — artifacts in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
