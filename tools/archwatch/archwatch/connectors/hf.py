"""HuggingFace connector — the model firehose, narrowed in two cheap phases.

Roughly 3.5k model repos are created on the Hub every day. Downloading a
``config.json`` for each one is thousands of requests per poll, so this
connector never does that. It narrows in two phases:

**Phase 1 — metadata only, zero extra requests.** ``list_models`` is asked for
the fields we need via ``expand=[...]``, which importantly includes ``config``:
the Hub keeps an *indexed* excerpt of every repo's ``config.json`` holding
``architectures``, ``model_type`` and ``quantization_config``. So the pipeline's
primary key (the architecture) is available for free, straight from the listing.
Two drops happen here, counted separately so each list can be tuned on its own:

* repo ids matching :data:`archwatch.config.DERIVATIVE_PATTERNS` — quants,
  merges, LoRAs and other repackagings of an existing architecture;
* repos that name no architecture *and* carry positive evidence of being
  something other than a language model (see :func:`is_non_lm_artifact`) —
  diffusers checkpoints, robotics policies, PEFT adapters, timm classifiers.

The second is deliberately timid. Roughly two thirds of the architecture-less
survivors on a given day publish neither a ``library_name`` nor a
``pipeline_tag``, and every one of those is kept: absence of evidence is not
evidence, and an unknown-shaped repo from an unexpected lab is precisely what
this pipeline is watching for.

**Phase 2 — full ``config.json``, budgeted.** The indexed excerpt has no shape
fields, so the novelty filter's "fields BLIS does not parse" check still needs
the real file. We fetch it for at most ``cfg.max_hf_config_fetches`` repos per
poll, choosing them by architecture: one representative per *distinct*
architecture, best-known org / most downloaded first, then the repos whose
architecture is still unknown but which look like language models. On a live
1-day window this is ~130 fetches for ~2900 surviving repos, because ~2900
repos only span ~130 architectures.

A Signal is emitted for **every** phase-1 survivor, whether or not its config
was fetched — the detector joins them on the architecture key, and
:attr:`Candidate.config` picks the richest config any sibling Signal carried.

Nothing here raises for ordinary source trouble (404, 429, gated repo,
malformed JSON, an outage mid-pagination). Such problems are logged and the
poll returns what it has, with ``config=None`` where the file was unobtainable.
"""

from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import DEFAULTS, DERIVATIVE_PATTERNS, DetectorConfig
from .base import Signal

log = logging.getLogger("archwatch.connectors.hf")

CONFIG_FILENAME = "config.json"
HF_BASE = "https://huggingface.co"

#: Listing fields we ask the Hub for. ``config`` is the cheap-arch trick; the
#: rest feed the significance gate and the fetch-priority ranking.
LIST_EXPAND: tuple[str, ...] = (
    "author", "createdAt", "lastModified", "downloads", "downloadsAllTime",
    "likes", "trendingScore", "tags", "library_name", "pipeline_tag",
    "config", "sha", "gated", "private",
)

#: Hard stop on pagination so a wide window cannot walk the whole Hub.
DEFAULT_MAX_LIST = 20_000

#: How many trending repos the S4 sweep looks at.
DEFAULT_TRENDING_LIMIT = 100

#: Safety cap on the org-popularity sweep (see ``_sweep_popular_orgs``). The
#: sweep normally stops itself long before this.
DEFAULT_ORG_SWEEP_LIMIT = 20_000

#: Cap on per-org fallback lookups, which cost one request each (~65 ms).
#: Sized so the eligibility gate in ``_org_top_downloads`` fits under it on a
#: normal day (~184 orgs measured over a 1-day window), because a cap that
#: truncates would make S2 depend on listing order rather than on the data.
DEFAULT_MAX_ORG_LOOKUPS = 200

#: How many of an org's top repos a fallback lookup reads to find its best.
ORG_TOP_N = 5

# Used only to rank phase-2 fetch priority among repos with no known
# architecture — never to drop anything.
_LM_LIBRARIES = frozenset({"transformers", "vllm", "sglang", "mlx", "nemo"})
_LM_PIPELINE_TAGS = frozenset({
    "text-generation", "text2text-generation", "image-text-to-text",
    "any-to-any", "video-text-to-text", "audio-text-to-text",
})

# ---------------------------------------------------------------------------
# The non-LM pre-filter's vocabularies.
#
# These are DROP lists, so every entry must be positive evidence that a repo
# holds something other than a language model. Membership was chosen against a
# live day's ``library_name`` / ``pipeline_tag`` histogram over the ~1,700
# architecture-less survivors; anything ambiguous was deliberately left out.
#
# Notably absent, and why:
#   transformers, pytorch, keras, onnx/onnxruntime, tensorrt, mlx, gguf,
#   llama.cpp, exllamav3, safetensors, custom, generic
#       — runtime/format/generic labels that say nothing about modality.
#       (The quant-packaging ones among them are DERIVATIVE_PATTERNS' job:
#       they hold a language model, just not a new architecture.)
#   nemo — hosts NVIDIA's Nemotron LLMs as well as speech models.
#   keras — KerasHub publishes LLMs under it.
#   one-off vendor library names (loom-py-rt, minimax-h3, karume, coreai, ...)
#       — an unrecognized library from an unexpected lab is the zero-day case
#       this system exists to catch, so it is never treated as evidence.
# ---------------------------------------------------------------------------

#: ``library_name`` values that positively identify a non-language-model repo.
NON_LM_LIBRARIES: frozenset[str] = frozenset({
    # image / video generation
    "diffusers", "diffusion-single-file", "trellis",
    # vision
    "timm", "ultralytics", "segmentation-models-pytorch", "open_clip", "doctr",
    "calamari", "paddleocr",
    # audio / speech
    "espnet", "speechbrain", "pyannote-audio", "asteroid", "k2", "gigaam",
    "whisper.cpp", "sherpa-onnx", "nemo-asr",
    # robotics / RL
    "lerobot", "openpi", "stable-baselines3", "ml-agents", "sample-factory",
    "unity-sentis",
    # classical ML / non-generative NLP
    "sklearn", "scikit-learn", "lightgbm", "xgboost", "fastai", "spacy",
    "flair", "stanza", "setfit", "span-marker", "bertopic",
    "sentence-transformers",
    # adapters: no architecture of their own, they point at a base model
    "peft", "adapter-transformers",
})

#: ``pipeline_tag`` values that positively identify a non-language-model task.
#: Text tasks an LM could plausibly serve (text-generation, translation,
#: fill-mask, feature-extraction, image-text-to-text, ...) are excluded.
NON_LM_PIPELINE_TAGS: frozenset[str] = frozenset({
    # image / video / 3D generation
    "text-to-image", "image-to-image", "text-to-video", "image-to-video",
    "image-text-to-video", "unconditional-image-generation", "text-to-3d",
    "image-to-3d",
    # vision understanding
    "image-classification", "image-segmentation", "semantic-segmentation",
    "object-detection", "zero-shot-image-classification",
    "zero-shot-object-detection", "depth-estimation", "keypoint-detection",
    "mask-generation", "image-feature-extraction", "video-classification",
    # audio / speech
    "automatic-speech-recognition", "audio-classification", "audio-to-audio",
    "text-to-speech", "text-to-audio", "voice-activity-detection",
    # not text at all
    "robotics", "reinforcement-learning", "tabular-classification",
    "tabular-regression", "time-series-forecasting", "graph-ml",
})


# ---------------------------------------------------------------------------
# helpers other components may reuse
# ---------------------------------------------------------------------------


def is_derivative(repo_id: str) -> bool:
    """True when a repo id advertises itself as a quant / merge / fine-tune.

    The phase-1 pre-filter, and the same test the novelty filter's
    "every model id matches DERIVATIVE_PATTERNS" suppressor wants.
    """
    low = (repo_id or "").lower()
    return any(pat in low for pat in DERIVATIVE_PATTERNS)


def architectures_of(config: dict[str, Any] | None) -> list[str]:
    """``architectures[]`` from a config, falling back to the text tower.

    Multimodal configs sometimes carry the causal-LM architecture only under
    ``text_config``; ``ParseHFConfig`` pivots that onto the top level, so we
    look there too rather than reporting no architecture at all.
    """
    if not isinstance(config, dict):
        return []
    for holder in (config, config.get("text_config")):
        if not isinstance(holder, dict):
            continue
        archs = holder.get("architectures")
        if isinstance(archs, str):
            archs = [archs]
        if isinstance(archs, (list, tuple)):
            out = [a.strip() for a in archs if isinstance(a, str) and a.strip()]
            if out:
                # preserve order, drop repeats
                return list(dict.fromkeys(out))
    return []


def is_non_lm_artifact(info: Any) -> bool:
    """Positive evidence that a repo holds something other than a language model.

    Deliberately asymmetric: absence of evidence is never evidence. A repo with
    no ``library_name`` and no ``pipeline_tag`` returns False and is kept, even
    though it is far more likely to be junk than a frontier release — an
    unknown-shaped repo from an unexpected lab is exactly the zero-day case this
    pipeline exists to catch, and a false drop is unrecoverable while a false
    keep only costs the detector one suppression.

    Only consulted for repos that expose no ``architectures[]`` at all; a repo
    naming an architecture is always kept, whatever library published it.
    """
    library = (getattr(info, "library_name", None) or "").strip().lower()
    if library in NON_LM_LIBRARIES:
        return True
    pipeline = (getattr(info, "pipeline_tag", None) or "").strip().lower()
    return pipeline in NON_LM_PIPELINE_TAGS


def _model_type_of(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    for holder in (config, config.get("text_config")):
        if isinstance(holder, dict):
            mt = holder.get("model_type")
            if isinstance(mt, str) and mt.strip():
                return mt.strip()
    return None


def _as_utc(value: datetime) -> datetime:
    """Coerce a datetime to aware UTC.

    ``Signal.observed_at`` and the timestamps in ``extra`` must be aware UTC:
    the emitter derives ``detected_at`` from the newest ``observed_at`` and
    deliberately refuses to guess an offset for a naive value, so a naive
    timestamp here would make the emitted output depend on the host's timezone.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: Any) -> str | None:
    return _as_utc(value).isoformat() if isinstance(value, datetime) else None


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _org_of(info: Any) -> str | None:
    """The repo owner, **lowercased**. Canonical repos (``gpt2``) have none.

    Lowercased because it is compared against
    :data:`archwatch.config.FRONTIER_ORGS`, which is all lowercase, and the
    other connectors emit ``Signal.org`` lowercased too — T3 and S2 must not
    depend on how a lab happens to capitalise its Hub namespace. The original
    casing is never lost: it survives in ``model_ids``, ``raw_ref`` and
    ``urls["hf"]``, which carry the real repo id.
    """
    author = getattr(info, "author", None)
    if not author:
        repo_id = getattr(info, "id", "") or ""
        author = repo_id.split("/")[0] if "/" in repo_id else None
    return author.strip().lower() if author else None


# ---------------------------------------------------------------------------
# connector
# ---------------------------------------------------------------------------


class HFConnector:
    """``poll(since)`` over newly created Hub repos; ``poll_trending()`` for S4.

    ``api`` and ``config_fetcher`` are injectable so tests run offline against
    recorded fixtures. ``config_fetcher(repo_id, revision) -> str | None``
    returns the raw ``config.json`` text; it is allowed to raise, and every
    exception it raises is caught here.
    """

    name = "hf"

    def __init__(
        self,
        cfg: DetectorConfig | None = None,
        *,
        api: Any | None = None,
        config_fetcher: Callable[[str, str | None], str | None] | None = None,
        clock: Callable[[], datetime] | None = None,
        max_list: int = DEFAULT_MAX_LIST,
        token: str | bool | None = None,
        org_downloads: bool = True,
        max_org_lookups: int = DEFAULT_MAX_ORG_LOOKUPS,
        org_sweep_limit: int = DEFAULT_ORG_SWEEP_LIMIT,
    ) -> None:
        self.cfg = cfg or DEFAULTS
        self.max_list = max_list
        self.org_downloads = org_downloads
        self.max_org_lookups = max_org_lookups
        self.org_sweep_limit = org_sweep_limit
        self._api = api
        self._fetcher = config_fetcher
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._token = token

    # -- lazily built collaborators ----------------------------------------

    @property
    def api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi  # imported late: tests never need it

            self._api = HfApi(token=self._token)
        return self._api

    def _fetch_text(self, repo_id: str, revision: str | None) -> str | None:
        if self._fetcher is not None:
            return self._fetcher(repo_id, revision)
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename=CONFIG_FILENAME,
            repo_type="model",
            revision=revision,
            token=self._token,
        )
        return Path(path).read_text(encoding="utf-8")

    # -- public API ---------------------------------------------------------

    def poll(self, since: datetime) -> list[Signal]:
        """Signals for every non-derivative repo created at or after ``since``."""
        infos = self._list_created_since(since)
        log.info("hf: %d repos created since %s", len(infos), since.isoformat())
        return self._to_signals(infos, phase="window")

    def poll_trending(self, limit: int = DEFAULT_TRENDING_LIMIT) -> list[Signal]:
        """Signals for currently trending repos, regardless of creation date.

        The S4 sweep: an architecture published quietly weeks ago and only now
        picking up attention never appears in a creation window.
        """
        infos = self._list_trending(limit)
        log.info("hf: %d trending repos", len(infos))
        # Rank is taken before the pre-filter, so it is the repo's true position
        # on the Hub's trending list rather than its position among survivors.
        ranks = {info.id: n for n, info in enumerate(infos, start=1)}
        return self._to_signals(infos, phase="trending", ranks=ranks)

    # -- phase 1: listing ---------------------------------------------------

    def _list_kwargs(self, *, newest_first_by: str, limit: int | None, **extra: Any) -> dict[str, Any]:
        """Sort/limit kwargs for the installed ``huggingface_hub``.

        The v1.x API renamed the sort keys to snake_case (``created_at``,
        ``trending_score``) and **removed** ``direction`` entirely — these sorts
        are descending server-side. 0.x used camelCase keys plus
        ``direction=-1``. Probe the signature rather than guessing.
        """
        kwargs: dict[str, Any] = {"expand": list(LIST_EXPAND), "limit": limit}
        kwargs.update(extra)
        try:
            params = inspect.signature(self.api.list_models).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic api objects
            params = {}
        if "direction" in params:  # huggingface_hub 0.x
            camel = {"created_at": "createdAt", "trending_score": "trendingScore"}
            kwargs["sort"] = camel.get(newest_first_by, newest_first_by)
            kwargs["direction"] = -1
        else:  # huggingface_hub 1.x
            kwargs["sort"] = newest_first_by
        return kwargs

    def _list_created_since(self, since: datetime) -> list[Any]:
        """Walk the createdAt-descending listing until it falls out of the window."""
        since = _as_utc(since)
        infos: list[Any] = []
        seen: set[str] = set()
        kwargs = self._list_kwargs(newest_first_by="created_at", limit=self.max_list)
        try:
            for info in self.api.list_models(**kwargs):
                created = getattr(info, "created_at", None)
                if created is None:
                    # Anomalous in a createdAt-sorted listing; cannot be placed
                    # in the window, so skip it rather than truncate the walk.
                    log.debug("hf: %s has no created_at, skipping", getattr(info, "id", "?"))
                    continue
                if _as_utc(created) < since:
                    break
                if info.id in seen:
                    continue
                seen.add(info.id)
                infos.append(info)
                if len(infos) >= self.max_list:
                    log.warning(
                        "hf: hit max_list=%d before reaching %s; window truncated",
                        self.max_list, since.isoformat(),
                    )
                    break
        except Exception as exc:  # rate limit, outage, unexpected payload
            log.warning("hf: listing new models failed after %d records: %s", len(infos), exc)
        return infos

    def _list_trending(self, limit: int) -> list[Any]:
        infos: list[Any] = []
        seen: set[str] = set()
        kwargs = self._list_kwargs(newest_first_by="trending_score", limit=limit)
        try:
            for info in self.api.list_models(**kwargs):
                if info.id in seen:
                    continue
                seen.add(info.id)
                infos.append(info)
                if len(infos) >= limit:
                    break
        except Exception as exc:
            log.warning("hf: listing trending models failed after %d records: %s", len(infos), exc)
        return infos

    # -- phase 2: budgeted config fetches ----------------------------------

    def _priority(self, info: Any) -> tuple[int, int, int, int, int]:
        """Fetch-order rank. Higher sorts first."""
        org = _org_of(info) or ""
        return (
            1 if org in self.cfg.frontier_orgs else 0,
            _int(getattr(info, "downloads", None)),
            _int(getattr(info, "downloads_all_time", None)),
            _int(getattr(info, "likes", None)),
            _int(getattr(info, "trending_score", None)),
        )

    def _looks_like_lm(self, info: Any) -> bool:
        """Cheap metadata guess, used only to rank unknown-architecture repos."""
        if _model_type_of(getattr(info, "config", None)):
            return True
        if (getattr(info, "library_name", None) or "") in _LM_LIBRARIES:
            return True
        if (getattr(info, "pipeline_tag", None) or "") in _LM_PIPELINE_TAGS:
            return True
        tags = getattr(info, "tags", None) or []
        return any(t in _LM_LIBRARIES or t in _LM_PIPELINE_TAGS for t in tags)

    def _plan_config_fetches(self, survivors: Sequence[Any]) -> list[Any]:
        """Pick at most ``cfg.max_hf_config_fetches`` repos to download configs for.

        Architecture-first: the pipeline's key is the architecture, so covering
        every *distinct* architecture once beats covering the newest N repos.
        """
        budget = max(0, int(self.cfg.max_hf_config_fetches))
        if budget == 0:
            return []
        by_arch: dict[str, list[Any]] = {}
        unknown: list[Any] = []
        for info in survivors:
            archs = architectures_of(getattr(info, "config", None))
            if archs:
                by_arch.setdefault(archs[0], []).append(info)
            else:
                unknown.append(info)
        reps = [max(group, key=self._priority) for group in by_arch.values()]
        reps.sort(key=self._priority, reverse=True)
        # Within the unknown-architecture bucket, a repo whose listing excerpt
        # already carries a model_type is one the Hub *did* parse a config.json
        # for, which simply declares no architectures[] — a custom or
        # non-transformers model_type. Measured live: those have a fetchable
        # config.json ~100% of the time, while the rest of the bucket (no
        # excerpt at all) is ~15%, so they go first. The rest are still kept as
        # insurance against Hub indexing lag on a brand-new release.
        extras = [i for i in unknown if self._looks_like_lm(i)]
        extras.sort(
            key=lambda i: (bool(_model_type_of(getattr(i, "config", None))), self._priority(i)),
            reverse=True,
        )
        plan = (reps + extras)[:budget]
        log.info(
            "hf: config-fetch plan %d/%d repos (%d architectures, %d unknown-arch candidates, cap %d)",
            len(plan), len(survivors), len(by_arch), len(extras), budget,
        )
        return plan

    # -- org track record for S2 --------------------------------------------

    def _sweep_popular_orgs(self) -> tuple[dict[str, tuple[int, int]], bool]:
        """One descending pass over the Hub's most-downloaded models.

        Returns ``({org: (best all-time, best 30-day)}, conclusive)``.

        This answers S2 for every org at once instead of one request per org.
        Because the listing is sorted by 30-day downloads descending, the walk
        can stop the moment it drops below ``min_org_top_downloads``: everything
        past that point is below the threshold anyway. So the cost is set by the
        threshold, not by how many orgs the window happened to contain — and any
        org *absent* from the result provably has no model over the threshold in
        the last 30 days. ``conclusive`` says whether the walk really did reach
        the threshold rather than being cut off by ``org_sweep_limit``.
        """
        floor = int(self.cfg.thresholds.min_org_top_downloads)
        best: dict[str, tuple[int, int]] = {}
        conclusive = False
        kwargs = self._list_kwargs(newest_first_by="downloads", limit=self.org_sweep_limit)
        kwargs["expand"] = ["author", "downloads", "downloadsAllTime"]
        seen = 0
        try:
            for info in self.api.list_models(**kwargs):
                thirty = _int(getattr(info, "downloads", None))
                if thirty < floor:
                    conclusive = True
                    break
                seen += 1
                org = _org_of(info)
                if not org:
                    continue
                candidate = (_int(getattr(info, "downloads_all_time", None)), thirty)
                if candidate > best.get(org, (-1, -1)):
                    best[org] = candidate
        except Exception as exc:
            log.warning("hf: org popularity sweep failed after %d records: %s", seen, exc)
        if not conclusive:
            log.warning(
                "hf: org popularity sweep stopped at org_sweep_limit=%d without reaching "
                "min_org_top_downloads=%d; absence from the map is inconclusive",
                self.org_sweep_limit, floor,
            )
        log.info("hf: org popularity sweep read %d models, mapped %d orgs (conclusive=%s)",
                 seen, len(best), conclusive)
        return best, conclusive

    def _lookup_org(self, org: str) -> tuple[int, int] | None:
        """One org's best download counts, or None when unobtainable.

        Returning None (rather than zeros) is deliberate: the caller omits the
        ``extra`` key entirely so a failed lookup reads as "unknown" instead of
        a measured absence of popularity.
        """
        kwargs = self._list_kwargs(newest_first_by="downloads", limit=ORG_TOP_N, author=org)
        kwargs["expand"] = ["downloads", "downloadsAllTime"]
        try:
            infos = list(self.api.list_models(**kwargs))
        except Exception as exc:
            log.warning("hf: org lookup failed for %s: %s: %s", org, type(exc).__name__, exc)
            return None
        if not infos:
            return None
        return (
            max(_int(getattr(i, "downloads_all_time", None)) for i in infos),
            max(_int(getattr(i, "downloads", None)) for i in infos),
        )

    def _org_top_downloads(self, survivors: Sequence[Any]) -> dict[str, tuple[int, str]]:
        """``{org: (top downloads, which metric)}`` for S2. Frontier orgs excluded.

        Frontier orgs satisfy S2 by membership, so looking them up would be a
        wasted request. Orgs the sweep did not cover get a capped number of
        per-org fallback lookups — the sweep is 30-day-sorted, so a dormant lab
        with a large all-time count but little current traffic can be missing
        from it.
        """
        if not self.org_downloads:
            return {}

        # Aggregate the window's footprint per org: whether it shipped anything
        # naming an architecture, and its best engagement numbers.
        agg: dict[str, dict[str, int]] = {}
        for info in survivors:
            org = _org_of(info)
            if not org or org in self.cfg.frontier_orgs:
                continue
            row = agg.setdefault(org, {"arch": 0, "likes": 0, "downloads": 0})
            if architectures_of(getattr(info, "config", None)):
                row["arch"] = 1
            row["likes"] = max(row["likes"], _int(getattr(info, "likes", None)))
            row["downloads"] = max(row["downloads"], _int(getattr(info, "downloads", None)))
        if not agg:
            return {}

        # The sweep answers every org at once, for three requests.
        sweep, conclusive = self._sweep_popular_orgs()
        out: dict[str, tuple[int, str]] = {}
        for org in agg:
            hit = sweep.get(org)
            if hit is not None:
                out[org] = self._pick_metric(*hit)

        # The sweep is sorted by 30-day downloads, so it is already conclusive
        # for that metric: an org missing from it provably has no model over
        # min_org_top_downloads in the last 30 days. The only thing a per-org
        # lookup can still add is the all-time figure for a *dormant* lab —
        # large lifetime count, little current traffic. Spending a request on
        # every unknown org to find those costs ~85 s per poll and answers
        # nothing for the thousand-odd individual accounts in a normal window,
        # so the fallback is gated on the org having actually shipped an
        # architecture that someone has engaged with. The gate is a property of
        # the data, not of listing order, so which orgs get answered is
        # reproducible from one run to the next — which matters because the
        # backtest calibrates min_org_top_downloads against these numbers.
        eligible = [
            org for org, row in agg.items()
            if org not in out and row["arch"] and (row["likes"] or row["downloads"])
        ]
        eligible.sort(key=lambda o: (agg[o]["likes"], agg[o]["downloads"]), reverse=True)
        budget = max(0, int(self.max_org_lookups))
        if len(eligible) > budget:
            log.warning(
                "hf: %d orgs are eligible for a fallback lookup but max_org_lookups=%d; "
                "%d will report no org_top_downloads (raise the cap to keep S2 reproducible)",
                len(eligible), budget, len(eligible) - budget,
            )
        n_ok = 0
        for org in eligible[:budget]:
            hit = self._lookup_org(org)
            if hit is None:
                continue
            out[org] = self._pick_metric(*hit)
            n_ok += 1
        log.info(
            "hf: org track record for %d/%d non-frontier orgs "
            "(%d from the sweep; %d of %d eligible fallback lookups answered, cap %d)",
            len(out), len(agg), len(out) - n_ok, n_ok, min(len(eligible), budget), budget,
        )
        return out

    @staticmethod
    def _pick_metric(all_time: int, thirty_day: int) -> tuple[int, str]:
        """Prefer the lifetime count; fall back to 30-day when it is unavailable."""
        if all_time:
            return all_time, "downloads_all_time"
        return thirty_day, "downloads"

    def _fetch_config(self, repo_id: str, revision: str | None) -> dict[str, Any] | None:
        """Download and parse one ``config.json``. Never raises."""
        try:
            text = self._fetch_text(repo_id, revision)
        except Exception as exc:  # 404, 429, gated, network, HFValidationError...
            log.warning("hf: could not fetch %s/config.json: %s: %s",
                        repo_id, type(exc).__name__, exc)
            return None
        if text is None:
            return None
        try:
            parsed = json.loads(text)
        except Exception as exc:  # malformed JSON, truncated download
            log.warning("hf: malformed config.json for %s: %s", repo_id, exc)
            return None
        if not isinstance(parsed, dict):
            log.warning("hf: config.json for %s is %s, not an object",
                        repo_id, type(parsed).__name__)
            return None
        return parsed

    # -- assembly -----------------------------------------------------------

    def _prefilter(self, infos: Sequence[Any]) -> tuple[list[Any], int, int]:
        """Phase 1. Returns (survivors, derivative dropped, non-LM dropped).

        The two drop counts are reported separately because they are tuned
        against different things: the derivative count against
        ``DERIVATIVE_PATTERNS``, the non-LM count against the vocabularies
        above. A single combined number would hide which list needs work.
        """
        survivors: list[Any] = []
        n_derivative = n_non_lm = 0
        for info in infos:
            if is_derivative(info.id):
                n_derivative += 1
                continue
            # Only judge modality when the listing excerpt names no architecture.
            # An architecture is the pipeline's primary key: once we have one,
            # the repo is the detector's business, whatever library shipped it.
            if not architectures_of(getattr(info, "config", None)) and is_non_lm_artifact(info):
                log.debug(
                    "hf: dropping non-LM repo %s (library=%s pipeline=%s)",
                    info.id, getattr(info, "library_name", None),
                    getattr(info, "pipeline_tag", None),
                )
                n_non_lm += 1
                continue
            survivors.append(info)
        return survivors, n_derivative, n_non_lm


    def _to_signals(
        self,
        infos: Sequence[Any],
        *,
        phase: str,
        ranks: dict[str, int] | None = None,
    ) -> list[Signal]:
        survivors, n_derivative, n_non_lm = self._prefilter(infos)
        log.info(
            "hf: pre-filter (%s) dropped %d derivative + %d non-LM of %d repos, %d survive",
            phase, n_derivative, n_non_lm, len(infos), len(survivors),
        )

        fetched: dict[str, dict[str, Any] | None] = {}
        for info in self._plan_config_fetches(survivors):
            fetched[info.id] = self._fetch_config(info.id, getattr(info, "sha", None))

        org_downloads = self._org_top_downloads(survivors)

        observed_at = _as_utc(self._clock())
        return [
            self._to_signal(
                info,
                fetched.get(info.id),
                phase=phase,
                observed_at=observed_at,
                org_top=org_downloads.get(_org_of(info) or ""),
                rank=(ranks or {}).get(info.id),
            )
            for info in survivors
        ]

    def _to_signal(
        self,
        info: Any,
        config: dict[str, Any] | None,
        *,
        phase: str,
        observed_at: datetime,
        org_top: tuple[int, str] | None = None,
        rank: int | None = None,
    ) -> Signal:
        repo_id: str = info.id
        hub_config = getattr(info, "config", None) or {}

        arch_ids = architectures_of(config) or architectures_of(hub_config)
        model_type = _model_type_of(config) or _model_type_of(hub_config)

        org = _org_of(info)
        name = repo_id.split("/")[-1]
        sha = getattr(info, "sha", None)
        downloads = _int(getattr(info, "downloads", None))
        likes = _int(getattr(info, "likes", None))
        trending_score = _int(getattr(info, "trending_score", None))
        created_at = _iso(getattr(info, "created_at", None))

        urls = {"hf": f"{HF_BASE}/{repo_id}"}
        if config is not None:
            urls["config"] = f"{HF_BASE}/{repo_id}/blob/{sha or 'main'}/{CONFIG_FILENAME}"

        if phase == "trending":
            evidence = (
                f"trending on HuggingFace (trending_score={trending_score}, "
                f"downloads={downloads}, likes={likes})"
            )
        else:
            evidence = (
                f"new HuggingFace repo created {created_at or 'at an unknown time'} "
                f"(downloads={downloads}, likes={likes})"
            )

        if config is not None:
            config_source = CONFIG_FILENAME
        elif arch_ids or model_type:
            config_source = "hub-listing"
        else:
            config_source = None

        extra: dict[str, Any] = {
            # --- significance gate inputs (F reads these by name) -----------
            "downloads": downloads,
            "likes": likes,
            "downloads_all_time": _int(getattr(info, "downloads_all_time", None)),
            "trending_score": trending_score,
            "trending": phase == "trending",
            # --- provenance / triage colour --------------------------------
            "phase": phase,
            "config_source": config_source,
            "created_at": created_at,
            "last_modified": _iso(getattr(info, "last_modified", None)),
            "library_name": getattr(info, "library_name", None),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "tags": list(getattr(info, "tags", None) or []),
            "gated": getattr(info, "gated", None),
            "private": getattr(info, "private", None),
            "sha": sha,
        }
        # S2's open-world path. Omitted, never zeroed, when unknown: a missing
        # key reads as "not measured" and simply fails to satisfy S2, whereas a
        # 0 would assert a measured absence of popularity. Always absent for
        # frontier orgs, which satisfy S2 by membership.
        if org_top is not None:
            extra["org_top_downloads"] = org_top[0]
            extra["org_top_downloads_basis"] = org_top[1]
        # S4's trending path.
        if rank is not None:
            extra["trending_rank"] = rank

        return Signal(
            source=self.name,
            observed_at=observed_at,
            arch_ids=arch_ids,
            model_type=model_type,
            model_ids=[repo_id],
            org=org,
            display_name=name,
            config=config,
            urls=urls,
            evidence=evidence,
            raw_ref=repo_id,
            extra=extra,
        )


def poll(since: datetime, cfg: DetectorConfig | None = None) -> list[Signal]:
    """Convenience wrapper matching the ``Connector`` protocol's free-function use."""
    return HFConnector(cfg).poll(since)


__all__ = [
    "HFConnector",
    "is_derivative",
    "is_non_lm_artifact",
    "architectures_of",
    "NON_LM_LIBRARIES",
    "NON_LM_PIPELINE_TAGS",
    "poll",
    "LIST_EXPAND",
    "DEFAULT_MAX_LIST",
    "DEFAULT_TRENDING_LIMIT",
    "DEFAULT_MAX_ORG_LOOKUPS",
    "DEFAULT_ORG_SWEEP_LIMIT",
]
