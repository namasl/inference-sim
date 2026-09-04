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
    """The repo owner. Canonical repos (``gpt2``) have no namespace at all."""
    author = getattr(info, "author", None)
    if author:
        return author
    repo_id = getattr(info, "id", "") or ""
    return repo_id.split("/")[0] if "/" in repo_id else None


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
    ) -> None:
        self.cfg = cfg or DEFAULTS
        self.max_list = max_list
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
        return self._to_signals(infos, phase="trending")

    # -- phase 1: listing ---------------------------------------------------

    def _list_kwargs(self, *, newest_first_by: str, limit: int | None) -> dict[str, Any]:
        """Sort/limit kwargs for the installed ``huggingface_hub``.

        The v1.x API renamed the sort keys to snake_case (``created_at``,
        ``trending_score``) and **removed** ``direction`` entirely — these sorts
        are descending server-side. 0.x used camelCase keys plus
        ``direction=-1``. Probe the signature rather than guessing.
        """
        kwargs: dict[str, Any] = {"expand": list(LIST_EXPAND), "limit": limit}
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
        org = (_org_of(info) or "").lower()
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


    def _to_signals(self, infos: Sequence[Any], *, phase: str) -> list[Signal]:
        survivors, n_derivative, n_non_lm = self._prefilter(infos)
        log.info(
            "hf: pre-filter (%s) dropped %d derivative + %d non-LM of %d repos, %d survive",
            phase, n_derivative, n_non_lm, len(infos), len(survivors),
        )

        fetched: dict[str, dict[str, Any] | None] = {}
        for info in self._plan_config_fetches(survivors):
            fetched[info.id] = self._fetch_config(info.id, getattr(info, "sha", None))

        observed_at = _as_utc(self._clock())
        return [
            self._to_signal(info, fetched.get(info.id), phase=phase, observed_at=observed_at)
            for info in survivors
        ]

    def _to_signal(
        self,
        info: Any,
        config: dict[str, Any] | None,
        *,
        phase: str,
        observed_at: datetime,
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
            extra={
                # --- significance gate inputs (F reads these by name) -------
                "downloads": downloads,
                "likes": likes,
                "downloads_all_time": _int(getattr(info, "downloads_all_time", None)),
                "trending_score": trending_score,
                "trending": phase == "trending",
                # --- provenance / triage colour ----------------------------
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
            },
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
]
