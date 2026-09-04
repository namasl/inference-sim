"""Novelty filter, trigger set, and significance gate — the heart of the pipeline.

What this module decides
------------------------
HuggingFace publishes thousands of new repos a day. Almost all of them are
fine-tunes, quantizations, and merges of architectures BLIS already handles. This
module answers two questions, in this order, and drops everything else:

1. **Is this architecture new?** (suppressors, then triggers T1-T4)
2. **Is it worth a human's attention?** (significance gate S1-S4)

Order is load-bearing. Suppressors run *first* and unconditionally: a suppressed
candidate is dropped and the reason recorded, no matter how many triggers it would
have fired. That is what makes the pipeline survive the firehose — without it, a
frontier org's FP8 repack of last month's model fires T3 (frontier org) and S1
(scale) and lands an issue every week.

Why T1 is the highest-value trigger
-----------------------------------
BLIS never dispatches on ``architectures[]`` or ``model_type``; everything flows from
numeric shape fields in ``config.json`` into ``sim.ModelConfig``, and **unknown config
fields are silently dropped**. So a new architecture usually does not crash BLIS — it
runs and produces confidently wrong numbers. A config field BLIS does not parse is
therefore direct evidence of un-modeled mechanism, which is exactly the failure mode
that has no other alarm. T1 fires on that.

The chain
---------
``join_signals()`` collapses raw Signals onto one Candidate per architecture, joining
across sources by union-find over three edge types (architecture id, normalized repo
id, family name) because the sources name the same release in disjoint vocabularies —
see that function. ``evaluate()`` then, per candidate:

**Suppressors** (any one drops the candidate)

=========================  ====================================================
``known_architecture``     ``surface.is_known_architecture(arch_id)`` — in the
                           cold-start seed set, so not news. Skipped when
                           ``cfg.recheck_known_architectures`` is set; see
                           "Re-checking known architectures" below.
``already_reported``       ``emitter.issue_exists(arch_id)`` — a stub is already
                           on disk. This *is* the dedup; the prototype keeps no
                           database. Delegated to the emitter so both sides share
                           one filename sanitizer.
``all_model_ids_derivative``  Every model id from an ``ARTIFACT_SOURCES`` signal
                           matches ``DERIVATIVE_PATTERNS`` (GGUF/AWQ/merge/LoRA/...).
                           Requires at least one such id, so a framework-PR-only
                           candidate is never dropped by vacuous truth, and ids merely
                           *mentioned* by a PR or changelog are not tested at all.
                           An **HF-noise suppressor**: see the exemption below.
``framework_title_only``    Every signal is a framework ``title_only`` PR — model code
                           touched, but no architecture name extractable from anywhere.
                           A PR title is not a model.
``not_a_language_model``   Bucket 0 *and* two or more of the four core transformer
                           dimensions absent under every spelling. BLIS refuses it
                           because it is a speech/video/codec config, not because it is
                           novel. See :class:`LmShapeEvidence`.
``structurally_identical``  A quantized repack of a known architecture: the class
                           name is a known one with a quantizer token spliced in,
                           or ``architectures[]`` names a known class and every
                           difference is confined to ``quantization_config``. See
                           :func:`structural_identity`.
``no_config_uncorroborated``  No Signal carried a config and only one source saw it.
                           An **HF-noise suppressor**: see the exemption below.
=========================  ====================================================

**The curated-source exemption.** ``all_model_ids_derivative`` and
``no_config_uncorroborated`` are HF-noise suppressors: they judge a candidate by the
shape of the HuggingFace repos that happen to exist. A candidate any curated source
(``vllm``/``sglang``/``inferencex``) has seen is exempt from that class as a whole,
because a human wrote reference code or ran a benchmark and that choice outranks the
repo shape. A framework PR carries no ``config.json`` at all, and the only HF repos for
a genuinely new architecture may all be community quants — in both cases the suppressor
would delete the candidate before the trigger phase, so T2/T5 and S3 never get a say.
Exemptions are logged.
=========================  ====================================================

**Triggers** (need >= 1; recorded on ``Candidate.triggers``)

- **T1** the config carries something BLIS cannot read correctly — a field it
  does not parse, or one it parses and silently misreads
  (``surface.check_silent_validators``)
- **T2** a framework support PR (``vllm``/``sglang``) for this unseen architecture
- **T3** a frontier org (``cfg.frontier_orgs``) publishing it
- **T4** corroboration — the same architecture from >= 2 distinct sources
- **T5** a curated benchmark entry (``inferencex``) for an unseen model. Kept separate
  from T2 because the follow-ups differ: a framework PR hands you reference code, a
  benchmark row hands you performance numbers and no architecture. With the
  cross-source join most InferenceX signals merge into an HF or framework candidate and
  fire T4 as well; T5 is what catches the genuine zero-day, where SemiAnalysis is
  benchmarking something before any config or PR is public.

**Significance gate** (need >= 1; recorded on ``Candidate.significance``)

- **S1** ``est_total_params >= thresholds.min_total_params``
- **S2** frontier org, or org top-model downloads over threshold
- **S3** any Signal from ``vllm``/``sglang``/``inferencex``
- **S4** model downloads/likes over threshold, or seen by the trending sweep

Re-checking known architectures
-------------------------------
``known_architecture`` is the filter's largest recall hole, and it is silent. A config
gains fields between point releases under an unchanged ``architectures[]`` string, BLIS
drops the ones it does not parse without a word, and archwatch never looks because the
architecture is "already supported". That is precisely the silent-wrong-numbers failure
T1 exists to catch, arriving in the one disguise the filter does not check.

``cfg.recheck_known_architectures`` (default ``False``) closes it. When set, a seeded
architecture is not dropped by that suppressor; it proceeds through every *other*
suppressor and is then evaluated for **T1 only**, recorded as
:data:`KNOWN_ARCH_TRIGGER` rather than ``"T1"`` so a human can tell the two findings
apart. T2/T3/T4 are deliberately withheld — a frontier org republishing Llama, or a
vLLM PR touching an architecture vLLM already supports, is not news. A re-checked
candidate that fires nothing is dropped as ``known_architecture_nothing_new``, distinct
from ``no_trigger``, so the backtest can weigh the sweep's cost against its yield. The
significance gate applies unchanged: a 1B fine-tune with one novel field is still not
worth a human's attention.

Thresholds are placeholders until the backtest calibrates them; nothing here bakes a
number in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from archwatch.config import (
    DERIVATIVE_PATTERNS,
    IGNORED_CONFIG_KEYS,
    DetectorConfig,
)
from archwatch.connectors.base import Candidate, Signal
from archwatch.sizing import estimate_params, pivot_text_config

log = logging.getLogger("archwatch.novelty")

__all__ = [
    "SurfaceLike",
    "Suppression",
    "EvaluationReport",
    "join_signals",
    "signal_edges",
    "normalize_repo_key",
    "normalize_family_key",
    "evaluate",
    "evaluate_detailed",
    "structural_identity",
    "lm_shape_evidence",
    "LmShapeEvidence",
    "matched_gaps",
    "normalize_arch_key",
    "alias_key",
    "TRIGGER_IDS",
    "CORE_TRIGGER_IDS",
    "KNOWN_ARCH_TRIGGER",
    "SIGNIFICANCE_IDS",
    "ALIAS_JOIN_MARKER",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The numbered triggers, for a candidate whose architecture is not in the seed set.
#: T1-T4 are PLAN.md's; T5 was added once the InferenceX connector landed (see
#: :data:`BENCHMARK_SOURCES`).
CORE_TRIGGER_IDS: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5")

#: Recorded instead of ``"T1"`` when the architecture *is* in the seed set and only
#: surfaced because ``cfg.recheck_known_architectures`` re-examined it. Distinct code
#: because the two findings mean different things to a human: "T1" is "a new
#: architecture BLIS has never seen", "T1-known-arch" is "an architecture BLIS thinks
#: it supports has grown a field BLIS does not parse" — the same silent-wrong-numbers
#: failure, arriving disguised as something already handled.
KNOWN_ARCH_TRIGGER = "T1-known-arch"

#: Every code that counts as a trigger. Use this (not :data:`CORE_TRIGGER_IDS`) to tell
#: triggers apart from provenance markers such as :data:`ALIAS_JOIN_MARKER`.
TRIGGER_IDS: tuple[str, ...] = CORE_TRIGGER_IDS + (KNOWN_ARCH_TRIGGER,)

SIGNIFICANCE_IDS: tuple[str, ...] = ("S1", "S2", "S3", "S4")

#: Recorded in ``Candidate.triggers`` when the candidate was keyed off a normalized
#: display name instead of an ``architectures[]`` entry (PLAN.md section F). It is a
#: provenance marker, NOT a trigger: the gate only counts entries in ``TRIGGER_IDS``.
ALIAS_JOIN_MARKER = "alias-join"

#: Sources that are themselves engineering effort on a model (T2).
FRAMEWORK_SOURCES: frozenset[str] = frozenset({"vllm", "sglang"})

#: Sources whose mere presence is significance (S3): somebody outside the model's own
#: lab spent effort on it.
CURATED_SOURCES: frozenset[str] = frozenset({"vllm", "sglang", "inferencex"})

#: Sources that publish measured performance rather than code (T5). Kept distinct from
#: FRAMEWORK_SOURCES because the two imply different follow-ups: a vLLM PR hands you
#: reference code to read, an InferenceX row hands you throughput numbers and no
#: architecture at all. Collapsing them into one trigger would lose that in the report.
BENCHMARK_SOURCES: frozenset[str] = frozenset({"inferencex"})

#: Framework ``Signal.extra["signal_strength"]`` values, weakest last. ``title_only``
#: means the connector found a PR that touches model code and matched a title pattern
#: but could not extract any architecture name from the registry, the changed classes,
#: the prose, or a new model file. There is no model in it — only a sentence.
FRAMEWORK_STRENGTH_TITLE_ONLY = "title_only"

#: The four dimensions every transformer language model declares under *some* spelling.
#: Broader than :mod:`archwatch.sizing`'s chains on purpose: sizing answers "can BLIS
#: read this config?" and must not invent numbers BLIS could not have read, whereas this
#: answers "is this a transformer LM at all?" and wants every plausible spelling. The
#: two must not be merged — a config whose fields BLIS cannot read is exactly the
#: interesting case, and collapsing the questions would hide it.
LM_CORE_SHAPE_FIELDS: dict[str, tuple[str, ...]] = {
    "num_hidden_layers": ("num_hidden_layers", "n_layer", "n_layers", "num_layers", "num_blocks"),
    "hidden_size": ("hidden_size", "n_embd", "d_model", "dim", "model_dim"),
    "num_attention_heads": ("num_attention_heads", "n_head", "n_heads", "num_heads", "num_q_heads"),
    "intermediate_size": (
        "intermediate_size", "ffn_hidden_size", "ffn_dim", "d_ff", "mlp_dim",
        "moe_intermediate_size",
    ),
}

#: How many of the four core dimensions must be absent, on a Bucket-0 config, before we
#: conclude it never described a language model. Two, not one: a genuinely novel LM might
#: omit an explicit FFN width, but a config missing two of the four is a different model
#: class. The observed margin is wide — audio/video configs miss all four, while
#: ``ParallaxOpen/Vela-Lumen-31M`` (nonstandard spellings throughout) misses none.
MIN_MISSING_CORE_DIMS_FOR_NON_LM = 2

#: A text model's tokenizer has thousands of entries. Below this, the "vocabulary" is a
#: phoneme or codec codebook (``Wav2Vec2``'s is 32). Used only to withhold a *rank*
#: bonus, never to suppress: a byte-level LM legitimately has ~256 tokens, and a false
#: suppression is unrecoverable while a bad rank only costs cap space.
LM_MIN_PLAUSIBLE_VOCAB = 1000

#: Rank bonus for the finding class that justifies this pipeline: BLIS runs the config
#: and reports confident nonsense. Larger than the Bucket-0 bonus because Bucket 0 is
#: loud — the user sees an abort — while this is silent by construction.
SILENTLY_WRONG_BUMP = 4

#: Rank bonus for Bucket 0 on a config that really is a language model: BLIS refuses to
#: run a model someone will want to run. Withheld when the config does not look like an
#: LM, because there "BLIS refuses" means "this was never a language model".
WOULD_NOT_RUN_LM_BUMP = 2

#: Sources whose ``model_ids`` name an *artifact* rather than merely mentioning one.
#: Only these are tested against DERIVATIVE_PATTERNS: a repo id from HuggingFace *is*
#: the thing published, whereas an id mined from a PR diff or a changelog line is a
#: mention — "[Model] Support Qwen3-8B-GGUF loading" describes work on a loader, not a
#: quantized upload. A new connector whose model_ids are real repos belongs here.
ARTIFACT_SOURCES: frozenset[str] = frozenset({"hf"})

#: Config keys that carry quantization metadata rather than architecture. Novelty
#: confined to these is a numeric-format change, not a new mechanism.
QUANT_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "quantization_config",
        "quantization",
        "quant_config",
        "compression_config",
        "compressed_tensors_config",
    }
)

#: Substrings a quantizer appends to (or splices into) an architecture class name when
#: it ships a repack under its own class. Stripping them recovers the base class, which
#: is usually already in the seed set.
QUANT_NAME_TOKENS: tuple[str, ...] = (
    "quantized", "quant", "fp8", "fp4", "mxfp4", "nvfp4", "int8", "int4",
    "w8a8", "w4a16", "awq", "gptq", "gguf", "bnb", "nf4", "marlin", "eetq",
)

# ``Signal.extra`` keys this module reads for S2/S4. ``extra`` is documented as
# never *required* by the detector, so every one of these is optional: when none is
# present the corresponding signal simply does not fire (PLAN.md: "if unavailable,
# S2 is simply not satisfied"). Connectors should prefer the first spelling in each
# tuple.
DOWNLOAD_KEYS: tuple[str, ...] = ("downloads", "downloads_all_time", "download_count")
LIKES_KEYS: tuple[str, ...] = ("likes", "like_count")
ORG_DOWNLOAD_KEYS: tuple[str, ...] = (
    "org_top_downloads",
    "org_top_model_downloads",
    "org_downloads",
)
TRENDING_KEYS: tuple[str, ...] = ("trending", "is_trending", "trending_score", "trending_rank")


# ---------------------------------------------------------------------------
# Surface contract (structural — archwatch.surface is written by another component)
# ---------------------------------------------------------------------------


class SurfaceLike(Protocol):
    """The subset of ``archwatch.surface.Surface`` this module needs.

    Declared as a Protocol rather than imported so novelty can be unit-tested against
    a stub, and so a missing/broken surface module cannot break importing this one.
    ``archwatch.surface.Surface`` satisfies it structurally.
    """

    def unparsed_fields(self, config: dict[str, Any]) -> list[str]: ...
    def check_hard_validators(self, config: dict[str, Any]) -> list[str]: ...
    def check_silent_validators(self, config: dict[str, Any]) -> list[str]: ...
    def is_known_architecture(self, arch_id: str) -> bool: ...
    def match_gaps(self, config: dict[str, Any]) -> list[Any]: ...


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class Suppression:
    """One candidate that did not make it out, and why.

    Every drop produces one of these. The CLI writes them to the run log, and they
    are the artifact the filter gets tuned from — a suppressor that never fires is
    dead weight, and one that fires on a frontier release is a bug.
    """

    arch_id: str
    display_name: str
    stage: str  # "suppressor" | "trigger" | "significance" | "cap"
    reason: str  # stable machine id, e.g. "known_architecture"
    detail: str = ""

    def __str__(self) -> str:
        tail = f": {self.detail}" if self.detail else ""
        return f"{self.arch_id} dropped at {self.stage} [{self.reason}]{tail}"


@dataclass
class EvaluationReport:
    """Full outcome of one ``evaluate()`` pass, for the run log.

    ``dropped`` is complete and unabridged — one record per candidate, so a
    calibration pass can ask "which repo did suppressor X eat?". It is also *big*: a
    real HuggingFace day is ~2,900 candidates of which the low thousands are dropped,
    and about 1,690 carry no architecture at all. Prefer :attr:`counts` /
    :meth:`summary` for anything a human reads; keep the per-record detail for the
    machine-readable half of the run log.
    """

    passed: list[Candidate] = field(default_factory=list)
    dropped: list[Suppression] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        """``{"passed": n, "<reason>": n, ...}`` — the per-stage tally."""
        out: dict[str, int] = {"passed": len(self.passed)}
        for s in self.dropped:
            out[s.reason] = out.get(s.reason, 0) + 1
        return out

    def summary(self) -> str:
        """One line: what came in, what got through, and why the rest did not."""
        total = len(self.passed) + len(self.dropped)
        tally = ", ".join(
            f"{reason}={n}"
            for reason, n in sorted(
                ((r, n) for r, n in self.counts.items() if r != "passed"),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
        return (
            f"{total} candidates -> {len(self.passed)} passed"
            + (f"; dropped: {tally}" if tally else "")
        )


# ---------------------------------------------------------------------------
# Key normalization and join edges
# ---------------------------------------------------------------------------

# Suffixes a quantizer appends to a repo id. Stripped so a repack and its base repo
# reduce to the same key: the join must not treat them as different releases.
QUANT_REPO_SUFFIXES: tuple[str, ...] = (
    "-fp8", "-fp4", "-mxfp4", "-nvfp4", "-awq", "-gptq", "-gguf", "-int4", "-int8",
    "-w4a16", "-w8a8", "-bnb-4bit", "-4bit", "-8bit", "-nf4", "-marlin", "-quantized",
    "-exl2", "-mlx", "-onnx", "-dynamic",
)

# Suffixes that mark a *variant* of one release rather than a different model.
# Deliberately excludes anything size-bearing: stripping "-8b" would merge Qwen3-8B
# with Qwen3-30B, which is the exact false merge this design most needs to avoid.
VARIANT_REPO_SUFFIXES: tuple[str, ...] = (
    "-instruct", "-base", "-chat", "-it", "-thinking", "-nonthinking", "-preview",
    "-exp", "-experimental", "-reasoning", "-pt", "-sft", "-dpo", "-rl", "-hf",
)

# Class-name suffixes stripped to recover the model *family* from an architecture
# class. Longest-first so "MTPModel" is not shortened to "MTP" + leftover "Model".
ARCH_CLASS_SUFFIXES: tuple[str, ...] = (
    "ForConditionalGeneration", "ForSequenceClassification", "ForCausalLM",
    "ForMaskedLM", "LMHeadModel", "MTPModel", "CausalLM", "MTP", "Model",
)

#: A family key shorter than this is too generic to be evidence — ``glm`` would merge
#: every GLM release ever. Below the floor no family edge is emitted at all.
MIN_FAMILY_KEY_LEN = 3

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_arch_key(name: str | None) -> str:
    """Case- and punctuation-insensitive architecture key.

    ``"KimiK3ForCausalLM"``, ``"kimik3forcausallm"`` and ``"Kimi_K3_ForCausalLM"`` are
    the same architecture seen through three sources' spelling conventions; joining on
    the raw string would emit three issues for one model.
    """
    if not name:
        return ""
    return _NON_ALNUM.sub("", name.strip().lower())


def normalize_repo_key(model_id: str | None) -> str:
    """``org/model`` reduced to one key per *release*, org prefix kept.

    All three sources emit repo ids — HF natively, the framework connector from PR
    diffs and bodies, InferenceX from ``configs/*master*.yaml`` ``model:`` fields — so
    this is the one edge every source can actually supply. Quantization and variant
    suffixes are stripped repeatedly (``-instruct-fp8`` sheds both), so
    ``moonshotai/Kimi-K3-Instruct``, ``moonshotai/Kimi-K3-Base`` and
    ``moonshotai/Kimi-K3`` all reduce to ``moonshotai/kimi-k3``.

    **The org prefix is deliberately kept.** It is the only thing preventing
    ``someorg/Mystery-8B`` and ``otherorg/Mystery-8B`` — two unrelated models that
    happen to share a basename — from collapsing into one issue.
    """
    if not model_id:
        return ""
    key = model_id.strip().lower().strip("/")
    for _ in range(8):  # bounded: "-instruct-fp8-dynamic" needs three passes
        for suffix in QUANT_REPO_SUFFIXES + VARIANT_REPO_SUFFIXES:
            if key.endswith(suffix) and len(key) > len(suffix):
                key = key[: -len(suffix)]
                break
        else:
            break
    return key


def normalize_family_key(name: str | None) -> str:
    """A model-family key: architecture class suffixes stripped, then punctuation.

    This is the edge that bridges the sources' disjoint vocabularies directly:
    ``KimiK3ForCausalLM`` -> ``kimik3`` and InferenceX's ``Kimi-K3`` -> ``kimik3``,
    with no repo id needed on either side. ``KimiK3MTPModel`` also lands on ``kimik3``,
    which is right — the MTP head ships with the release.

    It is the loosest of the three edges, so it is bounded on both ends: only
    architecture names and display names feed it (never repo basenames, which would
    merge across orgs), and a result shorter than :data:`MIN_FAMILY_KEY_LEN` is
    discarded as too generic. It does not over-merge real releases:
    ``Qwen3ForCausalLM`` -> ``qwen3`` vs ``Qwen3MoeForCausalLM`` -> ``qwen3moe``, and
    ``GLM-5.2`` -> ``glm52`` vs ``Glm5NextForCausalLM`` -> ``glm5next``.
    """
    if not name:
        return ""
    stem = name.strip()
    for suffix in ARCH_CLASS_SUFFIXES:
        if len(stem) > len(suffix) and stem.lower().endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
            break
    key = _NON_ALNUM.sub("", stem.lower())
    return key if len(key) >= MIN_FAMILY_KEY_LEN else ""


def alias_key(signal: Signal) -> tuple[str, str]:
    """``(family_key, label)`` for a Signal, from its display name or first repo id.

    A framework PR titled "Add support for MiniMax-M3" or an InferenceX changelog row
    names a *model*, not a class. The label is the bare name, not the ``org/model``
    form: it can become the candidate's ``arch_id`` and therefore the issue filename,
    and a slash there would only have to be sanitized back out. The full string stays
    on ``Signal.display_name``.
    """
    label = (signal.display_name or "").strip()
    if not label and signal.model_ids:
        label = signal.model_ids[0].strip()
    if not label:
        label = f"{signal.source}:{signal.raw_ref}".strip(":")
    bare = label.rsplit("/", 1)[-1].strip() if "/" in label else label
    return normalize_family_key(bare), (bare or label)


def signal_edges(signal: Signal) -> list[tuple[str, str]]:
    """Every join edge a Signal offers, as namespaced ``(kind, key)`` pairs.

    Namespaced so an architecture key can never collide with a repo or family key.
    Ordered strongest-first (``arch`` > ``repo`` > ``family``) purely so a merge is
    attributed to the most trustworthy edge that could explain it.
    """
    edges: list[tuple[str, str]] = []

    # (a) architecture identity — the strongest edge. Every arch_id, not just the
    # primary: a framework PR mining both KimiK3ForCausalLM and KimiK3MTPModel from one
    # patch should join either spelling seen elsewhere.
    for arch in signal.arch_ids:
        key = normalize_arch_key(arch)
        if key:
            edges.append(("arch", key))

    # (b) repo identity — the only edge all three sources can supply.
    for model_id in signal.model_ids:
        key = normalize_repo_key(model_id)
        if key:
            edges.append(("repo", key))

    # (c) family name — bridges class names to human names. Fed only from
    # architecture names and the display name; never from a repo basename, which
    # would drop the org prefix and merge unrelated models across orgs.
    family_sources = list(signal.arch_ids)
    if signal.display_name:
        display = signal.display_name.strip()
        # An org-qualified display name stays org-qualified, so two orgs' identically
        # named models cannot meet here. A bare name can only ever match bare.
        family_sources.append(display)
    for name in family_sources:
        key = normalize_family_key(name)
        if key:
            edges.append(("family", key))

    # Deduplicate, preserving strongest-first order.
    seen: set[tuple[str, str]] = set()
    return [e for e in edges if not (e in seen or seen.add(e))]


# ---------------------------------------------------------------------------
# join_signals
# ---------------------------------------------------------------------------

# Preference order for a merged candidate's arch_id. A real architectures[] spelling
# from HuggingFace is ground truth; a framework-mined name is a human's reading of a
# patch; a display name is all that is left. We never synthesize a CamelCase class
# name that no source actually published.
_LABEL_TIERS = (
    "hf primary architectures[0]",
    "hf secondary architectures[]",
    "framework-mined primary arch name",
    "framework-mined secondary arch name",
    "display name (alias path)",
)


def _canonical_label(group: list[Signal]) -> tuple[str, int]:
    """``(arch_id, tier)`` for a joined group. Tier 4 means the alias path was used.

    Within the winning tier the lexicographically first spelling is taken, so the
    result depends on neither signal arrival order nor how many signals carried each
    spelling. (ASCII puts uppercase before lowercase, so the canonical ``CamelCase``
    class name beats a lowercased variant.)
    """
    tiers: tuple[set[str], ...] = (set(), set(), set(), set(), set())
    for sig in group:
        archs = [a.strip() for a in sig.arch_ids if a and a.strip()]
        if archs:
            base = 0 if sig.source == "hf" else 2
            tiers[base].add(archs[0])
            tiers[base + 1].update(archs[1:])
        tiers[4].add(alias_key(sig)[1])
    for index, names in enumerate(tiers):
        if names:
            return min(names), index
    return "", 4


def join_signals(signals: Iterable[Signal]) -> list[Candidate]:
    """Collapse Signals onto one Candidate per architecture, joining across sources.

    The three sources describe one release in three disjoint vocabularies: InferenceX
    knows ``Kimi-K3`` and a checkpoint id but no architecture (its data has no
    ``config.json`` to read one from), the framework connectors know
    ``KimiK3ForCausalLM`` mined from a patch, and HuggingFace knows whatever
    ``architectures[]`` says. Grouping on a single key therefore filed the same release
    as three separate candidates and left T4 (corroboration) — the cheapest and most
    reliable noise killer in the filter — unable to fire at all.

    So the join is a union-find over three edge types (see :func:`signal_edges`): two
    Signals share a Candidate if they share **any** architecture key, normalized repo
    id, or family name. Edges are transitive, which is the point — an InferenceX row
    and a vLLM PR that never mention the same string both meet the HF signal, and so
    meet each other.

    **False merges are the danger this design accepts risk on.** Two architectures
    collapsed into one issue is worse than two issues, so every merge is auditable: the
    edge that caused it is recorded on ``Candidate.join_edges`` (as ``"kind:key"``
    strings, deduplicated, so the emitter can put them in the stub's front matter where
    the backtest can read them) and cross-source merges are logged. The edges are individually bounded too — the
    repo key keeps its org prefix, the family key is fed only by architecture and
    display names and has a minimum length, and no key strips a size token.

    ``arch_id`` is canonical, never first-seen: see :func:`_canonical_label`. A
    candidate that had to fall back to a display name carries
    :data:`ALIAS_JOIN_MARKER` in ``triggers``.

    Candidates come back in first-appearance order.
    """
    sigs = list(signals)

    members: dict[tuple[str, str], list[int]] = {}
    for index, sig in enumerate(sigs):
        edges = signal_edges(sig)
        if not edges:
            log.debug("dropping signal with no usable join key: source=%s raw_ref=%s",
                      sig.source, sig.raw_ref)
            continue
        for edge in edges:
            members.setdefault(edge, []).append(index)

    parent: dict[int, int] = {i: i for group in members.values() for i in group}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    # (member index, edge) for each union that actually merged two groups. The root is
    # resolved afterwards, because a later union can re-root an earlier merge.
    caused: list[tuple[int, tuple[str, str]]] = []
    for edge, group in members.items():
        first = group[0]
        for other in group[1:]:
            root_a, root_b = find(first), find(other)
            if root_a == root_b:
                continue
            # Smallest index becomes the root, so group order follows first appearance.
            low, high = sorted((root_a, root_b))
            parent[high] = low
            caused.append((first, edge))

    # Deduplicated: when three signals merge on one repo key that edge causes two
    # unions, but an auditor wants "which edges explain this group", not multiplicity.
    edges_by_root: dict[int, list[tuple[str, str]]] = {}
    for member, edge in caused:
        group_edges = edges_by_root.setdefault(find(member), [])
        if edge not in group_edges:
            group_edges.append(edge)

    grouped: dict[int, list[Signal]] = {}
    for index, sig in enumerate(sigs):
        if index in parent:
            grouped.setdefault(find(index), []).append(sig)

    out: list[Candidate] = []
    for root in sorted(grouped):
        group = grouped[root]
        label, tier = _canonical_label(group)
        if not label:
            continue
        display = next((s.display_name for s in group if s.display_name), label)
        cand = Candidate(arch_id=label, display_name=display, signals=group)
        if tier == 4:
            cand.triggers.append(ALIAS_JOIN_MARKER)

        merge_edges = edges_by_root.get(root, [])
        cand.join_edges = [f"{kind}:{key}" for kind, key in merge_edges]

        if merge_edges:
            sources = sorted({s.source for s in group})
            detail = ", ".join(f"{kind}:{key}" for kind, key in merge_edges)
            if len(sources) > 1:
                # Cross-source merges are what corroboration rests on and what a false
                # merge would corrupt, so they are the ones a human audits. Same-source
                # merges are routine (every Llama fine-tune in the window joins on one
                # arch key) and would bury this at INFO, so they stay at DEBUG.
                log.info("joined %d signals from %s into %r via %s",
                         len(group), sources, label, detail)
            else:
                log.debug("joined %d signals from %s into %r via %s",
                          len(group), sources, label, detail)
        out.append(cand)
    return out




# -------------------------------------------------------------------------
# Helpers over a Candidate
# ---------------------------------------------------------------------------


def _orgs(cand: Candidate) -> set[str]:
    """Every org this candidate can be attributed to, lowercased.

    Reads ``Signal.org`` and, because not every connector sets it, falls back to the
    owner prefix of each model id.

    Case folding happens **here**, and the comparison in the trigger phase folds
    ``cfg.frontier_orgs`` too, so T3 and S2 cannot miss ``Qwen`` vs ``qwen``. Doing it
    at the one point of comparison rather than trusting every connector to normalize is
    what makes that guarantee hold for connectors written later.
    """
    orgs: set[str] = set()
    for sig in cand.signals:
        if sig.org:
            orgs.add(sig.org.strip().lower())
        for mid in sig.model_ids:
            if "/" in mid:
                orgs.add(mid.split("/", 1)[0].strip().lower())
    return {o for o in orgs if o}


def _model_ids(cand: Candidate, sources: frozenset[str] | None = None) -> list[str]:
    """Distinct model ids across the candidate's signals, optionally source-filtered."""
    seen: list[str] = []
    for sig in cand.signals:
        if sources is not None and sig.source not in sources:
            continue
        for mid in sig.model_ids:
            if mid and mid not in seen:
                seen.append(mid)
    return seen


def _arch_entries(config: dict[str, Any] | None) -> list[str]:
    """``architectures[]`` from a config (after the ``text_config`` pivot)."""
    if not config:
        return []
    entries = pivot_text_config(config).get("architectures")
    if isinstance(entries, str):
        return [entries]
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, str) and e.strip()]
    return []


def _extra_int(cand: Candidate, keys: Iterable[str]) -> int:
    """Largest int found under any of ``keys`` in any Signal's ``extra``, else 0."""
    best = 0
    for sig in cand.signals:
        for key in keys:
            value = sig.extra.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                best = max(best, int(value))
    return best


def _is_trending(cand: Candidate) -> bool:
    """Whether any Signal came from (or was flagged by) the trending sweep."""
    for sig in cand.signals:
        for key in TRENDING_KEYS:
            if key not in sig.extra:
                continue
            value = sig.extra[key]
            if isinstance(value, bool):
                if value:
                    return True
            elif isinstance(value, (int, float)):
                if value > 0:
                    return True
            elif value:
                return True
    return False


def _newest(cand: Candidate) -> float:
    """Most recent ``observed_at`` as a POSIX timestamp; 0.0 when unreadable.

    ``Signal.observed_at`` is treated as UTC: a naive datetime is stamped UTC rather
    than interpreted in the host's local zone, so a scan does not rank differently
    depending on which machine ran it. Computed per-Signal rather than with ``max()``
    over the datetimes because a scan can mix naive and tz-aware timestamps across
    connectors, and comparing those raises.
    """
    best = 0.0
    for sig in cand.signals:
        observed = getattr(sig, "observed_at", None)
        if not isinstance(observed, datetime):
            continue
        try:
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            best = max(best, observed.timestamp())
        except (OSError, ValueError, OverflowError):
            continue
    return best


# ---------------------------------------------------------------------------
# Is this a language model at all?
# ---------------------------------------------------------------------------


@dataclass
class LmShapeEvidence:
    """Whether a config describes a transformer language model, by shape alone.

    BLIS's shape validators turn out to be a decent LM detector, but the polarity is
    counter-intuitive and getting it backwards is costly. For a speech or video repo,
    Bucket 0 means *"this was never a language model"* — not "a novel architecture BLIS
    cannot handle". Ranking those up put five audio and video repos at the top of a live
    HuggingFace run and consumed the whole per-run cap, which is how the error was found.

    So Bucket 0 splits in two, and only one half is interesting:

    * core shape fields **absent** -> not a transformer LM -> suppress
    * core shape fields **present and well-formed**, something else fatal (unrecognized
      dtype, non-SwiGLU activation) -> a real LM BLIS refuses -> rank up
    """

    present: dict[str, str]  # canonical dimension -> the spelling actually found
    missing: list[str]
    vocab_size: int | None

    @property
    def looks_like_lm(self) -> bool:
        """All four core dimensions resolve under some known spelling."""
        return not self.missing

    @property
    def probably_not_lm(self) -> bool:
        return len(self.missing) >= MIN_MISSING_CORE_DIMS_FOR_NON_LM

    @property
    def vocab_plausible(self) -> bool:
        return self.vocab_size is not None and self.vocab_size >= LM_MIN_PLAUSIBLE_VOCAB

    def describe(self) -> str:
        found = ", ".join(f"{canon}<-{spelling}" for canon, spelling in self.present.items())
        return (
            f"core shape dimensions missing: {self.missing or 'none'}"
            + (f"; found: {found}" if found else "")
            + (f"; vocab_size={self.vocab_size}" if self.vocab_size is not None else "")
        )


def lm_shape_evidence(config: dict[str, Any] | None) -> LmShapeEvidence:
    """Resolve the four core transformer dimensions under every known spelling.

    Reads the pivoted config, so a multimodal wrapper is judged on its text tower rather
    than on its top-level keys.
    """
    cfg = pivot_text_config(config) if config else {}
    present: dict[str, str] = {}
    missing: list[str] = []
    for canonical, spellings in LM_CORE_SHAPE_FIELDS.items():
        for spelling in spellings:
            value = cfg.get(spelling)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                present[canonical] = spelling
                break
        else:
            missing.append(canonical)
    vocab = cfg.get("vocab_size")
    vocab_size = (
        int(vocab)
        if isinstance(vocab, (int, float)) and not isinstance(vocab, bool) and vocab > 0
        else None
    )
    return LmShapeEvidence(present=present, missing=missing, vocab_size=vocab_size)


# ---------------------------------------------------------------------------
# Suppressors
# ---------------------------------------------------------------------------


def _signal_strength(signal: Signal) -> str:
    """A framework Signal's ``extra["signal_strength"]``, or "" when absent."""
    value = signal.extra.get("signal_strength")
    return value if isinstance(value, str) else ""


def _derivative_match(model_id: str) -> str | None:
    lowered = model_id.lower()
    for pattern in DERIVATIVE_PATTERNS:
        if pattern in lowered:
            return pattern
    return None


def _dequantized_names(arch_id: str) -> list[str]:
    """Plausible base class names for ``arch_id`` with quantizer tokens removed.

    A repacker that ships ``Qwen3MoeFp8ForCausalLM`` has not invented an
    architecture; stripping ``Fp8`` recovers ``Qwen3MoeForCausalLM``, which the seed
    set already knows. Tokens are matched case-insensitively but the *surviving*
    characters keep their original casing, so the suppression reason names a class a
    human can look up. Returns the single-token strips and the all-tokens strip; the
    caller tests each against the known set, so an over-eager strip that matches
    nothing is harmless.
    """
    out: list[str] = []
    stripped = arch_id
    for token in QUANT_NAME_TOKENS:
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        if not pattern.search(arch_id):
            continue
        one = pattern.sub("", arch_id)
        if one and one != arch_id and one not in out:
            out.append(one)
        stripped = pattern.sub("", stripped)
    if stripped and stripped != arch_id and stripped not in out:
        out.append(stripped)
    return out


def structural_identity(
    arch_id: str,
    config: dict[str, Any] | None,
    unparsed: list[str],
    surface: SurfaceLike,
) -> str | None:
    """Reason string when this is a quantized repack of a known architecture, else None.

    Two independent ways to recognize one, because they fail differently.

    **A — quantizer rename.** ``arch_id`` becomes a known class once quantizer tokens
    are stripped: ``Qwen3MoeFp8ForCausalLM`` -> ``Qwen3MoeForCausalLM``. The name is
    itself the evidence — nobody introducing a new mechanism calls their class
    ``...Fp8...`` — so this fires regardless of what the config contains.

    That "regardless" is load-bearing, and it is why this branch exists at all. A
    repack inherits every field of its base architecture, including fields BLIS does
    not parse: the real support surface reports ``decoder_sparse_step`` and
    ``norm_topk_prob`` as unparsed for *any* Qwen3-MoE config, base or repack. So a
    rule of the form "novelty must be confined to the quantization block" cannot
    suppress a repack of an architecture whose baseline already has unparsed fields —
    it would let every FP8 repack of a frontier MoE model through on T1+T3/S1+S2.
    The fields are attributable to the known base class, not to the repacker.

    **B — ``architectures[]`` names a known class** beside a novel primary entry (a
    wrapper class listed ahead of the real one). Here we have no reference config to
    diff against, so this branch keeps PLAN.md's conservative condition: a
    quantization block must be present, and nothing outside it may be novel. A second
    entry plus genuinely new fields is treated as real novelty and survives.

    Neither branch can fire without a *known* base architecture being named, which is
    what stops this from swallowing a brand-new frontier release that merely ships in
    FP8 using fields BLIS already parses.
    """
    cfg = pivot_text_config(config) if config else {}
    quant_keys = sorted(k for k in QUANT_CONFIG_KEYS if k in cfg)

    # --- A: quantizer rename of a known class ------------------------------
    for base in _dequantized_names(arch_id):
        if surface.is_known_architecture(base):
            tail = f"; config carries {', '.join(quant_keys)}" if quant_keys else ""
            return (
                f"{arch_id!r} is a quantizer rename of known architecture "
                f"{base!r}{tail} — structurally identical, no new mechanism"
            )

    # --- B: a known class listed in architectures[], novelty confined to quant ---
    if not quant_keys:
        return None
    if [f for f in unparsed if f not in QUANT_CONFIG_KEYS]:
        return None
    for entry in _arch_entries(cfg):
        if normalize_arch_key(entry) == normalize_arch_key(arch_id):
            continue
        if surface.is_known_architecture(entry):
            return (
                f"architectures[] also names the known architecture {entry!r} and every "
                f"difference from it is confined to {', '.join(quant_keys)}"
            )
    return None


def _emitter_dedup(issues_dir: Path | None) -> Callable[[str], bool]:
    """The default ``already_reported`` predicate: ``emitter.issue_exists``.

    The emitter owns filename sanitization — an arch id containing a space, slash,
    colon or non-ASCII character does **not** land at ``issues/<arch_id>.md`` (it is
    rewritten and given a digest suffix). Building that path here with an f-string
    would look for a file the emitter never writes, so the dedup would silently stop
    working and the same stub would be re-emitted every run. Delegating keeps both
    sides on one sanitizer by construction.

    Imported lazily so ``archwatch.novelty`` stays importable (and unit-testable)
    without the emitter; if the import fails the suppressor degrades to "nothing has
    been reported yet", which over-reports rather than silently under-reports.
    """
    try:
        from archwatch.emitter import issue_exists
    except Exception:  # pragma: no cover - emitter missing/broken
        log.warning(
            "archwatch.emitter unavailable; the already-reported suppressor is disabled "
            "for this run and stubs may be re-emitted"
        )
        return lambda _arch_id: False
    return lambda arch_id: issue_exists(arch_id, issues_dir)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _strength(cand: Candidate, lm: LmShapeEvidence | None = None) -> int:
    """Rank key: how much independent evidence backs this candidate.

    Triggers and corroboration weigh double because they are *independent* reasons to
    look, whereas the significance signals largely covary (a frontier org's new model is
    usually also big and usually also gets a vLLM PR).

    The finding bonuses are ordered by how *quiet* the failure is, which is the opposite
    of how loud it looks:

    * ``silently_wrong`` (silent failures, clean Bucket 0) gets the largest bonus. BLIS
      runs the config and reports confident nonsense; nothing else in the world warns
      anyone. This is the class the pipeline exists for, and it previously got no bonus
      at all while Bucket 0 got three.
    * ``would_not_run`` on a config that really is an LM gets a smaller bonus. BLIS
      refusing a model people will want to run is urgent, but it is at least visible.
    * ``would_not_run`` on anything else gets nothing. For a speech or video repo,
      Bucket 0 means "this was never a language model" — see :class:`LmShapeEvidence`.
      Such candidates are normally suppressed outright; this covers the residue that
      keeps all four core dimensions (an encoder, or a codec with a phoneme "vocabulary")
      and so cannot be separated from a real LM by shape alone.

    ``lm`` is passed in rather than recomputed so ranking cannot disagree with the
    suppressor that used the same evidence.
    """
    real_triggers = [t for t in cand.triggers if t in TRIGGER_IDS]
    score = 2 * len(real_triggers) + len(cand.significance) + 2 * len(cand.sources)
    if cand.silent_failures and not cand.bucket0_failures:
        score += SILENTLY_WRONG_BUMP
    elif cand.would_not_run:
        if lm is not None and lm.looks_like_lm and lm.vocab_plausible:
            score += WOULD_NOT_RUN_LM_BUMP
    return score


def evaluate_detailed(
    cands: list[Candidate],
    surface: SurfaceLike,
    cfg: DetectorConfig,
    *,
    issues_dir: Path | None = None,
    already_reported: Callable[[str], bool] | None = None,
) -> EvaluationReport:
    """:func:`evaluate`, plus the record of everything that was dropped and why.

    ``evaluate()`` is the interface PLAN.md fixes; this is the same pass with the run
    log attached, because "which suppressor fired how often" is the only way to tune
    the filter.
    """
    report = EvaluationReport()
    if already_reported is None:
        already_reported = _emitter_dedup(issues_dir)

    def drop(cand: Candidate, stage: str, reason: str, detail: str = "") -> None:
        # DEBUG, not INFO: on a real HuggingFace window this fires thousands of times
        # (~1,690 no-architecture repos a day alone). One line per drop would bury the
        # aggregated tally that is actually useful for tuning, so the per-candidate
        # detail lives in report.dropped and only the summary is logged at INFO.
        s = Suppression(cand.arch_id, cand.display_name, stage, reason, detail)
        report.dropped.append(s)
        log.debug("%s", s)

    # (candidate, lm-shape evidence) so ranking reuses exactly the evidence the
    # not_a_language_model suppressor judged on, rather than recomputing and possibly
    # disagreeing with it.
    survivors: list[tuple[Candidate, LmShapeEvidence]] = []
    for cand in cands:
        # --- suppressors: cheap identity checks first -----------------------
        # A known architecture is normally dropped here, unexamined. That is the
        # filter's largest recall hole: configs gain fields between point releases
        # under an unchanged architecture string, and BLIS silently drops fields it
        # does not parse — so the "silent wrong numbers" case can arrive disguised as
        # an architecture already handled. cfg.recheck_known_architectures keeps such a
        # candidate alive for a T1-only re-examination (see the trigger phase below);
        # every other suppressor still applies to it.
        known = surface.is_known_architecture(cand.arch_id)
        if known and not cfg.recheck_known_architectures:
            drop(cand, "suppressor", "known_architecture",
                 f"{cand.arch_id!r} is in the cold-start seed set")
            continue

        if already_reported(cand.arch_id):
            drop(cand, "suppressor", "already_reported",
                 f"a stub for {cand.arch_id!r} is already on disk")
            continue

        # --- a PR title is not a model -------------------------------------
        # The framework connector emits title_only signals: a PR that touches model code
        # and matched a title pattern, but from which no architecture name could be
        # extracted from the registry, the changed classes, the prose, or a new model
        # file. Those route through the alias path and key on the PR title, so T2 and S3
        # always fire and the stub ends up named "Find attention with a fuser and attach
        # vLLM's layer to it" or "gfx1250 on ROCM 10" — five out of five on a live run.
        #
        # Gated on the connector's own metadata rather than on whether the name reads
        # like a sentence, and narrowed further than strictly necessary: it fires only
        # when EVERY signal is a title_only framework signal. Any registry-,
        # model_class-, prose- or new_model_file-strength signal, any HF config, any
        # benchmark row, and the candidate is kept. A false suppression here is
        # unrecoverable, so the rule refuses to guess.
        framework_sigs = [sg for sg in cand.signals if sg.source in FRAMEWORK_SOURCES]
        if (
            ALIAS_JOIN_MARKER in cand.triggers
            and framework_sigs
            and len(framework_sigs) == len(cand.signals)
            and all(_signal_strength(sg) == FRAMEWORK_STRENGTH_TITLE_ONLY
                    for sg in framework_sigs)
        ):
            drop(cand, "suppressor", "framework_title_only",
                 f"only evidence is {len(framework_sigs)} title_only PR(s) with no "
                 f"extractable architecture: "
                 + "; ".join(f"{sg.source}#{sg.raw_ref}" for sg in framework_sigs))
            continue

        # --- HF-noise suppressors, and the curated-source exemption ----------
        # Both suppressors below exist to drop HuggingFace noise: repos that are
        # archless and metadata-free, or that are merely repacks. A curated source is
        # different in kind — a vLLM/SGLang PR or an InferenceX benchmark row means a
        # *person* decoded the architecture and wrote reference code or measured it.
        # That choice outranks whatever HF repos happen to exist, so a candidate any
        # curated source has seen is exempt from this class of suppressor as a whole.
        #
        # This is a class-wide rule rather than two independent conditions because the
        # same bug appeared twice: applying an HF-noise test to a curated signal deletes
        # the candidate before the trigger phase, so T2/T5 and S3 never get a say, and
        # the purest zero-day evidence we have is the thing that gets thrown away.
        curated = sorted({sg.source for sg in cand.signals} & CURATED_SOURCES)

        def exempt(reason: str, detail: str) -> bool:
            """True when a curated source spares the candidate from an HF-noise drop."""
            if not curated:
                return False
            # Volume is bounded by the number of curated signals in the window (tens,
            # not thousands), so this is safe at INFO — and it is worth seeing: it says
            # this candidate survived only because someone outside HF worked on it.
            log.info("exempting %r from %s (%s also saw it): %s",
                     cand.arch_id, reason, curated, detail)
            return True

        # Only ids from ARTIFACT_SOURCES: a repo id from HF is the artifact, while one
        # mined from a PR diff or changelog line is a mention. Testing a mention would
        # let a PR titled "[Model] Support Qwen3-8B-GGUF loading" suppress the very
        # architecture the PR adds support for.
        model_ids = _model_ids(cand, ARTIFACT_SOURCES)
        if model_ids:
            hits = [(mid, _derivative_match(mid)) for mid in model_ids]
            if all(pat for _, pat in hits):
                detail = "; ".join(f"{mid} matches {pat!r}" for mid, pat in hits)
                if not exempt("all_model_ids_derivative", detail):
                    drop(cand, "suppressor", "all_model_ids_derivative", detail)
                    continue

        config = cand.config
        if config is None and not cand.corroborated:
            detail = f"no config from any source and only {cand.sources} saw it"
            if not exempt("no_config_uncorroborated", detail):
                drop(cand, "suppressor", "no_config_uncorroborated", detail)
                continue

        # --- deterministic findings ----------------------------------------
        # Recomputing IGNORED_CONFIG_KEYS on top of surface.unparsed_fields() is
        # deliberate belt-and-braces: T1 firing on `transformers_version` would be a
        # silent, high-volume false positive, so the exclusion is enforced at both
        # ends of the contract.
        unparsed: list[str] = []
        bucket0: list[str] = []
        silent: list[str] = []
        if config is not None:
            unparsed = [f for f in surface.unparsed_fields(config) if f not in IGNORED_CONFIG_KEYS]
            bucket0 = list(surface.check_hard_validators(config))
            silent = list(surface.check_silent_validators(config))
            # The two sets must be disjoint: Candidate.would_not_run derives from
            # bucket0_failures, so a leak would make the emitter present a silent
            # misread as fatal — the exact confusion the split exists to prevent. The
            # surface partitions by severity so this cannot normally happen; if it does,
            # the fatal classification wins (it is the more severe claim) and the
            # contract violation is reported rather than silently absorbed.
            overlap = [m for m in silent if m in bucket0]
            if overlap:
                log.warning(
                    "surface contract violation: %d finding(s) reported as both fatal and "
                    "silent for %r; keeping them in bucket0_failures only: %s",
                    len(overlap), cand.arch_id, overlap)
                silent = [m for m in silent if m not in bucket0]
        est = estimate_params(config)

        cand.unparsed_fields = unparsed
        cand.bucket0_failures = bucket0
        cand.silent_failures = silent
        cand.est_total_params = est.total
        cand.est_active_params = est.active

        # --- suppressor: not a language model (needs the findings) ----------
        # Bucket 0 plus two or more of the four core transformer dimensions absent under
        # every known spelling: this config never described a language model. Suppressed
        # rather than merely down-ranked because these are pure noise AND they displace
        # real findings — the live run's entire per-run cap went to audio and video repos
        # while twelve other candidates were dropped as over_cap.
        lm = lm_shape_evidence(config)
        if bucket0 and lm.probably_not_lm:
            drop(cand, "suppressor", "not_a_language_model",
                 f"{len(bucket0)} Bucket-0 failure(s) and {lm.describe()} — BLIS refuses "
                 f"it because it is not a transformer LM, not because it is novel")
            continue

        # --- suppressor: structural identity (needs the findings) -----------
        why = structural_identity(cand.arch_id, config, unparsed, surface)
        if why:
            drop(cand, "suppressor", "structurally_identical", why)
            continue

        # --- triggers -------------------------------------------------------
        frontier = _orgs(cand) & {o.lower() for o in cfg.frontier_orgs}
        fired: list[str] = []
        # T1 evidence is "the config carries something BLIS cannot read correctly", and
        # that is two things, not one: a field BLIS never parses, or a field it parses
        # and misreads. A silent validator failure can fire with NO unparsed field at
        # all — num_key_value_heads given as the string "8", or num_experts_per_tok
        # exceeding the total — both recognized spellings, both silently misread. Gating
        # T1 on unparsed_fields alone would drop those candidates at no_trigger and lose
        # the highest-value finding class in the pipeline, which is the same
        # invisible-end-to-end failure as leaving silent_failures unpopulated.
        t1_evidence = bool(unparsed or silent)
        if known:
            # Re-check path: T1 ONLY. T2/T3/T4/T5 would fire on every fine-tune of every
            # seeded architecture — a frontier org republishing Llama is not news, and
            # a vLLM PR touching an architecture vLLM already supports is not either.
            # The one thing worth knowing is whether the config grew something BLIS
            # cannot read, so that is the only question we ask. A seeded architecture
            # that now trips a silent validator is exactly the trillion-parameter MoE
            # simulating as dense behind one logrus.Warnf.
            if t1_evidence:
                fired.append(KNOWN_ARCH_TRIGGER)
        else:
            if t1_evidence:
                fired.append("T1")
            if any(s.source in FRAMEWORK_SOURCES for s in cand.signals):
                fired.append("T2")
            if frontier:
                fired.append("T3")
            if cand.corroborated:
                fired.append("T4")
            if any(s.source in BENCHMARK_SOURCES for s in cand.signals):
                fired.append("T5")

        # Preserve the alias-join marker while keeping the trigger list ordered.
        markers = [t for t in cand.triggers if t not in TRIGGER_IDS]
        cand.triggers = fired + markers

        if not fired:
            if known:
                # Distinct from "no_trigger" so the run log can separate the cost of
                # the re-check sweep (known, nothing new) from its yield (known,
                # unparsed fields found) when the backtest calibrates the flag.
                drop(cand, "trigger", "known_architecture_nothing_new",
                     f"{cand.arch_id!r} is in the seed set and its config carries no "
                     f"unparsed field and no silent-misread finding")
            else:
                drop(cand, "trigger", "no_trigger",
                     f"nothing new: sources={cand.sources}, unparsed_fields=[], "
                     f"silent_failures=[], orgs={sorted(_orgs(cand))}")
            continue

        # --- significance gate ---------------------------------------------
        sig: list[str] = []
        min_total = cfg.thresholds.min_total_params
        if est.total is not None and est.total >= min_total:
            sig.append("S1")

        org_downloads = _extra_int(cand, ORG_DOWNLOAD_KEYS)
        if frontier or (org_downloads and org_downloads >= cfg.thresholds.min_org_top_downloads):
            sig.append("S2")

        if any(s.source in CURATED_SOURCES for s in cand.signals):
            sig.append("S3")

        downloads = _extra_int(cand, DOWNLOAD_KEYS)
        likes = _extra_int(cand, LIKES_KEYS)
        if (
            (downloads and downloads >= cfg.thresholds.min_model_downloads)
            or (likes and likes >= cfg.thresholds.min_model_likes)
            or _is_trending(cand)
        ):
            sig.append("S4")

        cand.significance = sig
        if not sig:
            drop(cand, "significance", "insignificant",
                 f"triggers={fired} but est_total_params={est.total}, "
                 f"downloads={downloads}, likes={likes}, orgs={sorted(_orgs(cand))}")
            continue

        survivors.append((cand, lm))

    # --- rank, then cap ----------------------------------------------------
    ranked = sorted(
        survivors,
        key=lambda pair: (-_strength(*pair), -_newest(pair[0]), pair[0].arch_id.lower()),
    )
    cap = max(0, cfg.max_issues_per_run)
    report.passed = [cand for cand, _ in ranked[:cap]]
    for cand, lm in ranked[cap:]:
        report.dropped.append(
            Suppression(
                cand.arch_id,
                cand.display_name,
                "cap",
                "over_cap",
                f"passed but ranked below max_issues_per_run={cfg.max_issues_per_run} "
                f"(strength {_strength(cand, lm)})",
            )
        )
    if len(ranked) > len(report.passed):
        log.debug(
            "capped run: %d candidates passed, emitting the strongest %d",
            len(ranked),
            len(report.passed),
        )
    log.info("%s", report.summary())
    return report


def evaluate(
    cands: list[Candidate],
    surface: SurfaceLike,
    cfg: DetectorConfig,
    *,
    issues_dir: Path | None = None,
    already_reported: Callable[[str], bool] | None = None,
) -> list[Candidate]:
    """Apply suppressors, then triggers, then the significance gate.

    Populates ``triggers``/``significance``/``unparsed_fields``/``bucket0_failures``/
    ``est_total_params``/``est_active_params`` on every candidate that reaches the
    findings stage (in place, so the run log can inspect rejects too). Returns only
    the candidates that pass, strongest-then-newest first, capped at
    ``cfg.max_issues_per_run``.

    ``issues_dir`` defaults to the emitter's ``default_issues_dir()`` and exists so
    tests and ``--out DIR`` can point the dedup somewhere else. ``already_reported``
    overrides the dedup predicate entirely (it defaults to
    ``archwatch.emitter.issue_exists``, so the two components can never disagree
    about how an arch id maps to a filename).
    """
    return evaluate_detailed(
        cands, surface, cfg, issues_dir=issues_dir, already_reported=already_reported
    ).passed


def matched_gaps(cand: Candidate, surface: SurfaceLike) -> list[Any]:
    """Known BLIS gaps whose keywords appear in this candidate's config.

    A cheap pre-hint for the stage-2 skill, **not** a classification. Offered as a
    function because ``Candidate`` (frozen contract) has no field to carry it.
    """
    config = cand.config
    if config is None:
        return []
    return list(surface.match_gaps(config))
