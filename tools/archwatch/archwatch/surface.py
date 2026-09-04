"""Loader for BLIS's support surface — what it parses, what it rejects, where it is wrong.

The three YAML files under ``support-surface/`` are the harvested record of BLIS's
HuggingFace-config surface. This module loads them and answers the three questions the
detector asks of every config, none of which needs an LLM:

* **Would BLIS even run this?**            :meth:`Surface.check_hard_validators`
* **Does this config carry fields BLIS drops on the floor?**  :meth:`Surface.unparsed_fields`
* **Does it look like a mechanism we already know we model badly?** :meth:`Surface.match_gaps`

The grounding fact (see ``PLAN.md``): BLIS never dispatches on ``architectures[]`` or
``model_type``. Everything flows from numeric shape fields into ``sim.ModelConfig``, and
an unrecognized field is silently dropped. So the interesting failure mode is not a
crash — it is a plausible-looking wrong number.

Parse semantics are deliberately a mirror of BLIS's Go code, not of Python's
conveniences:

* ``text_config`` is pivoted onto the top level exactly as ``ParseHFConfig`` does
  (``sim/latency/config.go:222-227``), so a multimodal config is judged on its text tower.
* An integer field is read the way ``HFConfig.GetInt`` reads it
  (``sim/latency/config.go:36-43``): only a JSON *number* counts. A numeric string, a
  bool, a list — anything else — reads as **0**, which is BLIS's "absent" value. That
  silent coercion is exactly the class of bug this tool exists to surface, so it is
  reproduced rather than smoothed over.
* Alias chains resolve to the *first non-zero* value in the order the Go code tries them
  (``HFConfig.mustGetIntFallback``, ``sim/latency/config.go:76-83``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from .config import IGNORED_CONFIG_KEYS

# ---------------------------------------------------------------------------
# Where the YAML lives, and the BLIS constants we mirror
# ---------------------------------------------------------------------------

#: ``tools/archwatch/support-surface`` — resolved relative to this file so the loader
#: works from a clean checkout regardless of the caller's working directory.
DEFAULT_SURFACE_DIR = Path(__file__).resolve().parent.parent / "support-surface"

PARSED_FIELDS_FILE = "parsed-fields.yaml"
KNOWN_GAPS_FILE = "known-gaps.yaml"
KNOWN_ARCHITECTURES_FILE = "known-architectures.yaml"

#: ``sim.MoEMinExperts`` (``sim/model_hardware_config.go:145``). A 1-expert config is
#: dense-equivalent in BLIS, so it must never reach the MoE formulas.
MOE_MIN_EXPERTS = 2

#: The closed ``torch_dtype`` -> bytes/param table (``sim/latency/config.go:320-329``).
#: Anything not a key here yields ``BytesPerParam = 0``, which is a hard rejection.
PRECISION_TO_BYTES_PER_PARAM: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
    "fp8": 1,
    "int4": 1,
    "nf4": 1,
}

#: The closed SwiGLU-family activation set (``sim/latency/kv_capacity.go:58-70``).
#: Case-sensitive, because the Go side is a plain map lookup.
SWIGLU_ACTIVATIONS: frozenset[str] = frozenset({"silu", "swiglu", "geglu", "situ", ""})

#: ``moeExpertCountFields`` — total routed-expert count, in resolution order
#: (``sim/latency/config.go:92-98``).
MOE_EXPERT_COUNT_FIELDS: tuple[str, ...] = (
    "num_experts",
    "moe_num_experts",
    "n_routed_experts",
    "num_local_experts",
    "num_routed_experts",
)
#: ``moeActiveExpertFields`` (``sim/latency/config.go:106``).
MOE_ACTIVE_EXPERT_FIELDS: tuple[str, ...] = ("num_experts_per_tok", "num_experts_per_token")
#: ``moeSharedExpertFields`` (``sim/latency/config.go:107``).
MOE_SHARED_EXPERT_FIELDS: tuple[str, ...] = ("n_shared_experts", "num_shared_experts")

#: The order ``ExtractKVCapacityParams`` scans for a "this is MoE" signal when no total
#: expert count resolved (``sim/latency/kv_capacity.go:779``).
MOE_SIGNAL_FIELDS: tuple[str, ...] = MOE_SHARED_EXPERT_FIELDS + MOE_ACTIVE_EXPERT_FIELDS

#: TP degree archwatch assumes when judging a bare config. The CLI default is 1, at
#: which every positive integer head count divides — see ``head_counts_tp_divisible``.
ASSUMED_TP = 1

_WORD_RE = re.compile(r"[a-z0-9_]+")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedField:
    """One HuggingFace ``config.json`` key BLIS actually reads."""

    name: str
    aliases: tuple[str, ...] = ()
    role: str = "other"  # shape | attention | moe | precision | rope | other
    source_ref: str = ""
    consumed_as: str = ""
    nested: bool = False
    nested_keys: tuple[str, ...] = ()
    notes: str = ""

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


#: A validator whose ``severity`` puts it in Bucket 0 — the run aborts.
FATAL = "fatal"
#: The run completes and reports numbers derived from a misread config. T1, not Bucket 0.
SILENT = "silent"
#: The default backend runs (silently wrong) while another supported backend aborts.
#: Grouped with :data:`SILENT`, deliberately: "BLIS would not run" must not be claimed
#: for a configuration that does run.
MIXED = "mixed"

#: Severities that :meth:`Surface.check_hard_validators` reports. Bucket 0 is exactly
#: this set, and it is a strict subset of the validators — see the severity block at the
#: top of ``parsed-fields.yaml`` for why ``mixed`` is excluded.
BUCKET0_SEVERITIES: frozenset[str] = frozenset({FATAL})
#: Severities that :meth:`Surface.check_silent_validators` reports.
SILENT_SEVERITIES: frozenset[str] = frozenset({SILENT, MIXED})


@dataclass(frozen=True)
class Validator:
    """One rule about a config shape BLIS cannot read correctly.

    Not every such rule is Bucket 0. ``severity`` records what a DEFAULT ``blis run``
    actually does — traced to a ``logrus.Fatalf`` or ``panic`` site, or to the
    demonstrated absence of any guard:

    * ``fatal`` — the process aborts. Bucket 0.
    * ``silent`` — the run completes on a misread config. T1 evidence, not Bucket 0.
    * ``mixed`` — the default backend (trained-physics) completes while roofline aborts.
      Counted with ``silent``.

    This is metadata. The predicate lives in :data:`_VALIDATOR_IMPLS`, keyed by ``id``,
    so the YAML stays readable documentation while behavior stays testable code. An id
    with no implementation is reported on :attr:`Surface.unimplemented_validators`
    rather than silently doing nothing.
    """

    id: str
    fields: tuple[str, ...] = ()
    rule: str = ""
    severity: str = FATAL
    source_ref: str = ""
    also_at: tuple[str, ...] = ()
    failure: str = ""
    observed: str = ""
    fatal_at: str = ""
    silent_when: str = ""
    inert_when: str = ""
    evaluated_by_archwatch: bool = True
    notes: str = ""

    @property
    def is_bucket0(self) -> bool:
        """True when a failure of this rule means BLIS would not run at all."""
        return self.severity in BUCKET0_SEVERITIES


@dataclass(frozen=True)
class Gap:
    """One known approximation: BLIS runs, but the number is wrong."""

    id: str
    mechanism: str = ""
    keywords: tuple[str, ...] = ()
    impact: str = ""
    direction: str = "unknown"  # pessimistic | optimistic | unknown
    scope: str = ""
    seam_refs: tuple[str, ...] = ()
    documented_in: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Config-reading helpers that mirror BLIS's Go semantics
# ---------------------------------------------------------------------------


def pivot_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy ``text_config``'s keys onto the top level, as ``ParseHFConfig`` does.

    Mirrors ``sim/latency/config.go:222-227``. The pivot **overwrites** colliding
    top-level keys (that is why ``model_type`` becomes ``gemma3_text`` for multimodal
    Gemma). Returns a new dict; the caller's config is never mutated.
    """
    merged = dict(config)
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        merged.update(text_cfg)
    return merged


def _as_number(value: Any) -> float | None:
    """The value as a JSON number, or None if BLIS's ``GetInt`` would see nothing.

    Go's ``json.Unmarshal`` decodes every JSON number into ``float64``, and
    ``HFConfig.GetInt`` type-asserts exactly that. So a numeric *string* (``"8"``), a
    bool, a list, or null all read as absent. ``bool`` is excluded explicitly because
    Python's ``bool`` is an ``int`` subclass and would otherwise sneak through as 1/0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def get_int(config: dict[str, Any], key: str) -> int:
    """``HFConfig.GetInt`` / ``MustGetInt(key, 0)``: truncate toward zero, else 0."""
    num = _as_number(config.get(key))
    return 0 if num is None else int(num)


def first_nonzero(config: dict[str, Any], *keys: str) -> int:
    """``HFConfig.mustGetIntFallback(0, keys...)``: first key resolving non-zero."""
    for key in keys:
        value = get_int(config, key)
        if value != 0:
            return value
    return 0


def get_string(config: dict[str, Any], key: str) -> str:
    """``HFConfig.MustGetString(key, "")``: only an actual JSON string counts."""
    value = config.get(key)
    return value if isinstance(value, str) else ""


def resolve_num_experts(config: dict[str, Any]) -> int:
    """``(*HFConfig).ResolveNumExperts`` (``sim/latency/config.go:127-134``).

    First field in :data:`MOE_EXPERT_COUNT_FIELDS` whose value is at least
    :data:`MOE_MIN_EXPERTS`; 0 for a dense model. Note this is *not* "first field
    present" — a total of 1 is skipped and the scan continues, because a single-expert
    config is dense-equivalent in BLIS.
    """
    for key in MOE_EXPERT_COUNT_FIELDS:
        value = get_int(config, key)
        if value >= MOE_MIN_EXPERTS:
            return value
    return 0


def linear_attn_full_layer_count(config: dict[str, Any]) -> int:
    """``(*HFConfig).LinearAttnFullLayerCount`` (``sim/latency/config.go:152-168``)."""
    block = config.get("linear_attn_config")
    if not isinstance(block, dict):
        return 0
    full = block.get("full_attn_layers")
    if not isinstance(full, list) or not full:
        return 0
    return len(full)


#: Task-head suffixes HuggingFace appends to an architecture name. Used ONLY by
#: :func:`_family_stem` for the informational sibling lookup — never by the suppressor.
#: Ordered longest-first so ``ForConditionalGeneration`` is matched before ``Model``.
TASK_HEAD_SUFFIXES: tuple[str, ...] = (
    "forconditionalgeneration",
    "forsequenceclassification",
    "fortokenclassification",
    "forcausallm",
    "formaskedlm",
    "lmheadmodel",
    "model",
)


def _family_stem(arch_id: str) -> str:
    """An architecture name with its task-head suffix removed, lowercased.

    ``Llama4ForConditionalGeneration`` -> ``llama4``. Returns the whole name when it
    carries no recognized suffix, and never returns an empty stem (a name that is
    nothing BUT a suffix, like ``Model``, is left intact) — an empty stem would match
    every other bare-suffix name.
    """
    name = arch_id.strip().lower()
    for suffix in TASK_HEAD_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _is_inert(value: Any) -> bool:
    """True when a config value declares a feature is OFF and so is not evidence.

    ``null``, ``false``, ``0``, and an empty string/list/dict all mean "this knob is not
    engaged". A detector hunting for *new mechanisms* must not fire on a mechanism the
    config explicitly disables — ``"sliding_window": null`` and ``"mlp_only_layers": []``
    describe a model that has neither. This also matches BLIS's own reading, whose alias
    chains treat 0 as absent.
    """
    if value is None or value is False:
        return True
    if isinstance(value, bool):  # True
        return False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value) == 0
    return False


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


@dataclass
class Surface:
    """The loaded support surface. Cheap to hold; construct once per run."""

    parsed_field_names: set[str] = field(default_factory=set)
    parsed_fields: list[ParsedField] = field(default_factory=list)
    hard_validators: list[Validator] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    known_architectures: set[str] = field(default_factory=set)  # lowercased

    # --- additive, beyond PLAN.md's dataclass sketch (see module docs / report) ---
    #: Keys BLIS deliberately never reads because they cannot move a BLIS number.
    #: An extension of the frozen ``config.IGNORED_CONFIG_KEYS``, not a replacement.
    ignored_field_names: set[str] = field(default_factory=set)
    #: Provenance blocks from ``known-architectures.yaml``.
    seeded_from: list[dict[str, Any]] = field(default_factory=list)
    #: Architectures BLIS itself ships configs for / validated against (original casing).
    blis_validated_architectures: list[str] = field(default_factory=list)
    #: Validator ids present in YAML with no implementation here — drift, made visible.
    unimplemented_validators: list[str] = field(default_factory=list)

    # -- lookups ------------------------------------------------------------

    def field_by_name(self, name: str) -> ParsedField | None:
        """The ParsedField that owns ``name``, whether as canonical name or alias."""
        for pf in self.parsed_fields:
            if name in pf.all_names:
                return pf
        return None

    def is_known_architecture(self, arch_id: str) -> bool:
        """EXACT membership in the cold-start seed set, case- and whitespace-insensitive.

        The task suffix is deliberately NOT normalized: ``FooForConditionalGeneration``
        is a different key from ``FooForCausalLM``. That was a considered decision, and
        the arguments for collapsing them are real, so here is the reasoning.

        The case FOR normalizing: BLIS never dispatches on the architecture name — it
        reads numeric shape fields — so a conditional-generation head over the same
        transformer body is not a new architecture as far as BLIS's arithmetic is
        concerned, and treating it as new risks a duplicate issue per multimodal variant.
        Every word of that is true.

        Why exact matching wins anyway, in order of weight:

        1. **Normalization is empirically unnecessary here.** vLLM's registry already
           enumerates BOTH variants for every family that has both — Gemma3, Gemma3n,
           Gemma4, Llama4, DeepseekV4, Glm4v, Glm5Next, Qwen3_5, Qwen3_5Moe, Qwen4Exp,
           MiniMaxM3Sparse, Inkling, MuseGlimmer. All 37 groups that suffix-stripping
           would merge consist entirely of names ALREADY in the seed set, so both
           variants are already suppressed and normalization changes no answer. The
           feared duplicate issue cannot arise from the seed set.
        2. **Where it would change an answer, the change is wrong.** Normalization only
           bites when vLLM lists variant A and a vendor ships variant B. But B's absence
           from the registry is exactly the signal this pipeline exists to detect — vLLM
           has not added support yet, which is what trigger T2 watches for. Collapsing B
           onto A fabricates knowledge nobody has.
        3. **Suppression is the unrecoverable direction.** The pipeline is stateless and
           window-based (``PLAN.md``): a wrongly-suppressed architecture is never
           revisited. A duplicate issue costs a human thirty seconds; a missed zero-day is
           the exact failure this tool was built to prevent. When the two errors are not
           symmetrical, prefer the recoverable one.
        4. **The variants carry different findings even when the numbers agree.** For
           BLIS's committed Llama-4-Scout config, ``unparsed_fields`` reports
           ``vision_config`` for ``Llama4ForConditionalGeneration`` and does not for a
           flattened ``Llama4ForCausalLM`` twin — the text-tower shape resolves
           identically through the ``text_config`` pivot, but the multimodal variant has
           an unmodeled vision tower that the causal one does not. One key per variant
           keeps that finding attached to the variant that has it.
        5. **It matches the frozen contract.** ``base.py`` sets ``Candidate.arch_id`` from
           ``arch_ids[0]`` verbatim and the emitter writes ``issues/<arch_id>.md``. A
           suppressor keyed on a normalized stem while the emitter keys on the raw name
           would suppress candidates under a key no file is ever named after, so the
           stateless "does the file exist" dedup could never agree with it.
        6. **Suffix stripping merges genuinely different models.** It would collapse
           ``JambaForCausalLM`` with ``JambaForSequenceClassification`` and the four
           ``Bert*`` heads onto one key — 432 names down to 387.

        On the over-stripping worry: it does not arise, because nothing is stripped. For
        the record, ``Qwen3MoeForCausalLM`` and ``Qwen3ForCausalLM`` would not have
        collided even under stripping (stems ``qwen3moe`` and ``qwen3``), but they are
        separate keys here for the simpler reason that they are separate strings.

        When family context is wanted for a REPORT rather than for a suppression
        decision, use :meth:`related_known_architectures`, which surfaces the sibling
        without dropping the candidate.
        """
        return bool(arch_id) and arch_id.strip().lower() in self.known_architectures

    def related_known_architectures(self, arch_id: str) -> list[str]:
        """Seed-set names from the same family, ignoring the task suffix. Informational.

        For an unknown ``FooForConditionalGeneration`` this returns ``FooForCausalLM``
        when the registry carries it. It answers "is this a new head on a body we already
        know?" — useful context for an issue stub or for stage 2, and deliberately NOT
        wired into :meth:`is_known_architecture`, so a family resemblance can inform a
        human without silently suppressing a candidate.

        Returns lowercased names, excluding an exact self-match (which
        :meth:`is_known_architecture` already covers), sorted for determinism.
        """
        stem = _family_stem(arch_id)
        if not stem:
            return []
        exact = arch_id.strip().lower()
        return sorted(
            name
            for name in self.known_architectures
            if name != exact and _family_stem(name) == stem
        )

    def gap_by_id(self, gap_id: str) -> Gap | None:
        for gap in self.gaps:
            if gap.id == gap_id:
                return gap
        return None

    # -- the three questions ------------------------------------------------

    def unparsed_fields(self, config: dict[str, Any]) -> list[str]:
        """Top-level config keys BLIS does not parse. Sorted, deterministic.

        ``text_config`` is pivoted first, exactly as ``ParseHFConfig`` does, so a
        multimodal config is judged on its text tower. Excluded from the result:

        * :data:`archwatch.config.IGNORED_CONFIG_KEYS` (frozen boilerplate);
        * :attr:`ignored_field_names` (keys that provably cannot move a BLIS number —
          epsilons, dropout, initializers, RoPE base frequency, bias flags);
        * every canonical name and alias in :attr:`parsed_field_names`;
        * keys whose value is inert — null, false, 0, or empty (see :func:`_is_inert`);
        * anything *inside* a nested dict. We never descend, so a nested key can never
          appear here; ``quantization_config`` / ``rope_scaling`` / ``linear_attn_config``
          are parsed leaves, and ``vision_config`` surfaces as one unparsed key rather
          than fifty.

        A non-empty result is trigger T1's evidence: this config describes something
        BLIS will read past in silence.
        """
        pivoted = pivot_text_config(config)
        out = [
            key
            for key, value in pivoted.items()
            if key not in IGNORED_CONFIG_KEYS
            and key not in self.ignored_field_names
            and key not in self.parsed_field_names
            and not _is_inert(value)
        ]
        return sorted(out)

    def _run_validators(
        self, config: dict[str, Any], severities: Iterable[str]
    ) -> list[tuple[Validator, str]]:
        """Every (validator, message) pair whose severity is in ``severities``."""
        wanted = set(severities)
        pivoted = pivot_text_config(config)
        out: list[tuple[Validator, str]] = []
        for validator in self.hard_validators:
            if validator.severity not in wanted:
                continue
            if not validator.evaluated_by_archwatch:
                continue
            impl = _VALIDATOR_IMPLS.get(validator.id)
            if impl is None:
                continue
            out.extend((validator, message) for message in impl(pivoted, validator))
        return out

    def check_hard_validators(self, config: dict[str, Any]) -> list[str]:
        """Bucket 0. Human-readable failure strings; empty means "BLIS would run it".

        Only ``severity: fatal`` rules are reported — the ones traced to a
        ``logrus.Fatalf`` or a ``panic`` on a default ``blis run``. A ``silent`` or
        ``mixed`` rule means BLIS *runs* and reports a wrong number, which is T1
        evidence rather than Bucket 0; get those from :meth:`check_silent_validators`.
        Claiming "would not run" for a configuration that does run would mislabel real
        models, so the split is enforced here rather than left to the caller.

        Evaluated against the pivoted config with BLIS's own coercion rules, in the
        order the validators appear in ``parsed-fields.yaml``.
        """
        return [message for _, message in self._run_validators(config, BUCKET0_SEVERITIES)]

    def check_silent_validators(self, config: dict[str, Any]) -> list[str]:
        """Shapes BLIS misreads *without* failing: the run completes, the number is wrong.

        These are the ``silent`` and ``mixed`` rules. They belong on the T1 side of the
        line with :meth:`unparsed_fields` — a config that trips one of these is more
        dangerous than one that trips Bucket 0, because nothing tells the user.
        """
        return [message for _, message in self._run_validators(config, SILENT_SEVERITIES)]

    def check_all_validators(self, config: dict[str, Any]) -> list[tuple[str, str]]:
        """``(severity, message)`` for every rule that fires, in YAML order.

        For the emitter, which renders the Bucket-0 verdict and the silent-misread
        findings as separate sections of one stub.
        """
        pairs = self._run_validators(config, BUCKET0_SEVERITIES | SILENT_SEVERITIES)
        return [(v.severity, message) for v, message in pairs]

    def match_gaps(self, config: dict[str, Any]) -> list[Gap]:
        """Gaps whose keywords appear in the config. A pre-hint, not a classification.

        Matching, over the pivoted config and recursively through nested dicts:

        * a single-token keyword matches a config **key** exactly (case-insensitive),
          and only when that key's value is not inert — ``"sliding_window": null``
          describes a model without sliding-window attention;
        * any keyword matches as a **substring** of a string value (``quant_method:
          "fp8"``, ``topk_method: "noaux_tc"``). Multi-word keywords such as
          ``"latent attention"`` exist only for this path.

        Exact key matching is load-bearing: substring matching would fire the explicit
        ``head_dim`` gap on every MLA config that merely declares ``qk_rope_head_dim``.

        Returns gaps in ``known-gaps.yaml`` order so output is deterministic.
        """
        keys, values = _harvest_keys_and_values(pivot_text_config(config))
        matched: list[Gap] = []
        for gap in self.gaps:
            for keyword in gap.keywords:
                kw = keyword.lower()
                if kw in keys or any(kw in value for value in values):
                    matched.append(gap)
                    break
        return matched

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Counts for ``archwatch surface`` and for sanity-checking a fresh checkout."""
        by_role: dict[str, int] = {}
        for pf in self.parsed_fields:
            by_role[pf.role] = by_role.get(pf.role, 0) + 1
        by_severity: dict[str, int] = {}
        for v in self.hard_validators:
            by_severity[v.severity] = by_severity.get(v.severity, 0) + 1
        by_direction: dict[str, int] = {}
        for gap in self.gaps:
            by_direction[gap.direction] = by_direction.get(gap.direction, 0) + 1
        return {
            "parsed_fields": len(self.parsed_fields),
            "parsed_field_names": len(self.parsed_field_names),
            "parsed_fields_by_role": dict(sorted(by_role.items())),
            "ignored_field_names": len(self.ignored_field_names),
            "hard_validators": len(self.hard_validators),
            "hard_validators_by_severity": dict(sorted(by_severity.items())),
            "bucket0_validators": sum(1 for v in self.hard_validators if v.is_bucket0),
            "unimplemented_validators": list(self.unimplemented_validators),
            "gaps": len(self.gaps),
            "gaps_by_direction": dict(sorted(by_direction.items())),
            "known_architectures": len(self.known_architectures),
            "blis_validated_architectures": len(self.blis_validated_architectures),
            "seeded_from": [
                f"{block.get('source', '?')}@{block.get('fetched_at', '?')}"
                f" ({block.get('count', '?')})"
                for block in self.seeded_from
            ],
        }


def _harvest_keys_and_values(config: dict[str, Any]) -> tuple[set[str], list[str]]:
    """Lowercased non-inert keys, and lowercased string values, recursively."""
    keys: set[str] = set()
    values: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and not _is_inert(value):
                    keys.add(key.lower())
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str) and node:
            values.append(node.lower())

    walk(config)
    return keys, values


# ---------------------------------------------------------------------------
# Validator implementations, keyed by the id in parsed-fields.yaml
# ---------------------------------------------------------------------------

ValidatorImpl = Callable[[dict[str, Any], Validator], list[str]]


def _describe_int_read(config: dict[str, Any], key: str) -> str:
    """Why BLIS read 0 out of ``key``: absent, or present-but-unreadable."""
    if key not in config:
        return "absent"
    raw = config[key]
    if _as_number(raw) is None:
        return f"present but not a JSON number ({raw!r}); HFConfig.GetInt reads it as 0"
    return f"{int(_as_number(raw) or 0)}"


def _v_positive_shape_fields(config: dict[str, Any], v: Validator) -> list[str]:
    """The three fields BOTH backends' constructors require. Unconditionally fatal."""
    out: list[str] = []
    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads"):
        value = get_int(config, key)
        if value > 0:
            continue
        out.append(
            f"[{v.id}] {key} must be > 0, resolved {value} "
            f"({_describe_int_read(config, key)}) -- {v.failure} ({v.source_ref})"
        )
    return out


def _v_positive_intermediate_size(config: dict[str, Any], v: Validator) -> list[str]:
    """intermediate_size, through its Falcon/GLM ffn_hidden_size alias."""
    value = first_nonzero(config, "intermediate_size", "ffn_hidden_size")
    if value > 0:
        return []
    detail = " / ".join(
        f"{k}: {_describe_int_read(config, k)}"
        for k in ("intermediate_size", "ffn_hidden_size")
    )
    return [
        f"[{v.id}] intermediate_size must be > 0, resolved {value} ({detail}) "
        f"-- {v.failure} ({v.source_ref})"
    ]


def _v_positive_vocab_size(config: dict[str, Any], v: Validator) -> list[str]:
    value = get_int(config, "vocab_size")
    if value > 0:
        return []
    return [
        f"[{v.id}] vocab_size must be > 0, resolved {value} "
        f"({_describe_int_read(config, 'vocab_size')}) -- {v.failure} ({v.source_ref})"
    ]


def _v_nonnegative_shape_fields(config: dict[str, Any], v: Validator) -> list[str]:
    out: list[str] = []
    for key in ("head_dim", "kv_lora_rank", "qk_rope_head_dim", "first_k_dense_replace"):
        num = _as_number(config.get(key))
        if num is not None and num < 0:
            out.append(
                f"[{v.id}] {key} must be >= 0, got {int(num)} -- {v.failure} ({v.source_ref})"
            )
    return out


def _v_recognized_torch_dtype(config: dict[str, Any], v: Validator) -> list[str]:
    # Go tries torch_dtype as a string, then dtype as a string; neither -> 0 bytes.
    for key in ("torch_dtype", "dtype"):
        value = config.get(key)
        if isinstance(value, str):
            if value in PRECISION_TO_BYTES_PER_PARAM:
                return []
            known = ", ".join(sorted(PRECISION_TO_BYTES_PER_PARAM))
            return [
                f"[{v.id}] {key}={value!r} is not in BLIS's closed precision table "
                f"({known}), so BytesPerParam = 0 -- {v.failure} ({v.source_ref})"
            ]
    present = [k for k in ("torch_dtype", "dtype") if k in config]
    detail = (
        f"{present[0]} is present but not a string ({config[present[0]]!r})"
        if present
        else "neither torch_dtype nor dtype is present"
    )
    return [
        f"[{v.id}] no readable compute dtype: {detail}, so BytesPerParam = 0 "
        f"-- {v.failure} ({v.source_ref})"
    ]


KV_HEAD_FIELDS: tuple[str, ...] = (
    "num_key_value_heads",
    "num_kv_heads",
    "multi_query_group_num",
)


def _v_head_counts_tp_divisible(config: dict[str, Any], v: Validator) -> list[str]:
    """Divisibility by TP. Inert at :data:`ASSUMED_TP` = 1, where everything divides.

    Kept live rather than deleted so the rule goes into effect the moment archwatch is
    given a real TP. The absent / unreadable / negative head-count cases live in
    ``positive_shape_fields``, ``kv_head_count_unreadable`` and
    ``nonnegative_kv_head_count`` — each of which has a *different* severity, which is
    why they are not folded in here.
    """
    if ASSUMED_TP == 1:
        return []
    num_heads = get_int(config, "num_attention_heads")  # pragma: no cover
    num_kv = first_nonzero(config, *KV_HEAD_FIELDS) or num_heads  # pragma: no cover
    out: list[str] = []  # pragma: no cover
    for label, value in (  # pragma: no cover
        ("num_attention_heads", num_heads),
        ("num_key_value_heads", num_kv),
    ):
        if value > 0 and value % ASSUMED_TP != 0:
            out.append(
                f"[{v.id}] {label}={value} is not divisible by TP={ASSUMED_TP} "
                f"-- {v.failure} ({v.source_ref})"
            )
    return out  # pragma: no cover


def _v_kv_head_count_unreadable(config: dict[str, Any], v: Validator) -> list[str]:
    """A KV-head field BLIS cannot read as a number. SILENT: it becomes MHA.

    ``HFConfig.GetInt`` type-asserts float64, so ``"num_key_value_heads": "8"`` reads 0,
    and 0 means "default to num_attention_heads" (``sim/latency/config.go:315-317``).
    Nothing warns. A fractional value is equally silent: ``int(float64)`` truncates.
    """
    out: list[str] = []
    for key in KV_HEAD_FIELDS:
        if key not in config:
            continue
        raw = config[key]
        num = _as_number(raw)
        if num is None:
            heads = get_int(config, "num_attention_heads")
            ratio = (
                f" (would be sized as {heads}-way MHA instead)" if heads > 0 else ""
            )
            out.append(
                f"[{v.id}] {key} is present but not a JSON number ({raw!r}); BLIS reads "
                f"0 and silently falls back to num_attention_heads{ratio} with no error "
                f"and no warning -- {v.source_ref}"
            )
        elif num >= 0 and num != int(num):
            out.append(
                f"[{v.id}] {key}={num} is not an integer; BLIS truncates it to "
                f"{int(num)} silently -- {v.source_ref}"
            )
    return out


def _v_nonnegative_kv_head_count(config: dict[str, Any], v: Validator) -> list[str]:
    """A negative KV-head count. FATAL on the default path only."""
    out: list[str] = []
    for key in KV_HEAD_FIELDS:
        num = _as_number(config.get(key))
        if num is not None and num < 0:
            out.append(
                f"[{v.id}] {key} must be >= 0, got {int(num)} -- {v.failure} "
                f"({v.source_ref})"
            )
    return out


def _v_hidden_size_divisible_by_heads(config: dict[str, Any], v: Validator) -> list[str]:
    if get_int(config, "kv_lora_rank") > 0:  # MLA latent path never forms the quotient
        return []
    if get_int(config, "head_dim") > 0:  # explicit head_dim is used directly
        return []
    hidden = get_int(config, "hidden_size")
    heads = get_int(config, "num_attention_heads")
    if hidden <= 0 or heads <= 0:  # owned by positive_shape_fields
        return []
    if hidden % heads == 0:
        return []
    return [
        f"[{v.id}] hidden_size ({hidden}) is not evenly divisible by "
        f"num_attention_heads ({heads}) and no explicit head_dim is declared "
        f"-- {v.failure} ({v.source_ref})"
    ]


def _v_swiglu_family_hidden_act(config: dict[str, Any], v: Validator) -> list[str]:
    act = get_string(config, "hidden_act")
    if act in SWIGLU_ACTIVATIONS:
        return []
    allowed = ", ".join(sorted(a for a in SWIGLU_ACTIVATIONS if a))
    return [
        f"[{v.id}] hidden_act={act!r} is not a SwiGLU-family activation "
        f"({allowed}, or absent); the weight estimator assumes a 3-matrix gated MLP "
        f"-- {v.failure} ({v.source_ref})"
    ]


def _v_moe_expert_count_resolvable(config: dict[str, Any], v: Validator) -> list[str]:
    if resolve_num_experts(config) >= MOE_MIN_EXPERTS:
        return []
    for key in MOE_SIGNAL_FIELDS:  # BLIS reports the first signal it finds, in this order
        value = get_int(config, key)
        if value > 0:
            tried = ", ".join(MOE_EXPERT_COUNT_FIELDS)
            return [
                f"[{v.id}] {key}={value} signals MoE but no total expert count resolved "
                f">= {MOE_MIN_EXPERTS} from any known spelling ({tried}) "
                f"-- {v.failure} ({v.source_ref})"
            ]
    return []


def _v_moe_active_expert_count_required(config: dict[str, Any], v: Validator) -> list[str]:
    total = resolve_num_experts(config)
    if total <= 1:
        return []
    active = first_nonzero(config, *MOE_ACTIVE_EXPERT_FIELDS)
    if active > 0:
        return []
    tried = ", ".join(MOE_ACTIVE_EXPERT_FIELDS)
    return [
        f"[{v.id}] {total} routed experts resolved but no active-expert count > 0 from "
        f"({tried}) -- {v.failure} ({v.source_ref})"
    ]


def _v_moe_active_not_exceeding_total(config: dict[str, Any], v: Validator) -> list[str]:
    total = resolve_num_experts(config)
    if total <= 1:
        return []
    active = first_nonzero(config, *MOE_ACTIVE_EXPERT_FIELDS)
    if active <= total:
        return []
    return [
        f"[{v.id}] active experts per token ({active}) exceeds the resolved total "
        f"expert count ({total}) -- {v.failure} ({v.source_ref})"
    ]


def _v_moe_total_required_when_active_present(
    config: dict[str, Any], v: Validator
) -> list[str]:
    if resolve_num_experts(config) != 0:
        return []
    active = first_nonzero(config, *MOE_ACTIVE_EXPERT_FIELDS)
    if active <= 0:
        return []
    return [
        f"[{v.id}] active experts per token ({active}) is set but the resolved total "
        f"expert count is 0 -- {v.failure} ({v.source_ref})"
    ]


_VALIDATOR_IMPLS: dict[str, ValidatorImpl] = {
    "positive_shape_fields": _v_positive_shape_fields,
    "positive_intermediate_size": _v_positive_intermediate_size,
    "positive_vocab_size": _v_positive_vocab_size,
    "nonnegative_shape_fields": _v_nonnegative_shape_fields,
    "recognized_torch_dtype": _v_recognized_torch_dtype,
    "head_counts_tp_divisible": _v_head_counts_tp_divisible,
    "kv_head_count_unreadable": _v_kv_head_count_unreadable,
    "nonnegative_kv_head_count": _v_nonnegative_kv_head_count,
    "hidden_size_divisible_by_heads": _v_hidden_size_divisible_by_heads,
    "swiglu_family_hidden_act": _v_swiglu_family_hidden_act,
    "moe_expert_count_resolvable": _v_moe_expert_count_resolvable,
    "moe_active_expert_count_required": _v_moe_active_expert_count_required,
    "moe_active_not_exceeding_total": _v_moe_active_not_exceeding_total,
    "moe_total_required_when_active_present": _v_moe_total_required_when_active_present,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"archwatch support surface: missing {path}")
    with path.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"archwatch support surface: {path} must contain a mapping")
    return doc


def load_surface(dir: Path | str = DEFAULT_SURFACE_DIR) -> Surface:  # noqa: A002
    """Load the three ``support-surface`` YAML files into a :class:`Surface`.

    ``dir`` keeps the name from ``PLAN.md``'s contract so other components can pass it
    by keyword. Raises ``FileNotFoundError`` if a file is missing and ``ValueError`` if
    one is structurally wrong — the surface is the tool's ground truth, so a broken
    file must fail loudly rather than degrade to an empty set that suppresses nothing.
    """
    base = Path(dir)

    parsed_doc = _load_yaml(base / PARSED_FIELDS_FILE)
    gaps_doc = _load_yaml(base / KNOWN_GAPS_FILE)
    archs_doc = _load_yaml(base / KNOWN_ARCHITECTURES_FILE)

    parsed_fields = [
        ParsedField(
            name=str(entry["name"]),
            aliases=_tuple_of_str(entry.get("aliases")),
            role=str(entry.get("role", "other")),
            source_ref=str(entry.get("source_ref", "")),
            consumed_as=str(entry.get("consumed_as", "")),
            nested=bool(entry.get("nested", False)),
            nested_keys=_tuple_of_str(entry.get("nested_keys")),
            notes=str(entry.get("notes", "")),
        )
        for entry in parsed_doc.get("parsed_fields") or []
    ]
    parsed_field_names = {name for pf in parsed_fields for name in pf.all_names}

    ignored_field_names: set[str] = set()
    for group in parsed_doc.get("ignored_fields") or []:
        ignored_field_names.update(_tuple_of_str(group.get("names")))

    overlap = parsed_field_names & ignored_field_names
    if overlap:
        raise ValueError(
            "archwatch support surface: keys listed as both parsed and ignored in "
            f"{PARSED_FIELDS_FILE}: {sorted(overlap)}"
        )

    hard_validators = [
        Validator(
            id=str(entry["id"]),
            fields=_tuple_of_str(entry.get("fields")),
            rule=str(entry.get("rule", "")),
            severity=str(entry.get("severity", FATAL)),
            source_ref=str(entry.get("source_ref", "")),
            also_at=_tuple_of_str(entry.get("also_at")),
            failure=str(entry.get("failure", "")),
            observed=str(entry.get("observed", "")),
            fatal_at=str(entry.get("fatal_at", "")),
            silent_when=str(entry.get("silent_when", "")),
            inert_when=str(entry.get("inert_when", "")),
            evaluated_by_archwatch=bool(entry.get("evaluated_by_archwatch", True)),
            notes=str(entry.get("notes", "")),
        )
        for entry in parsed_doc.get("hard_validators") or []
    ]

    bad_severity = sorted(
        {v.severity for v in hard_validators} - {FATAL, SILENT, MIXED}
    )
    if bad_severity:
        raise ValueError(
            f"archwatch support surface: unknown validator severity in "
            f"{PARSED_FIELDS_FILE}: {bad_severity} (expected {FATAL}/{SILENT}/{MIXED})"
        )
    missing_observed = sorted(v.id for v in hard_validators if not v.observed)
    if missing_observed:
        raise ValueError(
            f"archwatch support surface: hard validators missing an `observed:` line in "
            f"{PARSED_FIELDS_FILE}: {missing_observed}"
        )

    gaps = [
        Gap(
            id=str(entry["id"]),
            mechanism=str(entry.get("mechanism", "")),
            keywords=_tuple_of_str(entry.get("keywords")),
            impact=str(entry.get("impact", "")),
            direction=str(entry.get("direction", "unknown")),
            scope=str(entry.get("scope", "")),
            seam_refs=_tuple_of_str(entry.get("seam_refs")),
            documented_in=str(entry.get("documented_in", "")),
            notes=str(entry.get("notes", "")),
        )
        for entry in gaps_doc.get("gaps") or []
    ]

    raw_archs = _tuple_of_str(archs_doc.get("known_architectures"))
    if not raw_archs:
        raise ValueError(
            f"archwatch support surface: {KNOWN_ARCHITECTURES_FILE} declares no "
            "known_architectures; an empty seed set would flag every model as novel"
        )
    blis_validated = list(_tuple_of_str(archs_doc.get("blis_validated_architectures")))
    known_architectures = {name.strip().lower() for name in raw_archs if name.strip()}
    known_architectures.update(n.strip().lower() for n in blis_validated if n.strip())

    seeded_from = [dict(block) for block in archs_doc.get("seeded_from") or []]

    return Surface(
        parsed_field_names=parsed_field_names,
        parsed_fields=parsed_fields,
        hard_validators=hard_validators,
        gaps=gaps,
        known_architectures=known_architectures,
        ignored_field_names=ignored_field_names,
        seeded_from=seeded_from,
        blis_validated_architectures=blis_validated,
        unimplemented_validators=[
            v.id for v in hard_validators if v.id not in _VALIDATOR_IMPLS
        ],
    )


__all__ = [
    "BUCKET0_SEVERITIES",
    "DEFAULT_SURFACE_DIR",
    "FATAL",
    "Gap",
    "MIXED",
    "MOE_ACTIVE_EXPERT_FIELDS",
    "MOE_EXPERT_COUNT_FIELDS",
    "MOE_MIN_EXPERTS",
    "MOE_SHARED_EXPERT_FIELDS",
    "PRECISION_TO_BYTES_PER_PARAM",
    "ParsedField",
    "SWIGLU_ACTIVATIONS",
    "SILENT",
    "SILENT_SEVERITIES",
    "Surface",
    "TASK_HEAD_SUFFIXES",
    "Validator",
    "first_nonzero",
    "get_int",
    "get_string",
    "linear_attn_full_layer_count",
    "load_surface",
    "pivot_text_config",
    "resolve_num_experts",
]
