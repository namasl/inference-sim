"""The one contract between connectors and the detector.

Every connector normalizes its source into Signal records, so the detector never
needs to know which source it is looking at. This module is the frozen interface:
implementations must not change these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Protocol

# ---------------------------------------------------------------------------
# Signal — what a connector emits
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """One observation from one source about one model/architecture.

    A Signal is raw evidence. It has not been filtered, judged, or joined with
    other sources yet — that is the detector's job.
    """

    source: str  # "hf" | "vllm" | "sglang" | "inferencex"
    observed_at: datetime

    # --- identity -----------------------------------------------------------
    # arch_ids is the primary key of the whole pipeline. It comes from the
    # config's architectures[] array where available. May be empty when a source
    # discusses a model without exposing a config (see alias fallback in PLAN.md).
    arch_ids: list[str] = field(default_factory=list)
    model_type: str | None = None
    model_ids: list[str] = field(default_factory=list)
    org: str | None = None
    display_name: str = ""

    # --- payload ------------------------------------------------------------
    config: dict[str, Any] | None = None  # raw config.json when obtainable
    urls: dict[str, str] = field(default_factory=dict)  # {hf, pr, commit, docs}
    evidence: str = ""  # one line: why this source flagged it
    raw_ref: str = ""  # sha / PR number / file path, for provenance

    # --- source-specific extras (never relied on by the detector) -----------
    extra: dict[str, Any] = field(default_factory=dict)

    def primary_arch(self) -> str | None:
        """The join key, or None when this Signal carries no architecture."""
        return self.arch_ids[0] if self.arch_ids else None


class Connector(Protocol):
    """Connectors are stateless, side-effect-free, and independently failable.

    poll() must not raise for ordinary source problems (rate limits, an outage,
    an unexpected payload shape): log and return what it has. The detector
    tolerates a partial scan; the next overlapping window recovers.
    """

    name: str

    def poll(self, since: datetime) -> Iterable[Signal]: ...


# ---------------------------------------------------------------------------
# Candidate — what survives the filter and becomes an issue
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """An architecture that cleared novelty and significance.

    One Candidate per architecture, joined from every Signal that mentioned it.
    """

    arch_id: str  # the primary key, e.g. "KimiK3ForCausalLM"
    display_name: str
    signals: list[Signal] = field(default_factory=list)

    # why it got through — recorded so the issue can explain itself
    triggers: list[str] = field(default_factory=list)  # e.g. ["T1", "T4"]
    significance: list[str] = field(default_factory=list)  # e.g. ["S1", "S3"]

    # findings from the deterministic checks (no LLM involved)
    unparsed_fields: list[str] = field(default_factory=list)
    bucket0_failures: list[str] = field(default_factory=list)
    est_total_params: int | None = None
    est_active_params: int | None = None

    @property
    def sources(self) -> list[str]:
        return sorted({s.source for s in self.signals})

    @property
    def corroborated(self) -> bool:
        """True when two or more distinct sources saw this architecture."""
        return len(self.sources) >= 2

    @property
    def config(self) -> dict[str, Any] | None:
        """The richest config any Signal carried for this architecture."""
        best: dict[str, Any] | None = None
        for s in self.signals:
            if s.config and (best is None or len(s.config) > len(best)):
                best = s.config
        return best

    @property
    def would_not_run(self) -> bool:
        return bool(self.bucket0_failures)
