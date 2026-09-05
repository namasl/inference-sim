---
name: archwatch-deep-dive
description: Use when analyzing an archwatch stub issue for a newly detected model architecture - classifies it against BLIS's support surface into buckets 0-3, estimates fidelity impact, and appends the analysis to the stub. Stage 2 of the archwatch pipeline.
---

# archwatch deep-dive (stage 2)

You are analyzing ONE newly detected model architecture and reporting where — if anywhere
— BLIS would need to change to support it.

**This is tracking-only.** You write no simulator code, produce no patch, and modify nothing
outside the stub file you are appending to. Your output is analysis a human acts on.

## The division of labor

Stage 1 (the detector) already answered **"is this architecture new?"** — deterministically,
without an LLM. Your job is the question no rule can answer: **"does BLIS support it, and
if not, where exactly?"**

Stage 1 computes these deterministically, and you should **verify, not trust** — each has
been observed wrong in practice:

- **the Bucket 0 verdict.** Observed false positive: a config was rejected for
  `hidden_act: "gelu"` as "not SwiGLU-family", but the model's own `modeling_*.py` showed a
  3-matrix GEGLU (`down_proj(act(gate_proj(x)) * up_proj(x))`) — `gelu` names the gate
  nonlinearity exactly as Llama's `silu` names SwiGLU's. BLIS aborts on a model it would
  price correctly. Always read the modeling code before endorsing a Bucket 0 reason.
- **which config fields BLIS does not parse.** Reliable, but names only — you will usually
  need the *values*, and for list-valued keys the shape is the whole finding.
- **parameter estimates.** `est_total_params` is archwatch's own number, not BLIS's, and not
  the published one. All three have been observed to differ by >20%. Only BLIS's figure is
  relevant to a fidelity claim; derive it yourself from BLIS's formulas.

## The grounding fact you are reasoning against

BLIS is **architecture-name-agnostic but mechanism-sensitive.** It never dispatches on
`architectures[]` or `model_type`; everything flows from numeric shape fields in
`config.json` into `sim.ModelConfig`. Two consequences drive your whole analysis:

1. A new model usually **just runs**.
2. Unknown config fields are **silently dropped** — no error, no warning.

So the failure mode you are hunting is not a crash. It is **silent, confident, wrong
numbers**, and it occurs exactly where a novel *mechanism* is not represented in BLIS's
latency and KV basis functions.

## Procedure

### 1. Read the stub and the evidence

Read the `issues/<arch_id>.md` stub. Then actually read the sources it links — the HF model
card, the vLLM/SGLang PR (and its diff, which is often the clearest statement of what is
novel), the InferenceX entry, any linked paper.

**The highest-yield source is the model repo's own `modeling_*.py`, reachable via the config's
`auto_map`.** It is the only thing that settles what a field *means*: it revealed that a
Bucket 0 rejection was a false positive, and it was the only source for the expert shapes that
produced the largest finding in the first three stubs analyzed. Fetch it whenever the config
has an `auto_map`. Config field names alone are a guess; ratios between them are conventions,
not mechanisms.

**If the stub has `has_config: false`** — InferenceX-sourced stubs carry no config and often no
`model_ids` — you must locate the model repo yourself from the display name and fetch its
`config.json`. Do not attempt a fidelity estimate without a config; say what you could not
obtain instead. Do not classify from the config alone; the
config tells you *what fields exist*, and the PR or paper tells you *what they mean*.

### 2. Name the mechanism

State in two or three sentences what this architecture actually does differently. Be
specific and quantitative where the config supports it — "MLA with a 512-rank compressed KV
projection plus a 64-dim RoPE head, and every 4th layer replaced with linear attention" is
useful; "a novel efficient attention variant" is not.

If the architecture is *not* actually novel — a rename, a re-parameterization of something
BLIS already handles — say that plainly. A correct "nothing new here" is a valuable result,
not a failed analysis.

### 3. Load BLIS's support surface

Read `support-surface/parsed-fields.yaml` (what BLIS parses, with source refs) and
`support-surface/known-gaps.yaml` (BLIS's already-known approximations, each with its
mechanism, impact, and seam refs). These are the reference you classify against. They are
harvested from BLIS's own source and `docs/reference/models.md`.

### 4. Classify each mechanism, then give the architecture a headline bucket

Real architectures stack four to six mechanisms spanning several buckets. Classifying the
*architecture* into one bucket discards the mechanism-to-magnitude mapping that is the actual
decision input.

So: **list every mechanism you identified with its own bucket and its own magnitude**, then
give the architecture a headline bucket equal to the bucket of the **largest-magnitude**
finding. Say explicitly which mechanism sets the headline.

- **Bucket 0 — would not run.** The config violates a hard validator. Already determined by
  stage 1; if the stub says Bucket 0, confirm the reason reads correctly and explain what a
  user would see.
- **Bucket 1 — runs as-is.** Every architecturally meaningful field is one BLIS already
  parses, and the mechanism is one it already models (dense, uniform MoE, interleaved MoE,
  GQA, quantized weights). BLIS simulates this today. The only follow-up is validating the
  numbers.
- **Bucket 2 — known-gap mechanism.** The mechanism is real and unmodeled, and it matches an
  entry in `known-gaps.yaml`. Cite the gap id — and **always re-derive it from BLIS's source
  anyway.** `known-gaps.yaml` was harvested by a tool: its `file:line` refs have held up, but
  its *semantic* fields have been observed wrong, including a `direction` field inverted
  (BLIS was 0.53x optimistic where the entry claimed pessimistic) and an incomplete
  `seam_refs` list that would have sent a maintainer to fix half the bug. Re-deriving is how
  every one of those errors was found. Treat the file as a hypothesis, not an authority.
- **Bucket 3 — new mechanism, no seam.** Nothing in BLIS represents this and it is not on
  the known-gaps list. This is the highest-value finding. Name the functions that would
  need a new branch (see the seam map below).

**Tie-break on magnitude, not on bucket.** Do NOT prefer 2 when torn. The asymmetry runs the
other way than it first appears: a false Bucket 2 reads as "already tracked, file it away" and
**kills the finding**, while a false Bucket 3 costs a maintainer one minute of reading the gap
list. This was observed concretely — a known-gap entry named a real model and matched its
config exactly, producing a *seductively correct-looking* Bucket 2 that buried a 1.98x
parameter over-count no gap id covered. Check whether any gap entry covers the mechanism
**causing the largest error**, not merely a mechanism the model has.

### 5. Estimate fidelity impact

This is the part that converts "a new architecture exists" into a decision. Answer: **how
wrong is BLIS today for this model, and in which direction?**

Quantify wherever the surface supports it. The canonical example: BLIS sizes decode KV reads
as standard `2 * dKV * tokens`, so for an MLA model whose real per-token KV is
`kv_lora_rank + qk_rope_head_dim`, the KV-read term is over-counted by the ratio of those two
quantities — compute it from *this* config and state it, with the direction (over-counted KV
traffic makes decode step time and TTFT read pessimistically slow).

Three details that change the number, all observed to matter:

- **`dKV` uses integer division.** `hidden_size // num_attention_heads`, not the exact
  quotient. Compute it the way Go does.
- **Check which layer count the term is scoped to.** Some terms use all layers, others only
  the full-attention ones. Using the wrong one silently rescales the ratio.
- **For sparse or windowed attention the honest answer is a context-dependent band, not a
  scalar.** State the context length you evaluated at, and give the range across the range of
  contexts a user would actually run — the error can grow by an order of magnitude across it.

**Watch for interacting gaps.** Two entries filed separately can have opposite signs on the
same term, so fixing the cheap one first makes the model worse. If you spot that, say so
explicitly — nothing in `known-gaps.yaml` links its entries.

Be explicit about scope: BLIS often gets capacity right while getting step time wrong (MLA
is exactly this). "KV capacity is already correct; step time is over-counted ~20x" is far
more actionable than "MLA is unsupported."

State your uncertainty honestly. "I could not determine X from available sources" is a
legitimate and useful finding.

### 6. Append to the stub

Append a `## Stage 2 — deep-dive analysis` section to the stub file, placing it **strictly
below the `<!-- archwatch:stage2:append-below -->` marker.**

Two rules that the tooling depends on:

- **Never edit the YAML front matter**, and never modify anything above the marker. Everything
  above it is regenerated from the candidate on each scan; the emitter detects drift there and
  will start skipping the file. Your analysis lives below the marker and is preserved verbatim
  across re-renders.
- **Do not set a `stage2: complete` flag or otherwise mark status in the front matter.**
  Completion is detected by there being non-empty content after the marker.

Include: the headline bucket + one-line verdict; the per-mechanism table (mechanism, bucket,
magnitude, direction); the fidelity impact estimate; the affected seams **named by function**,
with a line number only where you read that exact line yourself; what you read to reach this;
and your confidence with any open questions.

Two things worth a line each when they apply, both cheap and both what a maintainer wants:
**inert matches** (a keyword fired but the field is degenerate or redundant — say so, so nobody
re-litigates it) and **an escape hatch** (e.g. a Bucket 0 on KV-capacity derivation can often
be bypassed with `--total-kv-blocks`, letting the model run today).

## Seam map — where a new mechanism lands in BLIS

For Bucket 3, name the specific functions. The full set of seams for adding a mechanism:

| Concern | Location (function, not line — see note) |
|---|---|
| New config field on the struct | `ModelConfig` in `sim/model_hardware_config.go` (+ an `Is*`/`Effective*` predicate) |
| Parse it (both paths, keep in sync) | `GetModelConfigFromHF` in `sim/latency/config.go` **and** `ExtractKVCapacityParams` in `sim/latency/kv_capacity.go` |
| Per-token KV sizing | `KVBytesPerToken` in `sim/latency/kv_capacity.go` (auto-propagates to PD transfer and KV offload) |
| Model weight footprint | `computeModelWeightBytes` in `sim/latency/kv_capacity.go` |
| Step-time physics | `StepTime` and `NewTrainedPhysicsModel` in `sim/latency/trained_physics_model.go`; roofline equivalents in `sim/latency/roofline.go` |
| User-facing warning | `cmd/root.go` (follow the existing warning pattern) |
| Docs | `docs/reference/models.md` |
| New learned coefficient | only if the new term needs its own beta: `defaults.yaml` + the beta-count logic in `trained_physics_model.go` |

> **Cite functions, not line numbers.** Line refs rot fast — the ones originally written into
> this skill were harvested from a different checkout of BLIS than the one being analyzed, and
> every one of them had drifted.
>
> `support-surface/parsed-fields.yaml` and `known-gaps.yaml` carry harvested `file:line` refs,
> and a test bounds-checks them — but **that test only verifies the line exists, not that it
> still says what we claim** (its own docstring says so). So a ref there is a pointer worth
> following, never evidence on its own. Open the function and read it. When you cite a line in
> your own analysis, cite one you have actually read in the checkout you are analyzing.

Note there is **no plugin or registry seam for architectures** in BLIS — adding mechanism
support means editing these shared functions, not registering a new type. Say so when
relevant, because it affects how invasive the change is.

## Hard exclusions

- **No implementation sketches.** Do not propose a formula, a code change, or a patch. An
  unverified formula stated confidently in a tracking issue misleads worse than no formula.
  Name the seam; stop there.
- **No edits outside the stub file.** Never touch `sim/`, `cmd/`, or any Go file.
- **No invented citations.** If you did not read it, do not cite it. If a paper is paywalled
  or a model card is empty, say so.

## Quality bar

A good analysis lets a BLIS maintainer decide, in under a minute, whether to act this week
or file it away — and if they act, tells them which function to open first.
