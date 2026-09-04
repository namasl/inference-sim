# archwatch prototype — implementation plan

**Status:** prototype / spike. Validates the idea end-to-end against real data.
Design discussion: `inference-sim/inference-sim` Discussion #1687.

## What this is

An early-warning pipeline that detects new model **architectures** and reports where
BLIS would need to change. **Tracking-only.**

## Hard rules — every agent must obey

1. **Never modify anything outside `tools/archwatch/`.** Nothing under `sim/`, `cmd/`,
   `docs/`, or any Go file. This tool reads BLIS as text and imports nothing from it.
2. **Dry-run only.** Nothing is ever posted to GitHub. The emitter writes markdown files
   into `tools/archwatch/issues/`. There is no code path that files an issue. No write
   scopes are needed or used.
3. **No live network calls in tests.** Tests read recorded fixtures from `tests/fixtures/`.
   Live calls happen only when a human runs `archwatch scan`.
4. **Do not edit these frozen files:** `archwatch/connectors/base.py`, `archwatch/config.py`.
   They are the shared contract. If you believe one is wrong, say so in your report
   instead of changing it.
5. **Stay in your lane.** Implement only the files your task assigns. Do not create or
   edit another component's files, even to "fix" them — report the problem instead.
6. Python >= 3.11. Dependencies are fixed in `pyproject.toml`: `huggingface_hub`,
   `requests`, `pyyaml`, `pytest`. Do not add others.
7. Write tests for your component under `tests/test_<component>.py`.

## Architecture recap

Two stages. Stage 1 is Python, no LLM, and answers **"is this architecture new?"**
Stage 2 is an LLM skill and answers **"does BLIS support it?"**

```
[connectors] -> [detector] -> [novelty + significance] -> Candidate -> [emitter] -> issues/*.md
                                                                            |
                                                        (human runs stage-2 skill)
```

**The primary key is the architecture** (`arch_ids[0]`, from the config's
`architectures[]`), not the model. One output file per architecture. Fine-tunes, quants,
and merges inherit their architecture string, so the HuggingFace firehose collapses at
this key before any expensive check.

**Stateless.** No database. Connectors are window-based: they ask "what changed in the
last N days?" rather than diffing against stored snapshots. Dedup for the prototype is
"does `issues/<arch_id>.md` already exist?"

## The grounding fact about BLIS

BLIS never dispatches on `architectures[]` or `model_type`. Everything flows from numeric
shape fields in `config.json` into `sim.ModelConfig`. Consequences:

- A new model usually **just runs** if it exposes the expected fields.
- **Unknown config fields are silently dropped** — no error, no warning.

So the risk is not a crash; it is silent wrong numbers. That is why T1 (below) is the
highest-value trigger.

## Components

Each is one agent's task. Interfaces between them are fixed here.

### B — support surface (`support-surface/*.yaml` + `archwatch/surface.py`)

Harvest, from BLIS source in the **parent repo** (read-only, at
`/ws/fork/inference-sim/.worktrees/archwatch/sim/`), what BLIS actually parses and where
it is known to be wrong.

`support-surface/parsed-fields.yaml`:
```yaml
parsed_fields:
  - name: num_key_value_heads
    aliases: [num_kv_heads, multi_query_group_num]
    role: attention          # shape | attention | moe | precision | rope | other
    source_ref: "sim/latency/config.go:302-308"
hard_validators:
  - id: positive_shape_fields
    fields: [num_hidden_layers, hidden_size, vocab_size, num_attention_heads, intermediate_size]
    rule: "must be > 0"
    source_ref: "sim/latency/config.go:419-431"
    failure: "rejected at config validation"
```
Cover at minimum: layers/dims, attention heads + KV heads + `head_dim`, MoE fields
(`ResolveNumExperts` alias set, `num_experts_per_tok`, shared experts,
`moe_intermediate_size`, `interleave_moe_layer_step`, `first_k_dense_replace`), MLA
(`kv_lora_rank`, `qk_rope_head_dim`), hybrid (`linear_attn_config.full_attn_layers`),
precision (`torch_dtype`, `quantization_config`), `hidden_act`, `tie_word_embeddings`,
`max_position_embeddings`, `rope_scaling`.

Hard validators to encode (these define Bucket 0): non-positive shape fields;
unrecognized `torch_dtype` (yields `BytesPerParam=0`, rejected at
`sim/latency/trained_physics_model.go:615`); `NumHeads % TP != 0` and
`NumKVHeads % TP != 0` (`:594-603`, assume TP=1 default so only report when the field is
absent/zero or non-integer); non-SwiGLU `hidden_act` (`sim/latency/kv_capacity.go:276`);
MoE signalled without a resolvable expert count (`kv_capacity.go:552`).

`support-surface/known-gaps.yaml` — BLIS's known approximations, from
`docs/reference/models.md` (the "known approximations" box) plus the code refs:
```yaml
gaps:
  - id: mla_step_time_kv_read
    mechanism: "Multi-head Latent Attention (MLA) KV compression"
    keywords: [kv_lora_rank, qk_rope_head_dim, latent attention, mla]
    impact: "decode/prefill KV-read sized as standard 2*dKV*tokens; ~21x over-counted for MLA -> pessimistic TTFT"
    scope: "step time only; KV capacity is already MLA-aware"
    seam_refs: ["sim/latency/trained_physics_model.go:297", "sim/latency/kv_capacity.go:125"]
```
Cover at minimum: MLA step-time pessimism; hybrid/linear-attention layers charged as full
attention in step time and weights; MTP / speculative decode not modeled; block-wise FP8
flattened; `first_k_dense_replace` capacity-only; explicit `head_dim` unused by step time;
novel MoE routing (`n_group`/`topk_group` unparsed).

`support-surface/known-architectures.yaml` — the cold-start seed set. Harvest architecture
names from **vLLM's model registry** (fetch
`https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py`
and parse the architecture-name string keys) merged with BLIS's validated set from
`docs/reference/models.md`. Record `seeded_from` provenance with a date. Expect a few
hundred names; if the fetch fails, fall back to a documented static list and say so.

`archwatch/surface.py` — the loader other components use:
```python
@dataclass
class Surface:
    parsed_field_names: set[str]      # canonical names + all aliases, flattened
    parsed_fields: list[ParsedField]
    hard_validators: list[Validator]
    gaps: list[Gap]
    known_architectures: set[str]     # lowercased for comparison

def load_surface(dir: Path = ...) -> Surface: ...

# Surface methods:
def unparsed_fields(self, config: dict) -> list[str]:
    """Top-level config keys BLIS does not parse, excluding
    archwatch.config.IGNORED_CONFIG_KEYS and anything under a nested dict we do not
    descend into. Descend into text_config exactly as ParseHFConfig does (pivot it
    onto the top level) so multimodal configs are judged on the text tower."""

def check_hard_validators(self, config: dict) -> list[str]:
    """Bucket 0. Returns human-readable failure strings; empty means 'would run'."""

def is_known_architecture(self, arch_id: str) -> bool: ...
def match_gaps(self, config: dict) -> list[Gap]:
    """Gaps whose keywords appear in the config's keys/values. A cheap pre-hint for
    stage 2 — NOT a substitute for the LLM's classification."""
```
**Acceptance:** `load_surface()` works from a clean checkout; `unparsed_fields()` returns
`[]` for a plain Llama config and flags `kv_lora_rank` as parsed (not unparsed) for a
DeepSeek-V3 config; `check_hard_validators()` returns a failure for a config with
`hidden_act: "gelu"` and for one with `torch_dtype: "float4_e2m1"`; tests use fixtures.

### C — HuggingFace connector (`archwatch/connectors/hf.py`)

`poll(since)` lists models created in the window (newest first) via `huggingface_hub`
(`HfApi().list_models(sort="createdAt", direction=-1, ...)`; verify the exact parameter
names against the installed version and adapt). Two-phase, because fetching a config for
every new HF model is thousands of requests:

1. **Cheap metadata pre-filter:** drop anything whose repo id matches
   `config.DERIVATIVE_PATTERNS`; keep the rest.
2. **Fetch `config.json`** (via `hf_hub_download` or `HfApi().hf_hub_download`) only for
   survivors, capped at `config.max_hf_config_fetches` per poll. A missing or unparseable
   config is not an error: emit the Signal with `config=None`.

Populate `arch_ids` from `architectures[]`, `model_type`, `org` from the repo owner, and
put downloads/likes into `extra` (S4 needs them). Add a `poll_trending()` method for the
S4 trending sweep, returning Signals for currently-trending models regardless of creation
date.

**Acceptance:** returns Signals with populated `arch_ids` and `config` from recorded
fixtures; obeys the fetch cap; never raises on a 404/429/malformed config; a
`Qwen3-8B-GGUF`-style repo id is dropped by the pre-filter.

### D — framework connector (`archwatch/connectors/frameworks.py`)

`poll(since)` searches `vllm-project/vllm` and `sgl-project/sglang` for
model-support PRs/commits in the window. Use the GitHub REST API with a token from
`GH_TOKEN`/`GITHUB_TOKEN` env or `gh auth token` (via subprocess) — read-only calls.
Target: PRs touching model-registry paths (`vllm/model_executor/models/`,
`python/sglang/srt/models/`) and/or titled with `[Model]`/"Add support for"/"Support ".
Extract candidate architecture names from PR titles and changed filenames (e.g. a new
`kimi_k3.py` and a title naming `KimiK3ForCausalLM`). Emit one Signal per (repo, PR),
`source` = `"vllm"` or `"sglang"`, `config=None`, `urls={"pr": ...}`.

**Acceptance:** from fixtures, extracts `arch_ids` from real PR **patch content** (registry
lines, `EntryClass` footers, added `class XxxForCausalLM`) — NOT from titles, which do not
carry architecture names (see addendum 10); handles 403 rate limits by returning partial
results; `raw_ref` carries the PR number.

### E — InferenceX connector (`archwatch/connectors/inferencex.py`)

`poll(since)` reads **commit diffs** (stateless — no snapshot comparison) in
`SemiAnalysisAI/InferenceX` touching `MODELS.md`, `configs/`, and `perf-changelog.yaml`.
Use the GitHub REST API (`/commits?path=...&since=...`, then the per-commit patch) and
parse added lines for model/architecture names. When the diff carries performance numbers
(throughput/latency/TTFT/cost), put them in `extra["perf"]` — they are future validation
ground truth for BLIS. `source="inferencex"`.

**Acceptance:** from fixtures, extracts added model names from a `MODELS.md` diff and a
new `configs/` file path; tolerates the repo restructuring (a missing path is a warning,
not an exception).

### F — novelty + significance (`archwatch/novelty.py`, `archwatch/sizing.py`)

`sizing.py`: estimate parameter counts from a config (total and, for MoE, active).
Standard transformer arithmetic — embeddings, attention projections, MLP/expert FFN,
accounting for MoE expert count and `num_experts_per_tok`. Approximate is fine; document
the formula. Return `None` when the config lacks what it needs rather than guessing.

`novelty.py`: the filter chain.
```python
def join_signals(signals: Iterable[Signal]) -> list[Candidate]:
    """Group by primary_arch(). Signals with no arch fall back to a normalized
    display_name key (the alias path); record that in Candidate.triggers as needed."""

def evaluate(cands: list[Candidate], surface: Surface, cfg: DetectorConfig) -> list[Candidate]:
    """Apply suppressors, then triggers, then the significance gate. Populate
    triggers/significance/unparsed_fields/bucket0_failures/est_*_params. Return only
    candidates that pass, newest/strongest first, capped at cfg.max_issues_per_run."""
```
**Triggers** (any one; record which fired):
- **T1** new architecture whose config carries fields BLIS does not parse
  (`surface.unparsed_fields`, excluding `IGNORED_CONFIG_KEYS`)
- **T2** a framework support PR (source `vllm`/`sglang`) for an unseen architecture
- **T3** a frontier org (`cfg.frontier_orgs`) publishing a new architecture
- **T4** corroboration — the same arch from >= 2 distinct sources

**Suppressors** (checked first; a suppressed candidate is dropped and logged):
- `surface.is_known_architecture(arch_id)` — in the cold-start seed set
- an `issues/<arch_id>.md` already exists (the stateless dedup)
- every model id matches `DERIVATIVE_PATTERNS`
- structurally identical to a known architecture: same `architectures[]` with differences
  confined to `quantization_config`
- no config **and** not corroborated — too weak

**Significance gate** (must pass >= 1; record which):
- **S1** `est_total_params >= thresholds.min_total_params`
- **S2** org in `frontier_orgs`, or org top-model downloads >= threshold (use whatever
  the Signals' `extra` carries; if unavailable, S2 is simply not satisfied)
- **S3** any Signal from `vllm`/`sglang`/`inferencex`
- **S4** model downloads/likes over threshold, or seen via the trending sweep

**Acceptance:** unit tests prove a quant repo is suppressed at the primary key; a
`LlamaForCausalLM` signal is suppressed as known; a synthetic novel arch with a big config
and 100B params passes with `["T1"]`/`["S1"]`; the per-run cap is honored. Tests are pure
(fixtures + synthetic dicts), no network.

### G — emitter (`archwatch/emitter.py`)

Render a `Candidate` to markdown and write `issues/<arch_id>.md`. **No GitHub calls, ever.**
The file is the stub issue: title line, sources table with links, why-it-fired (triggers +
significance), the deterministic findings (Bucket 0 verdict, unparsed fields, param
estimates), any `extra["perf"]` numbers, and a "stage 2 not yet run" placeholder section
the deep-dive skill later fills in. Include a machine-readable YAML front-matter block
(`arch_id`, `sources`, `triggers`, `significance`, `bucket`, `detected_at`) so the backtest
can parse results without regex-scraping prose. Idempotent: re-rendering the same
candidate produces identical bytes (no timestamps outside front matter).

**Acceptance:** golden-file test (render a fixture Candidate, compare to a checked-in
expected markdown); filenames are filesystem-safe; writing twice is a no-op diff.

### H — detector + CLI (`archwatch/detector.py`, `archwatch/cli.py`)

`detector.py` wires it together: instantiate connectors, poll each inside a try/except so
one failing source degrades to a partial scan (log and continue), join signals, evaluate,
emit. Return a run summary (counts per stage: signals in, candidates joined, suppressed
with reasons, passed, written).

`cli.py`: `archwatch scan [--window-days N] [--dry-run] [--sources hf,vllm,...] [--out DIR]`,
`archwatch show` (list what is in `issues/`), `archwatch surface` (print the loaded surface
summary — useful for sanity checks). `--dry-run` is the **only** mode; the flag exists for
explicitness and defaults to true. Write a run log to `.runlog/<timestamp>.json` (gitignored)
with the per-stage counts and every suppression reason — this is the artifact we tune the
filter from.

**Acceptance:** `archwatch scan --sources hf --window-days 1` runs end-to-end against the
live API and writes files plus a run log; a forced connector exception still produces a
summary; `archwatch surface` prints counts.

### I — stage-2 classifier skill (`skills/archwatch-deep-dive/SKILL.md`)

A Claude Code skill: given an `issues/<arch>.md` stub, read the config and the linked
PR/model card/paper, name the **mechanism**, classify into a bucket, estimate fidelity
impact, and append the analysis to the stub file (never overwrite the stub section).

Buckets: **0** would not run (already computed deterministically — trust the stub);
**1** runs as-is (only fields BLIS parses); **2** known-gap mechanism (cite the matching
`known-gaps.yaml` entry); **3** new mechanism, no seam (name the functions needing a new
branch: `KVBytesPerToken` `sim/latency/kv_capacity.go:91`; `StepTime` basis terms and
constructor freeze `sim/latency/trained_physics_model.go:190`,`:540`; a new `ModelConfig`
field `sim/model_hardware_config.go:6` parsed in both `GetModelConfigFromHF`
`sim/latency/config.go:289` and `ExtractKVCapacityParams` `kv_capacity.go:521`; weight
sizing `computeModelWeightBytes` `kv_capacity.go:394`).

Report = triage **plus a fidelity impact estimate** (how wrong BLIS is today, quantified
where the surface supports it). **Explicitly excluded: implementation sketches.**
Unverified formulas mislead worse than none.

**Acceptance:** the skill file is complete and self-contained; a dry read of it against
two stub files (one MLA, one plain dense) produces the expected buckets when a human runs it.

### J — validation harness (`tests/backtest.py`, `tests/test_*.py`)

Two validations, runnable as scripts:

1. **Backtest** — replay a historical window and assert the filter would have caught the
   frontier releases (Kimi K2/K3, DeepSeek V3/V4, GLM5, MiniMax M3, Qwen3.5) without
   flagging thousands of others. Report recall against that named list and total volume
   that survived. This is what calibrates the thresholds — **numbers are outputs, not inputs.**
2. **Live scan** — a real `scan` over a recent window; report what came through and
   eyeball the noise.

**Acceptance:** the backtest prints a per-target hit/miss table and a survivor count;
findings are written to `VALIDATION.md`.

## Sequencing

- **Wave 1** (done, by the orchestrator): scaffolding, `base.py`, `config.py`, this plan.
- **Wave 2** (parallel): B, C, D, E, F, G.
- **Wave 3**: H (needs all of wave 2).
- **Wave 4** (parallel): I, J.
- **Wave 5**: review each component, route feedback back to its implementer, verify fixes.
- **Wave 6**: run the backtest + live scan; audit whether they tested what we think;
  fold learnings back into plan and code; re-run until clean.

## Definition of done

`archwatch scan` runs end-to-end against live APIs, writes plausible `issues/*.md`, the
backtest shows high recall on the named frontier releases with a survivor count in the
low tens (not thousands), the classifier skill produces correct buckets on real stubs,
and `VALIDATION.md` records what worked and what did not.

---

## Contract addenda (found during wave 2 — binding on all later components)

These emerged from the completed emitter component and correct or pin down things the
original plan left ambiguous. They are binding.

1. **Dedup must use the emitter's path helpers, never a constructed path.**
   `emitter.issue_exists(arch_id, out_dir)` and `emitter.issue_path(...)`. The emitter
   sanitizes architecture ids (appending an 8-hex sha1 of the original when it must rewrite
   one), so `issues/<arch_id>.md` is not always the real filename. Building that path by hand
   makes the dedup silently fail for any id containing a space, slash, colon, or non-ASCII
   character — stubs would then be re-emitted forever with no error.

2. **`Signal.observed_at` is timezone-aware UTC.** Pinned in `base.py`. Naive values are
   treated as UTC by consumers; no consumer applies a host offset.

3. **Stage-2 handshake.** The deep-dive skill appends only *below* the
   `<!-- archwatch:stage2:append-below -->` marker and never edits front matter or any content
   above it. Completion is detected by non-empty content after the marker —
   `emitter.split_stub(text)` returns exactly that split. There is no `stage2: complete` flag.
   Component J must use this, not a front-matter field.

4. **`extra["perf"]` shape** (produced only by the InferenceX connector): a dict with a
   `hardware` key plus numeric metric keys (`output_tok_per_s`, `ttft_ms_p50`,
   `cost_per_mtok_usd`, ...) and a free-text `notes`; a list of such dicts when one commit
   yields several rows. Values stay numeric — units belong in the key name — so J can compare
   against them numerically.

5. **Validator severity: `fatal` vs `silent`.** Bucket 0 means "BLIS would not run." Some
   conditions originally listed for it may only mis-size silently (an unrecognized
   `torch_dtype` yielding `BytesPerParam=0`; MoE without a resolvable expert count). Each
   entry in `hard_validators` carries `severity: fatal | silent`. **Only `fatal` entries
   belong in Bucket 0**; `silent` ones are T1-style wrong-numbers risks and must be reported
   as such, not as a crash.

6. **`bucket` in front matter is `0` or `null` only.** Buckets 1-3 are stage-2 judgements and
   are never guessed by stage 1.

## Contract addenda, round 2 (found during wave 2 — binding)

7. **Curated sources are exempt from the `no_config_uncorroborated` suppressor.** Framework and
   InferenceX signals *always* carry `config=None` — a PR or a benchmark entry is not a model
   repo. As originally worded the suppressor therefore dropped every framework-only candidate,
   making **T2 incapable of ever firing** — deleting the purest zero-day signal we have (a vLLM
   PR means someone already decoded the architecture and wrote reference code). The suppressor
   exists to drop HuggingFace junk, so it applies **only to HF-only candidates**. Any signal from
   `vllm`, `sglang`, or `inferencex` exempts the candidate: a person chose to write that code or
   run that benchmark, and that choice is the evidence.

8. **`Signal.org` is lowercase.** `FRONTIER_ORGS` is all lowercase. Connectors emit lowercase,
   and `novelty.py` additionally lowercases at the comparison point, so T3/S2 cannot silently
   miss `Qwen` vs `qwen`.

9. **`arch_ids == []` from a curated source is normal**, not malformed. Umbrella PRs and
   follow-ups carrying only a registry key legitimately yield no architecture name; they route
   through the alias path on `display_name`.

10. **Architecture names come from patch content, not PR titles.** This corrects component D's
    acceptance criterion, which was written on a false assumption. Measured against both repos'
    full merged history: a GitHub search for `ForCausalLM in:title` returns **zero hits**. Real
    titles carry marketing names whose mapping to the class name is not derivable —
    `[Model] Support Qwen3.8-Flash-Next` yields `Qwen4ExpForCausalLM`;
    `[Model] add GLM-5.3-Flash support` yields `Glm5NextForCausalLM`;
    `[Model] Add native IFM K2 Horizon serving support` yields two unrelated names. Filenames are
    also insufficient (`k2_horizon.py` would require guessing the suffix). The reliable sources
    are vLLM's added registry lines, SGLang's `EntryClass = [...]` footers, and `+class
    XxxForCausalLM(` in added modules — **6/6 on genuine model-support PRs**. Titles serve only
    as a candidate gate; filenames only as a weak alias hint.

11. **New vLLM architectures land in `vllm/models/<name>/`**, not only
    `vllm/model_executor/models/<mod>.py`. Both prefixes must be watched; the registry file
    itself has not moved.

12. **Connectors accept an optional `until` bound** (constructor arg; the `poll(since)` protocol
    is unchanged). Component J's historical replay is impossible without it, and recorded
    fixtures drift without it.

13. **GitHub's `/search/issues` is limited to 30 requests/minute** (against 5,000/hr for core).
    The framework connector spends ~4 per poll, but **H must never poll in a loop.**

## Contract addenda, round 3 (found during wave 2 — binding)

14. **The cross-source join needs union-find over multiple edge types.** The original
    "primary key is `arch_ids[0]`, fall back to display name" produces **disjoint key spaces**
    across sources for the same release: InferenceX emits `Kimi-K3` with no architecture (it
    exposes no config), the framework connector emits `KimiK3ForCausalLM` mined from patches, HF
    emits whatever `architectures[]` says. Corroboration — T4 and the significance gate's
    corroboration path, meant to be the cheapest reliable noise killer — was therefore nearly
    unfireable. Signals now join if they share **any** of: (a) a case-folded `arch_id`;
    (b) a normalized HF repo id (lowercase, strip quantization and variant suffixes, keep the
    org, so `moonshotai/Kimi-K3-Instruct` ≡ `moonshotai/Kimi-K3`); (c) a normalized family name
    (strip `ForCausalLM`/`ForConditionalGeneration`/`MTPModel`/`Model` suffixes, punctuation and
    case, bridging `KimiK3ForCausalLM` ≡ `Kimi-K3`). **False merges are worse than duplicate
    issues**, so every merge records the edge that caused it and is logged for audit; the
    canonical `arch_id` prefers a real `architectures[]` spelling, then a framework-mined name,
    then the normalized display name — never an invented CamelCase guess.

15. **`T5` — a curated benchmark entry for an unseen model.** Added rather than widening T2,
    because the two imply different follow-ups: a framework PR hands you reference code, an
    InferenceX entry hands you performance numbers and no architecture. With addendum 14 most
    InferenceX signals merge into an HF or framework candidate and fire T4; T5 catches the
    genuine zero-day case where SemiAnalysis benchmarks something before any config or PR exists.

16. **`DERIVATIVE_PATTERNS` applies only to `model_ids` from HF signals.** A model id extracted
    from a PR diff or a changelog line is a *mention*, not the artifact. Otherwise a PR titled
    "[Model] Support Qwen3-8B-GGUF loading" would suppress a real candidate.

17. **`silent_failures` is a first-class finding, separate from `bucket0_failures`.** Fatal
    failures are loud and self-announcing; silent ones are the reason this pipeline exists. Real
    live instance: a config whose expert count uses an unrecognized spelling makes a
    trillion-parameter sparse MoE **simulate as a dense model** behind one `logrus.Warnf`. A
    clean Bucket 0 verdict with non-empty `silent_failures` is the *dangerous* case, and the stub
    must present it that way.

18. **Cite functions, not line numbers, in prose.** Every `file:line` ref in this plan's original
    component-B section was stale — harvested from a different BLIS checkout than the fork being
    analyzed. Verified refs live in `support-surface/parsed-fields.yaml`, which carries a test
    bounds-checking each one against the real Go files. That file is the source of truth.

19. **InferenceX's changelog prose is the highest-value payload in the pipeline** and must be
    preserved verbatim (`extra["perf_notes"]`), not discarded when no number parses. It names
    architecture *mechanisms* before HF configs are public — Kimi-K3's 896 routed experts and
    KDA layers holding no KV cache; Qwen3.8-Flash-Next's Mamba SSM state and built-in MTP
    module. This is stage-2 intelligence arriving inside the zero-day window.

20. **Worth watching later, deliberately deferred:** InferenceX's `golden_al_distribution/`
    (committed golden acceptance-length curves per model — direct speculative-decode ground
    truth) and `benchmarks/single_node/agentic/*.sh` (real serve flags: attention backend, KV
    dtype, `max-model-len`, MoE backend). Both are richer BLIS validation input than changelog
    prose.

## Contract addenda, round 4 (binding on component J)

21. **`silently_wrong` is the backtest's headline metric — not `bucket == 0`.** The emitter
    derives it as `silent_failures and not bucket0_failures`: BLIS accepts the config, runs, and
    reports confident nonsense with no error to show for it. Bucket 0 is the *loud* class that
    would have been noticed anyway; `silently_wrong: true` is the class that justifies this
    pipeline's existence. It is a plain boolean in the front matter, so J needs no taxonomy
    knowledge to count it.

22. **Front matter is `schema: archwatch/2`. Test for keys, not for the version string.** The
    schema will grow additively; a version pin turns every future field addition into a J failure.

23. **Third-party prose is defanged before rendering.** Verbatim changelog text containing the
    literal stage-2 marker would split the stub at the wrong point and let a later write silently
    drop real analysis — an injection through a data channel we control the format of. The emitter
    rewrites `archwatch:stage2:` in all third-party text. Any future component that embeds
    external text into a file with structural markers must do the same.
