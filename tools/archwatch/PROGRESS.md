# archwatch build — running progress log

Kept continuously so the build is resumable if the session dies. Newest entries at the bottom.

**Branch:** `feature/archwatch` in worktree
`/ws/fork/inference-sim/.worktrees/archwatch` (fork `namasl/inference-sim`).
**Nothing is pushed. Nothing is posted to GitHub.** Local only, by request.

## Orchestrator judgment calls (made without the user, who is away)

1. **The orchestrator wrote the plan, not a planning subagent.** The full design lives in
   the orchestrator's context from the brainstorming session; delegating planning would have
   meant re-transmitting it and risking fidelity loss. Agents implement; the plan is
   `PLAN.md`.
2. **`base.py` and `config.py` are frozen contracts written by the orchestrator.** Six
   agents work in parallel; if each invented its own `Signal` shape or threshold set the
   merge would be a rewrite. Agents are told not to edit these two files.
3. **`known-gaps.yaml`, not `known-gaps.md`** (design said `.md`). Structured YAML lets the
   novelty filter pre-match gap keywords deterministically and the stage-2 skill still reads
   it fine. Small deviation from Discussion #1687, recorded here.
4. **Dedup is "does `issues/<arch>.md` exist"** rather than a GitHub issue search, since the
   prototype never touches the tracker. Same stateless property, no write scopes.
5. **`issues/` is tracked in git** (not ignored) so generated stubs can be committed as
   review evidence. `.runlog/` is ignored.

## Status

- [x] Worktree + branch created
- [x] Scaffolding, `pyproject.toml`, venv
- [x] Frozen contracts: `archwatch/connectors/base.py`, `archwatch/config.py`
- [x] `PLAN.md`
- [x] **Wave 2 COMPLETE** — B, C, D, E, F, G all landed and committed. 573 tests green.
- [x] C org-stats DONE
- [~] Revisions in flight: F (union-find join, T5), G (silent_failures)
- [~] Wave 3: H detector + CLI launched
- [ ] Wave 3: H detector + CLI
- [x] Wave 4a: I classifier skill (`skills/archwatch-deep-dive/SKILL.md`) — written by orchestrator
- [ ] Wave 4b: J validation harness
- [ ] Wave 5: review + corrections
- [ ] Wave 6: backtest + live scan + audit loop

## Wave 2 log

### G — emitter: complete, accepted

56 tests green. Two things worth keeping in mind when reviewing it:

- **Network isolation is enforced structurally, not by convention.** A test AST-parses
  `emitter.py` and asserts its import set is a subset of an allowlist, then greps the source for
  `requests`, `urllib`, `httpx`, `socket`, `subprocess`, `gh api`, `create_issue`, and friends.
  A future edit that reaches for the network fails the test rather than quietly working. This is
  the right shape for a dry-run guarantee.
- **It is large** (823 lines, 56 tests) for something that renders markdown. The safety
  properties earn much of it (path traversal, idempotency, front-matter coercion), but it is
  flagged for the review wave to judge whether some can be trimmed.

Accepted deviations: `detected_at` derives from the newest `Signal.observed_at` rather than the
wall clock (so a re-render months later is still byte-identical); `write_issue` never overwrites
by default and re-appends any existing stage-2 analysis when forced, which is what protects the
deep-dive output; front matter is a superset of the required keys; a "Config at a glance"
section was added, which makes the stub genuinely readable by a human and by stage 2.

### Contract problems G found — all routed to the still-running agents

Recorded because they were real design gaps, not agent errors:

1. **Dedup path ownership was split** (the important one). `PLAN.md` told F to test
   `issues/<arch_id>.md` for existence, but only the emitter knows the filename sanitization
   rules. An f-string path would make the suppressor silently fail for any arch id with a
   space, slash, or non-ASCII character — re-emitting stubs forever with no error. Routed to F:
   use `emitter.issue_exists()`/`issue_path()`. Pinned in `PLAN.md` addendum 1.
2. **`Signal.observed_at` had no timezone requirement.** A connector emitting naive local time
   would make output depend on the host machine. Routed to C, D, E; pinned in `base.py` and
   addendum 2.
3. **The stage-2 handshake was unspecified.** Fixed in the skill: append only below the marker,
   never touch front matter; completion is detected by content after the marker, not a flag.
   Addendum 3.
4. **`extra["perf"]` had no shape** despite being BLIS validation ground truth. Convention
   routed to E; addendum 4.
5. **Bucket 0 may have been over-broad.** Two conditions the plan listed as "would not run" may
   only mis-size *silently* (unrecognized `torch_dtype`; MoE without a resolvable expert count).
   If so they belong on the T1 wrong-numbers side, not Bucket 0 — mislabelling them would
   misreport real models. Routed to B to verify against the Go source and tag each validator
   `severity: fatal | silent`. Addendum 5.

### C — HuggingFace connector: complete, accepted (resumed once)

57 tests green, verified offline with `HF_HUB_OFFLINE=1` and an autouse fixture that makes any
network call raise. Two genuine improvements on the plan:

- **`list_models(expand=["config"])` returns the Hub's indexed excerpt of each repo's
  `config.json` — including `architectures` — for zero extra requests.** The pipeline's primary
  key is therefore free from the listing. This is a better design than the plan's, which assumed
  the architecture was only obtainable by fetching each config.
- Consequently phase 2 fetches **one representative config per distinct architecture**, not per
  repo. Live, 2,895 phase-1 survivors collapse to **127 distinct architectures** — so the
  200-fetch cap is comfortable rather than tight. Full poll: 19.8s.

Measured live volume (1-day window): ~3,560 repos created, ~662 dropped as derivatives,
**~2,895 survivors, 127 architectures**. `poll_trending(30)` correctly surfaced
`DeepseekV4ForCausalLM`, `GlmMoeDsaForCausalLM`, `Qwen4ExpForConditionalGeneration`.

### The `.gitignore` bug — real data loss, already committed

**The repo-root `.gitignore` line 44 is a bare `*.json`** (BLIS's data-files section). It
silently excluded every JSON fixture in this tool: `git add` dropped them with no error or
warning. Verified concretely — the emitter's `expected_*.md` goldens were committed but
`kimi_k3_config.json` and `weirdact_config.json` were **not**, so a fresh checkout of this
branch would have failed G's test suite with missing-fixture errors.

Fixed centrally in `tools/archwatch/.gitignore` (a deeper .gitignore wins) with
`!tests/fixtures/**/*.json` and `!support-surface/**/*.json`. Confirmed with `git check-ignore`
that fixtures are now tracked and that `.runlog/` remains ignored — negation cannot resurrect
files under a directory-excluded path, which is why `.runlog/` is safely excluded by directory.

This is the kind of failure worth remembering: no error, no warning, and the tests keep passing
locally because the files exist on disk. It would only have surfaced on a clean clone.

### One agent claim that was wrong, and checked

C reported that BLIS "presumably reads only `torch_dtype`", making the 2026-era `dtype` rename a
live Bucket-0 gap. **Verified against the source: false.** `sim/latency/config.go:334-336` reads
`torch_dtype` and falls back to `dtype`, with a comment naming GLM-5. B was told not to record it
as a gap, but to encode the real two-key resolution so the validator is not dead code.

### Decisions taken from C's findings

- `DERIVATIVE_PATTERNS` extended with `-4bit`, `-8bit`, `-mlx`, `heretic` (real live misses).
  **`-mtp` deliberately excluded** — multi-token prediction is a mechanism BLIS does not model,
  so an MTP variant is signal, not noise.
- **The non-LM filter belongs in the connector, not F's suppressors.** Knowing a repo is a
  diffusers/peft/robotics artifact is source-specific knowledge. C resumed to add it, with the
  rule that absence of evidence is never grounds to drop — an unknown-shaped repo from an
  unexpected lab is the zero-day case this system exists for.
- `text_config` pivoting is load-bearing for both B and F: frontier models increasingly expose
  only `architectures`/`model_type`/`dtype`/`text_config`/`vision_config` at the top level.
  Without the pivot, T1 fires on every multimodal frontier release and sizing returns `None` for
  exactly the models S1 must catch.

### F — novelty + sizing: complete, accepted (resumed for a recall fix)

147 tests green. Sizing validated against **published** parameter counts, not asserted loosely:
Llama-3.1-70B 70.55B vs 70.55B (four significant figures), Mixtral-8x7B +0.0%, DeepSeek-V3
+0.0% total / +1.5% active, Qwen3-30B-A3B +0.1%.

The Llama-4-Scout miss (−6.7% total, −34.5% active) is a genuine limitation handled the right
way — tested explicitly rather than hidden. Llama-4 runs an always-on shared expert per MoE
layer that **nothing in `config.json` declares**, so no config-only estimator can see it. Total
still clears any plausible S1 threshold; the active figure should not be quoted for that family.

### The most valuable finding so far: a suppressor that could not work as specified

F caught that my structural-identity suppressor was **unimplementable as worded** — "same
`architectures[]` with differences confined to `quantization_config`" presumes a reference config
for the known architecture, and nothing in the pipeline has one. Worse, the obvious reading
*fails on the exact case it targets*: an FP8 repack inherits its base's unparsed fields, and the
real surface reports `['decoder_sparse_step','norm_topk_prob']` for **any** Qwen3-MoE config.

It found this only by running against the real `surface.py` instead of its own stub — which is
the difference between a test that passes and a test that means something. Fixed with two
branches: name-based (strip quantizer tokens, `Qwen3MoeFp8ForCausalLM` → `Qwen3MoeForCausalLM`)
and config-based (known base named, novelty confined to the quant block). Both require the base
to be *identifiable*, which is what stops it swallowing a real frontier release shipping FP8.

### A real recall hole, now being closed rather than noted

F's second finding is one I acted on. Suppressor #1 (`is_known_architecture`) drops a candidate
**before any config analysis** — but a point release can add config fields under an *unchanged*
architecture string, and BLIS silently drops fields it does not parse. So the "silent wrong
numbers" case can arrive disguised as a known architecture, and archwatch never looks.

Added `recheck_known_architectures` to `DetectorConfig` (default `False`) and tasked F to
implement a T1-only re-check for known architectures, with a distinct suppression reason so the
run log separates "known, nothing new" from "known, unparsed fields found". **The wave-6 backtest
measures both settings** — recall gained vs noise added — before we pick a default.

### Declined deliberately

F's finding #3 asked for a `Candidate` field to carry `surface.match_gaps()` output into the
stub. Declined: a keyword match rendered into the report would read as analysis while being only
a string match, and the stage-2 skill reads `known-gaps.yaml` directly anyway.

### C — non-LM pre-filter: accepted, with an honest negative result

73 tests green. Vocabularies (39 libraries, 31 pipeline tags) chosen against a **live day's
histogram**, not from memory, and everything ambiguous is documented as deliberately excluded:
`transformers`, `pytorch`, `keras`, `nemo`, runtime/format labels, and every one-off vendor
library observed live — because an unrecognized library from an unexpected lab *is* the zero-day
case.

**The honest headline: the filter only closes about a fifth of the gap.** 1,171 of ~1,690
archless survivors publish neither `library_name` nor `pipeline_tag`, so under the
"only drop on positive evidence" rule every one is kept. C also measured whether extending to
`tags` would help and reported that it would **not** (tags carry no modality information for
those repos), declining to write code that looks like it helps. That is the right instinct.

Live: 3,564 repos → 682 derivative → 326 non-LM → **2,556 survive** (1,189 naming an
architecture, 125 distinct).

I checked whether the remaining ~1,367 archless signals are actually a problem and concluded
they are not: F's `no_config_uncorroborated` suppressor already drops them, F now aggregates
suppression logging, and continuing to emit them preserves the corroboration path — an archless
HF repo can still be rescued if InferenceX or a vLLM PR names the same model. So the volume is
harmless and the design stands. Declined C's offered `extra["lm_evidence"]` field on YAGNI
grounds since F already buckets by reason.

Accepted C's own judgment call to apply the filter to `poll_trending()` as well: a trending
*language* model always carries an architecture in the listing excerpt and so can never be
wrongly dropped, while a trending diffusion checkpoint is junk to the detector either way.

### D — framework connector: complete, accepted

104 tests, offline-verified with sockets *and* subprocess monkeypatched to raise.

**D disproved one of my design assumptions with measurement.** I specified extracting
architecture names from PR titles. Across both repos' entire merged history, a GitHub search for
`ForCausalLM in:title` returns **zero hits**. Real titles carry marketing names with no derivable
mapping: `[Model] Support Qwen3.8-Flash-Next` → `Qwen4ExpForCausalLM`; `[Model] Add native IFM K2
Horizon serving support` → two unrelated architectures. Filenames are also insufficient
(`k2_horizon.py` would require guessing the suffix). Patch content works instead — added registry
lines, SGLang `EntryClass` footers, added `class XxxForCausalLM` — **6/6 on genuine model-support
PRs**. My acceptance criterion was unrealistic and is now corrected in PLAN.md rather than
quietly satisfied.

Also found that new vLLM architectures now land in `vllm/models/<name>/` packages, not only
`vllm/model_executor/models/`, so both prefixes must be watched.

### The T2 bug — my plan silently disabled its own best trigger

D's most important finding. Framework signals always carry `config=None` (a PR is not a model
repo), so the `no_config_uncorroborated` suppressor deleted **every** framework-only candidate
before the trigger phase — making T2 unfireable. T2 is the purest zero-day signal there is: a
vLLM PR adding an architecture before any HF config is public means someone already decoded it
and wrote reference code.

Fixed by scoping the suppressor to its actual purpose (HF junk) and exempting all curated
sources. Notably **F's own unit tests had ratified the bug** — one used a `vllm` signal to assert
the suppression — which is why only the cross-component report caught it. A good argument for
integration checks over per-component confidence.

### B — support surface: complete, accepted, and it overruled me twice

90 tests. 26 parsed fields with verified refs, 14 gaps, 432 architectures harvested live from
vLLM's registry via `ast.parse`.

**B was right and I was wrong, twice — both verified against the source before accepting:**

1. I diagnosed its failing test as suffix normalization. There was none; matching was exact, and
   `KimiK3ForConditionalGeneration` returned `True` because **it is literally in vLLM's registry**.
   B had already fixed the test by correcting its own wrong assumption rather than weakening the
   assertion, which is the right instinct.
2. I proposed normalizing architecture suffixes. B measured it: all 37 suffix-groups consist
   entirely of names **already in the seed set** (vLLM enumerates both heads for every family
   that has both), so normalization changes no answer for the duplicate-issue case I was worried
   about — that case cannot arise. It would only change an answer when vLLM lists A but a vendor
   ships B, and B's absence *is* the T2 signal. Suppression is unrecoverable in a stateless
   pipeline. It added a non-destructive `related_known_architectures()` instead.

**B also corrected PLAN.md on a Bucket-0 claim, and the correction is the best finding of the
build.** MoE-without-a-resolvable-expert-count is NOT fatal: all three `ExtractKVCapacityParams`
call sites only `logrus.Warnf`. So **a trillion-parameter sparse MoE with a novel expert-count
spelling is simulated as a dense model, behind one warning line nobody reads.** That is exactly
the silent-wrong-numbers failure archwatch exists to catch, and it is live in BLIS today.
Validators now carry `severity: fatal | silent`, and `Candidate` gained `silent_failures` so the
class has a first-class home.

Related subtlety B found: dtype resolution is `else-if`, so `{torch_dtype: "mxfp4", dtype:
"bfloat16"}` still aborts — the readable `dtype` is never consulted.

**My line refs were all stale.** My original seams scan ran against `/ws/inference-sim`, a
different checkout than this fork, so every `file:line` in the plan had drifted. B verified each
one and added a test bounds-checking them against the real Go files. The stage-2 skill now cites
functions, not lines, and points at that tested data file.

### E — InferenceX connector: complete, accepted

53 tests. Model-name extraction from `MODELS.md` is exact (28 rows, zero false positives). Perf
numbers come from English prose in `perf-changelog.yaml` — only 2.6% of 2,215 description strings
yield numbers — and E audited its extractor against all 820 entries, killing real false positives
(`MI325X`→`325x`, `TP8 x PP2`→`8x`). Verbatim prose is always retained because the prose is more
trustworthy than the parser. Good calibration of confidence.

E also declined to guess CamelCase architecture names from marketing names, on the grounds that a
wrong guess creates a phantom issue file. Correct.

### The join key was broken, and it took three connectors to see it

**E's finding is the one that invalidated a core design decision.** The three sources produce
disjoint key spaces for the same release — `Kimi-K3` (InferenceX), `KimiK3ForCausalLM`
(framework), `architectures[]` (HF) — so joining on the architecture with a display-name fallback
put one model in three candidates and made corroboration nearly unfireable.

Fixed with union-find over three edge types (arch id, normalized repo id, normalized family
name), with merge edges recorded and logged because **false merges are worse than duplicate
issues**. Architecture-as-primary-key remains right for *issue identity*; it was never sufficient
as a *join* key. That distinction is the kind of thing only real data from three sources exposes.

### The best unanticipated find

InferenceX's changelog prose **names architecture mechanisms before HF configs are public**:
Kimi-K3's "896 routed experts, 93 layers, KDA layers keep per-token KV small — only the 24
gated-MLA layers hold cache"; Qwen3.8-Flash-Next's "512-expert MoE, float32/bfloat16 Mamba SSM
state, built-in 4B NEXTN MTP module". That is stage-2 intelligence arriving *inside* the zero-day
window, from a source I had originally rated as merely a significance signal. E kept it as
`extra["perf_notes"]` rather than discarding non-numeric prose — the single best judgment call
any agent made.

### C — S2 org track record: accepted, and the design improved under measurement

My spec said "query per distinct org, cached within the poll, capped." C measured that and found
it does not fit: a 1-day window holds **1,310 distinct non-frontier orgs** — ~85 s and 1,310
requests. It also found two API facts that constrain the approach: `sort="downloadsAllTime"` is
rejected outright (HTTP 400 — the only download sort is 30-day), and repeated `author=` params do
not batch.

The replacement is better than what I asked for. **A bulk descending sweep answers every org at
once in 3 requests** by walking `sort="downloads"` and stopping the moment 30-day downloads fall
below the threshold — everything past that point is under it *by construction*. That makes
absence from the sweep **conclusive** for the 30-day metric, and the walk's depth is set by the
threshold rather than by how many orgs the window happened to contain: 2,471 models read, 688
orgs mapped, 0.1 s.

A capped per-org fallback then covers the one case a 30-day sort structurally cannot see: a
**dormant lab** with a large lifetime count but little current traffic. That is a real case, not
a hypothetical — MBZUAI sits at 198,796 all-time against 76,159 in 30 days.

**The judgment call I most want to keep** is that C *gated* the fallback rather than letting the
cap truncate it. With 1,310 orgs needing lookups and a cap of 200, you answer an arbitrary 200 of
them — so S2 would depend on listing order, and the backtest's threshold calibration would rest
on which orgs happened to be enumerated first. The gate ("shipped an architecture in this window
and has any traffic") selects 182 orgs, comfortably under the cap, so nothing truncates and the
selection is reproducible from the data. A cap that silently truncates is the one setting to
avoid, and C identified that unprompted.

Measured cost: default adds **+7.0 s and +185 requests**, answering 117 of 1,310 non-frontier
orgs. **Decision: keep it on.** Seven seconds is irrelevant to a 6-hourly job, and the dormant-lab
path is precisely the "credible lab not on my hand-written allowlist" recall route that S2 exists
to provide. `max_org_lookups=0` remains available: it keeps the conclusive 30-day answer for
+0.1 s and loses only the dormant case.

C also caught a flaw in its own test scaffolding worth recording: its first fake API ignored both
`author` and `sort`, so **every org test passed while measuring nothing** — the connector was
being handed the window fixture as though it were an org's catalogue. It rewrote the fake to
dispatch the way the real endpoint does. Tests that pass for the wrong reason are the most
expensive kind, and catching one in your own work is harder than catching it in someone else's.
