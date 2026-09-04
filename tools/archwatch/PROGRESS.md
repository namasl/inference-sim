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
- [~] Wave 2: **G emitter DONE**; **C hf DONE** (resumed for a non-LM pre-filter); B, D, E, F in flight
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
