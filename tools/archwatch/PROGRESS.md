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
- [ ] Wave 2: B support-surface, C hf, D frameworks, E inferencex, F novelty, G emitter
- [ ] Wave 3: H detector + CLI
- [ ] Wave 4: I classifier skill, J validation harness
- [ ] Wave 5: review + corrections
- [ ] Wave 6: backtest + live scan + audit loop
