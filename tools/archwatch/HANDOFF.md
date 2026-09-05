# archwatch — handoff

Prototype built autonomously while you were away. **Read this first**, then `VALIDATION.md`
for the measurements. Design discussion: `inference-sim/inference-sim` Discussion #1687.

- **Where:** branch `feature/archwatch`, worktree `/ws/fork/inference-sim/.worktrees/archwatch`,
  all of it under `tools/archwatch/`. Fork `namasl/inference-sim`.
- **Nothing pushed. Nothing posted to GitHub.** 21 commits, 917 tests, suite green, tree clean.
- **Nothing outside `tools/archwatch/` was modified.** The tool reads BLIS as text and imports
  nothing from it.

## Run it

```bash
cd /ws/fork/inference-sim/.worktrees/archwatch/tools/archwatch
.venv/bin/python -m archwatch.cli scan --sources inferencex --window-days 5 --out /tmp/aw
.venv/bin/python -m archwatch.cli scan --sources hf --window-days 1 --out /tmp/aw
.venv/bin/python -m archwatch.cli surface     # sanity-check the harvested BLIS surface
.venv/bin/pytest -q                            # 917 tests, no network
```

Dry-run is the only mode; `--no-dry-run` exits 2 explaining there is no live mode. Output is
markdown in `--out`. Five real stubs are committed in `issues/` as evidence.

## Does the idea work?

**Yes, with the caveat that its value concentrates in one source and one finding class.**

| Question | Measured |
|---|---|
| Suppresses the HF firehose? | Yes — 14,144 signals → 90 survivors uncapped |
| Noise at the shipped cap of 10 | **10%** (1 of 10) |
| Genuine frontier releases inside the cap | **8** |
| Cross-source join works? | Yes — 5 of top 6 survivors are cross-source joins |
| Frontier recall | **9/9** with the calibrated defaults |
| Per-source noise | inferencex **0%**, sglang 0%, hf 70%, vllm — (0 survivors) |

A live 5-day InferenceX scan returns DeepSeek-V4-Pro, Kimi-K3, GLM-5.2, MiniMax-M3 and
Qwen3.5-397B-A17B, 33 API requests, ~6s. **InferenceX is the standout source** — 0% noise, and
its changelog names architecture *mechanisms* before HF configs are public. I had rated it
lowest of the three when we designed this. I was wrong.

## The part worth more than the tool: what it found in BLIS

Five findings, each verified by me against the fork's Go source, not taken on an agent's word.

1. **Five of fifteen current frontier configs abort BLIS.** DeepSeek-V4-Pro/Flash and
   Qwen3.5-397B/122B carry only `moe_intermediate_size`; `config.go:341` reads only
   `intermediate_size`/`ffn_hidden_size`, and `IntermediateDim <= 0` is fatal at
   `trained_physics_model.go:1027`. MiniMax-M3 uses `hidden_act: "swigluoai"`.
2. **A false-positive rejection.** BLIS aborts on `hidden_act: "gelu"` as non-SwiGLU, but that
   model's `modeling_spark.py` is a 3-matrix GEGLU — `gelu` names the gate nonlinearity exactly
   as Llama's `silu` names SwiGLU's. And `mlpMatrixCount` discards the string (`_ = hiddenAct`),
   so **the validator gates a value no downstream formula consumes.** `--total-kv-blocks`
   bypasses it and the model runs correctly.
3. **`--kv-cache-dtype fp8` never reaches step time.** `bytesPerKVElement` is a hardcoded 2.0
   (`trained_physics_model.go:491`) and `KVBytesPerParam` has **zero** occurrences in
   `trained_physics_model.go` or `roofline.go`. Needs no flag to hit: any fp8/int8 checkpoint
   gets capacity at 1 byte/element while step time charges 2. The constant's comment is stale
   since #1565.
4. **Latent MoE is unpriced.** `routed_expert_hidden_size` has zero hits across `sim/` and
   `cmd/`, so Kimi-K3's half-width experts (3584 vs `hidden_size` 7168) are charged full width —
   ~1.98x whole-model over-count, wrong on **both** capacity and step-time paths.
5. **A docs error.** `docs/reference/models.md` calls hybrid KDA weight handling "the one
   remaining **pessimism**". It is optimistic: a KDA layer carries ~440M params against BLIS's
   ~205M, so BLIS under-counts (~0.53x). A KDA layer holds an O(1) *state* rather than a growing
   KV *cache*, which says nothing about parameter count — the two senses of "linear attention"
   got conflated, and our reference data faithfully reproduced the wrong doc until it was
   re-derived.

**Suggested next step: file 1-5 as BLIS issues.** They are independent of whether archwatch
itself goes further, and 1 and 2 block real models today.

## Six design decisions that measurement reversed

Recorded because these are the reason a throwaway build was the right call.

1. **Architecture-as-primary-key doesn't work as a *join* key.** Right for issue identity, but
   the three sources emit disjoint key spaces (`Kimi-K3` / `KimiK3ForCausalLM` /
   `architectures[]`), so corroboration was nearly unfireable. Now union-find over architecture
   id, normalized repo id, and normalized family name.
2. **My `no_config_uncorroborated` suppressor made T2 unfireable** — framework signals always
   have `config=None`, so every framework-only candidate died before triggers, deleting the
   purest zero-day signal there is.
3. **PR titles do not carry architecture names.** `ForCausalLM in:title` returns **zero hits**
   across both repos' entire merged history. Patch content works (6/6).
4. **My ranking read "not a language model" as "novel architecture."** `+3` for `would_not_run`
   put audio and video repos at the top and consumed the whole cap. Now ordered by how *quiet* a
   failure is: `silently_wrong` +4, LM-shaped `would_not_run` +2, else +0.
5. **Both my headline thresholds were wrong.** `min_total_params=30B` and `recheck=False` **each
   independently hid the only `silently_wrong` finding in a live day.** Now 3B and True.
6. **My stage-2 tie-break would have buried the biggest finding.** I wrote "when torn between
   Bucket 2 and 3, prefer 2." A false Bucket 2 reads as "already tracked" and kills the finding;
   a false Bucket 3 costs a minute of reading. Now tie-break on magnitude.

## Known limitations — read before trusting any number

- **No true zero-day recall measurement.** The architecture seed set was harvested *today*, so
  it post-dates every release. Only a prospective run over a genuinely new model measures this.
  Everything else is inference from the zero-day arm (architecture removed from the seed).
- **No HF historical replay.** No server-side date filter; a 30-day-old window is ~100k records
  behind head. Recall is measured by fetching known releases' configs directly instead.
- **Two sources carry the value; two mostly carry noise.** hf 70% noise, vllm 0 survivors.
- **S2's download threshold and S4's thresholds never bound anything.** S4 is now the
  most-fired significance code, so much of the tail is admitted by fine-tune popularity — the
  least principled path in the gate.
- **The ranking function is now load-bearing and under-scrutinised.** Noise is 59% uncapped and
  10% at the cap, i.e. the cap is doing the quality work.
- **Encoder/seq2seq noise got worse** with `recheck=True` (12 of 53 survivors).
- **One stub per model per source, across scans.** The join is per-scan; dedup is file
  existence. So an InferenceX scan and an HF scan of the same model produce two
  non-cross-referencing stubs. Fixing this needs cross-scan state, which contradicts the
  stateless design — a real tension, not an oversight.
- **`silently_wrong` fired exactly once in 1,446 candidates/day.** The finding class that
  justifies the pipeline is genuinely rare. Whether one per day is worth the machinery is a
  judgment call I could not make for you.
- Stage 2 was executed against three real stubs and its output was critiqued, but **no human has
  reviewed a stage-2 analysis.**

## If you want to take this further

1. File the five BLIS findings. Highest value, independent of everything else.
2. Decide whether hf and vllm earn their keep, or whether this becomes an InferenceX +
   framework-PR watcher with HF used only to fetch configs for models the other two name.
3. Scrutinise the ranking function — it is carrying the precision story.
4. Give stage 2 a human review pass on real stubs.
5. Then, and only then, consider the scheduled workflow and real issue filing. Both were
   deliberately deferred; the prototype has no write scopes anywhere.

## Where things are

| File | What |
|---|---|
| `HANDOFF.md` | this |
| `VALIDATION.md` | the measurements, pre-fix and post-fix side by side |
| `PROGRESS.md` | full build log, judgment calls, and every reversal with its evidence |
| `PLAN.md` | component specs plus 27 binding contract addenda, most of them corrections |
| `support-surface/` | harvested BLIS surface: 26 parsed fields, 17 gaps, 432 architectures |
| `skills/archwatch-deep-dive/` | the stage-2 classifier |
| `issues/` | five real stubs from a live scan |
