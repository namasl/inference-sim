# archwatch — validation findings (component J)

Everything here was measured against live HuggingFace and GitHub APIs (read-only) on
2026-09-04, in **two rounds**. Round 1 measured the system as it stood and produced the
threshold recommendations and bug reports. Those were then applied by their owners, and
round 2 re-measured. **Both rounds are kept.** The before/after is the evidence that the
fixes worked, so no round-1 number has been overwritten — every table and section says
which round it belongs to.

| | round 1 (pre-fix) | round 2 (post-fix) |
|---|---|---|
| code | `ce1452c3` (`fix: stop reading non-LM configs as novelty signals`) | `1675b896` + working-tree `config.py`/`novelty.py` |
| `min_total_params` | 30B | **3B** |
| `recheck_known_architectures` | False | **True** |
| `max_issues_per_run` | 5 | **10** |
| `MIN_FAMILY_KEY_LEN` | 3 | **4** |
| union-find `arch:` edges | every `arch_ids` entry | **primary's family only** (`coherent_arch_ids`) |
| artifacts | `/tmp/awbt` | `/tmp/awbt2` |

**Harness:** `tests/backtest.py` — a script, never collected by pytest. Its pure logic is
unit-tested in `tests/test_backtest.py` (37 tests). Full suite at the end of round 2:
**863 pass, 0 fail**. Nothing was written to `issues/`; every step used a temp directory.

Reproduce:

```
.venv/bin/python tests/backtest.py all --out /tmp/awbt2 --window-days 7
.venv/bin/python tests/backtest.py precision --out /tmp/awbt2 --window-days 7 --no-recheck
```

The recall step pins `recheck_known_architectures` explicitly in both directions rather
than reading `DEFAULTS`, so its two arms keep measuring two things now that the default has
been flipped on its evidence. The precision step follows `DEFAULTS` unless told otherwise,
so its headline number always describes the system as shipped.

---

## Verdict in one paragraph (round 2, post-fix)

The pipeline works, the ranking is what makes it usable, and one false merge remains.
Over a 7-day window 14,055 signals collapse to 7,074 candidates and **90 survivors**
(156:1), of which the shipped cap of 10 shows **8 genuine frontier architectures, 1
plausible-minor model and 1 noise item — 10% noise at the cap** against 59% across the
whole survivor list. Five of the top six were assembled by the **cross-source join**
(HuggingFace + vLLM, HuggingFace + InferenceX), so the join is not just working, it is
carrying the top of the report. The single `silently_wrong` finding in the corpus now
**ranks 9th of 90 and is therefore reported**; under the round-1 defaults two unrelated
settings each independently hid it. Recall on the nine named frontier releases is **9/9**
against the shipped surface, with all three seeded controls still correctly suppressed.
Three of the four false merges round 1 found are structurally gone; **one survives** by a
different mechanism (a `repo:` edge mined from a PR's prose), and it files Muse Glimmer
under another model's name. The framework PR-title noise is gone entirely — vLLM now
yields zero survivors per week rather than stubs titled *"gfx1250 on ROCM 10"*. What got
worse is the tail: encoder and seq2seq models are now **12 of 53 noise survivors**, and T1
fires on as little as one unparsed field.

## Verdict in one paragraph (round 1, pre-fix — kept for the record)

The pipeline works, and **the two most important shipped defaults are both wrong.**
The novelty filter does suppress the HuggingFace firehose — 14,107 signals collapse to 37
survivors over a 7-day window, a 380:1 reduction, and the ranking puts genuine frontier
releases in the top 5. The cross-source join really joins: it merged HuggingFace + vLLM
into one `Qwen3.8-Flash-Next` candidate and HuggingFace + InferenceX into one `Kimi-K3`
candidate, firing T4 on real data. The deterministic classifier reaches correct,
independently verified verdicts: it found that **5 of 15 frontier model configs would make
BLIS abort**, and re-derived that from BLIS's Go source. But
`recheck_known_architectures=False` gives **0/9 recall on the named frontier releases**
(9/9 with it on), and `min_total_params=30B` **suppresses the only `silently_wrong`
finding in a live day** — the single metric addendum 21 calls the pipeline's reason to
exist. There is also one reproducible **false merge** that reaches the output and would
file four unrelated architectures as one issue.

---

# Round 2 — re-measurement after the fixes

Same harness, same targets, same windows. Round 1's sections follow unchanged below.

## Recall (round 2)

**Unchanged, and now delivered by the shipped default.** Per-target recall (a target hits
when any of its repos passes), re-measured against the fixed code:

| arm | frontier releases | seeded controls |
|---|---|---|
| `recheck_off` (explicit False) | 0 / 9 | 0 / 3 (correct) |
| `recheck_on` — **the shipped default now** | **9 / 9** | 0 / 3 (correct) |
| `zero_day` | 9 / 9 | 3 / 3 |

The 3B threshold is visible here too: every one of the twelve frontier repos now satisfies
S1 as well as S2, where in round 1 `Qwen/Qwen3.5-9B` (8.21B) missed S1 and passed on S2+S4
alone. The three seeded controls still die at `known_architecture_nothing_new` with zero
unparsed fields and zero validator findings, so lowering the threshold did not blunt the
discriminator. The five bucket-0 findings on frontier configs — the DeepSeek-V4 / Qwen3.5-MoE
missing `intermediate_size` and MiniMax-M3's `swigluoai` — are unchanged; they are properties
of the configs and of BLIS, not of these thresholds.

## Precision under the shipped defaults (round 2)

7-day window, all four sources, uncapped so the number describes the filter rather than
the reporting budget (`recheck=True`, `min_total_params=3B`).

```
14,055 signals -> 7,074 candidates -> 90 survivors
drops: no_config_uncorroborated 6,877 | insignificant 61 | not_a_language_model 32
       known_architecture_nothing_new 8 | framework_no_architecture 5 | no_trigger 1
GitHub budget: well within 400
```

### What the shipped cap of 10 actually shows

| # | verdict | survivor | est. params | unparsed | triggers / significance | sources |
|---|---|---|---|---|---|---|
| 1 | **genuine** | `KimiK3ForCausalLM` | 5.47T | 32 | T1+T3+T4+T5 / S1+S2+S3+S4 | hf+inferencex |
| 2 | **genuine** | `DeepseekV4ForCausalLM` (bucket 0) | 291B | 34 | T1-known-arch / all four | hf+vllm |
| 3 | **genuine** | `Glm5NextForConditionalGeneration` (GLM-5.3-Flash) | 313B | 36 | T1-known-arch / all four | hf+vllm |
| 4 | **genuine** | `HYV4ForCausalLM` (tencent/Hy4-preview) | 771B | 28 | T1-known-arch / all four | hf+vllm |
| 5 | **genuine** | `Qwen4ExpForCausalLM` (Qwen3.8-Flash-Next) | 13.5B | 29 | T1-known-arch / all four | hf+vllm |
| 6 | **genuine** | `K2HorizonForCausalLM` (IFM/K2-Horizon) | 9B | 8 | T1-known-arch / S1+S3+S4 | hf+vllm |
| 7 | **genuine** | `Qwen3_5MoeForCausalLM` (bucket 0) | 34.1B | 13 | T1-known-arch / S1+S2+S4 | hf |
| 8 | plausible-minor | `Spark2_5ForCausalLM` (bucket 0) | 3.17B | 5 | T1 / S1+S2+S4 | hf |
| 9 | **genuine — `silently_wrong`** | `Qwen3NextForCausalLM` | 4.02B | 9 | T1-known-arch / S1 | hf |
| 10 | noise | `BertForMaskedLM` | 236M | 4 | T1-known-arch / S2+S4 | hf |

**8 genuine + 1 plausible-minor + 1 noise: 10% noise at the cap**, and the headline
`silently_wrong` finding is inside it. Five of the top six are cross-source joins, which is
the strongest evidence in either round that the union-find join earns its complexity: on its
own, each HuggingFace signal for those releases is one more repo among 14,000, and each vLLM
PR is one more PR. Joined, they are the top of the report.

### Before / after

| measurement | round 1 (30B, recheck=False, cap 5) | round 2 (3B, recheck=True, cap 10) |
|---|---|---|
| signals (7d) | 14,144 | 14,055 |
| candidates | 6,993 | 7,074 |
| survivors, uncapped | 37 | **90** |
| genuine / minor / noise | 6 / 5 / 26 | **24 / 13 / 53** |
| noise rate, uncapped | 70% | **59%** |
| noise rate at the shipped cap | 20% (1 of 5) | **10% (1 of 10)** |
| genuine frontier inside the cap | 3 | **8** |
| `silently_wrong` reported | **0** | **1** |
| bucket-0 stubs | 16 of 37 | 24 of 90 |
| front-matter keys missing | 0 | 0 |

### Noise rate per source (round 2, 7-day single-source scans)

| source | signals | candidates | survivors | genuine | minor | noise | noise rate | round 1 |
|---|---|---|---|---|---|---|---|---|
| hf | 14,017 | 7,063 | 85 | 19 | 13 | 53 | **62%** | 70% |
| vllm | 8 | 8 | **0** | 0 | 0 | 0 | **n/a — nothing emitted** | 100% |
| sglang | 5 | 5 | 1 | 1 (`XllmForCausalLM`) | 0 | 0 | **0%** | 67% |
| inferencex | 24 | 5 | 5 | 5 | 0 | 0 | **0%** | 0% |

vLLM going from 1 survivor to 0 is the `framework_no_architecture` fix: its 8 signals now
split into 5 `known_architecture_nothing_new` (real model PRs for architectures already
seeded — correct) and 3 `framework_no_architecture` (kernel and backend PRs with no
extractable architecture — also correct). Zero survivors is the right answer for a week in
which vLLM added support only for things vLLM's own registry already lists.

## `silently_wrong` (round 2)

Still **exactly one** across the corpus, and still the same candidate — but now it is
*reported* rather than suppressed:

```
Qwen3NextForCausalLM   from  arianraje/qwen3-4b-gdn-hybrid-*
est_total_params: 4.02B   triggers: [T1-known-arch]   significance: [S1]   rank: 9 of 90
silent_failures: moe_expert_count_resolvable + moe_total_required_when_active_present
bucket0_failures: []      bucket: null      silently_wrong: true
```

| configuration | reported? | why not |
|---|---|---|
| round 1 shipped (`recheck=False`, 30B) | **no** | dropped at `known_architecture` |
| `recheck=True`, 30B | **no** | dropped at `insignificant` (4.02B < 30B) |
| **round 2 shipped (`recheck=True`, 3B)** | **yes, rank 9 of 90** | — |

Both threshold changes were necessary and neither was sufficient. This is the finding class
addendum 21 calls the pipeline's reason to exist, and it is now visible end to end.

## False-merge audit (round 2)

Join step, 14-day curated window plus the recall targets: **17 merges, 3 flagged** (round 1:
16 merges, 8 flagged). All three remaining flags are legitimate same-family variant merges —
`Qwen4Exp{ForCausalLM, ForConditionalGeneration, MTP}` (one release, three heads) and
`DeepseekV4{ForCausalLM, ForConditionalGeneration}`.

| round-1 false merge | status | evidence |
|---|---|---|
| DeepSeek V3 + V4 + Qwen3-MoE + Qwen3.5-MoE as one candidate, ranked first (SGLang #35634) | **GONE** | now three separate candidates; the joined set went 33 → 39 candidates as the blob split. `DeepseekV3ForCausalLM`, `DeepseekV4ForCausalLM` and `Qwen3_5MoeForConditionalGeneration` each keep their own findings |
| `MinistralForCausalLM` fused with Qwen3.5-MoE and Bittensor spam via `family:asd` | **GONE** | `MIN_FAMILY_KEY_LEN` 3 → 4. `MinistralForCausalLM` is now one repo (`datahtarov/test1`); the `Affine-*` spam sits correctly under `Qwen3_5MoeForCausalLM`, which is what those repos actually are |
| `DeepseekV32MTPModel` fused with `GlmMoeDsaForCausalLM` (vLLM #52861, a DSA routing backend PR) — found in round 1's replay | **GONE** | replay now yields them as two candidates, refs `52861` and `30519` |
| Muse Glimmer filed under `DFlashLagunaForCausalLM` (SGLang #34262) | **SURVIVES, different mechanism** | see below |

### The one that survives, and why the fix did not reach it

The `arch:` leg of this merge *is* fixed: SGLang #34262 names
`DFlashLagunaForCausalLM, MuseGlimmerForCausalLM, MuseGlimmerForConditionalGeneration`, and
`coherent_arch_ids` now confines it to family `dflashlaguna`, ignoring the two Muse Glimmer
names. But the merge re-forms one edge over:

```
sglang #35371  "DFlash2: local convolution + candidate selector"
  arch_ids  = [DFlashLagunaForCausalLM]                        <- coherent, fine
  model_ids = [z-lab/Qwen3.8-27B-DFlash2, RadixArk/Qwen3.8-27B-DSpark,
               z-lab/Muse-Glimmer-30B-DFlash2, meta-models/Muse-Glimmer-30B]
  edges     = ... ('repo', 'meta-models/muse-glimmer-30b') ...

vllm  #51655  "Add Muse Glimmer model support"
  arch_ids  = [MuseGlimmerForConditionalGeneration, MuseGlimmerForCausalLM]
  model_ids = [meta-models/Muse-Glimmer-30B, meta-models/Muse-Glimmer-30B-assistant]
  edges     = ... ('repo', 'meta-models/muse-glimmer-30b') ...

-> one candidate, arch_id 'DFlashLagunaForCausalLM', refs 34262 + 35371 + 51655
```

A speculative-decoding PR names the **base model it drafts for**. That is a mention, not a
claim of identity — and **addendum 16 already states exactly this principle**
("a model id extracted from a PR diff or a changelog line is a *mention*, not the artifact")
but scopes it only to `DERIVATIVE_PATTERNS`. The join applies no such distinction: `repo:`
edges are emitted for framework `model_ids` as if they were artifacts. So the same class of
bug — one signal mentioning several things it is not about — persists at the `repo:` edge
after being closed at the `arch:` edge.

The harm is unchanged from round 1: Muse Glimmer *is* surfaced, but under
`issues/DFlashLagunaForCausalLM.md`, so a human looking for it will not find the file.
Confining framework `repo:` edges the way `arch:` edges are now confined is the natural
next step; component J has not attempted it (that is the join owner's file).

### One new benign flag, and one correction to my own auditor

* `GLM-5.2` fusing orgs `nvidia` and `amd` on `family:glm52` alone — InferenceX rows for
  `nvidia/GLM-5.2-NVFP4` and `amd/GLM-5.2-MXFP4`. Two quantizers of one model, so merging is
  correct; the heuristic fires because it cannot tell a repacker from a lab. Benign.
* **My own audit heuristic was wrong after the fix and would have inverted this section's
  conclusion.** `audit_merge` counted every entry of every signal's raw `arch_ids`, so a
  *confined* multi-architecture signal still read as a fusion: the first re-run reported the
  DeepSeek/Qwen merge as still present when it had in fact split into three candidates. It
  now counts only the names `coherent_arch_ids` makes eligible to join. A second heuristic
  was also over-eager: differing size tokens are expected when two scales of one release
  share an `architectures[]` string (Qwen3.5-397B-A17B and Qwen3.5-122B-A10B are both
  `Qwen3_5MoeForConditionalGeneration`, and one issue per architecture is the design), so it
  now fires only when no `arch:` edge explains the merge. Three unit tests pin both
  corrections.

## Round-1 findings that are now fixed

| round-1 finding | status in round 2 |
|---|---|
| 1. `recheck=False` gives 0/9 frontier recall | **fixed** — default True; recall 9/9, controls still suppressed |
| 2. `min_total_params=30B` hides the only `silently_wrong` finding | **fixed** — 3B; it now ranks 9 of 90 |
| 3. false merge reaching the output (`arch:` clique) | **fixed** for `arch:`; see the `repo:` leg above |
| 4. a successful join can *destroy* a signal | **fixed** — `known_architecture` now exempts benchmark-source candidates. Re-verified on the same cached signals: HF + InferenceX for Kimi-K3 now survives at **both** recheck settings, where round 1 lost it at `recheck=False` |
| 5. `framework_title_only` keyed on the wrong field | **fixed** — renamed `framework_no_architecture` and keyed on "no signal yielded an architecture". vLLM survivors 1 → 0; the *"gfx1250 on ROCM 10"*, *"Switch output projection gemm (oproj_a) to fp8"* and *"Find attention with a fuser…"* stubs are gone |
| 7. `DERIVATIVE_PATTERNS` missing modern quant tokens | **fixed** — `nvfp4`, `mxfp4`, `-fp4`, `aqlm` added. `Glm5vForConditionalGeneration`, which reached round 1's survivor list only through `jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid`, is no longer a survivor |

## What still does not work (round 2)

1. **One false merge survives, on the `repo:` edge** — mechanism, evidence and suggested
   direction above. It is the same class of bug as the fixed one, at a different edge type.

2. **Encoder and seq2seq noise got worse, not better: 12 of the 53 noise survivors (23%).**
   `BertForMaskedLM`, `BertForTokenClassification`, `DistilBertForMaskedLM`,
   `ModernBertForMaskedLM`, `RobertaForCausalLM`, `XLMRobertaForMaskedLM`,
   `DebertaV2ForMaskedLM`, `QiushiDualPathDebertaV2ForMaskedLM`, `DesklibAIDetectionModelV2`,
   `GPT2ForSequenceClassification`, `T5ForConditionalGeneration`, `Qwen3BidirectionalModel`.
   Round 1 had 8 of 26. `recheck=True` made it worse because these families *are* seeded, so
   the re-check path readmits them on any config drift, and the 3B threshold no longer
   excludes them because they satisfy S4 on fine-tune download counts instead of S1. Round
   1's diagnosis stands and is now more urgent: an encoder keeps all four core transformer
   dimensions, so `lm_shape_evidence` cannot separate it from a decoder, and its
   `hidden_act: "gelu"` reliably trips the fatal `swiglu_family_hidden_act` validator —
   which for a 236M BERT means "this was never a decoder", not "BLIS cannot handle a novel
   architecture". A positive decoder test is needed: `model_type` against a non-causal family
   list (the spirit of `STRONG_NON_LM_MODEL_TYPES`) or an
   `is_decoder` / `is_encoder_decoder` config check.

3. **T1 fires on a single unparsed field, and on the known-architecture path that is almost
   always config-authoring noise.** 10 of the 90 survivors reach the report with ≤2 unparsed
   fields and no other trigger: `Qwen2ForCausalLM` (1), `OlmoeForCausalLM` (1),
   `OlmoForCausalLM` (1), `Ministral3ForCausalLM` (1), `LlamaForCausalLM` (2 — on a 5.08M
   toy), `Olmo3ForCausalLM` (2), `BertForTokenClassification` (2), `RobertaForCausalLM` (2),
   `MinistralForCausalLM` (2), `Qwen3BidirectionalModel` (2). Meanwhile the **lowest**
   unparsed-field count among the config-bearing genuine survivors is **8**
   (`K2HorizonForCausalLM`), and the `silently_wrong` candidate has 9. A minimum of 3 unparsed
   fields on the `T1-known-arch` path alone would drop all ten without touching a single
   genuine survivor. It must **not** apply to T1 on a genuinely new architecture, and it must
   **never** gate the silent-failure half of T1 — addendum 24's whole point is that a silent
   misread can arrive with zero unparsed fields.

4. **90 survivors a week is more than a human will read**, and the cap of 10 is now doing
   heavy lifting: 80 candidates are dropped as `over_cap`. The cap is well placed (10% noise
   inside it, 8 genuine frontier architectures), but "the filter produces 90 and we show 10"
   means the ranking function, not the filter, is the component that decides what a human
   sees. That is worth stating plainly because the ranking has had far less scrutiny than the
   suppressors, and nothing in the run log tells you what the 80 dropped candidates were
   without re-reading the JSON.

5. **Unchanged from round 1, still open:** `InferenceXConnector` has no `until=` (addendum
   12) and its `max_commits=80` silently truncated an 11-day window holding 133 commits, with
   the truncation reaching only a log line rather than `RunSummary`; the InferenceX name
   parser still emits arch ids like `2.7`, `2.7-Code` and `kimik2.6`; `CustomResearchModel`
   still collapses 18 unrelated `model_type` values from dozens of orgs onto one generic
   architecture string, showing that the architecture primary key only works when labs choose
   distinctive class names.

6. **A one-day window still measures only the noise floor.** Unchanged.

---

# Round 1 — the original measurement (pre-fix)

Everything below was measured at `ce1452c3` with `min_total_params=30B`,
`recheck_known_architectures=False`, `max_issues_per_run=5` and `MIN_FAMILY_KEY_LEN=3`. It is
the evidence the fixes were made on and is kept unchanged.

---

## (A) Recall — would the filter have flagged these releases? (round 1)

Targets resolved live on the Hub; the exact repo ids used are in `recall.json`. For each,
the harness fetched the real `config.json` and Hub metadata by repo id and built the
Signal through the HF connector's **own** `_to_signals` (so the real pre-filter, budgeted
config fetch and org sweep all ran), then `join_signals` + `evaluate` against the real
`load_surface()`.

### Why there is no HuggingFace historical replay

PLAN.md section J says "replay a historical window". For HuggingFace that is
**infeasible, not merely slow**: the Hub has no server-side date filter, so
`list_models(sort="created_at")` must be walked from the newest repo backwards. A live day
is ~2,500 repos after the pre-filter, so a 30-day-old window sits behind ~75,000-100,000
records. The recall question does not need the listing — it needs each release's config,
which costs one request. The listing walk is exercised in step (B) instead, where the
window is recent and the walk is cheap. **This is a correction to the plan, not a
shortcut.**

### The seed set is from the future

`support-surface/known-architectures.yaml` was seeded from vLLM's registry on
**2026-09-04** — today. vLLM already supports every architecture on the target list, so
**all 15 targets are in the seed set** and the `known_architecture` suppressor drops every
one. That is correct behaviour answering a useless question ("would archwatch flag a
release vLLM shipped support for months ago?" — no, by design). Each target is therefore
measured in three arms:

| arm | what it is |
|---|---|
| `as_shipped` | the shipped config, `recheck_known_architectures=False` |
| `recheck` | `recheck_known_architectures=True` |
| `zero_day` | the target's own architecture strings removed from the seed set — the day before vLLM added support |

(The harness now calls the first two `recheck_off` and `recheck_on` and pins both settings
explicitly, because "as shipped" stopped meaning `False` once this measurement flipped the
default. The arms themselves are identical.)

`Surface.is_known_architecture` is exact lowercased membership, so removing the exact
strings is a clean, narrow counterfactual that cannot perturb any other verdict.

### Results

| target | repo | `architectures[0]` | est. params | as-shipped | recheck | zero-day |
|---|---|---|---|---|---|---|
| Kimi K2 | moonshotai/Kimi-K2-Instruct | `DeepseekV3ForCausalLM` | 1.03T | drop `known_architecture` | **PASS** T1-known-arch / S1+S2 | **PASS** T1+T3 / S1+S2+S4 |
| Kimi K3 | moonshotai/Kimi-K3 | `KimiK3ForConditionalGeneration` | 5.47T | drop | **PASS** | **PASS** |
| DeepSeek V3 | deepseek-ai/DeepSeek-V3 | `DeepseekV3ForCausalLM` | 671B | drop | **PASS** | **PASS** |
| DeepSeek V4 | deepseek-ai/DeepSeek-V4-Pro | `DeepseekV4ForCausalLM` | 1.61T | drop | **PASS** | **PASS** |
| DeepSeek V4 | deepseek-ai/DeepSeek-V4-Flash | `DeepseekV4ForCausalLM` | 291B | drop | **PASS** | **PASS** |
| GLM-5 | zai-org/GLM-5 | `GlmMoeDsaForCausalLM` | 743B | drop | **PASS** | **PASS** |
| GLM-5.2 | zai-org/GLM-5.2 | `GlmMoeDsaForCausalLM` | 743B | drop | **PASS** | **PASS** |
| GLM-5.3 | zai-org/GLM-5.3 | `GlmMoeDsaForCausalLM` | 743B | drop | **PASS** | **PASS** |
| MiniMax M3 | MiniMaxAI/MiniMax-M3 | `MiniMaxM3SparseForConditionalGeneration` | 301B | drop | **PASS** | **PASS** |
| Qwen3.5 | Qwen/Qwen3.5-397B-A17B | `Qwen3_5MoeForConditionalGeneration` | 394B | drop | **PASS** | **PASS** |
| Qwen3.5 | Qwen/Qwen3.5-122B-A10B | `Qwen3_5MoeForConditionalGeneration` | 121B | drop | **PASS** | **PASS** |
| Qwen3.5 | Qwen/Qwen3.5-9B | `Qwen3_5ForConditionalGeneration` | 8.21B | drop | **PASS** T1-known-arch / S2+S4 | **PASS** T1+T3 / S2+S4 |
| *Qwen3-14B* | Qwen/Qwen3-14B | `Qwen3ForCausalLM` | 14.8B | drop `known_architecture` | drop `known_architecture_nothing_new` | PASS T3 / S2+S4 |
| *Llama-3.1-70B* | meta-llama/Llama-3.1-70B | `LlamaForCausalLM` | 70.6B | drop | drop `..._nothing_new` | PASS T3 / S1+S2+S4 |
| *Mixtral-8x7B* | mistralai/Mixtral-8x7B-v0.1 | `MixtralForCausalLM` | 46.7B | drop | drop `..._nothing_new` | PASS T3 / S1+S2+S4 |

**Per-target recall (a target hits when any of its repos passes):**

| arm | frontier releases | seeded controls |
|---|---|---|
| as-shipped | **0 / 9** | 0 / 3 (correct) |
| recheck=True | **9 / 9** | 0 / 3 (correct) |
| zero-day | **9 / 9** | 3 / 3 |

The `recheck` row is the important one and it is close to ideal: **full recall on the nine
frontier releases while all three seeded controls stay suppressed.** Qwen3-14B,
Llama-3.1-70B and Mixtral-8x7B each have **zero** unparsed fields and zero validator
findings, so they die at `known_architecture_nothing_new` — the discriminator is real, not
a coincidence of thresholds. The nine frontier configs carry 10-32 unparsed fields each.

### The classifier's substantive finding: 5 of 15 frontier configs abort BLIS

| repo | fatal validator | cause |
|---|---|---|
| deepseek-ai/DeepSeek-V4-Pro | `positive_intermediate_size` | no `intermediate_size`; only `moe_intermediate_size: 3072` |
| deepseek-ai/DeepSeek-V4-Flash | `positive_intermediate_size` | same |
| Qwen/Qwen3.5-397B-A17B | `positive_intermediate_size` | same (`moe_intermediate_size: 1024`) |
| Qwen/Qwen3.5-122B-A10B | `positive_intermediate_size` | same |
| MiniMaxAI/MiniMax-M3 | `swiglu_family_hidden_act` | `hidden_act: "swigluoai"` |

I re-derived the first four from BLIS source rather than trusting the surface:
`sim/latency/config.go:341` resolves the intermediate dim from **only**
`intermediate_size` / `ffn_hidden_size`; `moe_intermediate_size` is read separately at
`:354` into a different field. `IntermediateDim <= 0` is then a hard error at
`sim/latency/trained_physics_model.go:1027-1029` **and** `sim/latency/kv_capacity.go:419-421`.
So modern MoE-only configs — which omit a dense FFN width entirely — abort BLIS today.
This is a real BLIS bug the pipeline found on its first real run, on four current frontier
checkpoints. It is Bucket 0 (loud), so it would eventually have been noticed; finding it
before someone tries the config is the point.

### `silently_wrong` on the target list: 0 / 15

Not one of the fifteen frontier configs trips a *silent* validator. See (F) — the headline
metric fires elsewhere.

---

## (B) Precision — live scans (round 1)

`detector.scan()`, uncapped (`max_issues_per_run=500`) so the measurement is of the filter
and not of the reporting budget. Shipped defaults otherwise.

### 7-day window (the shipped default), all four sources

```
14,144 signals -> 6,993 candidates -> 37 survivors
drops: no_config_uncorroborated 6,776 | known_architecture 99 | insignificant 51
       not_a_language_model 27 | framework_title_only 2 | no_trigger 1
GitHub budget: 95 / 400 requests
```

**Hand categorization of all 37 survivors** (full detail in `precision-asis-7d.json`):

| # | verdict | survivor | why |
|---|---|---|---|
| 1 | **genuine** | `KimiK3ForCausalLM` (hf+inferencex, 5.47T, 32 unparsed) | Kimi K3; T1+T3+T4+T5, all four S codes |
| 2 | **noise (false merge)** | `MinistralForCausalLM` (hf+inferencex, 34.1B) | see (E) — Ministral fused with Qwen3.5-MoE and Bittensor spam |
| 3 | **genuine** | `Qwen3_8FlashNextForConditionalGeneration` (hf+vllm, 29 unparsed) | Qwen3.8-Flash-Next; the designed cross-source join |
| 4 | plausible-minor | `Spark2_5ForCausalLM` (3.17B) | real novel gating (`headwise_attn_output_gate`), small lab |
| 5 | **genuine** | `DeepSeek-V4-Pro` (inferencex) | frontier, already has a stub |
| 6 | noise | `DebertaV2ForMaskedLM` (70.4M) | encoder; `DebertaV2ForSequenceClassification` **is** seeded |
| 7 | noise | `DesklibAIDetectionModelV2` (433M) | AI-text detector |
| 8 | noise | `MetaLLMForCausalLM` | 550M hobby model |
| 9 | noise | `QiushiDualPathDebertaV2ForMaskedLM` (37.9M) | BabyLM entry |
| 10 | **genuine** | `Qwen4ExpTextForCausalLM` (47.8B, 40 unparsed) | Qwen3.8-Flash-Next text tower — a *split* of #3 |
| 11 | noise | `RobertaForCausalLM` (354M) | encoder; 4 Roberta siblings **are** seeded |
| 12 | plausible-minor | `RWKV7ForCausalLM` (1.5-13.3B, RWKV org) | genuinely novel non-transformer; BLIS cannot model it |
| 13 | noise | `T5ForConditionalGeneration` | seq2seq; `T5ForConditionalGeneration` absent from seed set |
| 14 | noise | `XLMRobertaForMaskedLM` (319M) | encoder; `XLMRobertaModel` **is** seeded |
| 15 | **genuine** | `Glm5vForConditionalGeneration` (743B) | GLM-5.3-Vision, reached only via a repacker's repo |
| 16 | noise | `ModernBertForMaskedLM` (320M) | encoder; `ModernBertModel` **is** seeded |
| 17 | plausible-minor | `SDARForCausalLM` (8.19B) | diffusion-LM research model |
| 18-34 | noise (17) | `BananaMind21{Coder,Lite25M,Test}ForCausalLM`, `BlazeForCausalLM`, `CustomResearchModel`, `GatedFlowLM`, `LilmForCausalLM`, `MuseGlimmerVisionModel`, `NanochronoForCausalLM`, `Pebble10MLM`, `PebbleForCausalLM`, `QForCausalLM`, `QUSSMForCausalLM`, `Qwen3BidirectionalModel`, `SpeckForCausalLM`, `TinyGDNForCausalLM`, `XoneLM` | 1M-484M hobby/toy/embedding/vision models and Bittensor subnet spam |
| 35 | noise | `Find attention with a fuser and attach vLLM's layer to it` | vLLM PR #54941 — a kernel change; the stub is named after the PR title |
| 36 | noise | `Switch output projection gemm (oproj_a) to fp8` | SGLang PR #37423 — same failure |
| 37 | noise | `gfx1250 on ROCM 10` | SGLang PR #36871 — same failure |

**Precision, uncapped:** 6 genuine + 5 plausible-minor + 26 noise = **70% noise**.
**Precision at the shipped cap of 5:** 3 genuine + 1 plausible-minor + 1 noise = **20%
noise**. The ranking function is doing real work — it is the only reason the survivor list
is usable.

### Noise rate per source (7-day, single-source scans)

| source | signals | candidates | survivors | genuine | minor | noise | noise rate |
|---|---|---|---|---|---|---|---|
| hf | 14,106 | 6,986 | 33 | 5 | 5 | 23 | **70%** |
| vllm | 8 | 8 | 1 | 0 | 0 | 1 | **100%** |
| sglang | 5 | 5 | 3 | 1 (`XllmForCausalLM`) | 0 | 2 | **67%** |
| inferencex | 24 | 5 | 5 | 5 | 0 | 0 | **0%** |

InferenceX is the highest-precision source by a wide margin — but 4 of its 5 survivors
(`Kimi-K3`, `GLM-5.2`, `MiniMax-M3`, `Qwen3.5-397B-A17B`) are **re-benchmarks of releases
that already have stubs in `issues/`**, so in a real run the `already_reported` suppressor
absorbs them. Its *new-information* rate is much lower than 100%.

### The non-LM precision bug is fixed

The instruction was to re-run if `Sam3VideoModel` or `Wav2Vec2ForPreTraining` appeared.
The fix landed as `ce1452c3` mid-run; the final runs show **neither**, and the new
`not_a_language_model` suppressor drops 26-27 candidates per 7-day window. The
`strong_non_lm_evidence` pre-filter and the `lm_shape_evidence` suppressor both work as
described. What they do **not** catch is encoder models — an encoder keeps all four core
transformer dimensions, so it is indistinguishable from an LM by shape. Six of the 26 noise
survivors are encoders (see below).

### 1-day window, for comparison

```
2,494 signals -> 1,447 candidates -> 5 survivors
survivors: Spark2_5ForCausalLM, RobertaForCausalLM, XLMRobertaForMaskedLM,
           MicroIvoireTransformer17M (17.1M), Qwen2_5_VLForConditionalGenerationWithVGGT
```
On this particular day **0 of 5 were genuine frontier releases** (1 plausible-minor, 4
noise). A one-day window is too narrow to contain a frontier release, so it measures only
the noise floor. **Use the 7-day default.**

---

## (C) GitHub-source historical replay (round 1)

Window `2026-08-10 .. 2026-08-21`, chosen because a `gh api search/issues` query showed it
contains several merged model-support PRs. `FrameworkConnector(until=...)` bounds it
server-side; `InferenceXConnector` has **no `until=`** (addendum 12 is unimplemented
there), so the harness bounds it client-side on `observed_at`.

```
52 signals in window (vllm 14, sglang 6, inferencex 32 of 78 fetched)
GitHub budget: 178 / 400
27 candidates -> 10 passed
drops: framework_title_only 9 | known_architecture 8
```

**All four expected model-support PRs are present in the signal set** — #51655 (Muse
Glimmer), #51255 (Dots3 NOTE), #52114 (Ling MXFP4), #52706 (GraniteSWA). The connector
found them; **none passed**, because all eight architectures they add are in the
today-harvested seed set. Same seed-set-from-the-future effect as (A).

Zero-day arm (forget the 20 architectures the window's PRs name): **18 passed, 8 of them
framework-backed**, including #51655, #51255 and #52706:

```
DeepseekV32MTPModel        T2+T3+T4 / S2+S3   (PRs 30519, 52861)
DFlashLagunaForCausalLM    T2+T4    / S3      (PRs 34262, 35371, 51655)
GraniteMoeSWAForCausalLM   T2+T3    / S2+S3   (PR 52706)
Dots3NoteForCausalLM       T2       / S3      (PR 51255)
+ DeepseekV3ForCausalLM, LlavaNextForConditionalGeneration,
  TransformersForSequenceClassification, Qwen3_5ForCausalLM
```

So **the framework path works end to end** — T2 fires, T4 fires when two frameworks land
the same architecture — once the seed set does not already contain the answer.

**Idempotence and the stateless dedup, measured:** re-writing the same 10 candidates
produced 10 × `unchanged` (zero bytes changed), and re-evaluating against the now-populated
directory suppressed all 10 as `already_reported`, 0 passing. Both work.

**One truncation risk:** InferenceX logged `133 commits in window, capped at 80`
(`max_commits=80`) for an 11-day window. A 30-day backtest window silently loses commits.
`max_commits` needs to scale with the window, or the truncation needs to reach
`RunSummary` rather than only a log line.

---

## (D) Threshold calibration (round 1)

One live HF poll (2,494 signals, 1-day window incl. the trending sweep) reused across all
16 sweep points, so every row differs only in the swept value. Recall measured in both the
zero-day and as-shipped arms.

| `recheck` | `min_total_params` | zero-day recall | as-shipped recall | HF survivors | of which `silently_wrong` | of which bucket 0 |
|---|---|---|---|---|---|---|
| False | 1B | 9/9 | **0/9** | 11 | 0 | 3 |
| False | 3B | 9/9 | **0/9** | 10 | 0 | 3 |
| False | 7B | 9/9 | **0/9** | 6 | 0 | 3 |
| False | **30B** (shipped) | 9/9 | **0/9** | 5 | 0 | 3 |
| False | 400B | 9/9 | **0/9** | 5 | 0 | 3 |
| True | 1B | 9/9 | **9/9** | 46 | **1** | 10 |
| True | 3B | 9/9 | **9/9** | 40 | **1** | 10 |
| True | 7B | 9/9 | **9/9** | 31 | 0 | 10 |
| True | 15B | 9/9 | **9/9** | 27 | 0 | 10 |
| True | 30B | 9/9 | **9/9** | 27 | 0 | 10 |
| True | 400B | 9/9 | **9/9** | 25 | 0 | 9 |

### `min_total_params` — recommendation: **3 × 10⁹ (3B)**, down from 30B

* It has **no effect on recall at all** across three orders of magnitude, in either arm.
  Every frontier target satisfies S2 by frontier-org membership, so S1 never decides
  anything for them. **S1 is not a recall gate; it is a volume knob.**
* As a volume knob it is nearly flat above ~15B: 11 → 10 → 6 → 5 survivors from 1B to 15B,
  then unchanged to 400B. The shipped 30B buys nothing over 15B.
* And it has a real cost. The **only** `silently_wrong` candidate in the window is a 4.02B
  model whose only significance code is S1. At `min_total_params ≥ 7B` it is dropped as
  `insignificant`. **The shipped threshold suppresses the headline metric.**
* 3B keeps it at a cost of 5 extra survivors per day (10 vs 5 at `recheck=False`, 40 vs 27
  at `recheck=True`).

**Caveat, stated plainly:** every target on the list is from a `FRONTIER_ORGS` lab, so this
sweep cannot measure what `min_total_params` does for a *non*-frontier org's release —
where S1 is the only gate that could fire. That case is unmeasured; the recommendation is
sound for the measured population only.

**Better than tuning the number:** exempt the `silently_wrong` class from the significance
gate entirely. A silent misread is a BLIS *correctness* defect, and correctness does not
scale with parameter count. A 4B model that makes BLIS silently mis-size KV is the same bug
as a 4T one.

### `recheck_known_architectures` — recommendation: **True**

| | recheck=False | recheck=True |
|---|---|---|
| as-shipped recall, frontier targets | **0 / 9** | **9 / 9** |
| seeded controls wrongly surfaced | 0 / 3 | **0 / 3** |
| HF survivors per day (at 30B) | 5 | 27 |
| `silently_wrong` found per day | 0 | 1 (only at ≤3B) |
| bucket-0 findings per day | 3 | 10 |

Recall gained: **+9 of 9**. Extra volume: **+22 survivors/day (5.4×)**. And the extra
volume is *better*, not worse: at `recheck=False` the 5 survivors on this day were 4 noise
+ 1 plausible-minor and **zero** genuine frontier releases. At `recheck=True` the 27 include
`DeepseekV4ForCausalLM`, `Qwen3_5MoeForCausalLM`, `Qwen4ExpForCausalLM`,
`Glm5NextForConditionalGeneration`, `GlmMoeDsaForCausalLM`, `HYV4ForCausalLM`,
`KimiK3ForConditionalGeneration`, `MuseGlimmerForConditionalGeneration`,
`K2HorizonForCausalLM`, `Gemma4ForConditionalGeneration`, `BailingMoeV3ForCausalLM`,
`NemotronHForCausalLM`, `GptOssForCausalLM`, `HCXVisionV2ForCausalLM`, `OuroForCausalLM`,
`Qwen3_5ForCausalLM`, `Qwen3DSparkModel` — hand-categorized as **17 genuine frontier
architectures, 3 plausible-minor and 7 noise**. The added noise is old architectures whose
fine-tunes carry a stray field (`BertForMaskedLM`, `GPT2LMHeadModel`, `Qwen2ForCausalLM`,
`Qwen3ForCausalLM`, `RobertaForCausalLM`, `XLMRobertaForMaskedLM`) plus one toy model.

The argument is not marginal: **with the shipped default the pipeline cannot see any
release vLLM already supports, which after a few weeks is every release.** Because the seed
set is regenerated from vLLM's registry, `recheck=False` means archwatch can only ever
report architectures in the days-long gap before vLLM lands support — and it must
re-harvest the seed set to stay useful, which shrinks that gap further. `recheck=True` makes
the pipeline useful continuously.

**Recommended shipped defaults:**

```python
min_total_params: int = 3_000_000_000        # was 30_000_000_000
recheck_known_architectures: bool = True     # was False
max_issues_per_run: int = 10                 # was 5; see below
```

`max_issues_per_run=5` is too small at `recheck=True`: 27 survivors means 22 `over_cap`
drops per day, and the false-merged `MinistralForCausalLM` occupied slot 2 of 5 in the
7-day run. 10 keeps the genuine frontier set inside the budget. (`config.py` is frozen — I
did not change it. These are recommendations for whoever owns it.)

> **All three were applied** by `config.py`'s owner, with this evidence quoted in the code
> comments. Round 2 above re-measures the result: recall 9/9, the `silently_wrong` finding
> reported at rank 9 of 90, and 10% noise inside the cap.

---

## (E) False-merge audit (round 1)

`Candidate.join_edges` was inspected across every candidate in every run, with an
over-eager pure heuristic (`audit_merge`) flagging: two distinct architecture spellings
fused, different parameter-size tokens fused, or two orgs fused on a family edge alone.
Across the join step: **16 merges, 8 family-edge-only, 8 flagged.** Hand adjudication:

### Confirmed false merge — reaches the output

**`DeepseekV3ForCausalLM` fusing DeepSeek V3 + DeepSeek V4 + Qwen3-MoE + Qwen3.5-MoE.**
33 signals, 4 sources, 6 architecture spellings, one candidate, one stub. Under
`recheck=True` it **passes** with `T1-known-arch / S1+S2+S3+S4` and ranks first.

Root cause, isolated: **SGLang PR #35634, "[Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A
backend"** — an all-to-all *backend* change, not model support. It touches one
model-registry file, and the connector's prose extractor mines four unrelated architecture
names from it. `signal_edges` emits an `arch:` edge for **every** entry in `arch_ids`, so
that single signal is a **clique** joining all four families; the union-find then pulls in
the HF signals for `deepseek-ai/DeepSeek-V4-Pro`, `Qwen/Qwen3.5-397B-A17B` and
`Qwen/Qwen3.5-122B-A10B` transitively. The size-token check caught it independently
(`17b/397b` fused with `10b/122b`).

**This contradicts the task's expectation that `family:` is the edge most likely to be
wrong.** The dominant false-merge mechanism observed live is *one signal carrying many
architectures*, and it fires on `arch:` — the edge the design calls "the strongest". A
one-line reproduction is pinned in `tests/test_backtest.py::test_multi_architecture_signal_fuses_unrelated_families_and_is_flagged`.

Suggested fix (for the join's owner, not applied here): emit `arch:` edges for **all**
`arch_ids` only when the signal has ≤ N of them (2-3), or only for `arch_ids[0]` when the
signal is not a `registry`-strength framework signal. A backend PR mentioning four
families is evidence about none of them.

### Second confirmed false merge — wrong name on a real finding

**`DFlashLagunaForCausalLM` swallowing Muse Glimmer** (replay zero-day arm, PRs 34262,
35371, 51655). SGLang PR #34262 is titled "Muse Glimmer" and exports
`DFlashLagunaForCausalLM`, `MuseGlimmerForCausalLM` and `MuseGlimmerForConditionalGeneration`
in one `EntryClass`. Muse Glimmer *is* surfaced — but the stub is named
`DFlashLagunaForCausalLM`, so a human looking for Muse Glimmer will not find the file. Same
mechanism (one signal, many architectures), less damage.

### Third confirmed false merge — a 3-character family key

**`MinistralForCausalLM`** (7-day precision, slot 2 of 5). Merge edges include
`family:asd`, `family:albedoqwen3635bwonderc4`, `family:sn120524fd7702d1f` and
`arch:qwen35moeforconditionalgeneration`; model types are `ministral`, `qwen3_5_moe`,
`qwen3_5_moe_text`; model ids are dozens of Bittensor `Affine-*` subnet repos across dozens
of orgs. **`MIN_FAMILY_KEY_LEN = 3` is too low** — a 3-character family key such as `asd`
carries no information and merges everything that happens to be named that way. Raising the
floor to 5-6 would have prevented this merge without touching any real family key on the
target list (shortest observed: `kimik3`, `glm52`, `qwen4exp` — all ≥ 5).

### Adjudicated benign

* `Qwen4ExpForCausalLM` ← `Qwen4ExpForConditionalGeneration` + `Qwen4ExpMTP` — one release,
  three heads. Correct.
* `KimiK3ForConditionalGeneration` ← `KimiK3LinearForCausalLM` — same SGLang PR, plausibly
  the same release's two attention variants. Marginal; worth a human's eye, not a bug.
* `K2HorizonForCausalLM` ← `XllmForCausalLM` — one SGLang PR exporting both. Addendum 10
  already documents this PR as yielding "two unrelated names". Marginal.
* `Glm5Next*` (4 spellings from vLLM PR #53906) — one release. Correct.
* `family:kimik3` bridging InferenceX `Kimi-K3` to HF `moonshotai/Kimi-K3` — **the join
  working exactly as designed**, and the only reason T4 can fire across sources.

### One thing the join does that nobody designed: it can *destroy* a signal

Reproduced deterministically. Polling InferenceX alone over a 1-day window yields
`Kimi-K3`, which **passes** (T5 / S3). Polling HF + InferenceX over the same window, the
InferenceX signal correctly joins the HF trending signal for `moonshotai/Kimi-K3` on
`family:kimik3`, the merged candidate takes the HF architecture name
`KimiK3ForConditionalGeneration` — which is in the seed set — and the whole candidate is
**dropped as `known_architecture`**. Candidate count went from 1,451 (disjoint) to 1,450;
exactly one merge happened and it cost the only zero-day signal in the run.

Addendum 26 deliberately does not exempt `known_architecture` from the curated-source
exemption, so this is the design working as written — but nobody intended a *successful*
cross-source join to convert a pass into a suppression. Under `recheck=True` it survives as
`T1-known-arch`, which is a third independent argument for that default.

### The other naming defect

`_canonical_label` prefers `arch_ids[0]` of an HF signal, breaking ties lexicographically.
In the 7-day run that named the Kimi K3 candidate `KimiK3ForCausalLM` — a class published
by a third party's `kimi_k3_test_*` repos — over moonshotai's own
`KimiK3ForConditionalGeneration`. In the replay it named a DeepSeek V3.2 candidate
`DeepseekV32MTPModel`, i.e. after the MTP side-module rather than the model, because that
is what the miner put first in `arch_ids`. Cosmetic, but the arch id is the filename and
the dedup key.

---

## (F) `silently_wrong` — the headline metric (round 1)

Counted from stub front matter, testing for **keys, not the schema string** (addendum 22 —
which was already necessary: the emitter ships `archwatch/3` where the plan said
`archwatch/2`). `missing_front_matter_keys` was empty for all 14 required keys on every
stub in every run, and the stage-2 handshake was read through `emitter.split_stub`
(addendum 3), never a front-matter flag.

**Across every run: exactly one `silently_wrong` candidate.**

```
Qwen3NextForCausalLM   from  arianraje/qwen3-4b-gdn-hybrid-stage3-200M-OPD-dtfix
est_total_params: 4.02B     triggers: [T1-known-arch]     significance: [S1]
silent_failures:
  [moe_expert_count_resolvable]  num_experts_per_tok=10 signals MoE but no total expert
      count resolved >= 2 from any known spelling (num_experts, moe_num_experts,
      n_routed_experts, num_local_experts, num_routed_experts)
      -> warned and degraded on trained-physics; rejected on roofline
  [moe_total_required_when_active_present]  active experts per token (10) is set but the
      resolved total expert count is 0
bucket0_failures: []        bucket: null
```

This is the class addendum 21 says justifies the pipeline: BLIS accepts the config, runs,
warns once through `logrus`, and simulates a sparse MoE as a dense model. It is exactly the
case addendum 24 predicted — **zero unparsed fields**, only recognized spellings, silently
misread. If T1 had stayed gated on `unparsed_fields` alone it would have been invisible.

**Rate: 1 in 1,446 candidates (1 in 2,494 raw signals) per HuggingFace day. Zero on the
15-target frontier list.** And it is visible only with **both** recommended defaults:

| configuration | is it reported? |
|---|---|
| shipped (`recheck=False`, `min_total=30B`) | **no** — dropped at `known_architecture` |
| `recheck=True`, `min_total=30B` | **no** — dropped at `insignificant` (4.02B < 30B) |
| `recheck=True`, `min_total=3B` | **yes** |

Two unrelated shipped defaults each independently hide the one finding class the pipeline
exists for. That is the strongest single result in this report.

For context, bucket 0 (the loud class) fires on **57 of 1,446** candidates per day — 57× more
often than `silently_wrong`. Most are not interesting: 26-27 are non-LM repos the new
suppressor drops, and most of the rest are encoders. `silently_wrong` is rare, and rarity is
the argument for reporting it, not against.

---

## What did not work (round 1) — see round 2 above for status

1. **`recheck_known_architectures=False` gives zero recall on every named frontier
   release.** Not degraded — zero. `0/9`. The cold-start seed set is regenerated from
   vLLM's registry, so it converges on "everything already released", and the suppressor
   that consults it first drops everything.

2. **`min_total_params=30B` suppresses the only `silently_wrong` finding in a live day**
   and buys no volume reduction over 15B and no recall at all.

3. **A confirmed false merge reaches the output.** One SGLang backend PR fuses DeepSeek V3,
   DeepSeek V4, Qwen3-MoE and Qwen3.5-MoE into a single candidate that ranks first. Cause:
   `arch:` edges are emitted for every entry of a multi-architecture signal's `arch_ids`.
   Two more confirmed (Muse Glimmer under the wrong name; Ministral via a 3-character
   family key).

4. **A successful cross-source join can convert a pass into a suppression.** Joining
   InferenceX's `Kimi-K3` into HF's `moonshotai/Kimi-K3` renames the candidate to a seeded
   architecture and drops it. Reproduced deterministically.

5. **`framework_title_only` is keyed on the wrong field.** It suppresses signals whose
   `extra["signal_strength"] == "title_only"`, but the failing class is **`arch_ids == []`**
   at any strength. Three `new_model_file`-strength PRs with zero extracted architectures
   survived a 7-day run and were emitted as stubs titled *"Find attention with a fuser and
   attach vLLM's layer to it"*, *"Switch output projection gemm (oproj_a) to fp8"* and
   *"gfx1250 on ROCM 10"*. The condition should be "no signal yielded an architecture
   name", not "the connector labelled it title_only".

6. **Encoder and seq2seq models are 8 of the 26 noise survivors (31%), in two distinct
   sub-cases, and neither existing suppressor sees them.** An encoder keeps all four core
   transformer dimensions, so `lm_shape_evidence` cannot separate it from a real LM, and its
   `hidden_act: "gelu"` reliably trips the fatal `swiglu_family_hidden_act` validator — which
   for a 183M-parameter BERT variant means "this was never a decoder", not "BLIS cannot
   handle a novel architecture".

   *Sub-case 6a — the task-suffix gap (3 survivors).* `RobertaForCausalLM`,
   `XLMRobertaForMaskedLM` and `ModernBertForMaskedLM` are each absent from the seed set
   while 3-4 of their own siblings are present, and
   `Surface.related_known_architectures()` names those siblings. This is where the
   docstring's defence of exact matching contains a factual error: point 1 argues "All 37
   groups that suffix-stripping would merge consist entirely of names ALREADY in the seed
   set, so normalization changes no answer." That is true *within the seed set*, but the seed
   set is not the population — the population is what HuggingFace publishes, and it published
   all three of these. The suppression-asymmetry argument (points 3-5) still stands, so the
   fix should be narrow rather than general suffix-stripping: suppress only a candidate whose
   *only* trigger is T1, whose `est_total_params` is under a floor, and whose family has a
   seeded sibling differing solely in task head.

   *Sub-case 6b — families vLLM's registry never covered (5 survivors).*
   `DebertaV2ForMaskedLM`, `QiushiDualPathDebertaV2ForMaskedLM`, `DesklibAIDetectionModelV2`
   (all `model_type: deberta-v2`), `T5ForConditionalGeneration` and `Qwen3BidirectionalModel`.
   The seed set contains **no** Deberta and **no** T5 entry at all, so no sibling rule can
   help. These need a positive "is this a decoder?" test — `model_type` against a non-causal
   family list, in the spirit of the new `STRONG_NON_LM_MODEL_TYPES`, or an
   `is_decoder`/`is_encoder_decoder` config check.

7. **`DERIVATIVE_PATTERNS` is missing modern quantization tokens.** `nvfp4`, `mxfp4`, bare
   `-fp4` and `aqlm` are absent, so `jarrelscy/GLM-5.3-Vision-NVFP4-AQLM-hybrid` reached the
   survivor list as the sole evidence for `Glm5vForConditionalGeneration`.
   `novelty.QUANT_REPO_SUFFIXES` already lists `-nvfp4` and `-mxfp4`; the two lists have
   drifted.

8. **`InferenceXConnector` does not implement addendum 12's `until=`**, and its
   `max_commits=80` truncated an 11-day window that held 133 commits. The truncation is a log
   line only; it does not reach `RunSummary`, so a caller cannot tell a quiet window from a
   truncated one.

9. **The InferenceX name parser produces junk arch ids.** The replay emitted stubs named
   `2.7` and `2.7-Code` (from Kimi K2.7 rows) and `kimik2.6`. Harmless to the filter,
   embarrassing in a filename, and it defeats the family edge.

10. **A one-day window measures only the noise floor.** 0 of 5 survivors were genuine on the
    day tested. Frontier releases are weekly-to-monthly events; the 7-day default is the
    minimum useful window.

11. **`CustomResearchModel` shows the primary-key assumption failing.** 18 distinct
    `model_type` values and dozens of orgs collapsed onto one arch string that unrelated
    student projects all chose. The architecture string is only a good primary key when labs
    pick distinctive names.

---

---

# Known limitations

Carried forward from both rounds. Everything here is either unmeasured or unmeasurable with
the method available, and none of it was closed by the fixes.

* **A true HuggingFace historical replay.** No server-side date filter; a 30-day window is
  ~100,000 records behind the head of a `created_at`-descending listing. Replaced by the
  direct-fetch recall test, which measures the same question on the same data.
* **`min_total_params` for a non-frontier org.** Every target is a `FRONTIER_ORGS` lab, so
  S2 always fires and S1 never decides. The threshold's behaviour where it is the *only*
  gate is untested.
* **True zero-day recall.** The seed set post-dates every release on the list, so recall
  had to be measured against a counterfactual surface. It is a narrow and honest
  counterfactual (exact-string removal against exact-match membership), but it is not a
  measurement of the real system on the real day. The only way to get that number is to run
  archwatch prospectively and wait for the next frontier release.
* **S2's open-world path** (`min_org_top_downloads`, the org-download sweep). Every target
  and every survivor satisfied S2 by frontier-org membership or failed it outright; the
  sweep never decided an outcome, so `100_000` is untested.
* **S4's threshold values.** S4 fired on 32 of 37 round-1 survivors and 59 of 90 round-2
  survivors, almost always via the trending sweep or large fine-tune download counts, so
  `min_model_downloads`/`min_model_likes` were never the binding constraint and could not be
  calibrated. Worth noting for round 2 specifically: S4 is now the most-fired significance
  code, so a large share of the survivor tail is admitted by *fine-tune popularity* rather
  than by anything about the architecture.
* **Whether stage 2 reaches correct buckets.** That is component I's acceptance criterion and
  needs a human running the skill. What J verifies is the handshake: `split_stub` finds the
  marker, the appendix is empty on every stub written, and `bucket` is `0` or `null` only
  (addendum 6) — 16 of 37 round-1 stubs and 24 of 90 round-2 stubs carried `bucket: 0`, the
  rest `null`, never 1-3. Required front-matter keys were present on every stub in both
  rounds, tested by key rather than by `schema:` string (addendum 22 — necessary, since the
  emitter ships `archwatch/3` where the plan said `archwatch/2`).
* **Sustained multi-week behaviour, and the dedup's effect on it.** Every number in this
  document is one or two windows on a single day, in each round. Because the stateless dedup
  suppresses any architecture that already has a stub, the *second* week's survivor list is a
  different and probably much smaller population than the first. Round 2's 90 survivors are a
  cold-start number; the steady-state weekly number is unknown and could be far lower. Nothing
  here measures it, and only running archwatch on a schedule for a month will.
* **Whether the ranking function is right.** Round 2 makes it load-bearing — 90 survivors, 10
  shown — but the suppressors have had far more scrutiny than `_strength()`. The cap's 10%
  noise rate is one week's sample.
* **The surviving `repo:`-edge false merge is characterized but not bounded.** One instance
  was found and traced. How often a framework PR mentions a base model it is not about, across
  a longer history, is unmeasured.
