# EDPP coefficient calibration

Offline, pure-regime fitting of the E3 latency-law coefficients that EDPP
consumes as inputs:

    T_iter = alpha + c0*B_dec + c1*KV + c_pf*S_pf  (+ c_attn*pf_ctx)

(see `docs/superpowers/specs/2026-06-18-edpp-dpp-routing-design.md`, §3).

## 1. Tap the per-step data

The simulator emits one CSV row per executed engine step when `BLIS_STEP_CSV`
is set (off otherwise; no effect on normal runs or determinism). Columns:
`step_idx, t_iter_us, b_dec, kv, s_pf, pf_ctx, batch_size`.

`kv` is the exact summed resident decode context (Σ ProgressIndex). `pf_ctx` is
the attention basis Σ tᵢ·(sᵢ+tᵢ/2) for recovering `c_attn`.

> **Flag-range gotcha:** `blis run` rejects a config unless the *mean* AND the
> *stdev* both fall inside `[--prompt-tokens-min, --prompt-tokens-max]` (same for
> output). Defaults are mean 512 / stdev 256 / min 2 / max 7000. So: for prompts
> > 7000 raise only `--prompt-tokens-max` (leave min/stdev default); for ~1-token
> outputs set `--output-tokens-min 1`. Setting `min` above the default `stdev`
> (256) triggers a confusing "should be in range" fatal — don't.

## 2. Calibration runs (fit the coefficients)

Keep each run in one regime so coefficients are independently identifiable, and
spread `(b_dec, kv)` in 2-D across the decode runs — otherwise `KV ≈ B·Lbar` is a
thin line and the `c0/c1` split is meaningless (the fitter warns on this).

```bash
# Decode fit (alpha, c0, c1): 4 runs spanning batch size × context length.
BLIS_STEP_CSV=D1.csv ./blis run --model qwen/qwen3-14b --num-requests 150 --rate 6  --prompt-tokens 128  --output-tokens 2000 --max-num-scheduled-tokens 8192 --max-model-len 8192
BLIS_STEP_CSV=D2.csv ./blis run --model qwen/qwen3-14b --num-requests 150 --rate 24 --prompt-tokens 128  --output-tokens 2000 --max-num-scheduled-tokens 8192 --max-model-len 8192
BLIS_STEP_CSV=D3.csv ./blis run --model qwen/qwen3-14b --num-requests 150 --rate 24 --prompt-tokens 2000 --output-tokens 2000 --max-num-scheduled-tokens 8192 --max-model-len 8192
BLIS_STEP_CSV=D4.csv ./blis run --model qwen/qwen3-14b --num-requests 150 --rate 6  --prompt-tokens 2000 --output-tokens 500  --max-num-scheduled-tokens 8192 --max-model-len 8192

# Prefill fit (alpha_p, c_pf, c_attn): vary chunk size AND prompt length so c_pf
# (∝ S_pf) separates from c_attn (∝ pf_ctx). Low rate → one prefill at a time → B_dec=0.
BLIS_STEP_CSV=P1.csv ./blis run --model qwen/qwen3-14b --num-requests 120 --rate 3 --prompt-tokens 1000  --max-num-scheduled-tokens 512  --max-model-len 20000 --output-tokens 2 --output-tokens-stdev 0 --output-tokens-min 1 --output-tokens-max 4
BLIS_STEP_CSV=P2.csv ./blis run --model qwen/qwen3-14b --num-requests 120 --rate 3 --prompt-tokens 4000  --max-num-scheduled-tokens 2048 --max-model-len 20000 --output-tokens 2 --output-tokens-stdev 0 --output-tokens-min 1 --output-tokens-max 4
BLIS_STEP_CSV=P3.csv ./blis run --model qwen/qwen3-14b --num-requests 120 --rate 3 --prompt-tokens 16000 --max-num-scheduled-tokens 4096 --max-model-len 20000 --prompt-tokens-max 20000 --output-tokens 2 --output-tokens-stdev 0 --output-tokens-min 1 --output-tokens-max 4
```

## 3. Fit and freeze

```bash
python scripts/calibration/fit_coeffs.py D1.csv D2.csv D3.csv D4.csv P1.csv P2.csv P3.csv -o coeffs.json
```

`coeffs.json` holds `alpha, alpha_p, c0, c1, c_pf, c_attn`, R², row counts, and
collinearity diagnostics (`cond_*` should be well under 30).

> **Provenance of the frozen file.** The commands above use `qwen/qwen3-14b` as a
> worked example, but the committed `coeffs-llama70b-h100-tp4.json` was fit from
> `meta-llama/llama-3.3-70b-instruct --hardware H100 --tp 4` (its `source_csvs`
> point at a since-deleted `/tmp/llama70/*.csv` set). `repro_llama70b.sh` is the
> exact Llama-70B instantiation of this procedure; because the calibration runs
> are deterministic, it regenerates all six coefficients **bit-exactly** — the
> trust-check that the frozen file is reproducible. Run:
> `bash scripts/calibration/repro_llama70b.sh` (prints `CHECKPOINT: PASS`).

## 4. Validate the additive model across all three regimes

One mixed run biased to each regime; regime = **marginal** prefill share
`c_pf·S_pf / (c_pf·S_pf + c0·B_dec + c1·KV)` (α excluded — it's additive and
shared, so it carries no prefill-vs-decode information).

```bash
BLIS_STEP_CSV=M_decode.csv  ./blis run --model qwen/qwen3-14b --num-requests 250 --rate 22 --prompt-tokens 256  --output-tokens 1500 --max-num-scheduled-tokens 256
BLIS_STEP_CSV=M_ridge.csv   ./blis run --model qwen/qwen3-14b --num-requests 250 --rate 12 --prompt-tokens 1024 --output-tokens 512  --max-num-scheduled-tokens 1024
BLIS_STEP_CSV=M_prefill.csv ./blis run --model qwen/qwen3-14b --num-requests 200 --rate 5  --prompt-tokens 8000 --output-tokens 32 --output-tokens-stdev 8 --output-tokens-min 4 --output-tokens-max 64 --prompt-tokens-max 20000 --max-num-scheduled-tokens 8192 --max-model-len 12000

# --validate accepts multiple files; mixed rows are pooled into one report.
python scripts/calibration/fit_coeffs.py D*.csv P*.csv -o coeffs.json \
  --validate M_decode.csv M_ridge.csv M_prefill.csv
```

The `validation` block reports MAPE/RMSE stratified by regime. Expect a good fit
in the decode/prefill-bound corners and the worst error `near_ridge` — E3 is a
local linearization of the model's `max(compute, memory)` truth (design line 121).
That degradation is the empirical answer to whether frozen, additive
coefficients are safe to feed EDPP.
