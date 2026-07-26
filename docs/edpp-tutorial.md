# EDPP: A Tutorial

EDPP (the Lyapunov‑DPP disaggregation policy) decides, for each arriving request,
whether to run its **prefill** on a dedicated prefill server (*disaggregate*) or to
keep prefill co‑located with **decode** on the same server. This document builds the
policy up from first principles. We assume nothing about the reader except a working
mental model of LLM inference: a request is *prefilled* (its prompt is processed in
one or a few large forward passes) and then *decoded* (output tokens are generated one
per iteration).

This first part covers the foundations that everything else rests on:

1. The **coefficients** — the frozen numbers that describe the hardware/model.
2. **Prefill work** — what it is, its formula, and how each term is bookkept.
3. **Decode work** — same.
4. **Iteration time** — the per‑step time model, and how each term is tracked.

Later parts build the backlogs, the SLO virtual queues, and the actual routing
decision rule on top of these.

> **Scope note.** EDPP is implemented as a cluster‑level routing policy inside BLIS
> (the Blackbox Inference Simulator). Code references below point at
> `inference-sim/sim/edpp_coeffs.go` (the cost model) and `inference-sim/sim/edpp.go`
> (the policy). All times are in **microseconds (µs)** unless stated otherwise.

---

## 1. The coefficients

EDPP never measures latency at run time. Instead it carries a small set of **frozen
coefficients** that describe how long work takes on a given (model, GPU, tensor‑parallel)
combination. They are fit *offline* — once, from calibration data — and then loaded at
startup and never re‑estimated. Everything the policy computes (work, iteration time,
drain rates, the decision itself) is a closed‑form function of these six numbers.

They come from a single latency law (the "E3 law") that says the time of one model
iteration is an **affine** function of the batch's composition:

```
T_iter  =  α  +  c0 · (#decode requests)  +  c1 · (resident KV tokens)  +  c_pf · (prefill tokens this iteration)
```

The six coefficients are defined in `sim/edpp_coeffs.go:12‑19`:

| Symbol | Field   | Units        | Meaning |
|--------|---------|--------------|---------|
| α      | `AlphaD`| µs           | **Decode per‑iteration fixed cost.** Paid once every decode iteration regardless of batch size — weight loads, kernel launches, synchronization. Dominates at small batch. |
| α_p    | `AlphaP`| µs           | **Prefill per‑iteration fixed cost.** Same role for a prefill iteration. On the same hardware it is ≈ α (the loader enforces `|α − α_p|/α < 10%`, `edpp_coeffs.go:125`). |
| c0     | `C0`    | µs / request | **Decode per‑request overhead.** Marginal cost of having one more request in the decode batch, independent of how long its context is. |
| c1     | `C1`    | µs / token   | **Decode KV‑read cost per resident token.** Marginal cost per token of context already in the KV cache — this is the memory‑bandwidth term that makes long contexts expensive to decode. |
| c_pf   | `CPf`   | µs / token   | **Exposed prefill compute per token** (also written `k_p`). Marginal cost of processing one prompt token during prefill — compute‑bound. |
| c_attn | `CAttn` | µs / token²  | **Prefill attention term.** The quadratic part of prefill: attention cost grows with the *square* of prompt length. |

### Where they come from

They are loaded once from a calibration JSON via `LoadEDPPCoeffs` (`edpp_coeffs.go:38‑59`).
The on‑disk shape (e.g. `scripts/calibration/coeffs-llama70b-h100-tp4.json`) is nested
under `decode` and `prefill`:

```json
{
  "decode":  { "alpha_us": 16613.54, "c0_us_per_req": 5.347, "c1_us_per_token": 0.04761 },
  "prefill": { "alpha_p_us": 16617.95, "c_pf_us_per_token": 6.145, "c_attn_us_per_unit": 0.0001008 }
}
```

The loader maps these into the `EDPPCoeffs` struct and then **validates** them
(`edpp_coeffs.go:110‑129`): α, α_p, c_pf must be strictly positive; c0, c1, c_attn must
be non‑negative; and α and α_p must agree to within 10% (a sanity check that the decode
and prefill fits came from the same hardware/regime — they share the same physical
per‑iteration intercept).

### Why "frozen" matters

Because the coefficients never change during a run, the policy is **deterministic and
oracle‑safe**: it cannot "learn" from the future, and two runs with the same inputs
produce identical decisions. The coefficients are the *only* hardware knowledge in the
policy; everything downstream is arithmetic on top of them.

Reading the numbers above gives intuition for the regime:
- α ≈ 16.6 **ms** — the fixed per‑iteration cost utterly dominates the marginal terms
  (which are single‑ to tens‑of‑µs each). This is the small‑batch, latency‑bound regime
  where amortizing α across more useful work is the whole game.
- c_pf (6.1 µs/tok) > c1 (0.048 µs/tok): a prompt token costs ~130× more to *prefill*
  than a resident token costs to *re‑read* during a decode step. That asymmetry is why
  prefill is worth isolating.

---

## 2. Prefill work (`W_p`)

**Work** is EDPP's currency. It is not wall‑clock time and not GPU‑seconds — it is the
*marginal service demand* a request places on a server, measured in µs of exposed
compute. Prefill work answers: "how much prefill effort does this request require?"

### Formula

For a request with `a_p` **uncached** prompt tokens (`edpp_coeffs.go:61‑65`):

```
W_p(a_p)  =  c_pf · a_p  +  (c_attn / 2) · a_p²
```

```go
func (c EDPPCoeffs) Wp(ap int) float64 {
	a := float64(ap)
	return c.CPf*a + (c.CAttn/2.0)*a*a
}
```

### Each term

- **`c_pf · a_p`** — the linear compute term. Every uncached prompt token costs `c_pf`
  µs of prefill compute. This is the bulk of prefill work for short‑to‑medium prompts.
- **`(c_attn / 2) · a_p²`** — the quadratic attention term. Attention is a sum over token
  *pairs*, so its cost scales as `a_p²`; the factor of ½ reflects that each pair is
  counted once. Negligible for short prompts, dominant for very long ones.

### The one term to bookkeep: `a_p` (uncached tokens)

`W_p` has exactly one per‑request input, `a_p`, and getting it right is the whole
bookkeeping job. `a_p` is **not** the full prompt length — it is the number of prompt
tokens that are *not already served by the prefix cache*. If a request shares a prompt
prefix with something already cached, those tokens cost no prefill.

Crucially, `a_p` is knowable **at arrival**, from the input side only (prompt length
minus prefix‑cache hit). It never reads `Request.OutputTokens`, so it respects the
oracle boundary (INV‑9): the policy decides using only what a real router could know.

`W_p` itself is computed wherever the policy needs a request's prefill demand — at the
decision point and again at routing (`edpp.go`, the `Wp(ap)` / `Wp(apTokens)` call
sites). The resulting µs value is then added to whichever backlog the request lands on
(prefill pool vs. decode pool). Those backlogs are the subject of a later part; for now
the key fact is: **`W_p` is a pure function of `a_p` and the frozen coefficients.**

---

## 3. Decode work (`W_d`)

Decode work answers: "how much decode effort will this request require over its whole
output?" Unlike prefill, decode happens one token at a time across many iterations, so
we build it from a **per‑step** cost and a **count of steps**.

### Per‑step decode cost

The marginal cost of advancing one request by one token at context length `L`
(`edpp_coeffs.go:100‑104`):

```
δ̄_dec(L)  =  c0  +  c1 · L
```

```go
func (c EDPPCoeffs) deltaBarDecode(ctxLen float64) float64 {
	return c.C0 + c.C1*ctxLen
}
```

- **`c0`** — the fixed per‑request cost of being in the decode batch for this step.
- **`c1 · L`** — the KV‑read cost, proportional to the resident context length `L`. As a
  request generates more tokens, `L` grows, so each successive decode step is slightly
  more expensive.

### Total decode work

Total decode work is the per‑step cost times the number of output tokens:

```
W_d  =  N_out · δ̄_dec(L_nom)  =  N_out · (c0 + c1 · L_nom)
```

where `L_nom` is a nominal/representative context length (configured, not per‑step
exact — the policy uses a fixed nominal so it doesn't need to simulate the growing
context).

### The terms to bookkeep

`W_d` has two inputs, and they are bookkept very differently:

- **`L_nom` (nominal context length)** — a static configuration value (`NomDecodectx`).
  No per‑request tracking; it's plugged straight into `δ̄_dec`.

- **`N_out` (realized output length)** — this is the interesting one. The number of
  output tokens a request will generate is **not known at decision time**. EDPP handles
  this in two ways:
  1. **At routing**, it uses a running mean estimate, `N̂_out`, maintained per SLO class
     (`edppRunningMean`). With no completions yet it conservatively returns 1. So the
     decode work estimate at routing is `N̂_out · δ̄_dec(L_nom)`.
  2. **At completion**, the realized output length `len(req.OutputTokens)` is read and
     folded into the running mean (`nHatFor(class).update(...)`). This is a *post‑hoc*
     read — it happens after the request is done, so it never influences that request's
     own decision and stays oracle‑safe (INV‑9).

The deeper reason `N_out` can be estimated loosely is that **`W_d` does not appear in
the routing decision rule** — it cancels out (both the "keep on decode" and "send to
prefill" branches incur the same eventual decode work). It is tracked for backlog
accounting, not for the choice itself. We will see exactly where it cancels when we
derive the decision rule in a later part.

---

## 4. Iteration time (`T_iter`)

The coefficients and the work formulas come together in the **iteration‑time model** —
the estimate of how long one model forward pass takes given the batch it is running.
This is the E3 latency law made concrete, and it has two specializations: a decode
server (which may also be co‑scheduling some prefill) and a dedicated prefill server.

### Decode iteration time

```
T_iter_decode  =  α  +  c0 · B_dec  +  c1 · KV  +  c_pf · S_pf
```

(`edpp_coeffs.go:67‑70`)

```go
func (c EDPPCoeffs) tIterDecode(bDec int, kv, sPf int64) float64 {
	return c.AlphaD + c.C0*float64(bDec) + c.C1*float64(kv) + c.CPf*float64(sPf)
}
```

Each term and how it is tracked:

| Term | Meaning | Bookkeeping |
|------|---------|-------------|
| `α` | Fixed per‑iteration cost | Constant (coefficient). |
| `c0 · B_dec` | Cost of the decode batch | `B_dec` = number of requests currently decoding on the server. |
| `c1 · KV` | KV‑read cost | `KV` = total resident context tokens across the decode batch (sum of per‑request context lengths). |
| `c_pf · S_pf` | Prefill co‑scheduled this step | `S_pf` = number of prefill tokens processed in this iteration (a "chunk"). On a pure decode step `S_pf = 0`; when prefill is piggy‑backed onto a decode server, `S_pf > 0` and **inflates inter‑token latency** — this is precisely the cost disaggregation avoids. |

### Prefill iteration time

A dedicated prefill server runs *no* decode work, so `B_dec = 0` and `KV = 0`, leaving
(`edpp_coeffs.go:72‑76`):

```
T_iter_prefill  =  α_p  +  c_pf · S_pf
```

```go
func (c EDPPCoeffs) tIterPrefill(sPf int64) float64 {
	return c.AlphaP + c.CPf*float64(sPf)
}
```

Here `S_pf` is the prefill chunk size being processed this iteration.

### Drain rate `μ`: turning iteration time into throughput

Iteration time has a fixed part (α) that does no "useful" marginal work and a variable
part that does. The fraction of each iteration spent on useful work is the **drain
rate** (`edpp_coeffs.go:78‑86`):

```
μ  =  1  −  α / T_iter
```

```go
func (c EDPPCoeffs) muDecode(bDec int, kv, sPf int64) float64 {
	return clampMu(1.0 - c.AlphaD/c.tIterDecode(bDec, kv, sPf))
}
func (c EDPPCoeffs) muPrefill(sPf int64) float64 {
	return clampMu(1.0 - c.AlphaP/c.tIterPrefill(sPf))
}
```

`μ` is the rate at which a server *drains its work backlog* (work‑µs cleared per wall‑µs).
Intuition: if α is the whole iteration, almost no marginal work gets done, so `μ → 0`;
as the batch grows and the variable terms dominate, `μ → 1` (work‑conserving limit).
Because α here is ~16.6 ms and the marginal terms are tiny at small batch, `μ` is small
in the latency‑bound regime — amortizing α across more work is how the system earns
throughput.

`clampMu` keeps `μ` in `[edppMinMu, 1]` (`edpp.go`, `edppMinMu = 1e‑3`) so that when
`μ` appears in a denominator later (predicting how long a backlog takes to drain) it can
never collapse to zero or exceed the work‑conserving limit.

> There are also two **nominal** drain rates — `muDNom` (decode at the ITL‑critical
> batch, `1 − α/τ_itl`) and `muPNom` (prefill at a nominal chunk size). These are fixed
> reference rates used to normalize the backlogs in the decision rule, and we will
> introduce them when we build that rule. They are listed here only so the `μ` family
> is complete.

---

## Summary so far

| Quantity | Formula | Code |
|----------|---------|------|
| Prefill work | `W_p = c_pf·a_p + (c_attn/2)·a_p²` | `edpp_coeffs.go:62` |
| Per‑step decode | `δ̄_dec = c0 + c1·L` | `edpp_coeffs.go:102` |
| Decode work | `W_d = N_out·(c0 + c1·L_nom)` | (via `deltaBarDecode`) |
| Decode iter time | `T_iter = α + c0·B_dec + c1·KV + c_pf·S_pf` | `edpp_coeffs.go:68` |
| Prefill iter time | `T_iter = α_p + c_pf·S_pf` | `edpp_coeffs.go:74` |
| Drain rate | `μ = 1 − α/T_iter` (clamped) | `edpp_coeffs.go:80` |

Everything above is a deterministic function of the six frozen coefficients plus a few
per‑request counts (`a_p`, `N_out`) and batch‑state counts (`B_dec`, `KV`, `S_pf`). The
**next part** builds the prefill and decode **backlogs** (`Q_p`, `Q_d`) out of these work
units and shows how they are kept current as requests are admitted and drained.
