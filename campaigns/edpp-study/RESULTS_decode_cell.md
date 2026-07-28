# Decode cell, homogeneous fleet — fair policy comparison

Recorded 2026-07-28. All numbers below come from a single post-fix run set (seed 42) and
supersede any earlier decode-cell table in this repo. Two simulator fixes landed with this
measurement; both are described under "Fixes this table depends on".

## Setup

**Fleet (topology).** 1 dedicated prefill instance + 2 mixed instances, all homogeneous
H100 at tensor-parallel degree 4 (`llama-3.3-70b-instruct`, coefficients
`scripts/calibration/coeffs-llama70b-h100-tp4.json`).

```
--num-instances 3 --prefill-instances 1 --decode-instances 2
--decode-routing-scorers queue-depth:1 --max-num-running-reqs 16
```

The decode-instance scorer is the same load-balancing (`queue-depth:1`) profile for every
policy. It is only consulted by the policies that do not choose the decode instance
themselves (`never`, `always`, `kairos`); the joint policies pick it as part of their action.

**Workload.** Single SLO class `standard`, Poisson arrivals.

| parameter | value |
|---|---|
| input (prompt) tokens | **256, constant** |
| output tokens | **lognormal**, `mu=6.1583`, `sigma=0.4`, min 4, max 4112 (mean ≈ 512) |
| prefix reuse | none (`prefix_length: 0`) |
| arrival process | Poisson |
| workload spec seed | 42 (the spec is reseeded per seed, not just the simulator RNG) |

A 256-token prompt is a single prefill chunk under the 8192-token per-iteration budget.

**Measured capacity knee: 3.0 req/s** (highest rate where achieved/offered ≥ 0.95, measured
with the static corners; see `repro_knee_sweep.sh` and `specs/grid_v2/cells.txt`). The sweep
below spans 0.5× to 1.33× the knee.

**SLOs (derived, not asserted).** A zero-contention probe at rate 0.2 gives the baseline
p99s, and the targets are fixed multiples of them (Method A slowdown multiples, as in
Splitwise/Sarathi):

| dimension | baseline p99 | multiple | **target** |
|---|---|---|---|
| TTFT | 51.78 ms | 5× | **259 ms** |
| ITL (per-request mean) | 16.79 ms | 4× | **67 ms** |
| E2E | 22 773 ms | 3× | **68 320 ms** |

```
--slo-ttft standard=259ms --slo-itl standard=67ms --slo-e2e standard=68320ms
```

A request is *good* only if it meets all three. Goodput is the fraction of injected requests
that are good.

**Horizon (steady state).** `num_requests = rate × max(120 s, 10 × E2E SLO) = rate × 683.2 s`,
giving 1025 / 1435 / 1708 / 2050 / 2392 / 2733 requests at rates
1.5 / 2.1 / 2.5 / 3.0 / 3.5 / 4.0. Verified horizon-stable: doubling the request count moves
goodput by ≤ 0.006.

**Caveats.** Single seed (42), which is the harshest of {42, 7, 123} on this cell — e.g.
`never` at rate 3.0 scores 0.678 on seed 42 versus 0.852 on seed 7. Treat sub-0.02 margins as
provisional until multi-seeded.

## Policies

| arm | rule | decode instance chosen by |
|---|---|---|
| `never` | never disaggregate (prefill where you decode) | load-balancing scorer |
| `always` | always disaggregate through the prefill pool | load-balancing scorer |
| `kairos` | Kairos prefill deflection, β = 0.5 | its own deflection search |
| `lt-joint` | minimize the deciding request's predicted TTFT over the full joint action set | joint (hardware-aware) |
| `dpvar` | **the paper's rule**: congestion + SLO deficits + goodput penalty R(a) = VaR(a) − good_r(a) | joint |

Every arm is deployable: none reads a realized output length. `dpvar` uses the censored
per-class N̂_out for co-resident remaining steps (`--edpp-var-deployable`).

## Results

Latencies in ms. Miss columns are absolute counts out of the injected total for that rate.
`z_ttft` is the TTFT deficit queue in normalized form (z = Z/τ), `act` the fraction of
decisions on which it was non-empty, `leak` the number of requests still registered as
awaiting a first token at the end of the run (must be 0). Those three columns are blank for
`never` and `always` because admission and SLO-feedback instrumentation is wired only for the
EDPP decider.

| rate | arm | achieved | goodput | TTFT mean | ITL mean | E2E mean | disagg% | miss ttft | miss itl | miss e2e | mean z_ttft | z act | leak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | never | 1.465 | **1.0000** | 42.6 | 16.94 | 8763 | — | — | — | — | — | — | — |
| 1.5 | always | 1.465 | **1.0000** | 51.8 | 16.92 | 8694 | 100% | 0 | 0 | 0 | — | — | — |
| 1.5 | kairos | 1.465 | **1.0000** | 51.8 | 16.92 | 8694 | 100% | 0 | 0 | 0 | 0 | 0.000 | 0 |
| 1.5 | lt-joint | 1.465 | **1.0000** | 42.5 | 16.93 | 8757 | 0% | 0 | 0 | 0 | 0 | 0.000 | 0 |
| 1.5 | dpvar | 1.465 | **1.0000** | 42.8 | 16.93 | 8757 | 0% | 0 | 0 | 0 | 0 | 0.000 | 0 |
| 2.1 | never | 2.107 | 0.9686 | 75.2 | 17.05 | 8867 | — | — | — | — | — | — | — |
| 2.1 | always | 2.107 | **1.0000** | 51.9 | 17.01 | 8789 | 100% | 0 | 0 | 0 | — | — | — |
| 2.1 | kairos | 2.107 | **1.0000** | 51.9 | 17.01 | 8789 | 100% | 0 | 0 | 0 | 0 | 0.000 | 0 |
| 2.1 | lt-joint | 2.107 | 0.9819 | 77.1 | 17.03 | 8860 | 0% | 26 | 0 | 0 | 11.9 | 0.157 | 0 |
| 2.1 | dpvar | 2.107 | **1.0000** | 42.8 | 17.03 | 8844 | 2.6% | 0 | 0 | 0 | 0 | 0.000 | 0 |
| 2.5 | never | 2.561 | 0.9198 | 172.1 | 17.11 | 9026 | — | — | — | — | — | — | — |
| 2.5 | always | 2.561 | **1.0000** | 52.0 | 17.08 | 8947 | 100% | 0 | 0 | 0 | — | — | — |
| 2.5 | kairos | 2.561 | **1.0000** | 52.0 | 17.08 | 8947 | 100% | 0 | 0 | 0 | 0 | 0.000 | 0 |
| 2.5 | lt-joint | 2.561 | 0.9333 | 198.5 | 17.10 | 9048 | 0% | 114 | 0 | 0 | 216 | 0.717 | 0 |
| 2.5 | dpvar | 2.561 | 0.9947 | 47.0 | 17.10 | 8987 | 6.1% | 9 | 0 | 0 | 0.023 | 0.017 | 0 |
| 3.0 | never | 3.024 | 0.6776 | 781.5 | 17.18 | 9577 | — | — | — | — | — | — | — |
| 3.0 | always | 3.024 | **1.0000** | 52.0 | 17.14 | 9459 | 100% | 0 | 0 | 0 | — | — | — |
| 3.0 | kairos | 3.024 | **1.0000** | 52.0 | 17.14 | 9459 | 100% | 0 | 0 | 0 | 0 | 0.000 | 0 |
| 3.0 | lt-joint | 3.024 | 0.7039 | 1064 | 17.17 | 9856 | 0% | 607 | 0 | 0 | 4202 | 0.938 | 0 |
| 3.0 | dpvar | 3.024 | 0.9932 | 53.1 | 17.16 | 9487 | 29.8% | 14 | 0 | 0 | 0.056 | 0.034 | 0 |
| 3.5 | never | 3.468 | 0.1656 | 6893 | 17.23 | 15683 | — | — | — | — | — | — | — |
| 3.5 | always | 3.468 | **1.0000** | 52.1 | 17.19 | 14986 | 100% | 0 | 0 | 0 | — | — | — |
| 3.5 | kairos | 3.468 | **1.0000** | 52.1 | 17.19 | 14986 | 100% | 0 | 0 | 0 | 0 | 0.000 | 0 |
| 3.5 | lt-joint | 3.469 | 0.4778 | 3213 | 17.22 | 16370 | 31.0% | 1249 | 0 | 0 | 13500 | 0.975 | 0 |
| 3.5 | dpvar | 3.468 | 0.9933 | 82.5 | 17.19 | 15034 | 81.1% | 16 | 0 | 0 | 2.20 | 0.147 | 0 |
| 4.0 | never | 3.540 | 0.0245 | 45764 | 17.25 | 54549 | — | — | — | — | — | — | — |
| 4.0 | always | 3.564 | 0.6070 | 52.1 | 17.20 | 52697 | 100% | 0 | 0 | 1074 | — | — | — |
| 4.0 | kairos | 3.564 | 0.6070 | 52.1 | 17.20 | 52697 | 100% | 0 | 0 | 1074 | 0 | 0.000 | 0 |
| 4.0 | lt-joint | 3.356 | 0.2151 | 3540 | 17.22 | 55785 | 50.3% | 1271 | 0 | 874 | 16120 | 0.984 | 0 |
| 4.0 | dpvar | 3.491 | **0.6623** | 125.5 | 17.20 | 53040 | 97.5% | 5 | 0 | 920 | 28.4 | 0.303 | 0 |

## Analysis

`always` and `kairos` are identical everywhere (Kairos never deflects here) and are optimal
up to 3.5, because full offload pins TTFT at 52 ms. `dpvar` matches them at 1.5 and 2.1,
trails by ≤ 0.007 at 2.5–3.5, and wins at 4.0. `lt-joint` is the cautionary arm: it stays
collocated through the knee because the roll-forward estimator under-predicts collocated
admission delay, and pays 607 TTFT misses at 3.0. `never` is the floor.

Two honest notes. First, `dpvar`'s adaptive disaggregation is visible and monotone
(0 → 2.6 → 6.1 → 29.8 → 81.1 → 97.5%) and its mean TTFT stays at 43–125 ms while
`never`/`lt-joint` go into seconds — but on this cell that adaptivity buys nothing over the
static corner until the fleet is past capacity. Second, the ITL column still never moves
(16.9–17.25 ms against a 67 ms target), so that dimension remains inert here regardless of
the fix.

### Supporting observations

- **Every miss below rate 4.0 is a TTFT miss.** E2E only binds at 4.0, where the fleet is
  past capacity (achieved ~3.5 against 4.0 offered) and `always`/`kairos`/`dpvar` each lose
  ~1000 requests to E2E with zero TTFT misses. So this "decode-bound" archetype actually
  exercises prefill admission delay, not decode.
- **TTFT on this cell is near-binary**: ~42.6 ms collocated versus ~51.9 ms disaggregated
  (the difference is the KV transfer). Collocation is *cheaper* per request while the
  instance is idle; it is the tail under load that fails. Any TTFT target between 43 and
  52 ms would measure "did you collocate" rather than a latency goal.
- **The disaggregated path's TTFT excludes the wait to enter the decode batch.** TTFT is
  computed as prefill TTFT + KV transfer + first decode step (`projectPDMetrics`). The
  decode-side admission wait — `decode_t_adm` p99 reaches 71 s at rate 4.0 — lands in E2E
  only. This is a deliberate definitional choice, recorded here because it means a
  disaggregating policy is measured more leniently on the TTFT dimension than a collocating
  one.
- **The prefill pool never congests on this workload.** 32 of these 256-token prompts fit
  in one 8192-token prefill iteration (~67 ms), giving ~477 req/s of prefill throughput
  against a fleet decode capacity of ~3.6 req/s — over-provisioned ~133×. Measured
  `prefill_t_adm` p99 is flat at 15.6 ms across the whole sweep.

## Fixes this table depends on

1. **`awaitingFirstToken` leak (`sim/cluster/cluster.go`, `firstTokenKey`).** A PD request's
   first token is produced by its prefill sub-request, whose first-token event previously
   resolved to its own ID — a key the decider never registered (registration used the parent
   ID). The parent's record was therefore never trued up or deleted, and every later credit
   pass kept adding `(now − arrival − τ)` for an already-finished request, so the TTFT deficit
   queue integrated phantom lateness without bound. Before the fix, `dpvar` at rate 3.0 had
   mean z_ttft = 3.0×10⁵ with 591 leaked records (exactly its disaggregated count); after,
   0.056 and 0 leaked. Fixing it **improved** `dpvar`: 0.9873 → 0.9932 at rate 3.0 and
   0.6132 → **0.6623** at rate 4.0. `always`, `kairos`, and `lt-joint` are unchanged, since
   none consumes the TTFT deficit term.
2. **Deficit-queue logging** (`EDPPDeficitStats`, `ClusterSimulator.EDPPDeficitStats`, and an
   `EDPP_DEFICIT` line in `cmd`). Without it there was no way to tell whether the rule's
   time-average SLO constraints ever bound. They are what produced the `z_ttft` / `act` /
   `leak` columns, and they are how the leak was found.

## How to reproduce

Requires the two fixes above (commit `f15e314` or later on
`feat/edpp-estimator-validation`).

```bash
cd inference-sim
go build -o blis main.go

S=/tmp/fair; mkdir -p $S
COEF=scripts/calibration/coeffs-llama70b-h100-tp4.json
T=259; I=67; E=68320
TOPO="--num-instances 3 --prefill-instances 1 --decode-instances 2 \
      --decode-routing-scorers queue-depth:1 --max-num-running-reqs 16"
SLO="--slo-ttft standard=${T}ms --slo-itl standard=${I}ms --slo-e2e standard=${E}ms"
EC="--edpp-coeffs $COEF --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware \
    --edpp-tau-itl ${I}ms"
VVF="--pd-decider edpp $EC --edpp-rule var --edpp-var-metric util --edpp-joint \
     --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 \
     --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft ${T}ms --edpp-tau-e2e ${E}ms"

for R in 1.5 2.1 2.5 3.0 3.5 4.0; do
  WF=campaigns/edpp-study/specs/policy_curves/w_decode_${R}_42.yaml
  for arm in never always kairos lt-joint dpvar; do
    case $arm in
      never)    F="--pd-decider never" ;;
      always)   F="--pd-decider always" ;;
      kairos)   F="--pd-decider edpp $EC --edpp-tau-ttft ${T}ms --edpp-rule kairos --kairos-beta 0.5" ;;
      lt-joint) F="--pd-decider edpp $EC --edpp-tau-ttft ${T}ms --edpp-rule least-ttft --edpp-joint" ;;
      dpvar)    F="$VVF" ;;
    esac
    ./blis run --log info --model meta-llama/llama-3.3-70b-instruct \
      --workload-spec $WF $TOPO $SLO $F --seed 42 \
      --pd-outcome-trace $S/${R}_${arm}.csv --metrics-path $S/${R}_${arm}.json \
      2> $S/${R}_${arm}.log >/dev/null
  done
done
```

`--log info` is required: the `EDPP_DEFICIT` line is logged at info level to stderr (stdout
stays deterministic per INV-6). The workload specs are committed under
`campaigns/edpp-study/specs/policy_curves/`; regenerate them with `repro_policy_curves.sh`
if absent.

Then read the table off the outputs: goodput and mean latencies from
`per_class.standard.slo_attainment` and `ttft_mean_ms` / `itl_mean_ms` / `e2e_mean_ms` in the
metrics JSON; disaggregation share and per-dimension miss counts from the outcome-trace CSVs
(`disaggregated`, `realized_ttft`, `realized_mean_itl`, `realized_e2e` versus the three
targets); z_ttft columns from the `EDPP_DEFICIT` line in each `.log`. `never` emits no
outcome trace (it bypasses the PD routing path), so its miss columns are unavailable — a
known instrumentation gap.

## Open items

- Multi-seed this table (seed 42 only; it is the pessimistic seed of the three).
- `never` emits no outcome-trace rows, so its placements and per-dimension misses cannot be
  decomposed.
- `never` / `always` / `prefix-threshold` record no admission timestamps and no deficit
  stats, because that path is wired only for the EDPP decider.
- The E2E target rests on a p99 over 60 samples, which is that probe's maximum. It is stable
  for TTFT and ITL (no contention at rate 0.2) but swings 52.4–68.3 s for E2E across
  probe sizes and seeds, and this cell happens to use the loosest draw. E2E is the only
  dimension that binds at rate 4.0, so that arbitrariness is load-bearing.
