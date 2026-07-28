 ▐▛███▜▌   Claude Code v2.1.178
▝▜█████▛▘  Opus 4.8 · API Usage Billing
  ▘▘ ▝▝    ~/git-repos/llm-git-repos/edpp-fresh/inference-sim


❯  Let's start again with the dpp/edpp algorithm. I think I've lost track of    
  what we are trying to do. But this time, let's start simple. And please be    
  direct and concise in your answers. And push back if my thinking is flawed in 
  terms of queueing theory. Your goal is to guide me to first come up with an   
  empirical algorithm that takes into account the observable metrics, but then  
  also map it to the queueing theory formulation.                               
                                                                                
                                                                                
  We have a remote prefill server and a decode server. When the request arrives 
  with prefill work W_p and decode work W_d, a                                  
    decision needs to be made whether the request does its prefill on the       
  remote server or goes directly to the decode server                           
    that also does its prefill. The prefill work is measured in uncontended     
  GPU-time (i.e. how long would this request take to be serviced if this was    
  the only request. One can also think of this as the demand from the request.  
  The prefill work depends upon input tokens (particualrly the uncached tokens) 
  and decode step work increases with each decode step since it depends on the  
  current context length). But the servers concurrently process many requests.  
  So the wall clock service time of the request is not the same as GPU-time. I  
  believe BLIS already has some latency model that provides the GPU-time as     
  well as StepTime that given the current batch composition determines the      
  iteration time. I am not sure how these quantities translate to observable    
  metrics(both from blis and vllm server).                                      
                                                                                
                                                                                
  Additionally, each server has a queue that has the amount of work pending.    
  The pending work includes the pending work from requests currently being      
  processed and possibly from the requests waiting in the queue. The incoming   
  requests wait if the server is at capacity. So for instance, for the remote   
  prefill server, it could be that since the prefill is compute bound, the      
  server is working at its peak compute. It could also be the case that because 
  of some very long prompts, the KV cache at the prefill server is fully        
  occupied. In any case, if the remote prefill server is at capacity, then the  
  requests would wait in a queue to finish their prefill work. Similarly, the   
  decode server could be saturated or capacity. In this case, the pending work  
  is from requests that are or will be in the prefill phase. But the pending    
  work is also from the requests that are currently undergoing decode. Though   
  in the latter case, the pending work cannot be apriori determined in case the 
  output length is not known. But if we simplify the system and assume that     
  the output length is known, then the decode pending work of each request is   
  also known. Again the BLIS latency model might tell us the total work pending 
  in each server. For a tagged request that has just arrived to one of the      
  servers, its TTFT is then determined by time it waits in the queue and/or its 
  prefill time. To calculate the TTFT of this tagged request, one needs to      
  know the work drain rate at each server. Again not sure how to obtain this    
  from observable metrics.                                                      
                                                                                
  We'll exapnd on system model more later, but that's how far I can think       
  clearly at the moment.                                                        

⏺ I'll start by invoking the brainstorming skill since we're designing an
  algorithm — but I'll keep things tight and direct per your request.

⏺ Skill(superpowers:brainstorming)
  ⎿  Successfully loaded skill

⏺ Good — let me ground this before we build anything. I'll start with one
  correction and one question.

  One pushback up front

  You're modeling each server as a single work-conserving queue: a pile of
  pending work W that drains at some rate μ, so wait ≈ W/μ. That's a reasonable
  approximation for the prefill server (prefill is compute-bound; throughput in
  tokens/sec is roughly constant regardless of how many requests are batched, so
  it behaves like a work-conserving server with a near-fixed drain rate).

  It is not a good model for the decode server. Decode is a batched, 
  processor-sharing-like system: many requests decode concurrently, and the
  per-request drain rate drops as the batch grows (memory bandwidth is split
  across the batch). So the "drain rate" is state-dependent — it's a function of
  current batch size/composition, not a constant. A single scalar W/μ will
  mislead you there.

  This matters because it's the crux of the whole PD-disaggregation decision:
  the reason to send prefill remote is to protect the decode batch's drain rate
  from being disrupted by prefill compute — not primarily to reduce the tagged
  request's own TTFT. If you model both servers as identical work-conserving
  queues, the algorithm will systematically undervalue disaggregation.

  My question

  Before we pick metrics or math: what is the routing decision actually 
  optimizing?

  - (A) Minimize the tagged request's TTFT only.
  - (B) Minimize the tagged request's end-to-end (TTFT + total decode time).
  - (C) Meet per-class SLOs (e.g. TTFT target and ITL/decode target), routing to
  whichever placement keeps both within budget.
  - (D) Maximize system goodput (cluster-wide), treating the tagged request's
  latency as secondary.

  The honest answer changes the model: if it's (A), disaggregation usually looks
  bad and the algorithm is simple. The interesting cases — and the reason PD
  disagg exists — are (C)/(D), where protecting the decode server's drain rate
  is the payoff. Which one are we solving?

✻ Crunched for 56s

❯ Can't we also assume the batched, processor ssharing like system for the      
  prefill server as well? And I completely agree with This matters because it's 
  the crux of the whole PD-disaggregation decision: the reason to send prefill  
  remote is to protect the decode batch's drain rate from being disrupted by    
  prefill compute — not primarily to reduce the tagged request's own TTFT. And  
  yes, ideally we'd like to optimize the problem stated in c. Possibly also in  
  d. But we can start simple with c.                                            

⏺ On modeling prefill as processor-sharing too

  You can, but it buys you nothing — and that's the point. The question isn't
  "is it processor-sharing?" but "is the aggregate drain rate constant or
  state-dependent?"

  - Prefill is compute-bound. A single prefill request (hundreds–thousands of
  tokens) already saturates the GPU's FLOPs. Batching more prefills doesn't
  raise aggregate token throughput — they share a fixed compute ceiling. So
  aggregate drain rate μ_p ≈ peak-compute-tokens/sec ≈ constant. The per-request
  rate drops under sharing, but the total work drains at a fixed rate → the
  work-conserving W/μ model is accurate and simpler. PS adds complexity without
  changing the wait prediction.
  - Decode is bandwidth-bound. Each step reloads weights once and amortizes that
  across the whole batch, so aggregate throughput rises with batch size (until
  you hit a limit). The drain rate is genuinely state-dependent. That's why W/μ
  with constant μ fails here and not for prefill.

  So: model prefill as a work-conserving queue (constant rate), model decode as
  a state-dependent batched server. Different models, for a real physical
  reason.

  Objective (C) confirmed; I'll keep (D) in the back pocket.

  The observable substrate (my proposed mapping)

  Here's what each abstract quantity maps to. Tell me where this is wrong or
  unavailable:

  ┌──────────────┬─────────────────────────┬────────────────────────────────┐
  │  Abstract    │          BLIS           │        vLLM observable         │
  │   quantity   │                         │                                │
  ├──────────────┼─────────────────────────┼────────────────────────────────┤
  │ Pending      │ exact uncached input    │ num_requests_waiting × prompt  │
  │ prefill work │ tokens, queued +        │ tokens (sum of prompt lengths) │
  │  W_p         │ in-progress             │                                │
  ├──────────────┼─────────────────────────┼────────────────────────────────┤
  │ Prefill      │ StepTime model at       │ measured prefill tokens/sec    │
  │ drain rate   │ compute saturation      │ (recent window)                │
  │ μ_p          │                         │                                │
  ├──────────────┼─────────────────────────┼────────────────────────────────┤
  │ Decode batch │ running-request count   │ num_requests_running           │
  │  size B      │                         │                                │
  ├──────────────┼─────────────────────────┼────────────────────────────────┤
  │ Decode step  │ StepTime(batch          │ measured inter-token latency   │
  │ time /       │ composition)            │ at current B                   │
  │ ITL(B)       │                         │                                │
  ├──────────────┼─────────────────────────┼────────────────────────────────┤
  │ KV pressure  │ allocated/total blocks  │ gpu_cache_usage_perc           │
  └──────────────┴─────────────────────────┴────────────────────────────────┘

  With this, the prefill-server TTFT is clean:
  TTFT_remote ≈ (W_p_pending + W_p_tagged) / μ_p + transfer_time

  The decode side is the hard part, and it forks on one decision.

  My question

  For the decode server's rate-vs-batch relationship — the thing W/μ can't
  capture — do we get it from:

  - (a) Black-box / empirical: measure throughput and ITL online at the
  currently-observed batch size, and assume it locally. Model-agnostic, works
  against real vLLM. But a single operating point doesn't tell you how ITL will
  change after you add the tagged request (or its prefill) to the batch.
  - (b) White-box: use BLIS's StepTime(batch composition) to predict ITL at the
  hypothetical post-routing batch state. Gives you the counterfactual ("what
  happens to in-flight ITL if I add this prefill here"), which is exactly the
  protection signal — but it's a model, not a measurement.

  Since you want empirical-first, my instinct is (a) as the substrate, (b)'s 
  functional form as the interpolator — measure the rate online, but use the
  known monotone shape of ITL(B) to extrapolate one step. Which way do you want
  to lean?

✻ Sautéed for 1m 13s

❯ Let's keep the questions aside for a moment, it woud also be helpfl to        
  understand the units each quantitiy carries.                                  

⏺ Good instinct — unit discipline is exactly where queueing-theory reasoning
  about this system goes wrong. Let me lay it out, and flag the one trap.

  The trap: two currencies for "work"

  There are two consistent ways to measure work, and mixing them silently is the
  #1 error:

  1. Tokens — work [tokens], rate [tokens/s], so wait = work/rate [s].
  2. GPU-seconds (uncontended service time) — work [s of GPU-time], and the
  "rate" is dimensionless utilization/parallelism [GPU-s drained per wall-s], so
  wait = work/rate [s].

  Your phrasing "work = uncontended GPU-time" picks currency 2. That's fine, but
  then the rate is not "tokens/sec" — it's "how many GPU-seconds of backlog
  clear per wall-second," which equals the effective parallelism / number of 
  busy server-equivalents. Don't accidentally divide GPU-seconds by tokens/sec.

  Units table (GPU-seconds currency, since that's your framing)

  Request demands (intrinsic to the request):

  ┌───────────────┬────────┬──────────────┬────────────────────────────────┐
  │   Quantity    │ Symbol │     Unit     │             Notes              │
  ├───────────────┼────────┼──────────────┼────────────────────────────────┤
  │ Prefill       │ W_p    │ GPU-s        │ uncontended prefill service    │
  │ demand        │        │              │ time; ∝ uncached input tokens  │
  ├───────────────┼────────┼──────────────┼────────────────────────────────┤
  │ Decode demand │ W_d    │ GPU-s        │ uncontended decode service     │
  │               │        │              │ time; not constant per token   │
  ├───────────────┼────────┼──────────────┼────────────────────────────────┤
  │ Per-step      │ w_d(L) │ GPU-s/step   │ grows with context length L    │
  │ decode demand │        │              │ (attention over KV)            │
  ├───────────────┼────────┼──────────────┼────────────────────────────────┤
  │ Output length │ N_out  │ tokens (=    │ one decode step per output     │
  │               │        │ steps)       │ token                          │
  └───────────────┴────────┴──────────────┴────────────────────────────────┘

  So W_d = Σ_{i=1..N_out} w_d(L_0 + i) — the sum grows because each step's
  demand rises with context. Decode demand is a ramp, not a rectangle.

  Server rates (property of server + current state):

  ┌────────────┬─────────────┬───────────────────┬──────────────────────────┐
  │  Quantity  │   Symbol    │       Unit        │          Notes           │
  ├────────────┼─────────────┼───────────────────┼──────────────────────────┤
  │ Prefill    │             │ GPU-s/wall-s (≈   │ ≈ constant ≈ 1           │
  │ drain rate │ μ_p         │ dimensionless)    │ (compute-saturated,      │
  │            │             │                   │ single GPU)              │
  ├────────────┼─────────────┼───────────────────┼──────────────────────────┤
  │ Decode     │ μ_d(B)      │ GPU-s/wall-s      │ state-dependent; rises   │
  │ drain rate │             │                   │ with batch B             │
  ├────────────┼─────────────┼───────────────────┼──────────────────────────┤
  │ Decode     │ t_step(B,L⃗) │ wall-s/step       │ what vLLM/BLIS actually  │
  │ step time  │             │                   │ emit                     │
  ├────────────┼─────────────┼───────────────────┼──────────────────────────┤
  │ ITL        │ —           │ wall-s/token      │ = t_step (one            │
  │            │             │                   │ token/request/step)      │
  └────────────┴─────────────┴───────────────────┴──────────────────────────┘

  The cleanest physical observable is t_step [s/step], not an abstract "drain
  rate." μ_d is a derived convenience.

  Queue state (observable):

  ┌──────────────────────────────────┬────────────────────────────────────┐
  │             Quantity             │                Unit                │
  ├──────────────────────────────────┼────────────────────────────────────┤
  │ Pending prefill work W_p^pending │ GPU-s (or tokens, then convert)    │
  ├──────────────────────────────────┼────────────────────────────────────┤
  │ Batch size B                     │ requests                           │
  ├──────────────────────────────────┼────────────────────────────────────┤
  │ Context lengths L⃗                │ tokens                             │
  ├──────────────────────────────────┼────────────────────────────────────┤
  │ KV usage                         │ fraction (dimensionless) or blocks │
  ├──────────────────────────────────┼────────────────────────────────────┤
  │ Transfer time                    │ wall-s                             │
  └──────────────────────────────────┴────────────────────────────────────┘

  Outputs (what we predict, all wall-s):

  ┌──────────┬──────────────┐
  │ Quantity │     Unit     │
  ├──────────┼──────────────┤
  │ TTFT     │ wall-s       │
  ├──────────┼──────────────┤
  │ ITL      │ wall-s/token │
  ├──────────┼──────────────┤
  │ E2E      │ wall-s       │
  └──────────┴──────────────┘

  The one consequence worth stating now

  Because decode demand is a ramp in L (per-step demand grows with context) and
  the decode rate is state-dependent in B, the decode server has two coupled
  state dependencies. A static W_d/μ_d scalar collapses both and will be wrong.
  The prefill server has neither (rectangle demand, constant rate) — which is
  why it's the easy half.

  Want me to write down the per-placement TTFT and ITL predictors next, in these
  units, so we can see exactly which observables each one consumes?

✻ Crunched for 1m 33s

❯ Let's have the state dependent rates for prefill server as well? This is      
  again pushing back/trying to convince you and myself that we loose nothing if 
  we model the prefill server similarly to decode server. Again you can push    
  back.                                                                         

⏺ I'll concede most of this — but with a sharp boundary on what you're unifying.

  Where you're right (I'll update)

  "State-dependent rate" with a flat function is the constant-rate model —
  constant is just the degenerate case. So modeling prefill as μ_p(state) loses
  zero correctness. And there are two real regimes where prefill rate is
  genuinely not constant, so the symmetric model is actually more faithful:

  1. Under-saturation. A single short prompt (~tens of tokens) doesn't saturate
  FLOPs; it's memory-bound-ish. Aggregate prefill throughput rises with batching
  until compute saturates, then plateaus. So μ_p(B) is rising-then-flat, not
  flat.
  2. KV-capacity blocking. Long prompts fill the prefill server's KV; new
  prefills can't admit until blocks free. That's head-of-line blocking — a
  state-dependent throughput collapse, exactly the regime you flagged earlier.

  So: unify the interface. Both servers expose rate = f(observable state). One
  abstraction, one code path, one estimator framework. You lose nothing there,
  and you gain uniformity. Agreed.

  Where I push back (don't over-unify)

  Unifying the interface is not unifying the function. Three asymmetries survive
  no matter how you model rate, and they're the ones that actually drive the
  routing decision:

  1. Different rate shapes. Decode rate rises with B and asymptotes
  (weight-reload amortization). Prefill rate rises then hits a hard FLOPs
  ceiling, plus a KV cliff. If you fit one shared curve you'll mismodel both.
  The shared thing is the signature f(state); the learned f must differ per
  server.
  2. Demand shape — the deeper asymmetry, and it's in the workload, not the 
  server. Prefill demand is a rectangle: W_p known at arrival, fixed. Decode
  demand is a ramp whose area depends on N_out, which is unknown. No amount of
  symmetric rate-modeling erases this. It's why the prefill side of the routing
  decision is near-deterministic and the decode side is fundamentally a
  prediction under uncertainty.
  3. Different question asked of each server. At prefill you want a completion 
  time (when does this job's prefill finish → TTFT) — a sojourn-time question.
  At decode you want a sustained per-token rate over many steps (ITL) — a
  throughput question. Same rate model, different functional of it: ∫ to a
  completion event vs. the instantaneous rate.

  Net

  Adopt your unification at the interface level — it's cleaner and I was wrong
  to resist it. But keep two separate learned f(state) curves, and stay alert
  that the rectangle-vs-ramp demand asymmetry (and the known-vs-unknown N_out)
  is the real structural difference. That one is not a modeling choice — it's
  physics of the workload.

  Want to write the two rate = f(state) estimators next, or pin down the demand
  model (especially how we handle unknown N_out) first?

✻ Brewed for 49s

❯ In short, your push back is on considering the same state dependent rate for  
  each server. And I'm saying that yes, each server will have its own state     
  dependent rate. I just want to make sure that we are on the same page.        

⏺ Yes — we're on the same page.

  - Same interface: both servers expose rate = f(observable state).
  - Different functions: f_prefill and f_decode are separate, with their own
  shapes (and learned/measured independently). I'm not asking you to share the
  function — only flagging that you shouldn't.

  My pushback was never "don't model prefill as state-dependent." It was the
  narrower "don't assume one shared curve." You've now said exactly that —
  separate per-server functions — so there's no disagreement left.

  The two asymmetries that remain (and that the symmetric interface does not
  erase) are workload-side, not server-side:
  - prefill demand = rectangle (W_p known at arrival) vs. decode demand = ramp
  (area depends on unknown N_out);
  - prefill asks a completion-time question (→ TTFT), decode asks a
  sustained-rate question (→ ITL).

  Agreed and locked. Where next — the two f(state) estimators, or the demand
  model with unknown N_out?

✻ Brewed for 10s

❯ Thanks. So lets go back to the problem of routing decision in order to        
  optimize TTFT and ITL and I mentioned first hand that I'm interested in       
  seeing the Lyapunov Optimization with Drift Plus Penalty (DPP) type algorithm 
  playing a role here. It would be ideal if we can show it analytically as      
  well as empirically.                                                          

⏺ Good — this is the right tool, and there's a clean story here. Let me affirm
  the fit, then give you the one structural guardrail that, if you get it wrong,
  sinks the whole formulation.

  DPP fits — this is textbook Neely

  "Route each arrival to one of several parallel queues to optimize a
  time-average objective subject to time-average constraints, keeping queues
  stable" is exactly the problem class Lyapunov drift-plus-penalty was built
  for. Routing-to-parallel-queues with MaxWeight-style decisions is a canonical
  Neely example. So you're not forcing a fit.

  The decision rule will come out as: each slot, choose placement x to minimize

  V · penalty(x)  +  Σ_i Q_i(t) · (drift contribution of x to queue i)

  i.e. a penalty-weighted join-the-weighted-shorter-queue rule. The V knob gives
  you Neely's [O(1/V) penalty gap, O(V) backlog] tradeoff — that's your
  analytical result.

  The guardrail: TTFT and ITL are different kinds of quantity

  This is the thing to get right.

  - TTFT is a backlog/delay quantity. It's waiting work / drain rate on the
  prefill path. Lyapunov handles this natively — it's literally a queue.
  TTFT-SLO ↔ "keep this queue's delay bounded," which Little's law ties to
  backlog. Clean.
  - ITL is a service-rate quantity, not a backlog. It's the steady-state
  per-token time at the decode server, a function of congestion B. There is no
  "ITL queue" that drains. If you try to model ITL as a physical backlog you'll
  get nonsense.

  The correct move: ITL enters as a virtual queue for a time-average constraint
  (the standard Neely trick). Define

  Z_itl(t+1) = max(Z_itl(t) + (ITL_observed(t) − ITL_target), 0)

  Stabilizing Z_itl ⟺ enforcing time-average ITL ≤ ITL_target. The control
  couples in because routing a prefill to the decode server raises B → raises
  ITL_observed via f_decode(B) → pushes Z_itl's drift. So your routing decision
  moves the ITL virtual queue, even though ITL was never a real backlog. Same
  construction for a TTFT-attainment constraint if you want fraction-meeting-SLO
  rather than mean.

  So the queue set is:

  ┌────────┬──────────┬─────────────────────┬───────────────────────────────┐
  │ Queue  │   Type   │        Unit         │           Driven by           │
  ├────────┼──────────┼─────────────────────┼───────────────────────────────┤
  │ Q_p    │ physical │ GPU-s               │ prefill work routed to remote │
  │        │          │                     │  server                       │
  ├────────┼──────────┼─────────────────────┼───────────────────────────────┤
  │ Q_d    │ physical │ GPU-s               │ prefill+decode work on decode │
  │        │          │                     │  server                       │
  ├────────┼──────────┼─────────────────────┼───────────────────────────────┤
  │ Z_ttft │ virtual  │ GPU-s (or           │ TTFT-SLO misses               │
  │        │          │ violation·s)        │                               │
  ├────────┼──────────┼─────────────────────┼───────────────────────────────┤
  │ Z_itl  │ virtual  │ s/token·s           │ ITL-SLO misses                │
  └────────┴──────────┴─────────────────────┴───────────────────────────────┘

  One gap you have to fill before DPP is even well-posed

  Objective (C) alone is a pure feasibility problem — "meet the SLOs." DPP needs
  a penalty to trade against stability, otherwise V is meaningless and any
  feasible policy is optimal. So you must borrow a sliver of (D): the penalty is
  the real objective, the SLOs are the constraints (virtual queues).

  Natural penalty candidates:
  - −goodput (maximize requests completed within SLO),
  - transfer/communication cost of remote prefill (the price of disaggregation),
  - GPU-time / energy.

  This isn't a detour — it's what makes "minimize penalty s.t. SLOs hold" a DPP
  problem instead of a constraint-satisfaction problem.

  My question

  What is the penalty penalty(x)? Pick the thing you actually want to minimize
  while the virtual queues enforce TTFT/ITL SLOs:

  - (i) transfer/comm cost of choosing remote prefill (clean, makes
  disaggregation "cost something," small and physical),
  - (ii) −goodput / drop cost (most aligned with (D), but goodput is itself
  SLO-defined → some circularity),
  - (iii) total GPU-time or energy (capacity-planning flavor).

  My recommendation is (i): it's the genuine economic cost of the decision
  you're making (you offload prefill and pay a KV-transfer), it's directly
  observable, and it keeps the SLOs cleanly in the constraint role. Which do you
  want?

✻ Baked for 1m 27s

❯ Yes, agree with option (i).                                                   

⏺ Locked: penalty g(x) = c_xfer if x = remote-prefill, else 0. Now watch the
  decision rule fall out — it's genuinely clean.

  The drift-plus-penalty rule

  Slot model: a request arrives with prefill work W_p and decode work W_d (both
  GPU-s). Binary decision x ∈ {P, D}. Key observation about where work lands:

  - x = P: prefill goes remote → Q_p += W_p; decode still goes to decode server
  → Q_d += W_d.
  - x = D: prefill and decode on decode server → Q_d += (W_p + W_d).

  So W_d hits Q_d either way — the decision only moves W_p between Q_p and Q_d.
  That's the whole lever.

  Lyapunov L = ½(Q_p² + Q_d² + Z_ttft² + Z_itl²). Minimize V·g(x) + Σ_i 
  Q_i·Δ_i(x) per slot. Cancel the common W_d·Q_d term and you get:

  Choose P  ⟺   V·c_xfer  +  Z_ttft·(TTFT_P − TTFT_D)  +  Z_itl·(ITL_P − ITL_D)
   <   (Q_d − Q_p)·W_p

  Why this is the right rule (every term is interpretable)

  - (Q_d − Q_p)·W_p (RHS) — backlog balancing. Decode server much more loaded
  than prefill server → offloading the W_p chunk relieves the hot server →
  favors P. This is MaxWeight / join-weighted-shorter-queue, scaled by the work
  actually being moved.
  - V·c_xfer — the price of offloading. Bigger V → more reluctant to pay
  transfer → fewer offloads. This is Neely's [O(1/V), O(V)] knob.
  - Z_itl·(ITL_P − ITL_D) — the disaggregation payoff, analytically. ITL_P < 
  ITL_D almost always (remote prefill keeps prefill compute off the decode
  batch), so this term is negative → pushes toward P, and more strongly when the
  ITL SLO is already being violated (Z_itl large). The reason PD-disagg exists
  shows up as a term that activates exactly when you need it.
  - Z_ttft·(TTFT_P − TTFT_D) — if remote prefill would worsen TTFT (congested
  prefill queue) and you're already missing TTFT-SLO, this term is positive →
  discourages P. Self-correcting.

  This is the analytical result you wanted: a closed-form, interpretable
  threshold where backlog balancing, SLO pressure, and transfer cost trade off
  explicitly.

  Two caveats I have to flag (both real)

  1. Dimensional heterogeneity across queues. Q·W_p is GPU-s², Z_itl·ΔITL is
  (s/token·s)·(s/token)… the four terms carry different units. Neely's math
  tolerates this (each ½Q_i² is just a nonneg term, the policy is still
  stabilizing), but the relative weighting is then arbitrary unless you
  normalize each queue (e.g., divide by its SLO/scale) or carry explicit
  conversion constants. V interacts with that scaling. We will have to pin
  normalization before any empirical run means anything — this is the units
  discussion coming back to bite.
  2. The rule needs counterfactual predictors TTFT_P, TTFT_D, ITL_P, ITL_D at 
  decision time. But — and this is the nice part — they only need to be roughly
  right: the virtual queues Z self-correct from realized SLO misses, so
  systematic predictor bias gets absorbed over time. The predictor can be crude;
  the feedback loop fixes the average.

  My question

  Given caveat 2 — how accurate do we make the counterfactual predictors TTFT_x,
  ITL_x?

  - (α) Crudest: W/μ-style with online-measured rates, lean hard on
  virtual-queue self-correction. Fastest to stand up, most "empirical," weakest
  per-decision accuracy.
  - (β) Use the per-server f(state) rate curves we discussed to predict the
  counterfactual post-routing batch state (the white-box interpolator). More
  accurate per decision, more to build.
  - (γ) Hybrid: α for TTFT (prefill side is near-deterministic anyway), β for
  ITL (the decode-side counterfactual is the whole game).

  I recommend (γ) — spend modeling effort only where the counterfactual is both
  uncertain and decision-critical (decode ITL), and keep the easy prefill side
  crude. Which way?

✻ Cooked for 1m 21s

❯ Before we focus on caveat 2, I want to focus on caveat 1.                     

⏺ Good — caveat 1 is more fundamental than it looks, and it's where the units
  discussion finally pays off. Let me be precise about what is and isn't broken.

  The math doesn't break — but the resolution is not automatic

  Neely's guarantees (mean-rate stability ⇒ constraints met; penalty within
  O(1/V) of optimal) hold for any positive weighting of the queues. Summing ½ Σ 
  Q_i² over heterogeneous-unit queues is a legal scalar potential. So nothing is
  wrong.

  What the heterogeneity costs you is meaning of the comparison: in Σ_i 
  Q_i·Δ_i(x) you're adding GPU-s² to (s/token)²-ish terms, and the relative
  weight among them is then implicitly set by whatever raw units you happened to
  pick. That's not a neutral choice — it silently encodes a priority policy.
  The honest form is

  L = ½ Σ_i w_i · Q_i²

  with explicit weights w_i. The heterogeneity forces you to choose w_i; you
  don't get to skip it. The w_i are exactly where "how much do I care about ITL
  vs TTFT vs backlog" lives, and where the unit conversion happens.

  Why you can't just "convert everything to one unit"

  The tempting move — express all four queues in GPU-seconds — doesn't work 
  cleanly, and it's worth seeing why. Backlogs Q_p, Q_d are work (GPU-s). SLO
  violations are delay (s) or rate (s/token). Work and time differ by a rate,
  and the rate μ_d(B) is state-dependent — the very thing we're modeling. So any
  "common currency" conversion factor is itself the unknown state-dependent
  quantity. Common-currency is circular here. Don't go down that road.

  The clean route: nondimensionalize each queue by its own scale

  Divide each queue by its characteristic scale so every term becomes a pure
  number:

  ┌────────┬───────────────────────────────────┬───────────────────────────┐
  │ Queue  │           Natural scale           │        Normalized         │
  ├────────┼───────────────────────────────────┼───────────────────────────┤
  │ Q_p,   │ reference work W* (e.g.           │ Q/W*                      │
  │ Q_d    │ target/capacity backlog in GPU-s) │                           │
  ├────────┼───────────────────────────────────┼───────────────────────────┤
  │ Z_ttft │ SLO target τ_ttft                 │ Z_ttft/τ_ttft (fractional │
  │        │                                   │  TTFT violation)          │
  ├────────┼───────────────────────────────────┼───────────────────────────┤
  │ Z_itl  │ SLO target τ_itl                  │ Z_itl/τ_itl (fractional   │
  │        │                                   │ ITL violation)            │
  └────────┴───────────────────────────────────┴───────────────────────────┘

  Now every term in Σ w_i Q_i Δ_i is dimensionless, the four are directly
  comparable, and V is a single dimensionless knob trading transfer-cost against
  (normalized) SLO pressure. The [O(1/V), O(V)] result states cleanly in pure
  numbers.

  The pushback to keep in view: normalizing by τ is not neutral — it declares "a
  10% TTFT overshoot and a 10% ITL overshoot are equally bad." That's a real
  policy stance. If it isn't what you want, that's exactly when you add a
  deliberate w_i ≠ 1/scale² to skew priority. Make it a conscious dial, not an
  accident of units.

  One structural question this forces (and I want your call)

  Do we even keep the physical queues Q_p, Q_d once Z_ttft, Z_itl enforce the
  SLOs?

  - TTFT-SLO bounded ⇒ prefill-path backlog implicitly bounded (Little's law).
  So Q_p is arguably redundant with Z_ttft.
  - But the SLO queues constrain delay/rate, not stability. They say nothing in
  the infeasible region. If load exceeds capacity, you want a backlog queue that
  visibly blows up so the policy sheds/throttles rather than silently violating
  everything.

  So: keep Q_p, Q_d for the stability role, Z_ttft, Z_itl for the SLO role (they
  answer different questions), or drop the physical queues and trust the SLO
  virtual queues alone?

  I recommend keeping both — they're not redundant, they guard different failure
  modes (instability vs SLO miss), and the cost is just two more normalized
  terms. But this is a genuine design fork. Which way?

✻ Churned for 2m 3s

❯ Yes, I agree with keeping both. I also agree with normalizing Z_ttft and      
  Z_itl with the corresponding SLO targets. What is not clear is the reference  
  work W*. can you expand more on that?                                         

⏺ Good question — W* is less obvious than the SLO normalizations, because
  backlog (work) and the SLO (delay) live in different dimensions. Here's the
  reasoning.

  What W* has to do

  Normalizing Z by τ was easy because Z and τ share units. Q_p, Q_d are in GPU-s
  (work); there is no SLO in work-units to divide by. So W* is really setting
  an exchange rate: in the decision rule, the backlog-balancing term (Q_d − 
  Q_p)·W_p competes against the SLO terms, and W* decides "how many GPU-s of
  backlog imbalance is worth one unit of SLO pressure." It is not a neutral
  scaling — it's the price of backlog relative to SLO.

  The principled choice: convert backlog to an SLO-window

  Backlog and delay are linked by the drain rate: delay ≈ work / rate. So the
  natural reference work is "the backlog that drains in exactly one TTFT-SLO 
  window":

  W*_p = μ_p^nom · τ_ttft        W*_d = μ_d^nom · τ_ttft

  (μ^nom = a fixed nominal drain rate, units GPU-s drained per wall-s). Then:

  Q_p / W*_p  ≈  (Q_p / μ_p) / τ_ttft  =  (predicted queueing delay) / (TTFT
  SLO)

  So a normalized physical queue reads as "how many SLO-windows deep the queue 
  is." > 1 means an arriving request's wait alone already blows the TTFT budget.
  This is the same philosophy as normalizing Z by τ — everything becomes 
  SLO-relative — and it makes the backlog term and the SLO terms directly
  comparable, which is exactly what caveat 1 needed.

  Three things to get right

  1. Use a nominal fixed rate, not the instantaneous μ(B). We established μ is
  state-dependent — but if W* moved with state, the Lyapunov geometry would
  breathe and the analysis breaks. Keep W* a constant (e.g. peak/saturated rate,
  or a fixed operating-point rate). The state-dependence belongs in the
  predictors (caveat 2), not in the normalization constant.
  2. This does not make Q redundant with Z — and it's worth seeing why, since
  the scales now look similar. Normalized Q is the instantaneous predicted
  fractional delay; Z is the time-integrated fractional violation. That's a
  proportional term and an integral term — a P/I pair, not a duplicate. Keeping
  both is what gives you both fast response (Q) and zero steady-state bias (Z).
  3. Q_d is a mixture (prefill-queueing work plus ongoing-decode work) — so
  μ_d^nom · τ_ttft is only strictly "the TTFT window" for its queueing part. For
  a stability scale that's fine; it's a reference magnitude, not a claim that
  all of Q_d is TTFT-bound. Just don't over-read normalized Q_d as a pure TTFT
  predictor.

  The alternative I'm rejecting (and why)

  The other candidate is capacity-relative: W* =  server work-capacity (e.g. KV
  budget expressed as work), so Q/W* = "fraction of capacity backlogged." It's
  defensible for the pure stability role, but it sets the backlog↔SLO exchange
  rate by hardware capacity rather than by the latency target you actually care 
  about — so the decision rule would weight backlog imbalance in capacity-units
  while weighting everything else in SLO-units. Less coherent. I'd only switch
  to it if you want the physical queues to track saturation explicitly rather
  than delay.

  My recommendation: SLO-window scale, W* = μ^nom · τ_ttft, with fixed nominal 
  rates. Do you want that, or the capacity-relative scale for the stability
  queues?
✻ Worked for 1m 16s

❯ It usually helps if we can ground the system in example real values and       
  sanity check our model choices.                                               

⏺ Excellent instinct — let me put real numbers in and see what survives. I'll
  use the repo's default model.

  Concrete setup (rough but realistic)

  Qwen3-14B, single H100, TP=1, bf16. Weights ≈ 28 GB; HBM ≈ 3.35 TB/s → one
  weight-read ≈ 8.4 ms.

  ┌───────────────┬──────────────────────────┬─────────────────────────────┐
  │   Quantity    │          Value           │            Basis            │
  ├───────────────┼──────────────────────────┼─────────────────────────────┤
  │ Prefill work  │ ~0.056 GPU-s / 1k prompt │ 2·P·N FLOPs at ~500         │
  │ W_p           │  tokens                  │ effective TFLOP/s           │
  ├───────────────┼──────────────────────────┼─────────────────────────────┤
  │ Prefill rate  │ ~18k tok/s               │ = 1k / 0.056s               │
  │ μ_p           │                          │                             │
  ├───────────────┼──────────────────────────┼─────────────────────────────┤
  │ Decode step,  │ ~8.4 ms/token            │ weight-read bound           │
  │ B=1           │                          │                             │
  ├───────────────┼──────────────────────────┼─────────────────────────────┤
  │ Decode step,  │ ~20 ms/step, aggregate   │ weight amortized,           │
  │ B=32          │ ~1600 tok/s              │ KV+compute grow             │
  ├───────────────┼──────────────────────────┼─────────────────────────────┤
  │ τ_ttft, τ_itl │ 500 ms, 150 ms           │ "standard" class from       │
  │               │                          │ CLAUDE.md                   │
  └───────────────┴──────────────────────────┴─────────────────────────────┘

  Two tagged requests, in uncontended GPU-s:

  ┌─────────────────────────────────┬───────┬──────────────────┬──────────┐
  │            Workload             │  W_p  │       W_d        │ Dominant │
  ├─────────────────────────────────┼───────┼──────────────────┼──────────┤
  │ Chat (1k prompt, 200 out)       │ 0.056 │ 200×8.4ms = 1.68 │ decode   │
  ├─────────────────────────────────┼───────┼──────────────────┼──────────┤
  │ Summarize (32k prompt, 100 out) │ 1.8   │ 100×8.4ms = 0.84 │ prefill  │
  └─────────────────────────────────┴───────┴──────────────────┴──────────┘

  Check 1 — does the normalization make terms commensurate? ✓

  W*_p = μ_p^nom · τ_ttft. In GPU-s currency a saturated single GPU drains 1
  GPU-s of work per wall-s, so μ_p^nom ≈ 1 → W*_p = 0.5 GPU-s ≈ 9 prompts of 1k.
  Sanity: at 18k tok/s you prefill 9k tokens in 500 ms. ✓

  Decision-rule coefficient on the backlog term: W_p/W*_p = 0.056/0.5 ≈ 0.11.
  The SLO terms after normalization are z·(ΔTTFT/τ) etc., also O(0.1). All four 
  terms land at the same order — the caveat-1 fix works numerically, not just
  formally.

  Check 2 — does the decision driver shift with regime? ✓ (and it's the right 
  shift)

  - Chat: W_d ≫ W_p. The movable lever (W_p) is tiny; decision is dominated by
  the z_itl·(ITL_P − ITL_D) term — i.e. ITL protection, exactly the physical
  reason disagg exists. Backlog-balancing barely moves.
  - Summarize: W_p > W_d. Now offloading prefill is a large lever;
  backlog-balancing and offload genuinely matter.

  The model predicts decisions are ITL-driven for chat and load-driven for
  long-context — which matches the physics. Good sign.

  Check 3 — the GPU-s work currency breaks for decode ✗ (the real finding)

  Here's where grounding earns its keep. Compute the decode server's drain rate
  in your chosen currency (uncontended GPU-s) at B=32:

  μ_d = B × (uncontended_step / actual_step) = 32 × (8.4ms / 20ms) ≈ 13.4 GPU-s
  drained per wall-s

  It's >> 1, and it climbs with B (≈1 at B=1, ≈13 at B=32). Weight-read
  amortization means the decode server processes "uncontended GPU-seconds"
  superlinearly. Consequences:

  1. μ_d^nom cannot be a stable constant — it ranges over an order of magnitude
  with batch. So W*_d = μ_d^nom · τ_ttft is shaky exactly where we need a fixed
  normalization constant.
  2. It signals that uncontended GPU-s is the wrong work unit for decode. It's
  clean for prefill (rate pinned at ~1, no amortization) but pathological for
  decode.

  The honest fix: measure decode backlog in pending output-tokens (Σ remaining 
  tokens over in-flight + queued decodes — a direct observable in both BLIS and
  vLLM), with rate = aggregate decode tok/s (state-dependent). This is the
  natural currency where the rate is a measurable throughput, not an
  amortization artifact. (Note this re-confirms the original asymmetry:
  prefill's currency is GPU-s/tokens, decode's is token-steps — same thing the
  rectangle-vs-ramp point predicted.)

  The decision this surfaces

  For the decode normalization constant μ_d^nom, I propose tying it to the SLO
  rather than guessing a batch — pick the SLO-critical operating point:

  B*  = largest batch with ITL ≤ τ_itl
  μ_d^nom = B* / τ_itl     (tokens/s)        W*_d = μ_d^nom · τ_ttft
  (pending-token scale)

  This makes the decode normalization self-consistently SLO-grounded (same
  philosophy as W*_p = μ_p·τ_ttft and the Z/τ normalizations): the nominal rate
  is "throughput at the batch where we're just meeting the ITL SLO."
  Numerically, if B*≈128 at τ_itl=150ms, μ_d^nom ≈ 850 tok/s.

  Two questions, but they're really one call: (a) switch the decode work
  currency from uncontended-GPU-s to pending-output-tokens, and (b) set μ_d^nom 
  = B*/τ_itl at the SLO-critical batch? Or do you want to keep a single GPU-s
  currency across both servers and eat the shaky μ_d^nom?

✻ Crunched for 2m 38s

※ recap: We're designing a drift-plus-penalty router for prefill/decode disaggregation, optimizing TTFT and ITL SLOs. Right now we're grounding the model in real Qwen3-14B numbers to fix queue normalization. Next: decide whether to switch the decode work unit to pending output-tokens with μ_d set at the SLO-critical batch. (disable recaps in /config)

❯ I don't understand: W*_p = μ_p^nom · τ_ttft. In GPU-s currency a saturated    
  single GPU drains 1 GPU-s of work per wall-s, so μ_p^nom ≈ 1 → W*_p = 0.5     
  GPU-s ≈ 9 prompts of 1k. Sanity: at 18k tok/s you prefill 9k tokens in 500    
  ms. And I don't think such an assumption is correct. Can you explain why do   
  you think it is correct?                                                      

⏺ You're right to distrust it. Let me separate what's tautologically true from
  what I smuggled in.

  Where "rate = 1" actually comes from

  If the prefill server processed requests strictly serially (one fully before
  the next), then by definition of "uncontended GPU-time," in T wall-seconds it
  completes requests whose uncontended times sum to T. Drain rate = 1 GPU-s of
  work per wall-s — tautologically, regardless of compute/memory bound. That's
  the only sense in which "1" is safe.

  Why that's wrong as a general claim

  Real prefill servers run requests concurrently/batched, and concurrency
  changes the rate:

  - Compute-bound prefill (large prompt, e.g. 1k tokens — arithmetic intensity
  ~1000 FLOP/byte, well above the H100 ridge): batching gives no amortization,
  requests time-share the compute, aggregate progress still sums to T → rate
  ≈ 1. Here my claim happens to hold.
  - Memory-bound prefill (short prompt, e.g. 50 tokens — ~50 FLOP/byte, below
  ridge, dominated by the one-time 8.4 ms weight read): a single such request
  under-saturates the GPU. Batching many of them amortizes the weight read →
  aggregate throughput > 1 in uncontended-GPU-s. Same amortization phenomenon as
  decode.

  So μ_p^nom ≈ 1 is only the compute-bound baseline. I presented it as a flat
  constant, which directly contradicts my own earlier point that prefill rate is
  "rising-then-flat" (under-saturation regime). I over-simplified — you caught
  a real inconsistency.

  The actual fix (and it's cleaner)

  The "GPU-s currency with rate 1" was a fragile reframing I shouldn't have
  introduced. Drop it. Measure prefill backlog in prompt-tokens with a measured 
  throughput, exactly as we just decided for decode:

  W*_p = μ_p^measured · τ_ttft  =  18,000 tok/s × 0.5 s  =  9,000 prompt-tokens

  Same "≈ 9 prompts of 1k" number — but now derived from a measured rate, with
  no rate-1 fiction and no assumption about which regime you're in. The
  state-dependence (compute- vs memory-bound) lives in μ_p^measured, which you
  observe, not in the normalization constant.

  What this resolves

  It unifies both servers and kills the artifact from Check 3 in one move:

  ┌─────────┬─────────────────┬───────────────────────┬────────────────────┐
  │ Server  │    Backlog      │   Rate (measured,     │         W*         │
  │         │    currency     │   state-dependent)    │                    │
  ├─────────┼─────────────────┼───────────────────────┼────────────────────┤
  │ Prefill │ prompt-tokens   │ μ_p (tok/s)           │ μ_p^nom · τ_ttft   │
  ├─────────┼─────────────────┼───────────────────────┼────────────────────┤
  │ Decode  │ pending         │ μ_d(B) (tok/s)        │ μ_d^nom · τ_ttft,  │
  │         │ output-tokens   │                       │ μ_d^nom = B*/τ_itl │
  └─────────┴─────────────────┴───────────────────────┴────────────────────┘

  Neither rate is assumed to be 1. "Rate = 1" was never a law — it was an
  artifact of choosing uncontended-GPU-s as the currency and implicitly assuming
  serial, compute-bound processing. Tokens-with-measured-throughput is the
  honest unit for both.

  Does this resolve your objection — and do you want to lock the token currency
  for both servers?

✻ Crunched for 1m 50s

❯ I am not looking for a fix, rather I'm looking for a good enough system model 
  with assumptions that make sense. You can't change the system model halfway.  

⏺ You're right, and the methodological point matters more than any single
  choice: I was patching reactively (GPU-s → "rate ≈ 1" → tokens), which is
  exactly the sin — you can't re-pick the currency every time a number looks
  off. Let me step back, commit to one model, and justify the foundation so it
  doesn't move.

  The foundational decision is the work currency — and there's a principled 
  criterion

  A queueing model must cleanly factor demand from supply: demand is a property
  of the request/workload; supply is a property of the hardware + current state.
  The currency should make that factorization clean. Judge the two candidates
  by it:

  - Tokens: demand = token counts (pure workload property,
  hardware-independent); supply = tokens/s (pure hardware+state property). All
  the physics — compute- vs memory-bound, batching amortization, KV pressure —
  lives entirely in μ(state). Clean.
  - Uncontended GPU-s: the "work" is itself a hardware-time, so hardware shows
  up in both the demand and the rate. The rate then degenerates into a residual
  amortization factor (the 1→13 artifact). Demand and supply are entangled, the
  model is over-determined — that's why "rate = 1" kept slipping out from under
  us.

  So work = tokens, chosen because it satisfies the factorization criterion —
  not because the last number looked nicer. We hold it.

  The committed model (one set of assumptions, fixed)

  ┌─────┬────────────────────────────────────────────────────┬───────────────┐
  │  #  │                     Assumption                     │ Load-bearing? │
  ├─────┼────────────────────────────────────────────────────┼───────────────┤
  │ A1  │ Topology: 1 prefill server + 1 decode server       │ no            │
  │     │ (pools later)                                      │               │
  ├─────┼────────────────────────────────────────────────────┼───────────────┤
  │ A2  │ Decision: x ∈ {P, D}, made once at arrival         │ no            │
  ├─────┼────────────────────────────────────────────────────┼───────────────┤
  │ A3  │ Time: discrete ticks; one decode iteration = one   │ no            │
  │     │ slot                                               │               │
  ├─────┼────────────────────────────────────────────────────┼───────────────┤
  │ A4  │ Currency = tokens; each server drains its own      │ yes           │
  │     │ token type at its own rate                         │               │
  ├─────┼────────────────────────────────────────────────────┼───────────────┤
  │     │ Demand: a_p = uncached prompt tokens (known); a_d  │               │
  │ A5  │ = output tokens, assumed known at arrival (oracle  │ yes           │
  │     │ on N_out) for the base model                       │               │
  ├─────┼────────────────────────────────────────────────────┼───────────────┤
  │ A6  │ Supply: each server has its own measured μ(state)  │ yes           │
  │     │ [tok/s]; f_p ≠ f_d; all hardware physics here      │               │
  ├─────┼────────────────────────────────────────────────────┼───────────────┤
  │     │ Decode dual role: in mode D, prefill on the decode │               │
  │ A7  │  server enters only as a state-dependent drop in   │ yes           │
  │     │ decode rate (= ITL inflation) while co-scheduled — │               │
  │     │  not as a separate queue                           │               │
  ├─────┼────────────────────────────────────────────────────┼───────────────┤
  │ A8  │ KV occupancy caps batch; at the cap, arrivals wait │ no (base)     │
  │     │  (HOL blocking)                                    │               │
  └─────┴────────────────────────────────────────────────────┴───────────────┘

  Queues, all in token currency:
  - Q_p = pending prefill-tokens; Q_d = pending decode-tokens.
  - Z_ttft, Z_itl virtual, normalized by τ.
  - W*_p = μ_p^nom · τ_ttft (tokens); μ_d^nom = B*/τ_itl, W*_d = μ_d^nom · 
  τ_ttft.

  What I want ratified before we build any dynamics

  Don't let me proceed until the foundation is accepted as a whole. The two
  assumptions I'd scrutinize hardest, because they're load-bearing and arguable:

  - A5 (known N_out): you proposed this simplification yourself earlier. It
  makes both demands known at arrival, which is a huge simplification — and a
  real distortion, since N_out uncertainty is intrinsic. Accept for the base
  model, relax later?
  - A7 (decode dual role as pure rate-inflation): this is the modeling
  sleight-of-hand that lets the decode server stay a single queue. If
  prefill-at-decode is bursty enough to need its own backlog term, A7 is too
  coarse.

  Does this assumption set hold up as the fixed foundation — and do you accept
  A5 and A7, or do you want to challenge either before we put dynamics on top?

✻ Crunched for 2m 8s

❯ So these are the metrics exposed by vllm servers:                             
                                                                                
                                                                                
  The following metrics are exposed:                                            
                                                                                
  General Metrics¶                                                              
                                                                                
  Metric Name    Type    Description                                            
  vllm:corrupted_requests    Counter    Corrupted requests, in terms of total   
  number of requests with NaNs in logits.                                       
  vllm:external_prefix_cache_hits    Counter    External prefix cache hits from 
  KV connector cross-instance cache sharing, in terms of number of cached       
  tokens.                                                                       
  vllm:external_prefix_cache_queries    Counter    External prefix cache        
  queries from KV connector cross-instance cache sharing, in terms of number of 
  queried tokens.                                                               
  vllm:generation_tokens    Counter    Number of generation tokens processed.   
  vllm:mm_cache_hits    Counter    Multi-modal cache hits, in terms of number   
  of cached items.                                                              
  vllm:mm_cache_queries    Counter    Multi-modal cache queries, in terms of    
  number of queried items.                                                      
  vllm:num_preemptions    Counter    Cumulative number of preemption from the   
  engine.                                                                       
  vllm:prefix_cache_hits    Counter    Prefix cache hits, in terms of number of 
  cached tokens.                                                                
  vllm:prefix_cache_queries    Counter    Prefix cache queries, in terms of     
  number of queried tokens.                                                     
  vllm:prompt_tokens    Counter    Number of prefill tokens processed.          
  vllm:prompt_tokens_by_source    Counter    Number of prompt tokens by source. 
  vllm:prompt_tokens_cached    Counter    Number of cached prompt tokens (local 
  + external).                                                                  
  vllm:request_success    Counter    Count of successfully processed requests.  
  vllm:engine_sleep_state    Gauge    Engine sleep state; awake = 0 means       
  engine is sleeping; awake = 1 means engine is awake; weights_offloaded = 1    
  means sleep level 1; discard_all = 1 means sleep level 2.                     
  vllm:kv_cache_usage_perc    Gauge    KV-cache usage. 1 means 100 percent      
  usage.                                                                        
  vllm:lora_requests_info    Gauge    Running stats on lora requests.           
  vllm:num_requests_running    Gauge    Number of requests in model execution   
  batches.                                                                      
  vllm:num_requests_waiting    Gauge    Number of requests waiting to be        
  processed.                                                                    
  vllm:num_requests_waiting_by_reason    Gauge    Number of waiting requests by 
  reason. Reason labels: 'capacity' = waiting for scheduling capacity;          
  'deferred' = deferred by transient constraints (LoRA budget, KV transfer,     
  blocked status). Sum of all reasons equals vllm:num_requests_waiting.         
  vllm:e2e_request_latency_seconds    Histogram    Histogram of e2e request     
  latency in seconds.                                                           
  vllm:inter_token_latency_seconds    Histogram    Histogram of inter-token     
  latency in seconds.                                                           
  vllm:iteration_tokens_total    Histogram    Histogram of number of tokens per 
  engine_step.                                                                  
  vllm:kv_block_idle_before_evict_seconds    Histogram    Histogram of idle     
  time before KV cache block eviction. Sampled metrics (controlled by           
  --kv-cache-metrics-sample).                                                   
  vllm:kv_block_lifetime_seconds    Histogram    Histogram of KV cache block    
  lifetime from allocation to eviction. Sampled metrics (controlled by          
  --kv-cache-metrics-sample).                                                   
  vllm:kv_block_reuse_gap_seconds    Histogram    Histogram of time gaps        
  between consecutive KV cache block accesses. Only the most recent accesses    
  are recorded (ring buffer). Sampled metrics (controlled by                    
  --kv-cache-metrics-sample).                                                   
  vllm:request_decode_time_seconds    Histogram    Histogram of time spent in   
  DECODE phase for request.                                                     
  vllm:request_generation_tokens    Histogram    Number of generation tokens    
  processed.                                                                    
  vllm:request_inference_time_seconds    Histogram    Histogram of time spent   
  in RUNNING phase for request.                                                 
  vllm:request_max_num_generation_tokens    Histogram    Histogram of maximum   
  number of requested generation tokens.                                        
  vllm:request_params_max_tokens    Histogram    Histogram of the max_tokens    
  request parameter.                                                            
  vllm:request_params_n    Histogram    Histogram of the n request parameter.   
  vllm:request_prefill_kv_computed_tokens    Histogram    Histogram of new KV   
  tokens computed during prefill (excluding cached tokens).                     
  vllm:request_prefill_time_seconds    Histogram    Histogram of time spent in  
  PREFILL phase for request.                                                    
  vllm:request_prompt_tokens    Histogram    Number of prefill tokens           
  processed.                                                                    
  vllm:request_queue_time_seconds    Histogram    Histogram of time spent in    
  WAITING phase for request.                                                    
  vllm:request_time_per_output_token_seconds    Histogram    Histogram of       
  time_per_output_token_seconds per request.                                    
  vllm:time_to_first_token_seconds    Histogram    Histogram of time to first   
  token in seconds.                                                             
  Speculative Decoding Metrics¶                                                 
                                                                                
  Metric Name    Type    Description                                            
  vllm:spec_decode_num_accepted_tokens    Counter    Number of accepted tokens. 
  vllm:spec_decode_num_accepted_tokens_per_pos    Counter    Accepted tokens    
  per draft position.                                                           
  vllm:spec_decode_num_draft_tokens    Counter    Number of draft tokens.       
  vllm:spec_decode_num_drafts    Counter    Number of spec decoding drafts.     
  NIXL KV Connector Metrics¶                                                    
                                                                                
  Metric Name    Type    Description                                            
  vllm:nixl_num_failed_notifications    Counter    Number of failed NIXL KV     
  Cache notifications.                                                          
  vllm:nixl_num_failed_transfers    Counter    Number of failed NIXL KV Cache   
  transfers.                                                                    
  vllm:nixl_num_kv_expired_reqs    Counter    Number of requests that had their 
  KV expire. NOTE: This metric is tracked on the P instance.                    
  vllm:nixl_bytes_transferred    Histogram    Histogram of bytes transferred    
  per NIXL KV Cache transfers.                                                  
  vllm:nixl_num_descriptors    Histogram    Histogram of number of descriptors  
  per NIXL KV Cache transfers.                                                  
  vllm:nixl_post_time_seconds    Histogram    Histogram of transfer post time   
  for NIXL KV Cache transfers.                                                  
  vllm:nixl_xfer_time_seconds    Histogram    Histogram of transfer duration    
  for NIXL KV Cache transfers.                                                  
  Model Flops Utilization (MFU) Performance Metrics¶                            
                                                                                
  These metrics are available via --enable-mfu-metrics:                         
                                                                                
  Metric Name    Type    Description                                            
  vllm:estimated_flops_per_gpu_total    Counter    Estimated number of floating 
  point operations per GPU (for Model Flops Utilization calculations).          
  vllm:estimated_read_bytes_per_gpu_total    Counter    Estimated number of     
  bytes read from memory per GPU (for Model Flops Utilization calculations).    
  vllm:estimated_write_bytes_per_gpu_total    Counter    Estimated number of    
  bytes written to memory per GPU (for Model Flops Utilization calculations).   
                                                                                
  I have to be honest here. Because my queueing model has a work based approach 
  (not a token based approach), I'd like to use that approach. Only when we     
  realize that this approach is not feasible, we consider the token based       
  approach.                                                                     
                                                                                
                                                                                
  Now, here is what I am thinking:                                              
  Each request comes with a demand (work) in GPU time. It possibly waits in the 
  queue. And if we consider the unnormalized system for now, the queue time     
  will be GPU time, since there will be a couple of other requests waiting      
  ahead, each requiring some GPU time. When the request is inserted in a batch, 
  the server batches this request along with a couple of other requests. When   
  the request is added in the batch, it adds it's own "marginal work" depending 
  on the request processing stage it is in. A request could be in its chunked   
  prefill stages, or it could be in any of the decode stages. When the server   
  adds the marginal work of this request, its iteration time increases. What    
  this means is that the iteration time depends on the number of requests       
  concurrently running. So in each iteration, the server processes marginal     
  work per request times the number of request concurrently running. If we      
  consider iteration time in wall-clock time, then the unit we have is GPU      
  time/wall clock time.                                                         
                                                                                
  Let me pause here and see whether this makes sense to you. Where I want your  
  help is given the set of metrics available in vllm, can we derive the         
  interesting quantities in the model described so far.                         

⏺ Your model makes sense — it's the standard "iteration time as a function of
  the concurrent batch" view, and it's the right mental picture. Let me validate
  it precisely, flag the two places it strains, then map it onto the vLLM
  metrics (your actual question).

  Verdict on the model: coherent, with two precise caveats

  The structure is right: each iteration the server advances all B running
  requests at once, and T_iter grows with the batch. Two corrections to keep it
  honest:

  1. Iteration time isn't just Σ marginal — there's a shared fixed term. T_iter 
  = O_fixed + Σ_i c_i(stage_i, L_i). O_fixed ≈ the weight read (~8.4 ms here),
  paid once per iteration, attributable to no single request. c_i is the
  per-request marginal (its KV read + compute). Drop O_fixed and you lose
  exactly the amortization that makes batching efficient.
  2. Therefore GPU-time "work" is not conserved under batching — and only on the
  decode side. A request's marginal c_i depends on its batchmates, so it isn't
  an intrinsic per-request demand. The deeper reason, and the clean criterion: a
  work currency is conserved iff the bottleneck resource scales with the
  request and isn't shared.
    - Prefill is compute-bound; FLOPs scale per-token and aren't shared → work 
  is conserved/additive. The work model is clean here.
    - Decode is bandwidth-bound; weight-bytes are read once and shared across 
  the batch → byte-work is not additive per request. The work model strains
  here.

  This is the same easy/hard split as always, now with a sharp cause. You
  haven't hit "infeasible" — you've located exactly where to watch.

  Mapping the model to vLLM metrics

  Directly observable:

  ┌─────────────────────┬───────────────────────────────────────────────────┐
  │   Model quantity    │                    vLLM metric                    │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Batch size B        │ num_requests_running                              │
  │ (concurrency)       │                                                   │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Queue length        │ num_requests_waiting (+ _by_reason: capacity =    │
  │ (count)             │ KV/sched blocking → your A8)                      │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Iteration time      │ inter_token_latency_seconds (≈ T_iter during      │
  │ T_iter              │ decode)                                           │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Tokens per          │ iteration_tokens_total (prefill chunks inflate    │
  │ iteration           │ this)                                             │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Queue wait          │ request_queue_time_seconds                        │
  │ (wall-clock)        │                                                   │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ TTFT                │ time_to_first_token_seconds                       │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Prefill phase time  │ request_prefill_time_seconds                      │
  │ (contended)         │                                                   │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Decode phase time   │ request_decode_time_seconds                       │
  │ (contended)         │                                                   │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Uncached prefill    │ request_prefill_kv_computed_tokens (or            │
  │ tokens a_p          │ prefix_cache_queries − hits)                      │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Output tokens a_d   │ request_generation_tokens (known only post-hoc)   │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ KV occupancy (A8)   │ kv_cache_usage_perc                               │
  ├─────────────────────┼───────────────────────────────────────────────────┤
  │ Transfer penalty    │ nixl_xfer_time_seconds, nixl_bytes_transferred    │
  │ c_xfer              │                                                   │
  └─────────────────────┴───────────────────────────────────────────────────┘

  That last row is a gift: the cost of choosing remote-prefill (g(x)=c_xfer) is
  directly measured by the NIXL connector. Your penalty term is observable, not
  assumed.

  Derivable (metric + model config / MFU):

  ┌───────────────────┬─────────────────────────────────────────────────────┐
  │     Quantity      │                         How                         │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ O_fixed           │ min T_iter ≈ ITL at B=1; or weight_bytes / peak_BW  │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ T_iter(B) law     │ regress inter_token_latency on num_requests_running │
  │                   │  → fit O_fixed + c·B                                │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ Compute work      │ estimated_flops_per_gpu_total                       │
  │ (FLOPs)           │ (--enable-mfu-metrics) — conserved, clean for       │
  │                   │ prefill                                             │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ Memory work       │ estimated_read_bytes_per_gpu_total — amortized, the │
  │ (bytes)           │  non-conserved decode term                          │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ Per-resource      │ MFU rate ÷ peak (FLOPs→compute ρ, bytes→bandwidth   │
  │ utilization ρ     │ ρ)                                                  │
  └───────────────────┴─────────────────────────────────────────────────────┘

  Not observable at all — and this is the honest gap for a work-based model:

  - Uncontended per-request GPU-time demand W. vLLM only exposes contended
  wall-clock phase times (request_prefill_time, request_decode_time). The
  intrinsic demand must be derived (roofline: W_p ≈ 2·P·a_p / peak_FLOPs), never
  directly measured.
  - Per-request marginal work c_i and the O_fixed split — only inferable in
  aggregate via the T_iter(B) regression, not per request.

  Where this leaves the work-based approach

  It's feasible to parameterize — but with a clear character: your demands are
  roofline/FLOP-derived quantities, your rates and queue waits are measured
  wall-clock, and the bridge between them is the state-dependent T_iter(B) law
  you fit from (inter_token_latency, num_requests_running). One asymmetry to
  internalize: SLOs and the observed queue wait are wall-clock; the work model 
  is GPU-time; the conversion factor is the (state-dependent) rate. So "queue
  time in GPU-time" is a latent quantity — what you observe is wall-clock wait,
  and you'd multiply by the rate to get GPU-time backlog.

  The one thing I'd test before trusting it: fit T_iter ≈ O_fixed + c·B from a
  live server and check whether a single linear c holds, or whether c drifts
  with context length / prefill-mix enough that "marginal work" can't be treated
  as a per-request constant. That regression is the make-or-break for the
  decode side.

  Do you want to (a) pin down how prefill demand W_p is derived (the clean half,
  roofline + request_prefill_kv_computed_tokens), or (b) go straight at the
  decode T_iter(B) regression — the half that decides whether the work currency
  survives?

✻ Baked for 3m 4s

※ recap: We're designing a DPP/Lyapunov routing algorithm (remote-prefill vs direct-decode) optimizing TTFT and ITL, using a work-based (GPU-time) queueing model. We've mapped the model to vLLM metrics. Next: decide whether to derive prefill demand W_p or tackle the decode T_iter(B) regression. (disable recaps in /config)

❯ 1. Iteration time isn't just Σ marginal — there's a shared fixed term. T_iter 
  = O_fixed + Σ_i c_i(stage_i, L_i). O_fixed ≈ the weight read (~8.4 ms here),  
  paid once per iteration, attributable to no single request. c_i is the        
  per-request marginal (its KV read + compute). Drop O_fixed and you lose       
  exactly the amortization that makes batching efficient.                       
                                                                                
  I agree that iteration time is T_iter(B) = alpha + \sum_i \delta(stage_i),    
  where \delta(stage_i) is the marginal cost/work from request i.               

⏺ Agreed — T_iter(B) = α + Σ_i δ(stage_i) is the right primitive. Now let me
  extract the one consequence that matters for the queueing formulation, because
  this equation settles the "is work conserved?" question exactly.

  α is the sole source of non-conservation

  Rearrange into a service rate. Per iteration the server advances every running
  request one step; aggregate useful throughput is

  μ_d(N) = N / (α + Σ_i δ_i)      [progress per wall-second]

  - If α = 0: μ_d = N/Σδ_i = 1/δ̄ — a constant rate. The server is a textbook
  work-conserving queue; demands are additive; backlog/μ is a fixed clearing
  time. No difficulty at all.
  - If α > 0: μ_d(N) is increasing and concave in N (rises from 1/(α+δ̄) at N=1
  toward 1/δ̄ as N→∞). The work needed to clear a fixed set of requests now
  depends on how they're batched (each iteration re-pays α), i.e. on the
  schedule — not on the jobs alone. That schedule-dependence is the 
  non-conservation, and α is its entire cause.

  So this isn't vague hand-waving anymore: non-conservation ⟺ α > 0.

  The δ/α ratio decides the regime — and it splits exactly along P/D

  ┌─────────┬──────────┬─────────────────┬─────┬───────────────────────────┐
  │         │    α     │                 │     │                           │
  │         │ (weight  │  δ (marginal)   │ δ/α │         Behavior          │
  │         │  read)   │                 │     │                           │
  ├─────────┼──────────┼─────────────────┼─────┼───────────────────────────┤
  │         │          │ ~28 ms /        │     │ α negligible → rate ≈     │
  │ Prefill │ ~8.4 ms  │ 512-tok chunk   │ ≫ 1 │ constant → conservative   │
  │         │          │ (compute)       │     │                           │
  ├─────────┼──────────┼─────────────────┼─────┼───────────────────────────┤
  │         │          │ ~1 ms / token   │     │ α-dominated → strongly    │
  │ Decode  │ ~8.4 ms  │ at L≈2k (KV     │ ≪ 1 │ load-dependent →          │
  │         │          │ read)           │     │ non-conservative          │
  └─────────┴──────────┴─────────────────┴─────┴───────────────────────────┘

  Decode aggregate throughput swings ~8× (1/(α+δ) → 1/δ) purely from amortizing
  α. Prefill barely moves. The asymmetry we kept circling is now a single
  number: δ/α.

  The reassurance this buys us

  The equation maps the decode server onto a known queue class: a single queue 
  with a load-dependent (increasing, concave) service rate μ_d(N) — equivalently
  load-dependent processor sharing / M/G/1 with state-dependent rate.
  Tractable.

  And critically for the DPP plan: DPP does not require work conservation. Its
  queue update Q(t+1) = max(Q(t) − b(t), 0) + a(t) accepts a state-dependent
  service term b(t) = μ_d(N(t))·Δt. So the thing we worried might sink the
  work-based model — non-conservation — is absorbed by DPP's structure, provided
  we can observe/estimate μ_d(state). And we can: it's exactly the
  (inter_token_latency, num_requests_running) regression — 1/T_iter vs B
  recovers α (intercept) and δ̄ (slope), hence μ_d(N).

  So the work currency survives, with one explicit obligation: carry α and δ 
  separately (the fixed/marginal split), never collapse them into one "work"
  scalar — that collapse is precisely what breaks conservation.

  Next, do you want to (a) write the decode queue update Q_d(t+1) with this
  μ_d(N) plugged in, or (b) nail the (ITL, B) regression that estimates α, δ
  from live metrics — the empirical anchor everything rests on?

✻ Cooked for 1m 59s

❯ So lets brainstorm on how to reconcile the real world implementation of the   
  algorithm as opposed to writing a performance modeling and analysis paper.    
                                                                                
  Also, I'd say that \mu(N) = \sum_i \delta_i / (\alpha + \sum_i \delta_i).     

⏺ Good pivot. And yes — your μ(N) = Σδ_i / (α + Σδ_i) is the cleaner choice,
  because it commits us to marginal-work currency (work = Σδ, with α as the
  server's per-iteration tax, not part of any request's demand). Two things fall
  out that matter for implementation:

  - It's bounded in (0,1) and increasing in N — much nicer than the
  token-currency rate that ran off to 13.
  - It collapses to an observable: μ = 1 − α/(α+Σδ) = 1 − α/T_iter ≈ 1 − α/ITL.
  So online you don't need the individual δ_i — just α (regression intercept)
  and the measured T_iter ≈ inter_token_latency. That's the whole rate from two
  scraped numbers.

  The paper-vs-reality gap, stated plainly

  The paper version silently assumes four things a real router doesn't have:

  ┌─────────────────────────┬───────────────────────────────────────────────┐
  │      Paper assumes      │                    Reality                    │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │ Fresh global queue      │ vLLM /metrics scraped periodically (≥         │
  │ state Q(t), Z(t)        │ 50ms–seconds); router sees stale state        │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │ Exact α, δ, μ           │ must be estimated online (rolling             │
  │                         │ regression), noisy, drift with context dist.  │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │ Known demands incl.     │ N_out unknown at decision time (A5 is a lie   │
  │ N_out                   │ in prod)                                      │
  ├─────────────────────────┼───────────────────────────────────────────────┤
  │ Instantaneous actuation │ many requests routed between metric updates → │
  │                         │  herd effect on the "good" server             │
  └─────────────────────────┴───────────────────────────────────────────────┘

  So the real question isn't "is the model right" — it's "does the decision rule
  degrade gracefully when fed stale, noisy, incomplete inputs."

  The reconciliation that makes me optimistic

  Split the decision rule's terms by where they actually come from in a real
  router (e.g. an llm-d-style EPP that owns controller state and scrapes vLLM):

  Choose P  ⟺  V·c_xfer + z_ttft·(TTFT_P−TTFT_D)/τ + z_itl·(ITL_P−ITL_D)/τ  <
  q_d·(W_p/W*_d) − q_p·(W_p/W*_p)

  ┌──────────────────────┬────────────────────────────────────┬─────────────┐
  │         Term         │               Source               │  Freshness  │
  ├──────────────────────┼────────────────────────────────────┼─────────────┤
  │                      │ controller-owned accumulators,     │ exact,      │
  │ z_ttft, z_itl        │ bumped on each completion's        │ fresh       │
  │                      │ realized TTFT/ITL vs τ             │             │
  ├──────────────────────┼────────────────────────────────────┼─────────────┤
  │ c_xfer               │ rolling mean of                    │ fresh-ish   │
  │                      │ nixl_xfer_time_seconds             │             │
  ├──────────────────────┼────────────────────────────────────┼─────────────┤
  │ W_p                  │ request's own prompt length −      │ known at    │
  │                      │ prefix-cache-hit estimate          │ arrival     │
  ├──────────────────────┼────────────────────────────────────┼─────────────┤
  │ q_p, q_d (backlogs)  │ num_requests_* × demand, from      │ stale       │
  │                      │ scrape                             │             │
  ├──────────────────────┼────────────────────────────────────┼─────────────┤
  │ TTFT_x, ITL_x        │ predicted from μ,α,δ + current     │ stale +     │
  │ (counterfactuals)    │ queues                             │ model error │
  └──────────────────────┴────────────────────────────────────┴─────────────┘

  The key observation: the SLO-pressure half of the rule (z terms) is 
  router-local and exact — no staleness at all. Only the load-balancing half
  (q_p, q_d) and the counterfactual predictors depend on stale server metrics.
  So under staleness the algorithm keeps tracking SLOs faithfully (the z
  feedback is ground-truth) and degrades only in load-balancing precision —
  which is the benign failure mode. And the z queues self-correct any systematic
  bias in the crude counterfactual predictors. The thing that's hardest to get
  right (predictions) is also the thing the feedback loop forgives.

  Where to point next

  The real-world agenda has four items: (1) stale-state / herd effects
  (optimistic local queue accounting), (2) online α, δ estimation from the (ITL,
  B, KV) stream, (3) unknown N_out, (4) where the controller physically lives
  and its decision-latency budget.

  My instinct: start with (4) — pin the controller's ownership boundary (what
  state it holds vs scrapes) and its per-decision latency budget — because that
  constrains everything else (it decides how crude the predictors must be). Do
  you want to start there, or is one of the others more pressing for your setup?

✻ Churned for 2m 17s

❯ The controller is llm-d EPP (always).                                         

⏺ Good — that pins the hot constraints, because the EPP is an Envoy ext_proc
  server in the synchronous request path. That dictates the whole shape of the
  implementation:

  What the EPP constrains

  - Hot path is sub-ms. Envoy blocks on the picker before forwarding. So the
  per-request decision must be O(#candidate pods) of cached arithmetic — no 
  regression, no scrape, no blocking I/O at decision time.
  - State is a periodically-scraped metrics cache (per-pod
  num_requests_running/waiting, kv_cache_usage_perc, the latency histograms) —
  stale by the scrape interval.
  - It already has a composable scorer/filter framework (the one BLIS mirrors:
  queue-depth, kv-utilization, precise-prefix-cache, …) and, in PD mode,
  separate prefill- and decode-scheduling profiles. The P-vs-D choice is a gate
  in front of / fused with those.
  - It sees the request body → prompt length, and via its prefix index →
  uncached tokens = W_p. Known, fresh, per-request.

  The reconciliation: three execution contexts

  The algorithm doesn't run "as one thing" — it splits across three clocks:

  ┌───────────────┬────────────────────────┬─────────────┬──────────────────┐
  │    Context    │          Runs          │  Frequency  │     Produces     │
  ├───────────────┼────────────────────────┼─────────────┼──────────────────┤
  │ Hot path (per │ evaluate the DPP       │ every       │ the P-vs-D       │
  │  request)     │ inequality with cached │ request, µs │ decision + pod   │
  │               │  numbers               │             │                  │
  ├───────────────┼────────────────────────┼─────────────┼──────────────────┤
  │               │ scrape /metrics;       │             │                  │
  │ Background    │ rolling (ITL, B)       │ scrape      │ cached           │
  │ loop          │ regression → α, δ, μ;  │ interval    │ coefficients     │
  │               │ refresh q_p, q_d       │             │                  │
  ├───────────────┼────────────────────────┼─────────────┼──────────────────┤
  │ Completion    │ realized TTFT/ITL →    │ per         │ the SLO feedback │
  │ observation   │ bump z_ttft, z_itl     │ completion  │  (controller     │
  │               │                        │             │ state)           │
  └───────────────┴────────────────────────┴─────────────┴──────────────────┘

  Only the hot-path arithmetic is latency-critical, and it's trivial. The
  expensive parts (regression, scraping) live in the background loop. This is
  exactly why DPP fits the EPP: the feedback and estimation are off the hot 
  path; the decision is a comparison.

  The reframe that makes this not a from-scratch router

  Your existing EPP scorers already compute most of the DPP terms — under other
  names:

  - queue-depth scorer ≈ the q_p, q_d backlog terms
  - kv-utilization scorer ≈ the capacity/α-state signal (drives μ(N))
  - precise-prefix-cache scorer ≈ the uncached-token count = W_p

  So DPP is not a new scorer — it's a principled rule for combining them, where 
  the combination weights are the virtual-queue states z_ttft, z_itl instead of 
  hand-tuned constants. The static weighted profile
  (precise-prefix-cache:2,queue-depth:1,kv-utilization:1) becomes a profile
  whose weights move under SLO feedback. That's the entire novelty over today's
  EPP, and it's a small, legible delta on the existing framework — not a
  rewrite.

  The one question that gates the rest

  The z queues are the heart (the only ground-truth feedback), so how the EPP
  learns realized TTFT/ITL decides their freshness and attribution:

  - (A) Response-stream observation: EPP is on the data path, so it can
  timestamp first-token (TTFT) and inter-token gaps (ITL) per request and
  attribute them to the placement it chose. Fresh, per-decision, exact
  attribution — but adds response-path work and assumes EPP processes the
  streaming body.
  - (B) Scraped vLLM histograms: read time_to_first_token_seconds /
  inter_token_latency_seconds per pod periodically. Cheap, no data-path cost —
  but aggregate (no per-request attribution to P vs D) and stale.

  A gives you a clean per-decision learning signal (and lets you later attribute
  SLO misses to the routing choice); B is cheaper but blurs exactly the signal
  that tells you whether disaggregation helped. Which does your EPP actually do
  today — is it processing response bodies, or metrics-only?

✻ Cooked for 1m 50s

❯ Okay, agree with A. Can we derive from the beginning the algorithm (math-wise 
  and implementation wise) along with the unit of measurement to paint a        
  cohesive story?                                                               

⏺ This is the synthesis. I'll build it from scratch so every symbol has a unit
  and a source, and the math and the EPP implementation are the same story told
  twice.

  ---
  1. Setup

  One remote prefill server, one decode server. Each arriving request gets one
  decision:

  - x = P — prefill remote, then KV-transfer to decode server, then decode.
  - x = D — prefill and decode together on the decode server (aggregated).

  Objective: minimize time-average transfer cost, subject to TTFT-SLO and
  ITL-SLO holding on time-average, with both servers' backlogs stable. (Penalty
  = (D)-flavored; SLOs = (C).)

  2. Units master table (the contract)

  ┌─────────────┬──────────────────┬──────────────┬───────────────────────┐
  │   Symbol    │     Meaning      │     Unit     │     Source in EPP     │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │ a_p         │ uncached prompt  │ tokens       │ request body − prefix │
  │             │ tokens           │              │  index (fresh)        │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │             │ per-iteration    │              │ regression intercept  │
  │ α           │ fixed cost       │ s            │ (bg loop)             │
  │             │ (weight read)    │              │                       │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │             │ per-request      │              │ regression slope (bg  │
  │ δ(L)        │ marginal         │ s            │ loop)                 │
  │             │ iteration cost   │              │                       │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │             │ prefill          │              │ a_p · per-token       │
  │ W_p         │ marginal-work of │ s            │ compute (fresh)       │
  │             │  this request    │              │                       │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │             │ decode           │              │ N_out · δ̄ — unknown   │
  │ W_d         │ marginal-work of │ s            │ at decision           │
  │             │  this request    │              │                       │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │ T_iter      │ iteration time   │ s            │ ≈ inter_token_latency │
  │             │                  │              │  (scrape)             │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │ μ           │ drain rate = 1 − │ dimensionles │ derived from α,       │
  │             │  α/T_iter        │ s ∈(0,1]     │ scraped ITL           │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │ Q_p,Q_d     │ server backlogs  │ s            │ scrape counts ×       │
  │             │ (marginal-work)  │              │ demand + optimistic   │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │ W*          │ normalizer =     │ s            │ constant (config + α) │
  │             │ μ^nom·τ_ttft     │              │                       │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │ Z_ttft,Z_it │ virtual queues   │              │ owned, bumped on      │
  │ l           │ (accumulated     │ s            │ completion (resp.     │
  │             │ violation)       │              │ stream)               │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │ c_xfer      │ KV-transfer cost │ s            │ nixl_xfer_time_second │
  │             │                  │              │ s (scrape)            │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │ τ_ttft,τ_it │ SLO targets      │ s            │ config                │
  │ l           │                  │              │                       │
  ├─────────────┼──────────────────┼──────────────┼───────────────────────┤
  │ V           │ penalty/stabilit │ dimensionles │ config                │
  │             │ y knob           │ s            │                       │
  └─────────────┴──────────────────┴──────────────┴───────────────────────┘

  Currency = marginal-work-seconds: work is Σδ; α is the server's per-iteration
  tax, owned by no request.

  3. The primitive → the rate

  The one physical law, fit from (num_requests_running, inter_token_latency):

  T_iter(B) = α + Σ_i δ_i      ⇒    μ(B) = Σδ / (α + Σδ) = 1 − α/T_iter

  μ is the fraction of wall-time spent on useful work. α=0 ⇒ μ=1 (textbook
  conservative queue); α>0 is the sole source of non-conservation. δ/α ≫ 1 for
  prefill (≈conservative), ≪ 1 for decode (state-dependent). Nominal rates:

  μ_d^nom = 1 − α/τ_itl     (efficiency at the ITL-SLO-critical batch B* =
  (τ_itl−α)/δ̄)
  μ_p^nom = 1 − α/T_iter^nom  (measured; ≈0.77 in our example, not 1)

  4. The four queues, normalized to pure numbers

  q_p = Q_p/W*_p     q_d = Q_d/W*_d        (W* = μ^nom·τ_ttft)   "backlog in
  SLO-windows"
  z_ttft = Z_ttft/τ_ttft     z_itl = Z_itl/τ_itl                  "accumulated
  fractional violation"

  Physical q = stability role (P-term, instantaneous). Virtual z = SLO role
  (I-term, integrated). Normalizing by τ and by μ^nom·τ makes all four
  dimensionless and commensurate — caveat-1 closed.

  5. The decision rule (DPP greedy minimization)

  Increments per choice: prefill W_p goes to Q_p (if P) or Q_d (if D); decode
  W_d goes to Q_d either way. Minimizing V·(c_xfer/τ_ttft) + Σ qᵢ·Δqᵢ and
  cancelling common terms:

              ┌─ backlog balancing ─┐     ┌── transfer ──┐  ┌──────── SLO
  pressure ────────┐
   choose P ⟺  q_d·(W_p/W*_d) − q_p·(W_p/W*_p)  >  V·(c_xfer/τ_ttft)
                                                     +
  z_ttft·(TTFT_P−TTFT_D)/τ_ttft
                                                     + z_itl ·(ITL_P −ITL_D
  )/τ_itl

  Everything dimensionless. Two facts that make this implementable, not just 
  elegant:

  - W_d cancelled. The decision is independent of the unknown N_out — decode
  work lands on the decode server regardless of P/D, so only W_p (known) enters.
  A5 is not needed for the per-request decision. N_out uncertainty only blurs
  the slow Q_d accounting, never the hot-path choice.
  - z·(ITL_P − ITL_D) is the disaggregation payoff, signed. ITL_P < ITL_D ⇒
  negative ⇒ pushes toward P, harder the more the ITL-SLO is being missed (z_itl
  large). The reason PD-disagg exists is one term that self-activates.

  6. The counterfactual predictors (crude is fine — z corrects bias)

  TTFT_P ≈ (Q_p + W_p)/μ_p + c_xfer            ITL_P ≈ current decode ITL
  (protected)
  TTFT_D ≈ (Q_d^pf + W_p)/μ_p^decode           ITL_D ≈ ITL_P + δ_pf-chunk
  (prefill disrupts batch)

  All inputs are scraped/owned. Predictor error is absorbed by the virtual
  queues over time.

  7. Controller dynamics (the only learned state, updated off hot-path)

  On each completion k, from the response stream (option A):

  Z_ttft ← max(Z_ttft + (TTFT_k − τ_ttft), 0)
  Z_itl  ← max(Z_itl  + (mean_ITL_k − τ_itl), 0)

  Q_p, Q_d from scrape + optimistic in-flight increments. α, δ from rolling
  (ITL,B) regression.

  8. The same story, as EPP code paths

  ┌──────────────┬──────────────┬───────────────────────────────────────────┐
  │   Context    │    Clock     │                   Does                    │
  ├──────────────┼──────────────┼───────────────────────────────────────────┤
  │ Hot path     │ per request, │ compute W_p; evaluate §5 inequality with  │
  │              │  µs          │ cached q,z,μ,c_xfer; emit P/D + pod       │
  ├──────────────┼──────────────┼───────────────────────────────────────────┤
  │ Background   │ scrape       │ refresh q_p,q_d; regress α,δ,μ; refresh   │
  │              │ interval     │ c_xfer                                    │
  ├──────────────┼──────────────┼───────────────────────────────────────────┤
  │ Response     │ per          │ measure TTFT / mean-ITL; update           │
  │ stream       │ completion   │ z_ttft,z_itl                              │
  └──────────────┴──────────────┴───────────────────────────────────────────┘

  DPP = the existing scorers (queue-depth≈q, kv-utilization≈μ-state,
  precise-prefix-cache≈W_p) recombined with weights that are the virtual-queue 
  states instead of static constants. Small delta on today's EPP.

  9. Honest residuals

  - A7 (prefill-on-decode as pure ITL inflation) lives entirely in ITL_D and
  Q_d^pf; if prefill bursts need their own backlog, this is where it breaks.
  - Optimistic accounting is the only defense against herd effects under stale
  q; the z half is immune (it's ground-truth).
  - δ(L) context-dependence: the regression must key on KV-tokens-in-batch
  (kv_cache_usage_perc × capacity), not just B.

  ---
  That's the cohesive derivation — one currency (marginal-work-seconds), one
  primitive (T_iter = α + Σδ), one rule, mapped onto three EPP clocks.

  Two things before I write this to a design doc: (1) the single free knob is V
  (transfer-cost vs SLO-pressure) — do you want it fixed, or adapted? And (2)
  shall I commit this to
  docs/superpowers/specs/2026-06-18-edpp-dpp-routing-design.md as the foundation
  we build the plan on?

✻ Brewed for 2m 37s

❯ Can you explain:                                                              
  μ_d^nom = 1 − α/τ_itl     (efficiency at the ITL-SLO-critical batch B* =      
  (τ_itl−α)/δ̄)                                                                  
  μ_p^nom = 1 − α/T_iter^nom  (measured; ≈0.77 in our example, not 1)           
                                                                                
  and why is this needed? Why can't we simply use \mu(B) from each server?      

⏺ Two separate things are going on here, and the key is that μ^nom is a fixed 
  yardstick, while μ(B) is a live measurement — they have different jobs. We do
  use live μ(B); just not in the normalizer. Let me explain both, then why the
  normalizer must be the fixed one.

  What μ^nom is

  Both come from the same primitive μ = 1 − α/T_iter, evaluated at a chosen
  operating point:

  - Decode: the natural operating point is where the server is just meeting its 
  ITL SLO — i.e. T_iter = τ_itl (since ITL ≈ T_iter). Substitute:
  μ_d^nom = 1 − α/τ_itl. The batch that sits there is B* = (τ_itl − α)/δ̄ (solve
  α + B·δ̄ = τ_itl). It's a closed form because the SLO pins T_iter.
  - Prefill: there's no inter-token SLO pinning a critical batch, so there's no
  closed form — you just measure the typical prefill iteration time T_iter^nom
  and read off μ_p^nom = 1 − α/T_iter^nom ≈ 0.77.

  So μ^nom is "the rate at the load we intend to run at," a single constant per
  server.

  Why not just use live μ(B) in W*?

  Recall W* = μ^nom·τ_ttft is the normalization constant for the physical queue:
  q = Q/W*. Three reasons it must be constant:

  1. Lyapunov theory requires fixed weights. The drift-plus-penalty guarantee
  comes from telescoping per-slot drifts of a fixed potential L = ½ Σ 
  Q_i²/W*_i². If W* moves with state, L's definition changes every slot, the
  telescoping collapses, and the O(1/V)/O(V) bound evaporates. The normalizer is
  a Lyapunov weight — weights are constants, not signals. This alone is
  decisive.

  2. Live μ would invert the congestion signal. μ(B) rises with load (batching
  amortizes α). So a live-μ normalizer gets bigger exactly when the server gets
  busier — hiding congestion when it matters most. Concretely, hold backlog
  fixed at Q_d = 200 ms:

  ┌───────┬────────┬────────┬────────┬──────────┐
  │ Load  │ T_iter │ live μ │ W*=μ·τ │ q_d=Q/W* │
  ├───────┼────────┼────────┼────────┼──────────┤
  │ B=20  │ 28 ms  │ 0.70   │ 350 ms │ 0.57     │
  ├───────┼────────┼────────┼────────┼──────────┤
  │ B=140 │ 148 ms │ 0.94   │ 470 ms │ 0.43     │
  └───────┴────────┴────────┴────────┴──────────┘

  Same backlog, but live-μ says q_d dropped as the server got more loaded — the
  signal moves the wrong way. With fixed μ_d^nom=0.94, q_d reads 0.43
  consistently, and only real backlog changes move it.

  3. We do use live μ(B) — just in its proper role. It's the supply estimate, so
  it belongs in the predictors (TTFT_P ≈ (Q_p+W_p)/μ_p, the drain accounting in
  the queue updates). That's where you want the true, current rate. The
  normalizer's job is different: set the fixed exchange rate between a unit of
  backlog and a unit of SLO pressure. Two roles, two quantities.

  The reassurance

  Because W* only sets a relative weighting (later re-absorbed by tuning V), its
  exact value isn't critical — "fixed" matters far more than "exactly right." A
  slightly-wrong but constant W* just nudges the backlog-vs-SLO trade; a moving
  W* breaks the theorem and inverts the load signal. That's why we pin it at a
  principled, computable anchor (the SLO-critical point) and freeze it.

  Want me to fold this explanation into the design doc when I write it?

✻ Churned for 1m 2s

❯ why did we decide Recall W* = μ^nom·τ_ttft is the normalization constant for  
  the physical queue: q = Q/W* ? And what's the reason behind the definition    
  W* = μ^nom·τ_ttft?                                                            
⏺ Two distinct "why"s here — why normalize the physical queue at all, and why
  this formula. Let me separate them, because the first is forced and the second
  is a (principled) choice.

  Why normalize Q at all — this part is forced

  Back to caveat 1: the four queues have different units, so the decision rule's
  sum Σ_i q_i·Δq_i is incoherent unless we attach weights — L = ½ Σ_i w_i Q_i².
  Writing q = Q/W* is exactly choosing that weight: w = 1/W*². You cannot skip
  it; heterogeneous units force a weight choice, and the normalizer is just how
  that choice is parameterized. So "normalize the physical queue" isn't optional
  — it's "pick its Lyapunov weight," restated.

  Why W* = μ^nom · τ_ttft specifically — this is the choice, and here's the 
  chain

  The desideratum: make q SLO-relative, so it's commensurate with the z terms
  (which are already fractions of an SLO budget). Then every term in the
  decision rule reads in one consistent currency — "fraction of SLO budget" —
  and the comparison is apples-to-apples.

  So ask: what does a work-backlog Q mean for the SLO? Two bridges:

  backlog Q  ──(÷μ)──►  induced queueing delay  Q/μ   [s]      (a backlog drains
  at rate μ)
  Q/μ        ──(÷τ)──►  fraction of TTFT budget  (Q/μ)/τ_ttft  (delay relative
  to its SLO)

  Combine:

  q  =  (Q/μ) / τ_ttft  =  Q / (μ · τ_ttft)   ⇒   W* = μ · τ_ttft

  So W* is the backlog whose induced queueing delay exactly equals one TTFT 
  window, and q = Q/W* reads as "predicted queueing delay ÷ TTFT SLO." q = 1 ⇔
  the backlog alone will blow the TTFT budget. That's the whole reason for the
  formula — it's the work→delay→SLO conversion, nothing more.

  Three details fall out:

  - Why μ appears: it's the only bridge from work-backlog to delay (delay = 
  work/rate). It must be the fixed nominal μ^nom, for the Lyapunov reasons from
  the last answer.
  - Why τ_ttft, not τ_itl: a backlog manifests as queueing delay — waiting
  before service — which is a TTFT phenomenon. ITL is a rate during decode, not
  a wait. So the backlog→SLO chain uses the TTFT target. (Q_d's ongoing-decode
  portion is the known mixture approximation; for the queueing part this is
  exact.)
  - The bonus this buys: with this scaling, q is literally "instantaneous
  predicted SLO-pressure" and z is "accumulated realized violation" — the P-term
  and I-term of the controller. The normalization choice is what gives them
  that clean proportional/integral reading.

  Honest caveat

  This is a coherent choice, not the only one. The rejected alternative
  (capacity-relative W*) would make q read "fraction of hardware capacity"
  instead of "fraction of latency budget" — defensible, but then q wouldn't be
  commensurate with z, and you'd lose the one-currency property. We picked
  μ^nom·τ_ttft precisely because it puts backlog and SLO violations on the same
  SLO-budget axis. That alignment is the entire justification — and V is still
  there to re-tune the relative emphasis afterward.

  Should I write the design doc now with these derivations (the μ^nom rationale
  and this W* chain) included as first-class explanations?

✻ Crunched for 55s