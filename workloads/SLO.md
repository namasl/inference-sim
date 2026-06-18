# Workload SLO Recommendations

## Reading-speed derivation (chat-tier ITL)

The 50 ms ITL target for human-streamed output comes from the following:

- Average adult reading speed is 220–350 WPM ([scholarwithin.com](https://scholarwithin.com/average-reading-speed#average-reading-speed-by-age-and-grade)).
- Take 400 WPM as a fast reader (high end of normal).
- Apply a 5× skim multiplier (the model should out-pace the reader, not match them) → 2000 WPM.
- Convert: 2000 words/min × 1.3 tokens/word ≈ 2600 tokens/min ≈ 23 ms/token.
- Round to **50 ms** as the p99 ceiling.

This is consistent with MLPerf v5.1's Interactive tier of 30–40 ms p99 TPOT.

## Reference Anchors

| Anchor | Value | Source |
|---|---|---|
| Average adult reading | 220–350 WPM | [scholarwithin.com](https://scholarwithin.com/average-reading-speed#average-reading-speed-by-age-and-grade) |
| "Faster than reading" | 100 ms TPOT ≈ 450 WPM | [Databricks LLM Inference blog](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) |
| MLPerf v5.1 Interactive (Llama 8B/70B) | TTFT ≤ 500 ms p99, TPOT ≤ 30–40 ms p99 | [MLCommons v5.0 announcement](https://mlcommons.org/2025/04/llm-inference-v5/) |
| MLPerf v5.0/5.1 Server (Llama 8B/70B) | TTFT ≤ 2 s p99, TPOT ≤ 100–200 ms p99 | [MLCommons v5.0 announcement](https://mlcommons.org/2025/04/llm-inference-v5/) |
| MLPerf v5.0 Llama 405B Server | TTFT ≤ 6 s p99, TPOT ≤ 175 ms p99 | [MLCommons v5.0 announcement](https://mlcommons.org/2025/04/llm-inference-v5/) |
| Chatbot "feels responsive" | TTFT < 500 ms; premium < 200 ms | [Databricks LLM Inference blog](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices), [AWS Bedrock latency-optimized](https://aws.amazon.com/blogs/machine-learning/optimizing-ai-responsiveness-a-practical-guide-to-amazon-bedrock-latency-optimized-inference/) |
| Batch / once-daily report | E2E ~30 s acceptable | [Databricks LLM Inference blog](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) |
| SLOs-Serve Tight / Loose | 50 ms / 100 ms TPOT | [arXiv 2504.08784](https://arxiv.org/abs/2504.08784) |

## Per-Workload SLOs

### interactive-chat

| Metric | p99 | Source |
|---|---|---|
| TTFT | 500 ms | [MLPerf v5.1 Interactive](https://mlcommons.org/2025/04/llm-inference-v5/) p99 ≤ 500 ms; matches [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) and [Bedrock](https://aws.amazon.com/blogs/machine-learning/optimizing-ai-responsiveness-a-practical-guide-to-amazon-bedrock-latency-optimized-inference/) "feels responsive" |
| ITL | 50 ms | Reading-speed derivation; [MLPerf Interactive](https://mlcommons.org/2025/04/llm-inference-v5/) |

### code-generation

| Metric | p99 | Source |
|---|---|---|
| TTFT | 30 s | [MLPerf Server](https://mlcommons.org/2025/04/llm-inference-v5/) baseline (2 s) extended ~15× because the lognormal ISL distribution has p99 near the 990 k cap; prefill is compute-bound and dominates first-token latency at this scale |
| ITL | 100 ms | [MLPerf Server tier](https://mlcommons.org/2025/04/llm-inference-v5/) (100 ms TPOT). Reading-speed does not apply: code is reviewed in blocks, not streamed prose, so the chat-tier 50 ms is over-spec'd |

### deep-research

| Metric | p99 | Source |
|---|---|---|
| TTFT | 10 s | [MLPerf Server](https://mlcommons.org/2025/04/llm-inference-v5/) baseline (2 s) extended ~5× because exponential ISL is capped at 150 k as the loop accumulates context |
| ITL | 100 ms | [MLPerf Server tier](https://mlcommons.org/2025/04/llm-inference-v5/) (100 ms TPOT). Reading-speed does not apply for an agentic loop where most turns are machine-consumed |

### reasoning

| Metric | p99 | Source |
|---|---|---|
| TTFT | 2 s | [MLPerf Server](https://mlcommons.org/2025/04/llm-inference-v5/) p99 ≤ 2 s; ISL mean 1 k makes prefill fast |
| ITL | 100 ms | [MLPerf Server tier](https://mlcommons.org/2025/04/llm-inference-v5/) (100 ms TPOT). Thinking phase is not user-visible (only matters for total wall-clock); final-answer reading-speed is dominated by total OSL, not per-token rate |

### batch-summarization-rag

| Metric | p99 | Source |
|---|---|---|
| TTFT | 15 s | [MLPerf Server](https://mlcommons.org/2025/04/llm-inference-v5/) tier (2 s) extended ~7× because ISL is bimodal up to 80 k (full-document reads dominate prefill) and the user expects a "summarizing…" wait |
| ITL | 100 ms | Output is structured/JSON, parsed by code rather than read live, so [MLPerf Server tier](https://mlcommons.org/2025/04/llm-inference-v5/) (100–200 ms) applies instead of reading-speed |
| E2E | 6 min | TTFT (15 s) + p99 OSL (2 k) × ITL (100 ms) ≈ 215 s minimum; doubled to 6 min for queueing and prefill jitter on loaded systems. [Databricks batch anchor](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (~30 s once-daily report) is the lower bound for short-doc cases, but the bimodal long-doc tail dominates p99 |

### batch-synthetic-data-generation

| Metric | Target | Source |
|---|---|---|
| Goodput | tokens/s/GPU (or tokens/s/$) | [MLPerf Offline](https://mlcommons.org/2025/04/llm-inference-v5/) treats this scenario as throughput-only; latency targets here actively hurt by preventing high-batch regimes |
| E2E p99 | 30 min | Runaway backstop only. OSL up to 8 k × 100 ms ITL ≈ 13 min worst-case generation; 30 min provides ~2× margin for queueing and prefill on long inputs without biting under normal load |

## Final Summary

| Workload | TTFT p99 | ITL p99 | E2E p99 |
|---|---|---|---|
| interactive-chat | 500 ms | 50 ms | — |
| code-generation | 30 s | 100 ms | — |
| deep-research | 10 s | 100 ms | — |
| reasoning | 2 s | 100 ms | — |
| batch-summarization-rag | 15 s | 100 ms | 6 min |
| batch-synthetic-data | — | — | 30 min |
