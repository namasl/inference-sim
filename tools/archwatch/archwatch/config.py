"""Shared, tunable configuration. Frozen interface — implementations read it, not redefine it.

Every threshold here is a PLACEHOLDER until the backtest calibrates it. Treat these
numbers as inputs to be measured, not as decisions already made.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Orgs whose next release is significant on track record alone (S2/T3).
FRONTIER_ORGS: set[str] = {
    "moonshotai", "deepseek-ai", "qwen", "zai-org", "thudm", "minimaxai",
    "meta-llama", "mistralai", "google", "microsoft", "nvidia", "openai",
    "ai21labs", "baidu", "tencent", "bytedance-seed", "xiaomimimo",
    "inclusionai", "allenai", "ibm-granite", "stepfun-ai", "internlm",
}

# Repo-name substrings that mark a derivative rather than a new architecture.
DERIVATIVE_PATTERNS: tuple[str, ...] = (
    "gguf", "awq", "gptq", "-bnb-", "bnb-4bit", "int4", "int8", "w4a16",
    "w8a8", "-fp8", "quantized", "-merge", "merged", "lora", "adapter",
    "-distill-", "abliterated", "uncensored", "-exl2", "mlx-", "-onnx",
    "-openvino", "-trtllm", "smashed", "-dpo", "-sft-",
    # Added after live measurement showed these slipping through (component C):
    # bare "-4bit"/"-8bit" suffixes and a "-MLX" suffix (only the "mlx-" prefix
    # was covered), plus one abliteration brand.
    "-4bit", "-8bit", "-mlx", "heretic",
    # Added after the backtest found these drifting out of sync with the
    # connectors' own quant-suffix lists:
    "nvfp4", "mxfp4", "-fp4", "aqlm",
)

# Deliberately NOT suppressed: "-mtp". Multi-token prediction is one of the
# mechanisms BLIS does not model, so an MTP variant is signal we want to see,
# not packaging noise.

# Config keys that are never architecture-relevant; excluded from the
# "fields BLIS does not parse" diff so T1 does not fire on boilerplate.
IGNORED_CONFIG_KEYS: set[str] = {
    "_name_or_path", "transformers_version", "architectures", "model_type",
    "torch_dtype", "dtype", "bos_token_id", "eos_token_id", "pad_token_id",
    "unk_token_id", "sep_token_id", "decoder_start_token_id", "auto_map",
    "tokenizer_class", "use_cache", "return_dict", "output_attentions",
    "output_hidden_states", "id2label", "label2id", "problem_type",
    "task_specific_params", "finetuning_task", "prefix", "chunk_size_feed_forward",
    "_attn_implementation_autoset", "transformers_weights",
}


@dataclass
class Thresholds:
    """Significance gate thresholds (S1-S4). Placeholders; calibrate by backtest."""

    # S1 - scale. Calibrated by the wave-6 backtest, not chosen.
    #
    # Measured: recall against the frontier target list is IDENTICAL from 1B to
    # 400B, because every target also satisfies S2 by org — so this is a volume
    # knob, not a recall knob, and it flattens above ~15B. The deciding evidence
    # is at the other end: at >=7B it SUPPRESSES the only silently_wrong finding
    # in 1,446 candidates/day (a 4.02B Qwen3-Next variant simulating a sparse MoE
    # as dense). A threshold that filters out the single finding that justifies
    # the pipeline is the wrong threshold, however reasonable "industry scale"
    # sounded when I picked 30B.
    min_total_params: int = 3_000_000_000  # 3B total

    # S2 - org track record (used when org is not in FRONTIER_ORGS)
    min_org_top_downloads: int = 100_000

    # S4 - late-blooming popularity
    min_model_downloads: int = 10_000
    min_model_likes: int = 200


@dataclass
class DetectorConfig:
    window_days: int = 7
    # Raised from 5 on backtest evidence: at recheck=True the genuine frontier
    # architectures number more than five per day, and a cap of 5 discards real
    # findings. Noise at the cap was 20% while uncapped noise was 70% — the
    # ranking does the work, so a slightly wider cap is cheap.
    max_issues_per_run: int = 10
    thresholds: Thresholds = field(default_factory=Thresholds)
    frontier_orgs: set[str] = field(default_factory=lambda: set(FRONTIER_ORGS))

    # Cap on how many HF configs to fetch per poll (rate-limit guard).
    max_hf_config_fetches: int = 200

    # Shared GitHub REST budget per poll, across the framework and InferenceX
    # connectors. Core is ~5000 req/hr but /search/issues is only 30 req/min, so
    # the search-based paths are the real constraint. Measured live: ~60 calls for
    # a 7-day framework poll, ~35 for InferenceX, ~105 for a 30-day backtest window.
    max_github_requests: int = 400

    # Closes the filter's largest recall hole, at a cost in precision.
    #
    # Normally a candidate whose architecture is already known is suppressed
    # before any config analysis. But a point release can add config fields under
    # an UNCHANGED architecture string — and BLIS silently drops fields it does
    # not parse. That is precisely the "silent wrong numbers" failure this whole
    # system exists to catch, arriving disguised as a known architecture, where
    # archwatch never looks.
    #
    # When True, known architectures are still re-checked for T1 (unparsed
    # config fields) instead of being dropped outright.
    #
    # Measured by the backtest and turned ON. At False, frontier recall was 0/9
    # and the five survivors contained ZERO genuine frontier architectures. At
    # True it is 9/9, and the 5.4x extra volume (5 -> 27/day) is better volume:
    # 17 genuine frontier architectures, 3 minor, 7 noise. Seeded controls
    # (Qwen3-14B, Llama-3.1-70B, Mixtral-8x7B) still drop as
    # known_architecture_nothing_new, so the recheck discriminates rather than
    # simply readmitting everything.
    recheck_known_architectures: bool = True


DEFAULTS = DetectorConfig()
