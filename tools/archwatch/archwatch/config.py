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

    # S1 - scale
    min_total_params: int = 30_000_000_000  # 30B total

    # S2 - org track record (used when org is not in FRONTIER_ORGS)
    min_org_top_downloads: int = 100_000

    # S4 - late-blooming popularity
    min_model_downloads: int = 10_000
    min_model_likes: int = 200


@dataclass
class DetectorConfig:
    window_days: int = 7
    max_issues_per_run: int = 5
    thresholds: Thresholds = field(default_factory=Thresholds)
    frontier_orgs: set[str] = field(default_factory=lambda: set(FRONTIER_ORGS))

    # Cap on how many HF configs to fetch per poll (rate-limit guard).
    max_hf_config_fetches: int = 200


DEFAULTS = DetectorConfig()
