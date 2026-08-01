"""
Configuration Management System

Centralized configuration loading with Pydantic validation, environment variable
override support, and type-safe config objects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# TRAINING CONFIGURATION MODELS
# =============================================================================


class TrainerConfig(BaseModel):
    """Hugging Face Trainer / SFTTrainer configuration."""

    output_dir: str = "./checkpoints"
    overwrite_output_dir: bool = True
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_safetensors: bool = True
    save_only_model: bool = False

    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"use_reentrant": False}
    )

    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    optim_args: dict[str, Any] | None = None

    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    lr_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)

    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    half_precision_backend: str = "auto"

    logging_strategy: str = "steps"
    logging_steps: int = 10
    logging_first_step: bool = True
    logging_nan_inf_filter: bool = True
    logging_dir: str = "./logs/tensorboard"
    log_level: str = "info"
    log_level_replica: str = "warning"
    log_on_each_node: bool = False
    disable_tqdm: bool = False
    report_to: list[str] = Field(default_factory=lambda: ["tensorboard", "wandb"])
    run_name: str = "llm-finetuning-qlora"

    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    eval_delay: int = 0
    eval_accumulation_steps: int = 1
    prediction_loss_only: bool = False

    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = False
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    remove_unused_columns: bool = False
    label_names: list[str] = Field(default_factory=lambda: ["labels"])
    data_seed: int = 42

    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 25
    ddp_timeout: int = 1800
    dataloader_shuffle: bool = True

    seed: int = 42
    full_determinism: bool = False

    torch_compile: bool = False
    torch_compile_mode: str = "max-autotune"
    torch_compile_backend: str = "inductor"
    auto_find_batch_size: bool = False

    push_to_hub: bool = False
    hub_model_id: str = ""
    hub_strategy: str = "every_save"
    hub_token: str = ""
    hub_private_repo: bool = False
    hub_always_push: bool = False

    resume_from_checkpoint: str | None = None
    ignore_data_skip: bool = False

    fsdp: str = ""
    fsdp_config: dict[str, Any] | None = None
    deepspeed: str | dict[str, Any] | None = None

    group_by_length: bool = False
    length_column_name: str = "length"
    include_inputs_for_metrics: bool = False
    include_for_metrics: list[str] = Field(default_factory=list)
    eval_do_concat_batches: bool = True
    skip_memory_metrics: bool = False
    use_legacy_prediction_loop: bool = False
    push_to_hub_model_id: str = ""
    push_to_hub_organization: str = ""
    mp_parameters: str = ""
    ray_scope: str = "last"


class LoRAConfig(BaseModel):
    """LoRA / PEFT configuration."""

    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_rslora: bool = True
    use_dora: bool = False
    init_lora_weights: bool | str = "gaussian"

    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False

    layers_to_transform: list[int] | None = None
    layers_pattern: str | None = None
    rank_pattern: dict[str, int] = Field(default_factory=dict)
    alpha_pattern: dict[str, int] = Field(default_factory=dict)

    megatron_config: Any | None = None
    megatron_core: bool = False

    lora_plus_scale: float | None = None
    lora_plus_lr_ratio: float | None = None

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v: str) -> str:
        allowed = ["none", "all", "lora_only"]
        if v not in allowed:
            raise ValueError(f"bias must be one of {allowed}")
        return v

    @field_validator("init_lora_weights")
    @classmethod
    def validate_init_weights(cls, v: bool | str) -> bool | str:
        if isinstance(v, str):
            allowed = ["gaussian", "loftq", "pissa", "olora", "true", "false"]
            if v not in allowed:
                raise ValueError(f"init_lora_weights must be one of {allowed}")
        return v


class QuantizationConfig(BaseModel):
    """BitsAndBytes quantization configuration."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: str = "uint8"

    load_in_8bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False
    llm_int8_skip_modules: list[str] = Field(default_factory=list)
    llm_int8_enable_fp32_cpu_offload: bool = False

    quantization_config_class: str = "BitsAndBytesConfig"
    bnb_4bit_quant_storage_dtype: str = "uint8"

    @field_validator("bnb_4bit_quant_type")
    @classmethod
    def validate_quant_type(cls, v: str) -> str:
        if v not in ["fp4", "nf4"]:
            raise ValueError("bnb_4bit_quant_type must be 'fp4' or 'nf4'")
        return v

    @field_validator("bnb_4bit_compute_dtype")
    @classmethod
    def validate_compute_dtype(cls, v: str) -> str:
        if v not in ["float16", "bfloat16", "float32"]:
            raise ValueError("bnb_4bit_compute_dtype must be 'float16', 'bfloat16', or 'float32'")
        return v


class SFTConfig(BaseModel):
    """TRL SFTTrainer configuration."""

    max_seq_length: int = 2048
    packing: bool = False
    packing_fn: Any | None = None
    dataset_text_field: str = "text"
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)
    formatting_func: Any | None = None
    neftune_noise_alpha: float | None = None

    dataset_num_proc: int = 4
    dataset_batch_size: int = 1000
    remove_unused_columns: bool = True
    shuffle_buffer_size: int = 10000

    tokenizer_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "padding": "max_length",
            "truncation": True,
            "max_length": 2048,
            "return_tensors": "pt",
        }
    )


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    type: str = "adamw_torch"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
            "foreach": True,
            "fused": True,
        }
    )


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    type: str = "cosine"
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"num_warmup_steps": 0, "num_cycles": 0.5, "last_epoch": -1}
    )


class EarlyStoppingConfig(BaseModel):
    """Early stopping callback configuration."""

    enabled: bool = True
    patience: int = 3
    threshold: float = 0.001
    metric_for_best: str = "eval_loss"
    greater_is_better: bool = False


class CheckpointConfig(BaseModel):
    """Checkpoint callback configuration."""

    enabled: bool = True
    save_steps: int = 100
    save_total_limit: int = 3
    save_best: bool = True
    metric_for_best: str = "eval_loss"


class LoggingCallbackConfig(BaseModel):
    """Logging callback configuration."""

    enabled: bool = True
    log_steps: int = 10
    log_gpu_memory: bool = True
    log_learning_rate: bool = True
    log_grad_norm: bool = True


class ProfilerConfig(BaseModel):
    """Profiler callback configuration."""

    enabled: bool = False
    profile_steps: int = 10
    profile_dir: str = "./logs/profiler"
    activities: list[str] = Field(default_factory=lambda: ["cpu", "cuda"])
    record_shapes: bool = True
    with_stack: bool = True


class CallbacksConfig(BaseModel):
    """All callbacks configuration."""

    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingCallbackConfig = Field(default_factory=LoggingCallbackConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    custom_callbacks: list[Any] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    """Runtime configuration."""

    device: str = "auto"
    device_map: str = "auto"
    max_memory: dict[str, str] | None = None
    low_cpu_mem_usage: bool = True

    local_rank: int = -1
    world_size: int = 1
    ddp_backend: str = "nccl"

    empty_cache_steps: int = 50
    gc_collect_steps: int = 100

    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False

    torch_compile: bool = False
    torch_compile_mode: str = "max-autotune"
    flash_attention: bool = True
    flash_attention_version: int = 2


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""

    name: str = "llm-finetuning-qlora"
    project: str = "llm-finetuning"
    entity: str | None = None
    tags: list[str] = Field(default_factory=lambda: ["qlora", "nf4", "llama3"])
    notes: str = ""
    config_override: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)


# =============================================================================
# MODEL CONFIGURATION MODELS
# =============================================================================


class ModelArchitectureConfig(BaseModel):
    """Model architecture configuration."""

    model_type: str = "llama"
    architecture: str = "LlamaForCausalLM"
    config_class: str = "LlamaConfig"
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1.0e-5
    rope_theta: float = 500000.0
    rope_scaling: dict[str, Any] | None = None
    vocab_size: int = 128256
    pad_token_id: int | None = None
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    tie_word_embeddings: bool = False


class ModelConfig(BaseModel):
    """Base model configuration."""

    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_revision: str = "main"
    model_cache_dir: str | None = None
    trust_remote_code: bool = False
    use_auth_token: bool = True

    architecture: ModelArchitectureConfig | str = Field(default_factory=ModelArchitectureConfig)

    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_memory: dict[str, str] | None = None
    offload_folder: str = "./offload"
    offload_state_dict: bool = False
    offload_buffers: bool = False
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    variant: str | None = None

    attn_implementation: str = "flash_attention_2"
    use_flash_attention_2: bool = True
    flash_attention_version: int = 2
    _attn_implementation: str = "flash_attention_2"

    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"use_reentrant": False}
    )

    use_cache: bool = False

    quantization_config: Any | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = True

    @field_validator("architecture", mode="before")
    @classmethod
    def parse_architecture(cls, v: dict[str, Any] | str) -> ModelArchitectureConfig:
        if isinstance(v, str):
            return ModelArchitectureConfig(architecture=v)
        return v


class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""

    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str = "main"
    tokenizer_cache_dir: str | None = None
    use_auth_token: bool = True

    use_fast: bool = True
    padding_side: str = "right"
    truncation_side: str = "right"
    model_max_length: int = 4096

    pad_token: str | None = None
    eos_token: str | None = None
    bos_token: str | None = None
    unk_token: str | None = None
    sep_token: str | None = None
    cls_token: str | None = None
    mask_token: str | None = None

    additional_special_tokens: list[str] = Field(default_factory=list)

    chat_template: str | None = None
    chat_template_name: str = "llama3"

    tokenizer_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "clean_up_tokenization_spaces": True,
            "add_special_tokens": True,
            "return_token_type_ids": False,
        }
    )

    @field_validator("padding_side")
    @classmethod
    def validate_padding_side(cls, v: str) -> str:
        if v not in ["left", "right"]:
            raise ValueError("padding_side must be 'left' or 'right'")
        return v

    @field_validator("truncation_side")
    @classmethod
    def validate_truncation_side(cls, v: str) -> str:
        if v not in ["left", "right"]:
            raise ValueError("truncation_side must be 'left' or 'right'")
        return v


class PEFTLoraConfig(BaseModel):
    """PEFT LoRA configuration."""

    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_rslora: bool = True
    use_dora: bool = False
    init_lora_weights: bool | str = "gaussian"

    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False

    layers_to_transform: list[int] | None = None
    layers_pattern: str | None = None
    rank_pattern: dict[str, int] = Field(default_factory=dict)
    alpha_pattern: dict[str, int] = Field(default_factory=dict)

    megatron_config: Any | None = None
    megatron_core: bool = False

    lora_plus_scale: float | None = None
    lora_plus_lr_ratio: float | None = None


class AdaLoRAConfig(BaseModel):
    """AdaLoRA configuration."""

    target_r: int = 8
    init_r: int = 12
    tinit: int = 0
    tfinal: int = 0
    deltaT: int = 10
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5
    total_step: int | None = None
    rank_pattern: dict[str, int] = Field(default_factory=dict)
    alpha_pattern: dict[str, int] = Field(default_factory=dict)


class IA3Config(BaseModel):
    """IA3 configuration."""

    target_modules: list[str] = Field(default_factory=list)
    feedforward_modules: list[str] = Field(default_factory=list)
    fan_in_fan_out: bool = False
    init_ia3_weights: bool = True


class PEFTConfig(BaseModel):
    """PEFT configuration."""

    peft_type: str = "LORA"
    lora: PEFTLoraConfig = Field(default_factory=PEFTLoraConfig)
    adalora: AdaLoRAConfig = Field(default_factory=AdaLoRAConfig)
    ia3: IA3Config = Field(default_factory=IA3Config)

    @field_validator("peft_type")
    @classmethod
    def validate_peft_type(cls, v: str) -> str:
        allowed = ["LORA", "ADALORA", "IA3", "PROMPT_TUNING", "PREFIX_TUNING"]
        if v not in allowed:
            raise ValueError(f"peft_type must be one of {allowed}")
        return v


class QuantizationConfigModel(BaseModel):
    """Quantization configuration for model loading."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: str = "uint8"

    load_in_8bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False
    llm_int8_skip_modules: list[str] = Field(default_factory=list)
    llm_int8_enable_fp32_cpu_offload: bool = False

    quantization_config_class: str = "BitsAndBytesConfig"
    bnb_4bit_quant_storage_dtype: str = "uint8"


class ModelLoadingConfig(BaseModel):
    """Model loading options."""

    revision: str = "main"
    token: str | None = None
    use_auth_token: bool = True

    cache_dir: str | None = None
    force_download: bool = False
    resume_download: bool | None = None
    proxies: dict[str, str] | None = None
    local_files_only: bool = False

    offload_folder: str = "./offload"
    offload_state_dict: bool = False
    offload_buffers: bool = False

    max_memory: dict[str, str] | None = None
    low_cpu_mem_usage: bool = True

    use_safetensors: bool = True
    variant: str | None = None

    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

    attn_implementation: str = "flash_attention_2"
    use_flash_attention_2: bool = True

    gradient_checkpointing: bool = True

    load_in_8bit: bool = False
    load_in_4bit: bool = True

    peft_config: Any | None = None
    is_trainable: bool = True


class ModelSavingConfig(BaseModel):
    """Model saving configuration."""

    save_adapter: bool = True
    adapter_path: str = "./adapters/best"
    save_tokenizer: bool = True

    merge_and_unload: bool = False
    merged_model_path: str = "./artifacts/models/merged/v1.0.0"
    merged_dtype: str = "bfloat16"
    safe_serialization: bool = True
    max_shard_size: str = "5GB"

    push_to_hub: bool = False
    hub_model_id: str = ""
    hub_private_repo: bool = False
    hub_token: str | None = None
    commit_message: str = "Upload model"


class InferenceConfig(BaseModel):
    """Inference configuration."""

    max_new_tokens: int = 512
    min_new_tokens: int = 1
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    num_beams: int = 1
    num_return_sequences: int = 1

    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    diversity_penalty: float = 0.0
    penalty_alpha: float = 0.0
    min_p: float = 0.0

    use_cache: bool = True
    return_dict_in_generate: bool = False
    output_scores: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False

    batch_size: int = 4
    max_batch_size: int = 32
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    use_flash_attention_2: bool = True

    stream: bool = False
    stream_options: dict[str, Any] | None = None


class ModelPresetConfig(BaseModel):
    """Model preset configuration."""

    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    target_modules: list[str]
    chat_template_name: str


class ModelConfigComplete(BaseModel):
    """Complete model configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    peft: PEFTConfig = Field(default_factory=PEFTConfig)
    quantization: QuantizationConfigModel = Field(default_factory=QuantizationConfigModel)
    loading: ModelLoadingConfig = Field(default_factory=ModelLoadingConfig)
    saving: ModelSavingConfig = Field(default_factory=ModelSavingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    model_presets: dict[str, ModelPresetConfig] = Field(default_factory=dict)


# =============================================================================
# DATA CONFIGURATION MODELS
# =============================================================================


class ColumnMappingConfig(BaseModel):
    """Dataset column mapping."""

    instruction: str = "instruction"
    input: str = "input"
    output: str = "output"
    text: str | None = None


class DatasetConfig(BaseModel):
    """Single dataset configuration."""

    name: str
    path: str
    config_name: str | None = None
    data_files: str | dict[str, str] | None = None
    split: str = "train"
    streaming: bool = False
    max_samples: int | None = None
    column_mapping: ColumnMappingConfig = Field(default_factory=ColumnMappingConfig)
    weight: float = 1.0
    is_eval: bool = False
    category: str = "general"
    description: str = ""


class PromptTemplateConfig(BaseModel):
    """Prompt template configuration."""

    template: str
    template_with_input: str | None = None
    system_message: str = "You are a helpful assistant."
    instruction_key: str | None = "instruction"
    input_key: str | None = "input"
    output_key: str | None = "output"
    text_key: str | None = "text"
    add_eos_token: bool = True


class DataDownloadConfig(BaseModel):
    """Data download configuration."""

    cache_dir: str = "./data/raw"
    force_redownload: bool = False
    resume_download: bool = True
    num_proc: int = 4
    max_retries: int = 3


class DataValidationConfig(BaseModel):
    """Data validation configuration."""

    enabled: bool = True
    required_columns: list[str] = Field(default_factory=lambda: ["instruction", "output"])
    min_instruction_length: int = 10
    max_instruction_length: int = 8192
    min_output_length: int = 5
    max_output_length: int = 16384
    drop_nulls: bool = True
    drop_empty_strings: bool = True
    check_duplicates: bool = True
    duplicate_subset: list[str] = Field(default_factory=lambda: ["instruction", "input", "output"])
    detect_language: bool = False
    expected_language: str = "en"


class DataCleaningConfig(BaseModel):
    """Data cleaning configuration."""

    enabled: bool = True
    remove_duplicates: bool = True
    duplicate_subset: list[str] = Field(default_factory=lambda: ["instruction", "input", "output"])
    remove_nulls: bool = True
    strip_whitespace: bool = True
    remove_html: bool = False
    normalize_unicode: bool = True
    custom_cleaners: list[str] = Field(default_factory=list)


class DataFormattingConfig(BaseModel):
    """Data formatting configuration."""

    enabled: bool = True
    template: str = "alpaca"
    system_message: str = "You are a helpful assistant."
    include_input: bool = True
    add_eos_token: bool = True
    formatted_field: str = "text"
    keep_original_columns: bool = False


class TokenizationConfig(BaseModel):
    """Tokenization configuration."""

    enabled: bool = True
    max_seq_length: int = 2048
    truncation: bool = True
    truncation_side: str = "right"
    padding: bool = False
    padding_side: str = "right"
    pad_to_multiple_of: int = 8
    return_tensors: str = "pt"
    add_special_tokens: bool = True
    num_proc: int = 4
    batch_size: int = 1000
    text_field: str = "text"
    compute_stats: bool = True


class StatisticsConfig(BaseModel):
    """Statistics configuration."""

    enabled: bool = True
    sample_size: int | None = 10000
    percentiles: list[int] = Field(default_factory=lambda: [0, 25, 50, 75, 90, 95, 99, 100])
    save_path: str = "./data/processed/statistics.json"


class SplittingConfig(BaseModel):
    """Dataset splitting configuration."""

    enabled: bool = True
    ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.90, "validation": 0.05, "test": 0.05}
    )
    seed: int = 42
    shuffle: bool = True
    stratify_by: str | None = None
    min_train_samples: int = 100
    min_val_samples: int = 10
    min_test_samples: int = 10
    method: str = "random"

    @model_validator(mode="after")
    def validate_ratios(self) -> SplittingConfig:
        total = sum(self.ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self


class OutputConfig(BaseModel):
    """Output configuration."""

    output_dir: str = "./data/processed"
    formats: list[str] = Field(default_factory=lambda: ["arrow", "jsonl"])
    save_splits: bool = True
    filenames: dict[str, str] = Field(
        default_factory=lambda: {
            "train": "train.arrow",
            "validation": "val.arrow",
            "test": "test.arrow",
        }
    )
    save_statistics: bool = True
    stats_filename: str = "dataset_statistics.json"
    save_tokenizer: bool = False
    tokenizer_dir: str = "./data/processed/tokenizer"
    compression: str | None = "lz4"
    overwrite: bool = True


class DataLoaderConfig(BaseModel):
    """DataLoader configuration."""

    train: dict[str, Any] = Field(
        default_factory=lambda: {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "timeout": 0,
        }
    )
    eval: dict[str, Any] = Field(
        default_factory=lambda: {
            "batch_size": 4,
            "shuffle": False,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
            "persistent_workers": True,
            "prefetch_factor": 2,
        }
    )
    test: dict[str, Any] = Field(
        default_factory=lambda: {
            "batch_size": 4,
            "shuffle": False,
            "num_workers": 4,
            "pin_memory": True,
        }
    )
    collator: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "DataCollatorForSeq2Seq",
            "padding": "longest",
            "max_length": 2048,
            "pad_to_multiple_of": 8,
            "return_tensors": "pt",
            "mlm": False,
            "mlm_probability": 0.15,
        }
    )


class MixingConfig(BaseModel):
    """Data mixing configuration."""

    enabled: bool = False
    strategy: str = "proportional"
    custom_weights: dict[str, float] = Field(default_factory=dict)
    interleave: bool = True
    seed: int = 42
    stop_on_shortest: bool = False


class StreamingConfig(BaseModel):
    """Streaming configuration."""

    enabled: bool = False
    buffer_size: int = 10000
    shuffle_seed: int = 42
    take_n: int | None = None


class DataConfigComplete(BaseModel):
    """Complete data configuration."""

    datasets: list[DatasetConfig] = Field(default_factory=list)
    prompt_templates: dict[str, PromptTemplateConfig] = Field(default_factory=dict)
    default_template: str = "alpaca"
    processing: dict[str, Any] = Field(default_factory=dict)
    output: OutputConfig = Field(default_factory=OutputConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    mixing: MixingConfig = Field(default_factory=MixingConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)


# =============================================================================
# LOGGING CONFIGURATION MODELS
# =============================================================================


class ConsoleLogConfig(BaseModel):
    """Console logging configuration."""

    enabled: bool = True
    level: str = "INFO"
    stream: str = "stdout"
    format: str = "colored"
    show_timestamp: bool = True
    show_level: bool = True
    show_logger_name: bool = True
    show_module: bool = False
    show_line_number: bool = False
    colors: dict[str, str] = Field(
        default_factory=lambda: {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    )


class FileLogConfig(BaseModel):
    """File logging configuration."""

    enabled: bool = True
    level: str = "DEBUG"
    filename: str = "training.log"
    rotation: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "max_bytes": 10485760,
            "backup_count": 10,
            "encoding": "utf-8",
            "delay": False,
        }
    )
    format: str = "json"
    compression: str | None = None


class ErrorLogConfig(BaseModel):
    """Error logging configuration."""

    enabled: bool = True
    level: str = "ERROR"
    filename: str = "errors.log"
    rotation: dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True, "max_bytes": 10485760, "backup_count": 5}
    )
    capture_traceback: bool = True
    traceback_format: str = "full"


class TrainingLogConfig(BaseModel):
    """Training metrics logging."""

    enabled: bool = True
    level: str = "INFO"
    filename: str = "training_metrics.log"
    log_metrics: bool = True
    metrics: list[str] = Field(
        default_factory=lambda: [
            "step",
            "epoch",
            "loss",
            "learning_rate",
            "grad_norm",
            "tokens_per_second",
            "samples_per_second",
            "gpu_memory_allocated",
            "gpu_memory_reserved",
            "gpu_utilization",
            "cpu_percent",
            "ram_percent",
        ]
    )
    log_every: int = 10
    log_first_step: bool = True


class TensorBoardConfig(BaseModel):
    """TensorBoard configuration."""

    enabled: bool = True
    log_dir: str = "./logs/tensorboard"
    experiment_name: str = "{run_name}"
    comment: str = ""
    host: str = "localhost"
    port: int = 6006
    purge_orphaned_data: bool = True
    max_queue: int = 10
    flush_secs: int = 30
    write_graph: bool = True
    write_images: bool = False
    profile_batch: int = 0
    histogram_freq: int = 1
    embeddings_freq: int = 0
    scalar_tags: dict[str, list[str]] = Field(default_factory=dict)
    histogram_tags: list[str] = Field(default_factory=list)
    image_tags: list[str] = Field(default_factory=list)


class WandBConfig(BaseModel):
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "llm-finetuning"
    entity: str | None = None
    name: str = "{run_name}"
    id: str | None = None
    resume: str = "allow"
    group: str | None = None
    job_type: str = "training"
    tags: list[str] = Field(
        default_factory=lambda: ["qlora", "llama3", "fine-tuning", "production"]
    )
    notes: str = ""
    config: dict[str, Any] = Field(default_factory=lambda: {"log_all": True, "exclude_keys": []})
    log_freq: int = 10
    log_model: bool | str = True
    log_gradients: bool = True
    log_parameters: bool = True
    watch: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "log": "all",
            "log_freq": 100,
            "log_graph": True,
            "log_param_shapes": True,
        }
    )
    artifacts: dict[str, Any] = Field(default_factory=dict)
    sweep: dict[str, Any] = Field(default_factory=dict)
    offline: bool = False
    anonymous: str = "never"
    sync_tensorboard: bool = True
    sync_wandb: bool = True
    api_key: str = "${WANDB_API_KEY}"
    base_url: str = "https://api.wandb.ai"
    timeout: int = 30


class MLflowConfig(BaseModel):
    """MLflow configuration."""

    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "llm-finetuning"
    run_name: str = "{run_name}"
    tags: dict[str, str] = Field(default_factory=dict)
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    log_models: bool = True
    register_model: bool = False
    model_name: str = "llm-finetuned-model"
    model_aliases: list[str] = Field(default_factory=lambda: ["staging"])
    artifact_location: str = "./mlruns"
    autolog: dict[str, Any] = Field(default_factory=dict)


class LoggingConfigComplete(BaseModel):
    """Complete logging configuration."""

    level: str = "INFO"
    format: str = "json"
    colored_console: bool = True
    timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f"
    timestamp_utc: bool = True
    log_dir: str = "./logs"
    log_file_pattern: str = "{name}_{timestamp}.log"
    file_timestamp_format: str = "%Y%m%d_%H%M%S"
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    rotation: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "max_bytes": 10485760,
            "backup_count": 10,
            "encoding": "utf-8",
            "delay": False,
        }
    )
    console: ConsoleLogConfig = Field(default_factory=ConsoleLogConfig)
    file: FileLogConfig = Field(default_factory=FileLogConfig)
    error_log: ErrorLogConfig = Field(default_factory=ErrorLogConfig)
    training_log: TrainingLogConfig = Field(default_factory=TrainingLogConfig)
    tensorboard: TensorBoardConfig = Field(default_factory=TensorBoardConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


# =============================================================================
# EVALUATION CONFIGURATION MODELS
# =============================================================================


class GenerationConfig(BaseModel):
    """Generation configuration."""

    max_new_tokens: int = 512
    min_new_tokens: int = 1
    early_stopping: bool = True
    max_length: int | None = None
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.0
    typical_p: float = 1.0
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    encoder_repetition_penalty: float = 1.0
    diversity_penalty: float = 0.0
    num_beams: int = 1
    num_beam_groups: int = 1
    num_return_sequences: int = 1
    beam_search: bool = False
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    use_cache: bool = True
    synced_gpus: bool = False
    penalty_alpha: float = 0.0
    top_k_contrastive: int = 0
    watermark: bool = False


class EvalDatasetConfig(BaseModel):
    """Evaluation dataset configuration."""

    name: str
    path: str
    config_name: str | None = None
    split: str = "test"
    subset: str | None = None
    max_samples: int | None = None
    streaming: bool = False
    prompt_template: str = "alpaca"
    system_message: str = "You are a helpful assistant."
    metrics: list[str] = Field(default_factory=list)
    weight: float = 1.0
    category: str = "general"
    description: str = ""


class RougeConfig(BaseModel):
    """ROUGE metric configuration."""

    enabled: bool = True
    rouge_types: list[str] = Field(
        default_factory=lambda: ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    )
    use_stemmer: bool = True
    use_aggregator: bool = True
    confidence_interval: float = 0.95
    bootstrap_samples: int = 1000
    tokenizer: str | None = None
    split_summaries: bool = True


class BleuConfig(BaseModel):
    """BLEU metric configuration."""

    enabled: bool = True
    max_order: int = 4
    smooth: bool = True
    smooth_method: str = "exp"
    smooth_value: float = 1e-4
    use_effective_order: bool = True
    tokenizer: str = "13a"
    lowercase: bool = False
    force: bool = False


class BertScoreConfig(BaseModel):
    """BERTScore metric configuration."""

    enabled: bool = True
    model_type: str = "microsoft/deberta-xlarge-mnli"
    num_layers: int = 17
    batch_size: int = 32
    device: str = "cuda"
    rescale_with_baseline: bool = True
    baseline_path: str | None = None
    lang: str = "en"
    idf: bool = False
    verbose: bool = False
    metrics: list[str] = Field(default_factory=lambda: ["precision", "recall", "f1"])


class PerplexityConfig(BaseModel):
    """Perplexity metric configuration."""

    enabled: bool = True
    model_id: str = "gpt2-large"
    use_eval_model: bool = False
    stride: int = 512
    max_length: int = 1024
    batch_size: int = 8
    device: str = "cuda"
    add_start_token: bool = True
    aggregation: str = "mean"


class DistinctConfig(BaseModel):
    """Distinct-n metric configuration."""

    enabled: bool = True
    n_grams: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    max_n: int = 4
    normalize: bool = True


class EvaluationConfigComplete(BaseModel):
    """Complete evaluation configuration."""

    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    datasets: list[EvalDatasetConfig] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    baseline: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# MAIN CONFIG MANAGER
# =============================================================================


class ConfigManager:
    """
    Centralized configuration manager with YAML loading, environment variable
    overrides, and Pydantic validation.
    """

    def __init__(self, config_dir: str | Path = "configs", env_file: str | Path | None = ".env"):
        self.config_dir = Path(config_dir)
        self.env_file = Path(env_file) if env_file else None
        self._configs: dict[str, Any] = {}
        self._load_env_file()
        self._load_all_configs()

    def _load_env_file(self) -> None:
        """Load environment variables from .env file."""
        if self.env_file and self.env_file.exists():
            with open(self.env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        """Load a YAML configuration file."""
        path = self.config_dir / filename
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _resolve_env_vars(self, value: Any) -> Any:
        """Recursively resolve environment variable references in config values."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            env_value = os.environ.get(env_var)
            # Only replace if env var is set and non-empty
            if env_value is not None and env_value != "":
                return env_value
            return value
        elif isinstance(value, dict):
            return {k: self._resolve_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_env_vars(v) for v in value]
        return value

    def _load_all_configs(self) -> None:
        """Load and validate all configuration files."""
        # Training config
        training_data = self._resolve_env_vars(self._load_yaml("training.yaml"))
        self._configs["training"] = (
            TrainingConfig(**training_data) if training_data else TrainingConfig()
        )

        # Model config
        model_data = self._resolve_env_vars(self._load_yaml("model.yaml"))
        self._configs["model"] = (
            ModelConfigComplete(**model_data) if model_data else ModelConfigComplete()
        )

        # Data config
        data_data = self._resolve_env_vars(self._load_yaml("data.yaml"))
        self._configs["data"] = (
            DataConfigComplete(**data_data) if data_data else DataConfigComplete()
        )

        # Logging config
        logging_data = self._resolve_env_vars(self._load_yaml("logging.yaml"))
        self._configs["logging"] = (
            LoggingConfigComplete(**logging_data) if logging_data else LoggingConfigComplete()
        )

        # Evaluation config
        eval_data = self._resolve_env_vars(self._load_yaml("evaluation.yaml"))
        self._configs["evaluation"] = (
            EvaluationConfigComplete(**eval_data) if eval_data else EvaluationConfigComplete()
        )

    @property
    def training(self) -> TrainingConfig:
        return self._configs["training"]

    @property
    def model(self) -> ModelConfigComplete:
        return self._configs["model"]

    @property
    def data(self) -> DataConfigComplete:
        return self._configs["data"]

    @property
    def logging(self) -> LoggingConfigComplete:
        return self._configs["logging"]

    @property
    def evaluation(self) -> EvaluationConfigComplete:
        return self._configs["evaluation"]

    def update(self, **kwargs: Any) -> None:
        """Update configuration sections programmatically with deep merge."""

        def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
            """Recursively merge update dict into base dict."""
            result = base.copy()
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        for section, values in kwargs.items():
            if section in self._configs:
                current = self._configs[section].model_dump()
                merged = deep_merge(current, values)
                # Re-validate
                if section == "training":
                    self._configs[section] = TrainingConfig(**merged)
                elif section == "model":
                    self._configs[section] = ModelConfigComplete(**merged)
                elif section == "data":
                    self._configs[section] = DataConfigComplete(**merged)
                elif section == "logging":
                    self._configs[section] = LoggingConfigComplete(**merged)
                elif section == "evaluation":
                    self._configs[section] = EvaluationConfigComplete(**merged)

    def save_resolved(self, output_path: str | Path) -> None:
        """Save fully resolved configuration to YAML."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = {name: config.model_dump() for name, config in self._configs.items()}
        with open(output_path, "w") as f:
            yaml.dump(resolved, f, default_flow_style=False, sort_keys=False)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation."""
        config = self._configs.get(section)
        if not config:
            return default
        keys = key.split(".")
        value = config
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def validate_all(self) -> bool:
        """Validate all configurations. Raises ValidationError if invalid."""
        for _name, config in self._configs.items():
            config.model_validate(config.model_dump())
        return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_config_manager: ConfigManager | None = None


def get_config_manager(
    config_dir: str | Path = "configs", env_file: str | Path | None = ".env"
) -> ConfigManager:
    """Get or create the global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir, env_file)
    return _config_manager


def load_config(
    config_dir: str | Path = "configs", env_file: str | Path | None = ".env"
) -> ConfigManager:
    """Load and return a new ConfigManager instance."""
    return ConfigManager(config_dir, env_file)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config classes
    "TrainingConfig",
    "TrainerConfig",
    "LoRAConfig",
    "QuantizationConfig",
    "SFTConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "CallbacksConfig",
    "EarlyStoppingConfig",
    "CheckpointConfig",
    "LoggingCallbackConfig",
    "ProfilerConfig",
    "RuntimeConfig",
    "ExperimentConfig",
    "ModelConfigComplete",
    "ModelConfig",
    "ModelArchitectureConfig",
    "TokenizerConfig",
    "PEFTConfig",
    "PEFTLoraConfig",
    "AdaLoRAConfig",
    "IA3Config",
    "QuantizationConfigModel",
    "ModelLoadingConfig",
    "ModelSavingConfig",
    "InferenceConfig",
    "ModelPresetConfig",
    "DataConfigComplete",
    "DatasetConfig",
    "ColumnMappingConfig",
    "PromptTemplateConfig",
    "DataDownloadConfig",
    "DataValidationConfig",
    "DataCleaningConfig",
    "DataFormattingConfig",
    "TokenizationConfig",
    "StatisticsConfig",
    "SplittingConfig",
    "OutputConfig",
    "DataLoaderConfig",
    "MixingConfig",
    "StreamingConfig",
    "LoggingConfigComplete",
    "ConsoleLogConfig",
    "FileLogConfig",
    "ErrorLogConfig",
    "TrainingLogConfig",
    "TensorBoardConfig",
    "WandBConfig",
    "MLflowConfig",
    "EvaluationConfigComplete",
    "GenerationConfig",
    "EvalDatasetConfig",
    "RougeConfig",
    "BleuConfig",
    "BertScoreConfig",
    "PerplexityConfig",
    "DistinctConfig",
    # Manager
    "ConfigManager",
    "get_config_manager",
    "load_config",
]
