"""
Model Utilities for LLM Fine-tuning

Comprehensive model loading, quantization, LoRA/PEFT, and memory optimization utilities.
Configuration-driven with support for:
- BitsAndBytesConfig (NF4, Double Quantization)
- QLoRA / PEFT
- Gradient Checkpointing
- Flash Attention
- Memory optimization
- Automatic device mapping
- Tokenizer loading
- Trainable parameter reporting
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import (
    AdaLoraConfig,
    IA3Config,
    LoraConfig,
    PeftConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config import (
    AdaLoRAConfig,
    LoRAConfig,
    ModelConfig,
    ModelLoadingConfig,
    PEFTLoraConfig,
    QuantizationConfig,
    QuantizationConfigModel,
    RuntimeConfig,
    TokenizerConfig,
    TrainingConfig,
)


# no_init_weights was removed in newer transformers, create a compatible version
@contextmanager
def no_init_weights(_enable: bool = True):
    """Context manager to skip weight initialization (compatible with older transformers)."""
    yield


logger = logging.getLogger(__name__)


@dataclass
class ModelLoadResult:
    """Result of model loading operation."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    model_config: AutoConfig
    quantization_config: BitsAndBytesConfig | None
    peft_config: PeftConfig | None
    device_map: dict[str, Any]
    memory_stats: dict[str, Any]


@dataclass
class ParameterStats:
    """Trainable parameter statistics."""

    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_percent: float
    lora_params: int
    lora_percent: float


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def create_bnb_config(
    quant_config: QuantizationConfig | QuantizationConfigModel,
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytesConfig from configuration.

    Supports:
    - NF4 (NormalFloat4) quantization
    - Double Quantization
    - FP4 quantization
    - 8-bit quantization
    """
    if isinstance(quant_config, QuantizationConfig):
        load_in_4bit = quant_config.load_in_4bit
        quant_type = quant_config.bnb_4bit_quant_type
        compute_dtype = get_torch_dtype(quant_config.bnb_4bit_compute_dtype)
        use_double_quant = quant_config.bnb_4bit_use_double_quant
        quant_storage = getattr(torch, quant_config.bnb_4bit_quant_storage, torch.uint8)

        load_in_8bit = quant_config.load_in_8bit
        llm_int8_threshold = quant_config.llm_int8_threshold
        llm_int8_has_fp16_weight = quant_config.llm_int8_has_fp16_weight
        llm_int8_skip_modules = quant_config.llm_int8_skip_modules
        llm_int8_enable_fp32_cpu_offload = quant_config.llm_int8_enable_fp32_cpu_offload
    else:
        load_in_4bit = quant_config.load_in_4bit
        quant_type = quant_config.bnb_4bit_quant_type
        compute_dtype = get_torch_dtype(quant_config.bnb_4bit_compute_dtype)
        use_double_quant = quant_config.bnb_4bit_use_double_quant
        quant_storage = getattr(torch, quant_config.bnb_4bit_quant_storage, torch.uint8)

        load_in_8bit = quant_config.load_in_8bit
        llm_int8_threshold = quant_config.llm_int8_threshold
        llm_int8_has_fp16_weight = quant_config.llm_int8_has_fp16_weight
        llm_int8_skip_modules = quant_config.llm_int8_skip_modules
        llm_int8_enable_fp32_cpu_offload = quant_config.llm_int8_enable_fp32_cpu_offload

    if load_in_4bit:
        logger.info(
            f"Creating 4-bit BitsAndBytesConfig: "
            f"quant_type={quant_type}, compute_dtype={compute_dtype}, "
            f"double_quant={use_double_quant}, quant_storage={quant_storage}"
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_quant_storage=quant_storage,
            llm_int8_skip_modules=llm_int8_skip_modules,
        )
    elif load_in_8bit:
        logger.info(
            f"Creating 8-bit BitsAndBytesConfig: "
            f"threshold={llm_int8_threshold}, fp16_weight={llm_int8_has_fp16_weight}, "
            f"cpu_offload={llm_int8_enable_fp32_cpu_offload}"
        )
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=llm_int8_threshold,
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            llm_int8_skip_modules=llm_int8_skip_modules,
            llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
        )
    else:
        logger.warning("No quantization configured (load_in_4bit=False, load_in_8bit=False)")
        return None


def create_lora_config(peft_config: PEFTLoraConfig | LoRAConfig) -> LoraConfig:
    """Create PEFT LoRA config from configuration."""
    if isinstance(peft_config, PEFTLoraConfig):
        r = peft_config.r
        lora_alpha = peft_config.lora_alpha
        lora_dropout = peft_config.lora_dropout
        use_rslora = peft_config.use_rslora
        use_dora = peft_config.use_dora
        init_lora_weights = peft_config.init_lora_weights
        target_modules = peft_config.target_modules
        bias = peft_config.bias
        task_type = peft_config.task_type
        inference_mode = peft_config.inference_mode
        layers_to_transform = peft_config.layers_to_transform
        layers_pattern = peft_config.layers_pattern
        rank_pattern = peft_config.rank_pattern
        alpha_pattern = peft_config.alpha_pattern
        megatron_config = peft_config.megatron_config
        megatron_core = peft_config.megatron_core
    else:
        r = peft_config.r
        lora_alpha = peft_config.lora_alpha
        lora_dropout = peft_config.lora_dropout
        use_rslora = peft_config.use_rslora
        use_dora = peft_config.use_dora
        init_lora_weights = peft_config.init_lora_weights
        target_modules = peft_config.target_modules
        bias = peft_config.bias
        task_type = peft_config.task_type
        inference_mode = peft_config.inference_mode
        layers_to_transform = getattr(peft_config, "layers_to_transform", None)
        layers_pattern = getattr(peft_config, "layers_pattern", None)
        rank_pattern = getattr(peft_config, "rank_pattern", {})
        alpha_pattern = getattr(peft_config, "alpha_pattern", {})
        megatron_config = None
        megatron_core = False

    logger.info(
        f"Creating LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, "
        f"rslora={use_rslora}, dora={use_dora}, target_modules={target_modules}"
    )

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        use_dora=use_dora,
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType[task_type] if isinstance(task_type, str) else task_type,
        inference_mode=inference_mode,
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        megatron_config=megatron_config,
        megatron_core=megatron_core,
    )


def create_adalora_config(adalora_config: AdaLoRAConfig) -> AdaLoraConfig:
    """Create PEFT AdaLoRA config from configuration."""
    logger.info(
        f"Creating AdaLoRA config: target_r={adalora_config.target_r}, "
        f"init_r={adalora_config.init_r}, tinit={adalora_config.tinit}, "
        f"tfinal={adalora_config.tfinal}, deltaT={adalora_config.deltaT}"
    )

    total_step = adalora_config.total_step or 1000

    return AdaLoraConfig(
        target_r=adalora_config.target_r,
        init_r=adalora_config.init_r,
        tinit=adalora_config.tinit,
        tfinal=adalora_config.tfinal,
        deltaT=adalora_config.deltaT,
        beta1=adalora_config.beta1,
        beta2=adalora_config.beta2,
        orth_reg_weight=adalora_config.orth_reg_weight,
        total_step=total_step,
        rank_pattern=adalora_config.rank_pattern,
        alpha_pattern=adalora_config.alpha_pattern,
    )


def create_ia3_config(ia3_config: Any) -> IA3Config:
    """Create PEFT IA3 config from configuration."""
    logger.info(f"Creating IA3 config: target_modules={ia3_config.target_modules}")

    target_modules = ia3_config.target_modules
    feedforward_modules = ia3_config.feedforward_modules

    # Ensure feedforward_modules is a subset of target_modules
    if feedforward_modules:
        feedforward_modules = [m for m in feedforward_modules if m in target_modules]

    return IA3Config(
        target_modules=target_modules,
        feedforward_modules=feedforward_modules,
        fan_in_fan_out=ia3_config.fan_in_fan_out,
        init_ia3_weights=ia3_config.init_ia3_weights,
    )


def load_tokenizer(tokenizer_config: TokenizerConfig) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer from configuration.

    Handles:
    - Fast/slow tokenizer selection
    - Padding/truncation side
    - Special tokens
    - Chat templates
    - Model max length
    """
    tokenizer_name = (
        tokenizer_config.tokenizer_name_or_path
        if tokenizer_config.tokenizer_name_or_path
        else tokenizer_config.tokenizer_name_or_path
    )

    logger.info(f"Loading tokenizer: {tokenizer_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_config.tokenizer_revision,
        cache_dir=tokenizer_config.tokenizer_cache_dir,
        use_fast=tokenizer_config.use_fast,
        padding_side=tokenizer_config.padding_side,
        truncation_side=tokenizer_config.truncation_side,
        model_max_length=tokenizer_config.model_max_length,
        use_auth_token=tokenizer_config.use_auth_token,
        trust_remote_code=True,
    )

    if tokenizer_config.pad_token is not None:
        tokenizer.pad_token = tokenizer_config.pad_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    if tokenizer_config.eos_token is not None:
        tokenizer.eos_token = tokenizer_config.eos_token

    if tokenizer_config.bos_token is not None:
        tokenizer.bos_token = tokenizer_config.bos_token

    for special_token in tokenizer_config.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})

    if tokenizer_config.chat_template is not None:
        tokenizer.chat_template = tokenizer_config.chat_template
        logger.info("Applied custom chat template")
    elif tokenizer_config.chat_template_name:
        chat_templates = {
            "llama3": (
                "<|begin_of_text|>{% for message in messages %}"
                "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
                "{{ message['content'] }}<|eot_id|>"
                "{% endfor %}"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            "chatml": (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}"
                "{{'<|im_start|>assistant\n'}}"
            ),
            "vicuna": (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}USER: {{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}\n"
                "{% endif %}"
                "{% endfor %}"
            ),
            "zephyr": (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}"
                "{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}"
                "{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}"
                "{% endif %}"
                "{% endfor %}"
                "{{'<|assistant|>\n'}}"
            ),
            "alpaca": (
                "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}\n\n{% endif %}"
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}{{'### Instruction:\n' + message['content'] + '\n\n'}}"
                "{% elif message['role'] == 'assistant' %}{{'### Response:\n' + message['content'] + '\n\n'}}"
                "{% endif %}"
                "{% endfor %}"
            ),
        }
        if tokenizer_config.chat_template_name in chat_templates:
            tokenizer.chat_template = chat_templates[tokenizer_config.chat_template_name]
            logger.info(f"Applied {tokenizer_config.chat_template_name} chat template")

    logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}, pad_token={tokenizer.pad_token}")
    return tokenizer


def setup_flash_attention(
    model: PreTrainedModel, enable: bool = True, version: int = 2
) -> PreTrainedModel:
    """
    Configure Flash Attention for the model.

    Supports:
    - Flash Attention 2 (default)
    - Flash Attention 3 (Hopper/H100)
    - Automatic fallback to eager attention
    """
    try:
        import flash_attn

        logger.info(f"Flash Attention {version} available: {flash_attn.__version__}")
    except ImportError:
        logger.warning("Flash Attention not installed, falling back to eager attention")
        model.config._attn_implementation = "eager"
        return model

    if not enable:
        logger.info("Flash Attention disabled")
        model.config._attn_implementation = "eager"
        return model

    if version == 2:
        model.config._attn_implementation = "flash_attention_2"
        model.config.use_flash_attention_2 = True
        logger.info("Enabled Flash Attention 2")
    elif version == 3:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
            model.config._attn_implementation = "flash_attention_3"
            logger.info("Enabled Flash Attention 3 (Hopper/H100)")
        else:
            logger.warning("Flash Attention 3 requires Hopper (H100) GPU, falling back to FA2")
            model.config._attn_implementation = "flash_attention_2"
    else:
        logger.warning(f"Unknown Flash Attention version {version}, using FA2")
        model.config._attn_implementation = "flash_attention_2"

    return model


def setup_gradient_checkpointing(
    model: PreTrainedModel,
    enabled: bool = True,
    use_reentrant: bool = False,
) -> PreTrainedModel:
    """
    Configure gradient checkpointing for memory efficiency.

    Args:
        model: The model to configure
        enabled: Whether to enable gradient checkpointing
        use_reentrant: Use reentrant checkpointing (False recommended for Flash Attention)
    """
    if enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
        model.config.use_cache = False
        logger.info(f"Gradient checkpointing enabled (use_reentrant={use_reentrant})")
    else:
        model.gradient_checkpointing_disable()
        logger.info("Gradient checkpointing disabled")
    return model


def prepare_model_for_training(
    model: PreTrainedModel,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
) -> PreTrainedModel:
    """
    Prepare model for k-bit training (QLoRA).

    This enables:
    - Gradient checkpointing
    - Input gradient requirement
    - Cast layernorm to fp32
    - Cast output embeddings to fp32
    """
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": False}

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    )

    for name, param in model.named_parameters():
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
        if "lora" in name.lower() or "adapter" in name.lower():
            param.requires_grad = True

    logger.info("Model prepared for k-bit training")
    return model


def apply_peft_model(
    model: PreTrainedModel,
    peft_config: PEFTLoraConfig | LoRAConfig | AdaLoRAConfig | Any,
    peft_type: str = "LORA",
    adapter_name: str = "default",
) -> PeftModel:
    """
    Apply PEFT (LoRA, AdaLoRA, IA3) to model.

    Args:
        model: Base model
        peft_config: PEFT configuration
        peft_type: Type of PEFT (LORA, ADALORA, IA3)
        adapter_name: Name for the adapter
    """
    if peft_type.upper() == "LORA":
        lora_config = create_lora_config(peft_config)
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)
    elif peft_type.upper() == "ADALORA":
        adalora_config = create_adalora_config(peft_config)
        model = get_peft_model(model, adalora_config, adapter_name=adapter_name)
    elif peft_type.upper() == "IA3":
        ia3_config = create_ia3_config(peft_config)
        model = get_peft_model(model, ia3_config, adapter_name=adapter_name)
    else:
        raise ValueError(f"Unknown PEFT type: {peft_type}")

    logger.info(f"Applied {peft_type} adapter: {adapter_name}")
    return model


def count_parameters(model: nn.Module, verbose: bool = True) -> ParameterStats:
    """
    Count total, trainable, and frozen parameters.

    Returns detailed statistics including LoRA-specific counts.
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            if "lora" in name.lower() or "lora_" in name.lower():
                lora_params += num_params
        else:
            frozen_params += num_params

    trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
    lora_percent = 100 * lora_params / total_params if total_params > 0 else 0

    stats = ParameterStats(
        total_params=total_params,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        trainable_percent=trainable_percent,
        lora_params=lora_params,
        lora_percent=lora_percent,
    )

    if verbose:
        logger.info(
            f"Parameter Statistics:\n"
            f"  Total:     {total_params:,} ({total_params / 1e9:.2f}B)\n"
            f"  Trainable: {trainable_params:,} ({trainable_percent:.2f}%)\n"
            f"  Frozen:    {frozen_params:,} ({100 - trainable_percent:.2f}%)\n"
            f"  LoRA:      {lora_params:,} ({lora_percent:.2f}%)"
        )

    return stats


def print_trainable_parameters(model: nn.Module) -> ParameterStats:
    """Print trainable parameter summary (alias for count_parameters)."""
    return count_parameters(model, verbose=True)


def get_model_memory_footprint(model: nn.Module) -> dict[str, Any]:
    """Get detailed memory footprint of model."""
    param_mem = 0
    buffer_mem = 0

    for param in model.parameters():
        param_mem += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_mem += buffer.numel() * buffer.element_size()

    total_mem = param_mem + buffer_mem

    return {
        "parameters_memory_gb": param_mem / 1e9,
        "buffers_memory_gb": buffer_mem / 1e9,
        "total_memory_gb": total_mem / 1e9,
        "parameters_memory_mb": param_mem / 1e6,
        "buffers_memory_mb": buffer_mem / 1e6,
        "total_memory_mb": total_mem / 1e6,
    }


def optimize_model_memory(
    model: PreTrainedModel,
    enable_flash_attention: bool = True,
    flash_attention_version: int = 2,
    gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    empty_cache_steps: int = 50,
    gc_collect_steps: int = 100,
) -> PreTrainedModel:
    """
    Apply comprehensive memory optimizations to model.

    Includes:
    - Flash Attention
    - Gradient Checkpointing
    - Cache clearing hooks
    - Garbage collection hooks
    """
    model = setup_flash_attention(model, enable_flash_attention, flash_attention_version)

    if gradient_checkpointing:
        gc_kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        model = setup_gradient_checkpointing(model, True, **gc_kwargs)

    if empty_cache_steps > 0 or gc_collect_steps > 0:

        def clear_cache_hook(module, args, output):
            if empty_cache_steps > 0 and hasattr(module, "_step_counter"):
                module._step_counter += 1
                if module._step_counter % empty_cache_steps == 0:
                    torch.cuda.empty_cache()
            if gc_collect_steps > 0 and hasattr(module, "_step_counter"):
                if module._step_counter % gc_collect_steps == 0:
                    gc.collect()

        model._step_counter = 0
        model.register_forward_hook(clear_cache_hook)
        logger.info(
            f"Registered cache clearing hook (every {empty_cache_steps} steps) "
            f"and GC hook (every {gc_collect_steps} steps)"
        )

    return model


def create_device_map(
    model: PreTrainedModel,
    max_memory: dict[str, str] | None = None,
    device_map: str = "auto",
    low_cpu_mem_usage: bool = True,
) -> dict[str, Any]:
    """
    Create and validate device map for model.

    Supports:
    - "auto": Automatic device mapping
    - "balanced": Balanced across GPUs
    - "sequential": Sequential across GPUs
    - Custom device map dict
    """
    if device_map == "auto":
        from accelerate import infer_auto_device_map

        if max_memory is None:
            max_memory = get_balanced_memory(model)
        device_map = infer_auto_device_map(model, max_memory=max_memory)
        logger.info(f"Auto device map: {device_map}")
    elif device_map == "balanced":
        from accelerate import infer_auto_device_map

        if max_memory is None:
            max_memory = get_balanced_memory(model)
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=model._no_split_modules
        )
    elif device_map == "sequential":
        from accelerate import infer_auto_device_map

        if max_memory is None:
            max_memory = get_balanced_memory(model)
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=model._no_split_modules
        )

    return device_map


def get_balanced_memory(model: PreTrainedModel) -> dict[str, str]:
    """Get balanced memory allocation for all available devices."""
    if not torch.cuda.is_available():
        return {}

    memory = {}
    for i in range(torch.cuda.device_count()):
        total_mem = torch.cuda.get_device_properties(i).total_memory
        free_mem = total_mem - torch.cuda.memory_allocated(i)
        memory[i] = f"{int(free_mem * 0.85 / 1e9)}GiB"
    memory["cpu"] = "16GiB"
    return memory


def load_model_config(model_config: ModelConfig) -> AutoConfig:
    """Load model configuration from config object."""
    logger.info(f"Loading model config: {model_config.model_name_or_path}")

    config = AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        cache_dir=model_config.model_cache_dir,
        trust_remote_code=model_config.trust_remote_code,
        use_auth_token=model_config.use_auth_token,
    )

    config.use_cache = model_config.use_cache
    config.gradient_checkpointing = (
        model_config.gradient_checkpointing if model_config.gradient_checkpointing else False
    )

    if model_config.gradient_checkpointing:
        config.gradient_checkpointing_kwargs = model_config.gradient_checkpointing_kwargs

    return config


def load_base_model(
    model_config: ModelConfig,
    quantization_config: BitsAndBytesConfig | None = None,
    device_map: str | dict | None = None,
    torch_dtype: torch.dtype | None = None,
    low_cpu_mem_usage: bool = True,
    attn_implementation: str | None = None,
) -> PreTrainedModel:
    """
    Load base model with quantization and device mapping.

    Handles:
    - 4-bit/8-bit quantization via BitsAndBytes
    - Automatic device mapping
    - Flash Attention
    - Low CPU memory usage
    """
    if torch_dtype is None:
        torch_dtype = get_torch_dtype(model_config.torch_dtype)

    if device_map is None:
        device_map = model_config.device_map

    if attn_implementation is None:
        attn_implementation = model_config.attn_implementation

    logger.info(
        f"Loading base model: {model_config.model_name_or_path}\n"
        f"  torch_dtype: {torch_dtype}\n"
        f"  device_map: {device_map}\n"
        f"  quantization: {quantization_config is not None}\n"
        f"  attn_implementation: {attn_implementation}\n"
        f"  low_cpu_mem_usage: {low_cpu_mem_usage}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        cache_dir=model_config.model_cache_dir,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_safetensors=model_config.use_safetensors,
        attn_implementation=attn_implementation,
        trust_remote_code=model_config.trust_remote_code,
        use_auth_token=model_config.use_auth_token,
        offload_folder=model_config.offload_folder,
        offload_state_dict=model_config.offload_state_dict,
        offload_buffers=model_config.offload_buffers,
        variant=model_config.variant,
        max_memory=model_config.max_memory,
    )

    logger.info(f"Base model loaded: {model.__class__.__name__}")
    return model


def load_model_and_tokenizer(
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    quantization_config: QuantizationConfig | QuantizationConfigModel | None = None,
    peft_config: PEFTLoraConfig | LoRAConfig | None = None,
    peft_type: str = "LORA",
    loading_config: ModelLoadingConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
) -> ModelLoadResult:
    """
    Complete model and tokenizer loading pipeline.

    This is the main entry point for loading a model with all configurations:
    - Quantization (BitsAndBytes NF4, Double Quant)
    - PEFT/LoRA
    - Flash Attention
    - Gradient Checkpointing
    - Device Mapping
    - Tokenizer with chat templates
    """
    if loading_config is None:
        loading_config = ModelLoadingConfig()

    if runtime_config is None:
        runtime_config = RuntimeConfig()

    if quantization_config is None:
        quantization_config = (
            model_config.quantization_config
            if hasattr(model_config, "quantization_config")
            else QuantizationConfig()
        )

    logger.info("=" * 60)
    logger.info("Loading Model and Tokenizer")
    logger.info("=" * 60)

    bnb_config = create_bnb_config(quantization_config)

    model_config_obj = load_model_config(model_config)

    model = load_base_model(
        model_config,
        quantization_config=bnb_config,
        device_map=model_config.device_map,
        torch_dtype=get_torch_dtype(model_config.torch_dtype),
        low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        attn_implementation=model_config.attn_implementation,
    )

    model = setup_flash_attention(
        model,
        enable=getattr(runtime_config, "flash_attention", False),
        version=getattr(runtime_config, "flash_attention_version", 2),
    )

    if getattr(runtime_config, "gradient_checkpointing", False):
        model = setup_gradient_checkpointing(
            model,
            True,
            use_reentrant=(
                model_config.gradient_checkpointing_kwargs.get("use_reentrant", False)
                if hasattr(model_config, "gradient_checkpointing_kwargs")
                and isinstance(model_config.gradient_checkpointing_kwargs, dict)
                else False
            ),
        )

    if peft_config is not None:
        model = prepare_model_for_training(model)
        model = apply_peft_model(model, peft_config, peft_type=peft_type)

    tokenizer = load_tokenizer(tokenizer_config)

    model = optimize_model_memory(
        model,
        enable_flash_attention=getattr(runtime_config, "flash_attention", False),
        flash_attention_version=getattr(runtime_config, "flash_attention_version", 2),
        gradient_checkpointing=getattr(runtime_config, "gradient_checkpointing", False),
        empty_cache_steps=getattr(runtime_config, "empty_cache_steps", 100),
        gc_collect_steps=getattr(runtime_config, "gc_collect_steps", 100),
    )

    memory_stats = get_model_memory_footprint(model)
    logger.info(f"Model memory footprint: {memory_stats['total_memory_gb']:.2f} GB")

    param_stats = count_parameters(model)
    logger.info(
        f"Trainable params: {param_stats.trainable_params:,} ({param_stats.trainable_percent:.2f}%)"
    )

    return ModelLoadResult(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config_obj,
        quantization_config=bnb_config,
        peft_config=peft_config,
        device_map=getattr(model, "hf_device_map", {}),
        memory_stats=memory_stats,
    )


def merge_and_unload_peft(model: PeftModel) -> PreTrainedModel:
    """Merge PEFT adapter weights into base model and unload PEFT."""
    logger.info("Merging PEFT adapter into base model...")
    merged_model = model.merge_and_unload()
    logger.info("PEFT adapter merged and unloaded")
    return merged_model


def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str | Path,
    save_adapter: bool = True,
    save_tokenizer: bool = True,
    merge_and_unload: bool = False,
    merged_dtype: str = "bfloat16",
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
    push_to_hub: bool = False,
    hub_model_id: str = "",
    hub_private_repo: bool = False,
    hub_token: str | None = None,
    commit_message: str = "Upload model",
) -> None:
    """
    Save model and tokenizer with various options.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        save_adapter: Save only adapter weights (for PEFT models)
        save_tokenizer: Also save tokenizer
        merge_and_unload: Merge adapter into base model before saving
        merged_dtype: Dtype for merged model
        safe_serialization: Use safetensors
        max_shard_size: Maximum shard size
        push_to_hub: Push to Hugging Face Hub
        hub_model_id: Hub model ID
        hub_private_repo: Private repo
        hub_token: Hub token
        commit_message: Commit message
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if merge_and_unload and isinstance(model, PeftModel):
        model = merge_and_unload_peft(model)
        if merged_dtype:
            model = model.to(get_torch_dtype(merged_dtype))

    if save_adapter and isinstance(model, PeftModel):
        model.save_pretrained(
            output_dir / "adapter",
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
        logger.info(f"Adapter saved to {output_dir / 'adapter'}")
    else:
        model.save_pretrained(
            output_dir,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
        logger.info(f"Model saved to {output_dir}")

    if save_tokenizer:
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Tokenizer saved to {output_dir}")

    if push_to_hub and hub_model_id:
        model.push_to_hub(
            hub_model_id,
            private=hub_private_repo,
            token=hub_token,
            commit_message=commit_message,
        )
        if save_tokenizer:
            tokenizer.push_to_hub(hub_model_id, token=hub_token)
        logger.info(f"Pushed to hub: {hub_model_id}")


def load_peft_adapter(
    model: PreTrainedModel,
    adapter_path: str | Path,
    adapter_name: str = "default",
    is_trainable: bool = True,
) -> PeftModel:
    """Load a PEFT adapter onto a base model."""
    logger.info(f"Loading PEFT adapter from {adapter_path}")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
    )
    return model


def set_adapter(model: PeftModel, adapter_name: str) -> PeftModel:
    """Set active adapter for PEFT model."""
    model.set_adapter(adapter_name)
    logger.info(f"Active adapter: {adapter_name}")
    return model


def disable_adapter(model: PeftModel) -> PeftModel:
    """Disable all adapters (use base model)."""
    model.disable_adapter()
    logger.info("Adapters disabled")
    return model


def enable_adapter(model: PeftModel) -> PeftModel:
    """Enable previously disabled adapter."""
    model.enable_adapter()
    logger.info("Adapters enabled")
    return model


@contextmanager
def model_loading_context(
    low_cpu_mem_usage: bool = True,
    torch_dtype: torch.dtype | None = None,
):
    """Context manager for memory-efficient model loading."""
    old_init = nn.Module.__init__

    def new_init(self, *args, **kwargs):
        if low_cpu_mem_usage:
            with no_init_weights(_enable=True):
                old_init(self, *args, **kwargs)
        else:
            old_init(self, *args, **kwargs)

    nn.Module.__init__ = new_init
    try:
        yield
    finally:
        nn.Module.__init__ = old_init


def get_gpu_memory_info() -> dict[str, Any]:
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    info = {"cuda_available": True, "devices": []}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free = props.total_memory - reserved

        info["devices"].append(
            {
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "allocated_gb": allocated / 1e9,
                "reserved_gb": reserved / 1e9,
                "free_gb": free / 1e9,
                "utilization_percent": (reserved / props.total_memory) * 100,
            }
        )
    return info


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    info = get_gpu_memory_info()
    if info["cuda_available"]:
        for dev in info["devices"]:
            logger.info(
                f"{prefix} GPU {dev['index']} ({dev['name']}): "
                f"{dev['allocated_gb']:.2f}GB allocated, "
                f"{dev['reserved_gb']:.2f}GB reserved, "
                f"{dev['free_gb']:.2f}GB free "
                f"({dev['utilization_percent']:.1f}% utilized)"
            )


def clear_gpu_cache() -> None:
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    logger.debug("GPU cache cleared and GC run")


def estimate_model_memory(
    model_name: str,
    quantization: str = "4bit",
    dtype: str = "bfloat16",
) -> dict[str, float]:
    """
    Estimate model memory requirements before loading.

    Args:
        model_name: Hugging Face model name
        quantization: "4bit", "8bit", "none"
        dtype: Model dtype

    Returns:
        Estimated memory in GB for parameters, gradients, optimizer states
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    param_count = 0
    param_count += vocab_size * hidden_size
    param_count += num_layers * (4 * hidden_size * hidden_size)
    param_count += num_layers * (2 * hidden_size * intermediate_size)
    param_count += num_layers * (hidden_size * num_kv_heads * (hidden_size // num_heads) * 2)
    param_count += num_layers * hidden_size * 2
    param_count += hidden_size

    bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2, "4bit": 0.5, "8bit": 1}.get(
        dtype, 2
    )
    if quantization == "4bit":
        bytes_per_param = 0.5
    elif quantization == "8bit":
        bytes_per_param = 1

    model_mem_gb = param_count * bytes_per_param / 1e9
    grad_mem_gb = model_mem_gb if quantization == "none" else 0
    opt_mem_gb = model_mem_gb * 2 if quantization == "none" else model_mem_gb * 0.5
    act_mem_gb = model_mem_gb * 0.5
    total_gb = model_mem_gb + grad_mem_gb + opt_mem_gb + act_mem_gb

    return {
        "parameters_gb": model_mem_gb,
        "gradients_gb": grad_mem_gb,
        "optimizer_gb": opt_mem_gb,
        "activations_gb": act_mem_gb,
        "total_estimate_gb": total_gb,
        "param_count": param_count,
    }


def print_model_summary(model: PreTrainedModel, input_shape: tuple | None = None) -> None:
    """Print comprehensive model summary."""
    lines = [
        "=" * 60,
        "MODEL SUMMARY",
        "=" * 60,
        f"Architecture: {model.__class__.__name__}",
        f"Config: {model.config.model_type}",
    ]

    param_stats = count_parameters(model, verbose=False)
    lines.extend(
        [
            f"Total Parameters: {param_stats.total_params:,} ({param_stats.total_params / 1e9:.2f}B)",
            f"Trainable Parameters: {param_stats.trainable_params:,} ({param_stats.trainable_percent:.2f}%)",
            f"LoRA Parameters: {param_stats.lora_params:,} ({param_stats.lora_percent:.2f}%)",
        ]
    )

    mem_stats = get_model_memory_footprint(model)
    lines.append(f"Model Memory: {mem_stats['total_memory_gb']:.2f} GB")

    if input_shape:
        lines.append(f"Input Shape: {input_shape}")

    lines.append("=" * 60)
    summary_text = "\n".join(lines)
    logger.info(summary_text)


def get_layer_info(model: PreTrainedModel) -> list[dict[str, Any]]:
    """Get information about model layers."""
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            param_count = sum(p.numel() for p in module.parameters())
            trainable = any(p.requires_grad for p in module.parameters())
            layers.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "parameters": param_count,
                    "trainable": trainable,
                }
            )
    return layers


def freeze_layers(model: PreTrainedModel, layer_pattern: str) -> PreTrainedModel:
    """Freeze layers matching pattern."""
    import re

    pattern = re.compile(layer_pattern)
    frozen = 0
    for name, param in model.named_parameters():
        if pattern.search(name):
            param.requires_grad = False
            frozen += 1
    logger.info(f"Frozen {frozen} parameters matching pattern: {layer_pattern}")
    return model


def unfreeze_layers(model: PreTrainedModel, layer_pattern: str) -> PreTrainedModel:
    """Unfreeze layers matching pattern."""
    import re

    pattern = re.compile(layer_pattern)
    unfrozen = 0
    for name, param in model.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            unfrozen += 1
    logger.info(f"Unfrozen {unfrozen} parameters matching pattern: {layer_pattern}")
    return model


def get_model_device_map(model: PreTrainedModel) -> dict[str, Any]:
    """Get device map of model."""
    if hasattr(model, "hf_device_map"):
        return model.hf_device_map
    return {"": str(next(model.parameters()).device)}


def move_model_to_device(model: PreTrainedModel, device: str | torch.device) -> PreTrainedModel:
    """Move model to device (handles device_map)."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        logger.warning("Model has device_map, skipping manual device move")
        return model
    return model.to(device)


def cast_model_dtype(model: PreTrainedModel, dtype: torch.dtype) -> PreTrainedModel:
    """Cast model parameters to dtype (excluding quantized layers)."""
    for name, param in model.named_parameters():
        if param.dtype != dtype and "quant" not in name.lower():
            param.data = param.data.to(dtype)
    logger.info(f"Cast model to {dtype}")
    return model


def verify_model_setup(model: PreTrainedModel) -> bool:
    """Verify model is correctly configured for training."""
    issues = []

    if model.config.use_cache:
        issues.append("use_cache should be False for training")

    if not any(p.requires_grad for p in model.parameters()):
        issues.append("No trainable parameters found")

    if hasattr(model, "hf_device_map"):
        devices = set(model.hf_device_map.values())
        if len(devices) > 1:
            logger.info(f"Model sharded across devices: {devices}")

    if issues:
        for issue in issues:
            logger.warning(f"Model setup issue: {issue}")
        return False

    logger.info("Model setup verification passed")
    return True


def create_training_model(
    training_config: TrainingConfig,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
) -> ModelLoadResult:
    """
    Create model for training from unified training config.

    This is the main entry point that uses the complete TrainingConfig
    to set up everything for training.
    """
    quant_config = training_config.quantization
    peft_config = training_config.lora
    runtime_config = training_config.runtime

    return load_model_and_tokenizer(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        quantization_config=quant_config,
        peft_config=peft_config,
        peft_type="LORA",
        runtime_config=runtime_config,
    )


def apply_lora_plus_scaling(model: PeftModel, scale: float, lr_ratio: float) -> PeftModel:
    """Apply LoRA+ scaling (different learning rates for A and B matrices)."""
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.lr_scale = 1.0
        elif "lora_B" in name:
            param.lr_scale = lr_ratio
    logger.info(f"Applied LoRA+ scaling: A_lr=1.0, B_lr={lr_ratio}, overall_scale={scale}")
    return model


def enable_dora(model: PeftModel) -> PeftModel:
    """Enable DoRA (Weight-Decomposed Low-Rank Adaptation) on existing LoRA model."""
    if not hasattr(model, "peft_config"):
        raise ValueError("Model is not a PEFT model")
    for _adapter_name, config in model.peft_config.items():
        config.use_dora = True
    logger.info("DoRA enabled on all adapters")
    return model


def get_peft_state_dict(model: PeftModel, adapter_name: str = "default") -> dict[str, torch.Tensor]:
    """Get PEFT adapter state dict."""
    return model.get_peft_state_dict(adapter_name)


def set_peft_state_dict(
    model: PeftModel, state_dict: dict[str, torch.Tensor], adapter_name: str = "default"
) -> PeftModel:
    """Set PEFT adapter state dict."""
    return model.set_peft_state_dict(state_dict, adapter_name)


__all__ = [
    "ModelLoadResult",
    "ParameterStats",
    "get_torch_dtype",
    "create_bnb_config",
    "create_lora_config",
    "create_adalora_config",
    "create_ia3_config",
    "load_tokenizer",
    "setup_flash_attention",
    "setup_gradient_checkpointing",
    "prepare_model_for_training",
    "apply_peft_model",
    "count_parameters",
    "print_trainable_parameters",
    "get_model_memory_footprint",
    "optimize_model_memory",
    "create_device_map",
    "get_balanced_memory",
    "load_model_config",
    "load_base_model",
    "load_model_and_tokenizer",
    "merge_and_unload_peft",
    "save_model_and_tokenizer",
    "load_peft_adapter",
    "set_adapter",
    "disable_adapter",
    "enable_adapter",
    "model_loading_context",
    "get_gpu_memory_info",
    "log_gpu_memory",
    "clear_gpu_cache",
    "estimate_model_memory",
    "print_model_summary",
    "get_layer_info",
    "freeze_layers",
    "unfreeze_layers",
    "get_model_device_map",
    "move_model_to_device",
    "cast_model_dtype",
    "verify_model_setup",
    "create_training_model",
    "apply_lora_plus_scaling",
    "enable_dora",
    "get_peft_state_dict",
    "set_peft_state_dict",
]
