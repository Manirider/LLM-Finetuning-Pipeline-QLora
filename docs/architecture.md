# Architecture Documentation

## Overview

The LLM Fine-Tuning Pipeline is a modular, production-ready system for training, evaluating, and deploying Large Language Models using QLoRA (Quantized Low-Rank Adaptation). The architecture emphasizes separation of concerns, configuration-driven behavior, and extensibility.

---

## Design Principles

1. **Configuration-First**: All behavior controlled via YAML configs with Pydantic validation
2. **Modularity**: Each component (data, model, training, evaluation) is independently testable
3. **Reproducibility**: Deterministic seeding, versioned configs, artifact tracking
4. **Scalability**: Supports single-GPU to multi-node distributed training
5. **Observability**: Structured logging, experiment tracking, profiling

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION LAYER                       │
│  configs/data.yaml  │  configs/training.yaml  │  configs/...    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONFIG MANAGER                              │
│  • YAML parsing + env var resolution                            │
│  • Pydantic validation                                          │
│  • Section accessors (.data, .training, .model, .logging)       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌───────────────┐ ┌───────────┐ ┌───────────┐
│ DATA PIPELINE │ │ MODEL     │ │ TRAINER   │
│               │ │ UTILS     │ │           │
│ • Load        │ │ • QLoRA   │ │ • SFT     │
│ • Validate    │ │ • PEFT    │ │ • CBs     │
│ • Clean       │ │ • Device  │ │ • Log     │
│ • Format      │ │ • Merge   │ │ • Ckpt    │
│ • Tokenize    │ │           │ │           │
│ • Split       │ │           │ │           │
└───────┬───────┘ └─────┬─────┘ └─────┬─────┘
        │               │             │
        └───────────────┼─────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXPERIMENT TRACKING                            │
│  W&B  │  TensorBoard  │  MLflow  │  Console  │  File (JSON)    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION ENGINE                             │
│  Generation  │  Metrics (ROUGE, BLEU, BERTScore, Perplexity)    │
│  Benchmark   │  Reporting (Tables, CSV, JSON, Markdown)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Details

### 1. Configuration Manager (`src/config.py`)

**Purpose**: Centralized configuration with type safety and environment overrides.

**Key Features**:
- Loads 5 YAML files: `data.yaml`, `training.yaml`, `model.yaml`, `logging.yaml`, `evaluation.yaml`
- Environment variable resolution: `${VAR_NAME}` → `os.environ["VAR_NAME"]`
- Pydantic v2 models for each config section
- Deep merge for programmatic updates
- Export resolved config for reproducibility

**Config Sections**:
```python
config_manager = ConfigManager("configs")
config_manager.data       # DataConfigComplete
config_manager.training   # TrainingConfig
config_manager.model      # ModelConfigComplete
config_manager.logging    # LoggingConfigComplete
config_manager.evaluation # EvaluationConfigComplete
```

---

### 2. Data Pipeline (`src/data_pipeline.py`)

**Purpose**: End-to-end data processing from raw datasets to model-ready format.

**Processing Stages**:

| Stage | Function | Config Section |
|-------|----------|----------------|
| Download | `load_dataset()` with caching | `processing.download` |
| Validate | Required cols, length limits, language | `processing.validation` |
| Clean | Whitespace, unicode, HTML, duplicates | `processing.cleaning` |
| Format | Apply prompt template | `processing.formatting` |
| Tokenize | HF tokenizer, truncation, stats | `processing.tokenization` |
| Split | Random/sequential/stratified | `processing.splitting` |
| Export | Arrow/JSONL/Parquet | `output.formats` |

**Prompt Formatters**:
- `AlpacaFormatter` - `### Instruction/Input/Response`
- `ChatMLFormatter` - `im_start>system/user/assistant`
- `Llama3Formatter` - `<|begin_of_text|><|start_header_id|>`
- `VicunaFormatter` - `USER:/ASSISTANT:`
- `ZephyrFormatter` - `<|system|>/<|user|>/<|assistant|>`
- `PlainFormatter` - Raw text (pretraining)
- `CustomFormatter` - User-defined templates

**CLI Interface**:
```bash
python -m src.data_pipeline --config configs/data.yaml [options]
```

---

### 3. Model Utilities (`src/model_utils.py`)

**Purpose**: Model loading, QLoRA configuration, PEFT application, memory optimization.

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `create_bnb_config()` | BitsAndBytesConfig for 4-bit NF4 / 8-bit |
| `create_lora_config()` | PEFT LoraConfig from config |
| `load_tokenizer()` | Tokenizer with chat templates |
| `setup_flash_attention()` | Enable FA2/FA3 with fallback |
| `setup_gradient_checkpointing()` | Memory-efficient training |
| `prepare_model_for_training()` | k-bit training prep |
| `apply_peft_model()` | LoRA/AdaLoRA/IA3 application |
| `load_model_and_tokenizer()` | **Main entry point** |
| `merge_and_unload_peft()` | Merge adapter for inference |
| `save_model_and_tokenizer()` | Save with options |
| `count_parameters()` | Trainable/frozen/LoRA stats |
| `get_model_memory_footprint()` | Parameter/buffer memory |
| `optimize_model_memory()` | Combined optimizations |
| `create_device_map()` | Accelerate device mapping |
| `estimate_model_memory()` | Pre-load memory estimation |

**Memory Optimizations**:
- Gradient checkpointing (use_reentrant=False)
- Flash Attention 2/3
- CPU offloading for optimizer states
- Empty cache / GC hooks
- Low CPU memory loading
- bfloat16 compute dtype

---

### 4. Trainer (`src/train.py`)

**Purpose**: Orchestrate training with SFTTrainer, callbacks, experiment tracking.

**Key Components**:

**Custom Callbacks**:
- `GradientNormCallback` - Log gradient norms
- `GPUMemoryCallback` - Log VRAM usage
- `LearningRateCallback` - Log LR schedule
- `ThroughputCallback` - Steps/sec, tokens/sec
- `ProfilerCallback` - PyTorch profiler trace
- `EarlyStoppingCallback` - Patience-based stopping

**TrainingArguments Mapping**:
All `TrainerConfig` fields map to `TrainingArguments`:
- Batch size, grad accum, epochs, steps
- LR, scheduler, warmup, weight decay
- FP16/BF16/TF32
- Logging, evaluation, saving strategies
- DDP, FSDP, DeepSpeed config
- Torch compile options

**SFTConfig Mapping**:
- Max seq length, packing
- Dataset text field, formatting func
- NEFTune noise alpha

**CLI Interface**:
```bash
python -m src.train --config configs [options]
# --resume-from-checkpoint, --merge-and-save, --dry-run, --push-to-hub
```

---

### 5. Evaluation (`src/evaluate.py`)

**Purpose**: Comprehensive model evaluation with multiple metrics and benchmarking.

**Metrics Calculators**:

| Metric | Implementation | Config |
|--------|---------------|--------|
| ROUGE | `rouge_score.RougeScorer` | `RougeConfig` |
| BLEU | `nltk.translate.bleu_score` | `BleuConfig` |
| METEOR | `nltk.translate.meteor_score` | (enabled) |
| BERTScore | `bert_score.score` | `BertScoreConfig` |
| Perplexity | `perplexity` library | `PerplexityConfig` |
| Distinct-n | Custom n-gram counting | `DistinctConfig` |

**Benchmark Metrics**:
- Latency: avg, median, p95 (ms)
- Throughput: tokens/second
- Memory: avg, peak (MB)

**Prompt Formatters**: Same as data pipeline (Alpaca, ChatML, Llama3, etc.)

**Output Formats**:
- JSON: Raw predictions + metrics
- Markdown: Comparison tables
- CSV: Metrics summary

**CLI Interface**:
```bash
python -m src.evaluate \
  --base-model meta-llama/Llama-3-8B \
  --finetuned-model ./checkpoints/best \
  --output-dir ./eval_results
```

---

### 6. Inference (`src/inference.py`)

**Purpose**: Production-ready inference server with FastAPI.

**Features**:
- Async batching with configurable max batch size
- Streaming responses (SSE)
- OpenAI-compatible API
- Prometheus metrics
- Health checks
- Request validation

**Endpoints**:
- `POST /v1/completions` - Completions
- `POST /v1/chat/completions` - Chat
- `GET /health` - Health
- `GET /metrics` - Prometheus

---

## Data Flow

### Training Flow
```
1. ConfigManager.load("configs")
2. DataPipeline(config.data).process()
   → DatasetDict{train, validation, test}
3. ModelUtils.load_model_and_tokenizer(config.model, config.training)
   → PreTrainedModel, PreTrainedTokenizer
4. TrainArguments = create_training_arguments(config.training)
5. SFTConfig = create_sft_config(config.training.sft)
6. Callbacks = create_callbacks(config.training.callbacks)
7. SFTTrainer(model, tokenizer, train_ds, eval_ds, args, sft_config, callbacks)
7. trainer.train(resume_from_checkpoint)
8. Save best model / merge adapter
9. Experiment tracking cleanup
```

### Evaluation Flow
```
1. ConfigManager.load("configs")
2. Load base model + tokenizer
3. Load fine-tuned model (+ merge adapter)
4. For each eval dataset:
   a. Format prompts with template
   b. Generate responses (base + ft)
   c. Benchmark performance
   d. Calculate metrics
5. Generate comparison table
6. Save reports (JSON, MD, CSV)
```

---

## Configuration Schema

### Data Config (`configs/data.yaml`)
```yaml
datasets:
  - name: "alpaca"
    path: "tatsu-lab/alpaca"
    split: "train"
    column_mapping:
      instruction: "instruction"
      input: "input"
      output: "output"
prompt_templates:
  alpaca:
    template: "### Instruction:\n{instruction}..."
    template_with_input: "..."
default_template: "alpaca"
processing:
  download: {cache_dir, num_proc, max_retries}
  validation: {required_columns, min/max_length, check_duplicates}
  cleaning: {strip_whitespace, normalize_unicode, remove_duplicates}
  formatting: {template, system_message, add_eos_token}
  tokenization: {max_seq_length, truncation, padding}
  splitting: {ratios: {train: 0.9, validation: 0.05, test: 0.05}}
output:
  output_dir: "./data/processed"
  formats: ["arrow", "jsonl"]
  save_splits: true
```

### Training Config (`configs/training.yaml`)
```yaml
trainer:
  output_dir: "./checkpoints"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  eval_steps: 100
  save_steps: 100
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  report_to: ["tensorboard", "wandb"]
  run_name: "llm-finetuning-qlora"

lora:
  r: 64
  lora_alpha: 16
  lora_dropout: 0.05
  use_rslora: true
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

callbacks:
  early_stopping: {enabled: true, patience: 3}
  checkpoint: {enabled: true, save_steps: 100}
  logging: {enabled: true, log_steps: 10, log_gpu_memory: true}
  profiler: {enabled: false}
```

### Model Config (`configs/model.yaml`)
```yaml
model:
  model_name_or_path: "meta-llama/Meta-Llama-3-8B-Instruct"
  torch_dtype: "bfloat16"
  device_map: "auto"
  attn_implementation: "flash_attention_2"
  gradient_checkpointing: true
  use_cache: false
  load_in_4bit: true

tokenizer:
  tokenizer_name_or_path: null  # Uses model path
  padding_side: "right"
  model_max_length: 4096
  chat_template_name: "llama3"

peft:
  peft_type: "LORA"
  lora: {r: 64, lora_alpha: 16, ...}
```

---

## Extensibility Points

### Adding a New Prompt Template
1. Add class in `data_pipeline.py` extending `PromptFormatter`
2. Register in `FORMATTERS` dict
3. Add template config in `data.yaml`

### Adding a New Metric
1. Add config class in `config.py` (e.g., `NewMetricConfig`)
2. Add method in `MetricsCalculator` (e.g., `calculate_new_metric`)
3. Call in `calculate_all()`
4. Add to `evaluation.yaml`

### Adding a New Callback
1. Extend `TrainerCallback` in `train.py`
2. Add config in `training.yaml`
3. Register in `create_callbacks()`

### Adding a New Model Type
1. Add architecture config in `config.py`
2. Handle in `load_base_model()` in `model_utils.py`
3. Update target modules for LoRA in `training.yaml`

---

## Error Handling & Validation

| Layer | Validation |
|-------|------------|
| Config | Pydantic type checking, required fields, cross-field validation |
| Data | Column existence, length limits, null checks, duplicate detection |
| Model | Quantization config validation, device map inference |
| Training | Argument validation, callback compatibility |
| Evaluation | Metric dependency checks, dataset format validation |

---

## Performance Considerations

### Memory Optimization Checklist
- [ ] 4-bit NF4 quantization (`load_in_4bit: true`)
- [ ] Double quantization (`bnb_4bit_use_double_quant: true`)
- [ ] Gradient checkpointing (`gradient_checkpointing: true`, `use_reentrant: false`)
- [ ] Flash Attention 2 (`attn_implementation: "flash_attention_2"`)
- [ ] bfloat16 compute (`bnb_4bit_compute_dtype: "bfloat16"`)
- [ ] LoRA only trainable params (99%+ frozen)
- [ ] Gradient accumulation for large effective batch
- [ ] CPU offload for optimizer states if needed

### Speed Optimization Checklist
- [ ] `num_proc` > 1 for data processing
- [ ] `dataloader_num_workers` > 0
- [ ] `dataloader_pin_memory: true`
- [ ] `dataloader_persistent_workers: true`
- [ ] `prefetch_factor: 2`
- [ ] Flash Attention 2
- [ ] Torch compile (`torch_compile: true`, `mode: "max-autotune"`)
- [ ] Packed sequences (`packing: true` in SFTConfig)

---

## Security Considerations

- No hardcoded secrets (use `${ENV_VAR}`)
- Tokenizer `trust_remote_code` configurable
- Model `trust_remote_code` configurable
- Hub push requires explicit `push_to_hub: true`
- Local file access restricted to configured paths

---

## Testing Strategy

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Unit | `tests/unit/` | Config, formatters, metrics, callbacks |
| Integration | `tests/integration/` | Pipeline stages, model loading, training loop |
| Smoke | `tests/smoke/` | CLI commands, end-to-end minimal run |

Run: `pytest tests/ -v --cov=src`