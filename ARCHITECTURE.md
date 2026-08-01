# Architecture Documentation

## System Overview

The LLM Fine-Tuning Pipeline is a modular, configuration-driven system for training, evaluating, and deploying Large Language Models using QLoRA (Quantized Low-Rank Adaptation). The architecture follows clean separation of concerns with explicit interfaces between components.

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONFIGURATION LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  data.yaml  │  │training.yaml│  │  model.yaml │  │logging.yaml │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONFIG MANAGER (Pydantic)                          │
│  • YAML loading with env var resolution                                      │
│  • Type validation & defaults                                                │
│  • Section accessors: .data, .training, .model, .logging, .evaluation       │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  DATA PIPELINE  │   │  MODEL UTILS    │   │   TRAINER       │
│                 │   │                 │   │                 │
│ • Download      │   │ • Load base     │   │ • SFTTrainer    │
│ • Validate      │   │ • QLoRA config  │   │ • Callbacks     │
│ • Clean         │   │ • PEFT apply    │   │ • Checkpointing │
│ • Format        │   │ • Device map    │   │ • Logging       │
│ • Tokenize      │   │ • Memory opt    │   │ • Resume        │
│ • Split         │   │ • Merge/save    │   │ • Distributed   │
│ • Export        │   │                 │   │                 │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                       │
         └─────────────────────┼───────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT TRACKING                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                          │
│  │   W&B       │  │ TensorBoard │  │  MLflow     │                          │
│  └─────────────┘  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION ENGINE                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Generation  │  │  Metrics    │  │ Benchmark   │  │  Reporting  │         │
│  │ (Base/FT)   │  │ (ROUGE,     │  │ (Latency,   │  │ (Tables,    │         │
│  │             │  │  BLEU,      │  │  Memory,    │  │  CSV, JSON) │         │
│  │             │  │  BERTScore, │  │  Throughput)│  │             │         │
│  │             │  │  Perplexity)│  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. Configuration Manager (`src/config.py`)

**Responsibilities:**
- Load YAML configs from `configs/` directory
- Resolve environment variables (`${VAR_NAME}`)
- Pydantic validation with type coercion
- Programmatic updates with deep merge
- Export resolved configs

**Key Classes:**
- `ConfigManager` - Main entry point
- `TrainingConfig` - Complete training configuration
- `DataConfigComplete` - Data pipeline configuration
- `ModelConfigComplete` - Model, tokenizer, PEFT, quantization
- `LoggingConfigComplete` - All logging backends
- `EvaluationConfigComplete` - Generation, metrics, datasets

**Configuration Flow:**
```python
config_manager = ConfigManager(config_dir="configs")
training_config = config_manager.training
data_config = config_manager.data
model_config = config_manager.model

# Programmatic override
config_manager.update(
    training={"trainer": {"learning_rate": 1e-4}},
    model={"model": {"model_name_or_path": "custom/model"}}
)
```

---

### 2. Data Pipeline (`src/data_pipeline.py`)

**Responsibilities:**
- Dataset acquisition from HF Hub or local files
- Validation, cleaning, deduplication
- Prompt formatting with multiple templates
- Tokenization with statistics
- Train/validation/test splitting
- Multi-format export

**Processing Stages:**

| Stage | Class/Function | Config Section |
|-------|----------------|----------------|
| Download | `download_datasets()` | `processing.download` |
| Validate | `validate_dataset()` | `processing.validation` |
| Clean | `clean_dataset()` | `processing.cleaning` |
| Format | `format_dataset()` | `processing.formatting` |
| Tokenize | `tokenize_dataset()` | `processing.tokenization` |
| Statistics | `compute_statistics()` | `processing.statistics` |
| Split | `split_dataset()` | `processing.splitting` |
| Export | `export_dataset()` | `output.formats` |

**Prompt Formatters (Strategy Pattern):**
- `AlpacaFormatter` - `### Instruction:/Input:/Response:`
- `ChatMLFormatter` - `<|im_start|>system/user/assistant<|im_end|>`
- `Llama3Formatter` - `<|begin_of_text|><|start_header_id|>...`
- `VicunaFormatter` - `USER: ... ASSISTANT: ...`
- `ZephyrFormatter` - `<|system|>...<|user|>...<|assistant|>...`
- `PlainFormatter` - Raw text for pretraining
- `CustomFormatter` - User-defined templates

**Output Formats:**
- Arrow (HuggingFace native, fast)
- JSONL (human-readable, streaming)
- Parquet (columnar, compressed)

---

### 3. Model Utilities (`src/model_utils.py`)

**Responsibilities:**
- Base model loading with quantization
- PEFT/LoRA configuration and application
- Device mapping and memory optimization
- Tokenizer loading with chat templates
- Model merging and saving
- Parameter counting and reporting

**Quantization (BitsAndBytesConfig):**
```python
# 4-bit NF4 (default)
load_in_4bit=True
bnb_4bit_quant_type="nf4"        # NormalFloat4
bnb_4bit_compute_dtype=bfloat16
bnb_4bit_use_double_quant=True   # Double quantization

# 8-bit alternative
load_in_8bit=True
llm_int8_threshold=6.0
```

**PEFT Configurations:**
```python
# LoRA (default)
LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05, use_rslora=True)

# AdaLoRA (adaptive rank)
AdaLoraConfig(target_r=8, init_r=12, tinit=0, tfinal=0, deltaT=10)

# IA3 (fewer parameters)
IA3Config(target_modules=["k_proj", "v_proj", "down_proj"])
```

**Memory Optimizations:**
- Gradient checkpointing (`use_reentrant=False`)
- Flash Attention 2/3
- CPU offloading for optimizer states
- Empty cache / GC hooks
- Low CPU memory usage loading

**Device Mapping:**
- `auto` - Accelerate infer_auto_device_map
- `balanced` - Even layer distribution
- `sequential` - Layer-by-layer
- Custom dict per layer

---

### 4. Trainer (`src/train.py`)

**Responsibilities:**
- SFTTrainer / Trainer instantiation
- TrainingArguments construction
- Callback management
- Experiment tracking setup
- Training loop execution
- Checkpoint management
- Model merging and export

**Callbacks:**
| Callback | Purpose | Config |
|----------|---------|--------|
| `EarlyStoppingCallback` | Stop on metric plateau | `callbacks.early_stopping` |
| `GradientNormCallback` | Log gradient norms | `callbacks.logging.log_grad_norm` |
| `GPUMemoryCallback` | Log GPU memory | `callbacks.logging.log_gpu_memory` |
| `LearningRateCallback` | Log LR schedule | `callbacks.logging.log_learning_rate` |
| `ThroughputCallback` | Log samples/sec | `callbacks.logging` |
| `ProfilerCallback` | PyTorch profiler | `callbacks.profiler` |

**Experiment Tracking:**
```python
# Weights & Biases
report_to=["wandb"]
wandb.init(project="llm-finetuning", name=run_name, config=config)

# TensorBoard
report_to=["tensorboard"]
logging_dir="./logs/tensorboard"

# MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("llm-finetuning")
```

**Checkpointing:**
- Save strategy: steps/epochs
- Save total limit: keep N best
- Load best model at end
- Safetensors format
- Resume from checkpoint

**Distributed Training:**
- DDP (nccl backend)
- FSDP (sharding)
- DeepSpeed (ZeRO-3)
- Gradient accumulation synchronization

---

### 5. Evaluation (`src/evaluate.py`)

**Responsibilities:**
- Load base and fine-tuned models
- Generate responses on evaluation datasets
- Compute NLP metrics
- Benchmark performance
- Generate comparison reports

**Metrics Pipeline:**
```
Predictions + References
        │
        ▼
┌───────────────────┐
│ MetricsCalculator │
├───────────────────┤
│ • ROUGE           │
│ • BLEU            │
│ • METEOR          │
│ • BERTScore       │
│ • Perplexity      │
│ • Distinct-n      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Benchmark         │
├───────────────────┤
│ • Latency (ms)    │
│ • Throughput      │
│ • Memory (MB)     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Report Generator  │
├───────────────────┤
│ • Markdown table  │
│ • JSON details    │
│ • CSV summary     │
└───────────────────┘
```

**Generation Config:**
```python
GenerationConfig(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    do_sample=True,
    num_beams=1,
    early_stopping=True
)
```

---

### 6. Inference (`src/inference.py`)

**Responsibilities:**
- FastAPI server for production inference
- Batched generation with continuous batching
- Streaming responses (SSE)
- Prometheus metrics
- Health checks

**API Endpoints:**
```
POST   /generate           - Single completion
POST   /generate/batch     - Batch completions
POST   /chat               - Chat completion (OpenAI compatible)
GET    /health             - Health check
GET    /metrics            - Prometheus metrics
```

**Features:**
- Continuous batching (vLLM-style)
- Prefix caching
- Speculative decoding support
- Tensor parallelism
- Dynamic batching

---

## Data Flow

### Training Flow
```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│ Config  │───▶│  Data    │───▶│  Model   │───▶│  Trainer │───▶│ Check-  │
│ Manager │    │ Pipeline │    │  Loader  │    │ (SFTTrainer)    │ points │
└─────────┘    └──────────┘    └──────────┘    └──────────┘    └─────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │ Experiment   │
                                            │ Tracking     │
                                            └──────────────┘
```

### Evaluation Flow
```
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│ Config  │───▶│  Load    │───▶│  Generate    │───▶│ Metrics  │
│ Manager │    │  Models  │    │  (Base + FT) │    │ Calculator│
└─────────┘    └──────────┘    └──────────────┘    └────┬─────┘
                                                         │
                                                         ▼
                                                ┌──────────────┐
                                                │  Reporting   │
                                                │ (Tables,     │
                                                │  JSON, CSV)  │
                                                └──────────────┘
```

---

## Configuration Schema

### Training Configuration (`configs/training.yaml`)
```yaml
trainer:                    # TrainingArguments
  output_dir: "./checkpoints"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  bf16: true
  gradient_checkpointing: true
  save_strategy: "steps"
  save_steps: 100
  evaluation_strategy: "steps"
  eval_steps: 100
  logging_steps: 10
  report_to: ["tensorboard", "wandb"]

lora:                       # LoRAConfig
  r: 64
  lora_alpha: 16
  lora_dropout: 0.05
  use_rslora: true
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

quantization:               # QuantizationConfig
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

callbacks:                  # CallbacksConfig
  early_stopping:
    enabled: true
    patience: 3
    threshold: 0.001
  logging:
    log_steps: 10
    log_gpu_memory: true
```

### Data Configuration (`configs/data.yaml`)
```yaml
datasets:
  - name: "alpaca"
    path: "tatsu-lab/alpaca"
    split: "train"
    max_samples: null
    column_mapping:
      instruction: "instruction"
      input: "input"
      output: "output"

prompt_templates:
  alpaca:
    template: "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    template_no_input: "### Instruction:\n{instruction}\n\n### Response:\n{output}"
  chatml:
    template: "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

processing:
  download:
    cache_dir: "./data/raw"
    num_proc: 4
  validation:
    enabled: true
    required_columns: ["instruction", "output"]
    min_instruction_length: 10
  cleaning:
    enabled: true
    strip_whitespace: true
    normalize_unicode: true
  formatting:
    enabled: true
    template: "alpaca"
    add_eos_token: true
  tokenization:
    enabled: true
    max_seq_length: 2048
  splitting:
    enabled: true
    ratios: {train: 0.9, validation: 0.05, test: 0.05}
    seed: 42

output:
  output_dir: "./data/processed"
  formats: ["arrow", "jsonl"]
```

---

## Extension Points

### Custom Prompt Template
```python
# In data.yaml
prompt_templates:
  my_template:
    template: "Instruction: {instruction}\nOutput: {output}"
    template_with_input: "Instruction: {instruction}\nInput: {input}\nOutput: {output}"

# Use in formatting
formatting:
  template: "my_template"
```

### Custom Metric
```python
# In src/metrics.py
class CustomMetric:
    def compute(self, predictions, references):
        # Implementation
        return {"custom_score": score}

# Register in evaluation.yaml
metrics:
  custom:
    enabled: true
    class: "src.metrics.CustomMetric"
```

### Custom Callback
```python
# In src/callbacks.py
class CustomCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Custom logic

# Register in training.yaml
callbacks:
  custom_callbacks:
    - "src.callbacks.CustomCallback"
```

---

## Performance Considerations

### Memory Optimization Checklist
- [ ] 4-bit quantization enabled
- [ ] Double quantization enabled
- [ ] Gradient checkpointing enabled
- [ ] Flash Attention 2 enabled
- [ ] CPU offloading for optimizer (if OOM)
- [ ] Low CPU memory usage loading
- [ ] Gradient accumulation for large effective batch

### Speed Optimization Checklist
- [ ] Flash Attention 2 (H100/A100)
- [ ] Torch compile (inductor backend)
- [ ] TF32 enabled (Ampere+)
- [ ] Dataloader prefetch_factor=2
- [ ] Persistent workers
- [ ] Pin memory
- [ ] Optimal batch size per GPU

### Multi-GPU Scaling
| GPUs | Strategy | Expected Speedup |
|------|----------|------------------|
| 1    | Single   | 1.0x             |
| 2    | DDP      | ~1.8x            |
| 4    | DDP      | ~3.5x            |
| 8    | DDP/FSDP | ~6.5x            |

---

## Security Considerations

- Model weights loaded with `trust_remote_code=True` only when necessary
- Authentication tokens via environment variables (never in config)
- Safetensors format for safe model loading
- No code execution in data pipeline
- Input validation on all user-facing endpoints