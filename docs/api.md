# API Reference

## Project: LLM Fine-Tuning Pipeline

## Version: 1.0.0

---

## 1. Core Modules

### 1.1 ConfigManager (`src.config`)

```python
from src.config import ConfigManager, TrainingConfig, EvaluationConfig, ModelConfig, DataConfig, LoggingConfig

# Load all configurations
config_manager = ConfigManager(
    config_dir="configs",
    env_file=".env"
)

# Access typed configs
training_config: TrainingConfig = config_manager.training
eval_config: EvaluationConfig = config_manager.evaluation
model_config: ModelConfig = config_manager.model
data_config: DataConfig = config_manager.data
logging_config: LoggingConfig = config_manager.logging

# Override programmatically
config_manager.update(training={"learning_rate": 1e-4}, model={"lora_r": 128})

# Export resolved config
config_manager.save_resolved("configs/resolved.yaml")
```

#### ConfigManager

| Method | Description |
|--------|-------------|
| `__init__(config_dir, env_file)` | Initialize with config directory and env file |
| `training` | Property: TrainingConfig |
| `evaluation` | Property: EvaluationConfig |
| `model` | Property: ModelConfig |
| `data` | Property: DataConfig |
| `logging` | Property: LoggingConfig |
| `update(**kwargs)` | Update config sections programmatically |
| `save_resolved(path)` | Save fully resolved config to YAML |

---

### 1.2 Logger (`src.logger`)

```python
from src.logger import Logger, LogLevel, setup_logging

# Setup global logging
logger = setup_logging(
    config=logging_config,
    run_name="my-training-run",
    log_dir="./logs"
)

# Or use directly
logger = Logger(
    name="training",
    level=LogLevel.INFO,
    log_file="./logs/training.log",
    json_format=True,
    tensorboard_dir="./logs/tensorboard",
    wandb_config={"project": "llm-finetuning", "name": "run-1"}
)

# Logging methods
logger.info("Training started", extra={"step": 0, "epoch": 0})
logger.debug("Batch processed", extra={"batch_size": 4, "loss": 2.34})
logger.warning("GPU memory high", extra={"memory_gb": 22.5})
logger.error("Training failed", exc_info=True, extra={"error": "OOM"})

# Metrics logging
logger.log_metrics({
    "train/loss": 2.34,
    "train/learning_rate": 2e-4,
    "train/grad_norm": 0.85
}, step=1000)

# Artifact logging
logger.log_artifact("./adapters/best", name="best-adapter", type="model")
logger.log_artifact("./evaluation_report.html", name="eval-report", type="report")

# Context manager for timing
with logger.timer("forward_pass"):
    outputs = model(input_ids)

logger.close()
```

#### Logger

| Method | Description |
|--------|-------------|
| `debug(msg, **kwargs)` | Debug level logging |
| `info(msg, **kwargs)` | Info level logging |
| `warning(msg, **kwargs)` | Warning level logging |
| `error(msg, **kwargs)` | Error level logging |
| `critical(msg, **kwargs)` | Critical level logging |
| `log_metrics(metrics, step)` | Log metrics dict to all backends |
| `log_artifact(path, name, type)` | Log artifact to W&B |
| `timer(name)` | Context manager for timing |
| `close()` | Flush and close all handlers |

---

### 1.3 DataPipeline (`src.data_pipeline`)

```python
from src.data_pipeline import DataPipeline, DataConfig

pipeline = DataPipeline(config=data_config)

# Full pipeline
dataset = pipeline.run(
    dataset_name="alpaca",
    dataset_path="data/raw/alpaca_data.json",
    output_dir="data/processed",
    formats=["arrow", "jsonl"],
    split_ratios={"train": 0.9, "val": 0.05, "test": 0.05}
)

# Individual stages
dataset = pipeline.download("tatsu-lab/alpaca", "data/raw")
dataset = pipeline.validate(dataset, schema="alpaca")
dataset = pipeline.clean(dataset, remove_duplicates=True, remove_nulls=True)
dataset = pipeline.format(dataset, template="alpaca")
dataset = pipeline.tokenize(dataset, tokenizer, max_length=2048)
dataset = pipeline.split(dataset, ratios={"train": 0.9, "val": 0.1})
pipeline.save(dataset, "data/processed", formats=["arrow"])

# Statistics
stats = pipeline.get_statistics(dataset)
print(f"Samples: {stats['num_samples']}")
print(f"Avg tokens: {stats['avg_tokens']}")
print(f"Token distribution: {stats['token_length_percentiles']}")
```

#### DataPipeline

| Method | Description |
|--------|-------------|
| `run(...)` | Execute full pipeline |
| `download(source, output_dir)` | Download dataset |
| `validate(dataset, schema)` | Validate schema & stats |
| `clean(dataset, ...)` | Deduplicate, remove nulls |
| `format(dataset, template)` | Apply prompt template (alpaca/chatml/vicuna) |
| `tokenize(dataset, tokenizer)` | Tokenize with stats |
| `split(dataset, ratios)` | Train/val/test split |
| `save(dataset, dir, formats)` | Save to JSONL/Arrow/Parquet |
| `get_statistics(dataset)` | Compute dataset statistics |

---

### 1.4 ModelManager (`src.model_utils`)

```python
from src.model_utils import ModelManager, ModelConfig

manager = ModelManager(config=model_config)

# Load model with QLoRA
model, tokenizer = manager.load_model(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True
    },
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# Apply LoRA
model = manager.apply_lora(
    model,
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    use_rslora=True
)

# Print trainable parameters
manager.print_trainable_parameters(model)

# Save adapter
manager.save_adapter(model, "./adapters/best", tokenizer=tokenizer)

# Merge adapter for deployment
merged_model = manager.merge_adapter(
    base_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="./adapters/best",
    output_path="./artifacts/models/merged/v1.0.0",
    dtype="bfloat16"
)

# Load merged model
merged_model, tokenizer = manager.load_merged_model("./artifacts/models/merged/v1.0.0")
```

#### ModelManager

| Method | Description |
|--------|-------------|
| `load_model(name, **kwargs)` | Load base model with quantization |
| `apply_lora(model, **kwargs)` | Inject LoRA adapters |
| `print_trainable_parameters(model)` | Print trainable/total params |
| `save_adapter(model, path, tokenizer)` | Save LoRA adapter |
| `load_adapter(model, path)` | Load LoRA adapter |
| `merge_adapter(base, adapter, output, dtype)` | Merge adapter into base |
| `load_merged_model(path)` | Load merged model |
| `get_device_map(model, max_memory)` | Auto device mapping |

---

### 1.5 TrainingPipeline (`src.train`)

```python
from src.train import TrainingPipeline, TrainingConfig

pipeline = TrainingPipeline(config=training_config)

# Prepare everything
trainer = pipeline.prepare(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        LoggingCallback(logger),
        CheckpointCallback(save_dir="./checkpoints")
    ]
)

# Train
train_result = pipeline.train(
    resume_from_checkpoint="./checkpoints/checkpoint-1000"
)

# Evaluate during training
eval_results = pipeline.evaluate()

# Save final model
pipeline.save_model("./adapters/final", tokenizer)

# Get training state
state = pipeline.get_state()
print(f"Best metric: {state.best_metric}")
print(f"Global step: {state.global_step}")
```

#### TrainingPipeline

| Method | Description |
|--------|-------------|
| `prepare(model, tokenizer, train_dataset, eval_dataset, callbacks)` | Setup SFTTrainer |
| `train(resume_from_checkpoint)` | Run training loop |
| `evaluate()` | Run evaluation |
| `save_model(path, tokenizer)` | Save adapter |
| `get_state()` | Get TrainerState |
| `push_to_hub(repo_id)` | Push to Hugging Face Hub |

---

### 1.6 EvaluationPipeline (`src.evaluate`)

```python
from src.evaluate import EvaluationPipeline, EvaluationConfig

pipeline = EvaluationPipeline(config=eval_config)

# Evaluate single model
results = pipeline.evaluate_model(
    model=model,
    tokenizer=tokenizer,
    datasets=["alpaca_eval", "mt_bench", "gsm8k"],
    output_dir="./outputs/evaluation"
)

# Compare base vs fine-tuned
comparison = pipeline.compare_models(
    base_model=base_model,
    ft_model=ft_model,
    tokenizer=tokenizer,
    datasets=["alpaca_eval", "mt_bench"],
    output_dir="./outputs/comparison"
)

# Generate report
report_path = pipeline.generate_report(
    results=comparison,
    output_path="./artifacts/evaluation_report.html",
    format="html"
)

# Access metrics
print(comparison["alpaca_eval"]["rougeL"])
print(comparison["mt_bench"]["score"])
print(comparison["gsm8k"]["accuracy"])
```

#### EvaluationPipeline

| Method | Description |
|--------|-------------|
| `evaluate_model(model, tokenizer, datasets, output_dir)` | Evaluate single model |
| `compare_models(base_model, ft_model, tokenizer, datasets, output_dir)` | Compare two models |
| `generate_report(results, output_path, format)` | Generate HTML/MD report |
| `compute_metrics(predictions, references, metrics)` | Compute specific metrics |
| `run_generation(model, tokenizer, prompts, gen_config)` | Batched generation |

---

### 1.7 InferenceEngine (`src.inference`)

```python
from src.inference import InferenceEngine, GenerationConfig

engine = InferenceEngine(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    use_bf16=True
)

# Single generation
output = engine.generate(
    prompt="Explain quantum computing:",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Batched generation
outputs = engine.generate_batch(
    prompts=["Prompt 1:", "Prompt 2:", "Prompt 3:"],
    batch_size=4,
    max_new_tokens=512,
    temperature=0.7
)

# Streaming generation
for token in engine.generate_stream(
    prompt="Write a story:",
    max_new_tokens=512
):
    print(token, end="", flush=True)

# Chat template
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's the weather?"}
]
output = engine.chat(messages, max_new_tokens=256)

# Benchmark
bench = engine.benchmark(
    prompts=["Benchmark prompt:"] * 10,
    max_new_tokens=128,
    batch_sizes=[1, 2, 4, 8]
)
print(f"Throughput: {bench['tokens_per_sec']} tok/s")
print(f"Latency P50: {bench['latency_p50']} ms")
```

#### InferenceEngine

| Method | Description |
|--------|-------------|
| `generate(prompt, **gen_kwargs)` | Single prompt generation |
| `generate_batch(prompts, batch_size, **gen_kwargs)` | Batched generation |
| `generate_stream(prompt, **gen_kwargs)` | Streaming token generator |
| `chat(messages, **gen_kwargs)` | Chat template generation |
| `benchmark(prompts, batch_sizes, max_new_tokens)` | Throughput/latency benchmark |
| `get_memory_usage()` | Current GPU memory usage |

---

### 1.8 MetricsComputer (`src.metrics`)

```python
from src.metrics import MetricsComputer, compute_all_metrics

computer = MetricsComputer(device="cuda")

# Individual metrics
rouge = computer.rouge(predictions, references)
bleu = computer.bleu(predictions, references)
bertscore = computer.bertscore(predictions, references)
perplexity = computer.perplexity(model, tokenizer, texts)
distinct = computer.distinct_n(predictions, n=4)

# All at once
metrics = compute_all_metrics(
    predictions=predictions,
    references=references,
    model=model,
    tokenizer=tokenizer,
    texts=texts,
    device="cuda",
    metrics=["rouge", "bleu", "bertscore", "perplexity", "distinct"]
)

# Output
{
    "rouge1": 0.45,
    "rouge2": 0.22,
    "rougeL": 0.41,
    "bleu": 0.18,
    "bertscore_precision": 0.89,
    "bertscore_recall": 0.87,
    "bertscore_f1": 0.88,
    "perplexity": 12.34,
    "distinct_1": 0.92,
    "distinct_2": 0.87,
    "distinct_3": 0.81,
    "distinct_4": 0.75
}
```

#### MetricsComputer

| Method | Description |
|--------|-------------|
| `rouge(preds, refs)` | ROUGE-1/2/L/Lsum |
| `bleu(preds, refs)` | BLEU-1/2/3/4 |
| `bertscore(preds, refs)` | BERTScore P/R/F1 |
| `perplexity(model, tokenizer, texts)` | Perplexity |
| `distinct_n(preds, n)` | Distinct-n diversity |
| `semantic_similarity(preds, refs)` | Embedding similarity |
| `toxicity(texts)` | Toxicity scores |

---

## 2. Callback Classes (`src.callbacks`)

### 2.1 LoggingCallback

```python
from src.callbacks import LoggingCallback

callback = LoggingCallback(
    logger=logger,
    log_every=10,
    log_metrics=["loss", "learning_rate", "grad_norm", "gpu_memory"]
)
```

### 2.2 CheckpointCallback

```python
from src.callbacks import CheckpointCallback

callback = CheckpointCallback(
    save_dir="./checkpoints",
    save_every=500,
    keep_last=3,
    save_best=True,
    metric_for_best="eval_loss",
    greater_is_better=False
)
```

### 2.3 EarlyStoppingCallback

```python
from src.callbacks import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
    metric_for_best="eval_loss",
    greater_is_better=False
)
```

### 2.4 ProfilerCallback

```python
from src.callbacks import ProfilerCallback

callback = ProfilerCallback(
    profile_every=100,
    profile_dir="./logs/profiler",
    activities=["cpu", "cuda"],
    record_shapes=True,
    with_stack=True
)
```

---

## 3. Utility Functions (`src.utils`)

```python
from src.utils import (
    set_seed,
    get_device,
    count_parameters,
    format_prompt,
    truncate_sequence,
    get_gpu_memory,
    format_time,
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    get_git_sha,
    get_git_diff
)

# Reproducibility
set_seed(42)

# Device
device = get_device()  # "cuda", "mps", "cpu"

# Model info
trainable, total = count_parameters(model)
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Prompt formatting
prompt = format_prompt(
    instruction="Write a poem",
    input="about cats",
    template="alpaca"  # or "chatml", "vicuna", "llama3"
)

# GPU memory
allocated, reserved = get_gpu_memory()
print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Time formatting
print(format_time(3661))  # "1h 1m 1s"

# Git info
sha = get_git_sha()
diff = get_git_diff()
```

---

## 4. Configuration Classes

All configuration classes are Pydantic models with validation.

### 4.1 TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Trainer
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    dataloader_num_workers: int = 4
    report_to: List[str] = ["tensorboard", "wandb"]
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = [...]
    use_rslora: bool = True
    
    # QLoRA
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # SFT
    max_seq_length: int = 2048
    packing: bool = False
```

---

## 5. CLI Entry Points

```bash
# Training
python -m src.train --config configs/training.yaml --resume_from_checkpoint ./checkpoints/checkpoint-1000

# Evaluation
python -m src.evaluate --config configs/evaluation.yaml --model_path ./adapters/best --base_model meta-llama/Meta-Llama-3-8B-Instruct

# Inference
python -m src.inference --model_path ./adapters/best --prompt "Hello!" --max_new_tokens 256

# Data preparation
python -m src.data_pipeline --config configs/data.yaml --dataset alpaca --output_dir ./data/processed

# Merge adapter
python -m src.model_utils --merge --base_model meta-llama/Meta-Llama-3-8B-Instruct --adapter_path ./adapters/best --output_path ./artifacts/models/merged/v1.0.0
```

---

## 6. Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | Hugging Face token | For private models |
| `WANDB_API_KEY` | Weights & Biases API key | For W&B logging |
| `WANDB_PROJECT` | W&B project name | Optional |
| `WANDB_ENTITY` | W&B entity/team | Optional |
| `OPENAI_API_KEY` | OpenAI API key | For MT-Bench judge |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | Optional |
| `TOKENIZERS_PARALLELISM` | Tokenizer parallelism | `false` recommended |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA memory config | Optional |

---

*End of API Reference*