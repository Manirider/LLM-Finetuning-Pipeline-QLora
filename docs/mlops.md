# MLOps Architecture

## Project: LLM Fine-Tuning Pipeline

## Version: 1.0.0

---

## 1. MLOps Overview

This document describes the MLOps architecture for the LLM fine-tuning pipeline, covering experiment tracking, model versioning, artifact management, monitoring, and deployment workflows.

---

## 2. Experiment Tracking

### 2.1 Dual Backend Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT TRACKING LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐          ┌─────────────────┐              │
│  │   TensorBoard   │          │  Weights&Biases │              │
│  │   (Local)       │          │   (Cloud)       │              │
│  ├─────────────────┤          ├─────────────────┤              │
│  │ • Loss curves   │          │ • Loss curves   │              │
│  │ • LR schedules  │          │ • LR schedules  │              │
│  │ • GPU metrics   │          │ • GPU metrics   │              │
│  │ • Histograms    │          │ • Artifacts     │              │
│  │ • Scalars       │          │ • Model registry│              │
│  │ • Offline ✓ No account needed          ✓ Collaboration        │
│  ✓ Zero config               ✓ Hyperparam sweeps   │
│                              ✓ Alerting           │
│                              ✓ Reports            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Logged Metrics

| Category | Metrics | Frequency |
|----------|---------|-----------|
| **Training** | `train/loss`, `train/learning_rate`, `train/grad_norm`, `train/tokens_per_sec` | Per step |
| **Validation** | `eval/loss`, `eval/rouge1`, `eval/rouge2`, `eval/rougeL`, `eval/bleu`, `eval/bertscore` | Per eval |
| **System** | `system/gpu_memory_allocated`, `system/gpu_memory_reserved`, `system/gpu_utilization`, `system/cpu_percent`, `system/ram_percent` | Per step |
| **Model** | `model/trainable_params`, `model/total_params`, `model/trainable_pct` | Once |

### 2.3 Logged Artifacts

| Artifact | Description | Retention |
|----------|-------------|-----------|
| `config.yaml` | Full resolved configuration | Forever |
| `adapter_config.json` | LoRA adapter configuration | Forever |
| `adapter_model.safetensors` | LoRA weights (best) | Forever |
| `checkpoint-*/` | Training checkpoints (last 3) | 30 days |
| `evaluation_report.html` | Full evaluation report | Forever |
| `predictions.jsonl` | Model predictions on eval sets | 90 days |

---

## 3. Model Versioning & Registry

### 3.1 Versioning Scheme

```
Model Version: {base_model}-{task}-v{major}.{minor}.{patch}-{git_sha}

Examples:
- meta-llama/Meta-Llama-3-8B-Instruct-alpaca-v1.0.0-a1b2c3d
- mistralai/Mistral-7B-Instruct-v0.2-code-v2.1.0-f4e5d6c
```

### 3.2 Model Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Train   │────▶│ Evaluate │────▶│  Stage   │────▶│ Deploy   │
│  (Dev)   │     │  (QA)    │     │ (Staging)│     │ (Prod)   │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
      │              │                │                │
      ▼              ▼                ▼                ▼
  Checkpoint    Metrics Pass      A/B Test         Canary
  Artifacts     Thresholds        Results          Deploy
```

### 3.3 W&B Model Registry

```python
# Automatic registration on best model
model_artifact = wandb.Artifact(
    name=f"{config.model.name}-{config.task.name}-v{version}",
    type="model",
    metadata={
        "base_model": config.model.base_model,
        "lora_config": config.lora.to_dict(),
        "training_config": config.training.to_dict(),
        "eval_metrics": eval_results,
        "git_sha": git_sha,
    }
)
model_artifact.add_dir("./adapters/best")
wandb.log_artifact(model_artifact, aliases=["latest", "staging"])
```

---

## 4. Artifact Management

### 4.1 Artifact Structure

```
artifacts/
├── models/
│   ├── adapters/
│   │   ├── best/
│   │   │   ├── adapter_config.json
│   │   │   ├── adapter_model.safetensors
│   │   │   └── README.md
│   │   └── checkpoints/
│   │       ├── checkpoint-100/
│   │       ├── checkpoint-200/
│   │       └── checkpoint-300/
│   └── merged/
│       └── v1.0.0/
├── datasets/
│   ├── raw/
│   │   └── alpaca_data_cleaned.jsonl
│   ├── processed/
│   │   ├── train.arrow
│   │   ├── val.arrow
│   │   └── test.arrow
│   └── statistics/
│       └── dataset_stats.json
├── evaluations/
│   ├── v1.0.0/
│   │   ├── metrics.json
│   │   ├── predictions.jsonl
│   │   ├── report.html
│   │   └── plots/
└── logs/
    ├── tensorboard/
    │   └── run_20260728_120000/
    └── wandb/
        └── run-20260728_120000-a1b2c3d/
```

### 4.2 Artifact Lineage

Each artifact tracks:
- **Source Code**: Git commit SHA
- **Configuration**: Full resolved config hash
- **Dependencies**: Pinned requirements hash
- **Hardware**: GPU type, CUDA version, driver
- **Parent Artifacts**: Input dataset, base model

---

## 5. Monitoring & Observability

### 5.1 Training Monitoring

| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| GPU Memory > 95% | Warning | Reduce batch size, enable gradient checkpointing |
| GPU Utilization < 50% | Warning | Check data loading bottleneck |
| Loss NaN/Inf | Critical | Stop training, investigate |
| Gradient Norm > 10 | Warning | Increase gradient clipping |
| Learning Rate = 0 | Warning | Check scheduler |
| ETA > 48h | Info | Consider early stopping |

### 5.2 Health Checks

```yaml
# docker-compose.yml health checks
healthcheck:
  test: ["CMD", "python", "-c", "import torch; torch.cuda.is_available() or exit(1)"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

### 5.3 Structured Logging

```python
# JSON log format for log aggregation
{
  "timestamp": "2026-07-28T12:34:56.789Z",
  "level": "INFO",
  "logger": "train",
  "message": "Training step completed",
  "context": {
    "step": 1000,
    "epoch": 1.5,
    "loss": 2.341,
    "learning_rate": 1.2e-4,
    "gpu_memory_gb": 18.2,
    "tokens_per_sec": 12450
  }
}
```

---

## 6. CI/CD Pipeline

### 6.1 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/ruff-action@v2
      - uses: actions/setup-python@v5
      - run: ruff check src/ tests/
      - run: black --check src/ tests/
      - run: mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e ".[test]"
      - run: pytest tests/unit -v --cov=src --cov-fail-under=90

  integration-test:
    runs-on: [self-hosted, gpu]
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/integration -v

  docker-build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: llm-finetuning:test
          load: true

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
```

---

## 7. Deployment Architecture

### 7.1 Deployment Targets

| Target | Use Case | Technology |
|--------|----------|------------|
| **Development** | Local testing | Docker Compose |
| **Staging** | Integration testing | Kubernetes (K3s) |
| **Production** | High-throughput inference | Triton Inference Server / vLLM |
| **Edge** | Low-latency local | llama.cpp / ONNX Runtime |

### 7.2 Production Inference Stack

```
┌────────────────────────────────────────────────────────────────┐
│                     PRODUCTION INFERENCE                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│  │  Load    │───▶│  API Gateway │───▶│  vLLM / Triton      │  │
│  │  Balancer│    │  (FastAPI)   │    │  Inference Engine   │  │
│  └──────────┘    └──────────────┘    └──────────┬──────────┘  │
│                                                  │             │
│                           ┌──────────────────────┼──────────┐  │
│                           │                      │          │  │
│                           ▼                      ▼          ▼  │
│                    ┌─────────────┐        ┌──────────┐ ┌──────┐ │
│                    │  Merged     │        │  LoRA    │ │ Base │ │
│                    │  Model      │        │  Adapters│ │ Model│ │
│                    │  (FP16/BF16)│        │  (PEFT)  │ │      │ │
│                    └─────────────┘        └──────────┘ └──────┘ │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 7.3 Deployment Configuration

```yaml
# deployment/config.yaml
inference:
  engine: "vllm"  # or "triton", "tgi"
  model_path: "./artifacts/models/merged/v1.0.0"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 4096
  dtype: "bfloat16"
  quantization: null  # or "awq", "gptq"
  
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  timeout: 300
  
monitoring:
  metrics_port: 9090
  health_endpoint: "/health"
  prometheus_enabled: true
```

---

## 8. Reproducibility

### 8.1 Reproducibility Checklist

- [ ] **Pinned Dependencies**: All versions in `requirements.txt` and `pyproject.toml`
- [ ] **Locked Docker Base**: Specific CUDA/Ubuntu versions
- [ ] **Deterministic Training**: `torch.manual_seed`, `torch.cuda.deterministic`
- [ ] **Data Versioning**: Dataset hashes recorded
- [ ] **Config Snapshots**: Full config saved with each run
- [ ] **Git SHA**: Recorded in all artifacts
- [ ] **Environment Capture**: `pip freeze`, `conda list`, GPU driver version

### 8.2 Reproduction Command

```bash
# Exact reproduction from artifacts
docker run --gpus all \
  -v ./artifacts:/artifacts \
  llm-finetuning:1.0.0 \
  python -m src.train \
    --config /artifacts/models/adapters/best/config.yaml \
    --resume_from_checkpoint /artifacts/models/adapters/best/checkpoint-last
```

---

## 9. Security & Compliance

### 9.1 Secrets Management

| Secret | Source | Rotation |
|--------|--------|----------|
| HF_TOKEN | `.env` (gitignored) | Manual |
| WANDB_API_KEY | `.env` / GitHub Secrets | 90 days |
| OPENAI_API_KEY | `.env` / GitHub Secrets | 90 days |
| Docker Registry | GitHub Secrets | N/A |

### 9.2 Security Scanning

```bash
# Dependency scanning
pip-audit -r requirements.txt

# Container scanning
trivy image llm-finetuning:latest

# Secret scanning
truffleHog filesystem .
```

---

## 10. Cost Optimization

### 10.1 Training Cost Estimation

| GPU | VRAM | Model (QLoRA) | Est. Cost/hr (Spot) | 8B Model 3 Epochs |
|-----|------|---------------|---------------------|-------------------|
| RTX 3090 | 24GB | ✅ | $0.30/hr | ~$15 |
| RTX 4090 | 24GB | ✅ | $0.40/hr | ~$12 |
| A10G (G5) | 24GB | ✅ | $1.20/hr | ~$36 |
| A100 40GB | 40GB | ✅ | $2.50/hr | ~$15 |
| A100 80GB | 80GB | ✅ | $3.50/hr | ~$10 |

### 10.2 Cost Optimization Strategies

- **Spot Instances**: 70-90% savings with checkpointing
- **Gradient Accumulation**: Simulate larger batch on small GPU
- **Mixed Precision**: BF16 reduces memory, speeds up compute
- **Flash Attention 2**: 2-3x speedup on Ampere+
- **Model Compilation**: `torch.compile` for 10-20% speedup

---

*End of MLOps Architecture Document*