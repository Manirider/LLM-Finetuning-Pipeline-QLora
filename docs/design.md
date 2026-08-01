# Design Decisions Document

## Project: LLM Fine-Tuning Pipeline with QLoRA/LoRA

## Version: 1.0.0
## Date: 2026-07-28

---

## 1. Key Design Decisions

### 1.1 QLoRA over Full Fine-Tuning

**Decision**: Use QLoRA (4-bit NF4 quantization + LoRA) instead of full fine-tuning.

**Rationale**:
- **Memory Efficiency**: 4-bit quantization reduces model memory by ~4x (e.g., Llama-3-8B: 16GB → 4GB)
- **Consumer GPU Support**: Enables training on 24GB VRAM (RTX 3090/4090) vs 80GB A100
- **Performance Parity**: QLoRA matches 16-bit LoRA performance within 1-2% on benchmarks
- **Speed**: Faster training due to reduced memory bandwidth

**Trade-offs**:
- Slightly more complex setup (BitsAndBytes integration)
- Requires compute dtype BF16/FP16 for training
- Adapter merging needed for deployment

**Alternatives Considered**:
| Approach | VRAM (8B) | Performance | Complexity |
|----------|-----------|-------------|------------|
| Full FT (BF16) | ~48GB | Best | Low |
| LoRA (BF16) | ~16GB | Excellent | Low |
| **QLoRA (NF4)** | **~6GB** | **Excellent** | **Medium** |
| GPTQ + LoRA | ~5GB | Good | High |

---

### 1.2 NF4 Quantization over FP4/INT4

**Decision**: Use NormalFloat4 (NF4) quantization with double quantization.

**Rationale**:
- **NF4**: Optimal for normally distributed weights (transformers)
- **Double Quantization**: Quantizes the quantization constants (saves 0.5 bits/param)
- **Superior to FP4**: Better accuracy for same bit-width
- **Superior to INT4**: Handles outlier weights better

**Configuration**:
```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
```

---

### 1.3 LoRA Configuration Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` (rank) | 64 | Balances capacity vs overfitting; 64 works well for 7B-13B |
| `lora_alpha` | 16 | Scaling factor; alpha/r = 0.25 standard ratio |
| `target_modules` | All linear | q,k,v,o,gate,up,down - full coverage for causal LM |
| `lora_dropout` | 0.05 | Light regularization; higher hurts convergence |
| `use_rslora` | true | Rank-stabilized LoRA; better scaling properties |
| `use_dora` | false | Weight-decomposed LoRA; adds complexity |

**Rank Selection Guide**:
| Model Size | Recommended Rank |
|------------|------------------|
| 1B-3B | 32 |
| 7B-8B | 64 |
| 13B | 64-128 |
| 30B+ | 128-256 |

---

### 1.4 SFTTrainer over Custom Training Loop

**Decision**: Use TRL's `SFTTrainer` instead of custom training loop.

**Rationale**:
- **Battle-tested**: Used in production at Hugging Face, NVIDIA, etc.
- **Built-in Features**: Formatting, packing, NEFTune, data collation
- **Callbacks Integration**: Native Hugging Face Trainer callbacks
- **Distributed Training**: FSDP, DeepSpeed, DDP support
- **Maintenance**: Community maintained, regular updates

**Custom Loop Would Require**:
- Data formatting & tokenization logic
- Dynamic padding collator
- Gradient accumulation handling
- Checkpointing & resume logic
- Distributed training boilerplate
- Callback system
- Logging integration

---

### 1.5 Configuration Architecture

**Decision**: YAML + Pydantic + Environment Variables with explicit precedence.

**Rationale**:
- **YAML**: Human-readable, hierarchical, supports comments
- **Pydantic**: Runtime validation, type coercion, IDE support
- **Environment Variables**: Secrets, deployment-specific overrides
- **Precedence**: Clear, predictable override behavior

**Why Not**:
- **Hydra/OmegaConf**: Adds heavy dependency, complex for simple needs
- **JSON**: No comments, less readable
- **Python Configs**: Security risk (code execution), not declarative

---

### 1.6 Data Pipeline Design

**Decision**: Single `DataPipeline` class with explicit stages and caching.

**Stage Architecture**:
```python
class DataPipeline:
    def download()      # Dataset download with cache
    def validate()      # Schema validation, statistics
    def clean()         # Deduplication, null handling
    def format()        # ChatML/Alpaca formatting
    def tokenize()      # Tokenization with stats
    def split()         # Train/val/test split
    def save()          # JSONL, Arrow, Parquet
```

**Why Not**:
- **Multiple Classes**: Over-engineering for linear pipeline
- **Functional**: Harder to maintain state (tokenizer, config)
- **Lazy/Eager Mix**: Explicit control over materialization

---

### 1.7 Evaluation Strategy

**Decision**: Comprehensive multi-metric evaluation on multiple datasets.

**Metrics Selection**:
| Metric | Purpose | Why Included |
|--------|---------|--------------|
| ROUGE-1/2/L | N-gram overlap | Standard for summarization/QA |
| BLEU | Precision-focused | Machine translation standard |
| BERTScore | Semantic similarity | Correlates with human judgment |
| Perplexity | Fluency | Language model quality |
| Distinct-n | Diversity | Detects degenerate repetition |
| Latency/Throughput | Production readiness | Real-world deployment |

**Datasets**:
- **AlpacaEval**: Instruction following
- **MT-Bench**: Multi-turn conversation
- **GSM8K**: Mathematical reasoning
- **HumanEval**: Code generation
- **TruthfulQA**: Factuality/hallucination

---

### 1.8 MLOps Integration

**Decision**: Dual logging to TensorBoard + Weights & Biases.

**Rationale**:
- **TensorBoard**: Local, free, standard, good for loss curves
- **W&B**: Cloud, collaboration, artifact versioning, sweeps
- **Both**: Redundancy, different strengths, team flexibility

**Logged Artifacts**:
- Model checkpoints (best, last)
- LoRA adapters
- Evaluation reports (HTML/MD)
- Prediction samples
- Configuration snapshots

---

### 1.9 Container Strategy

**Decision**: Multi-stage Docker build with docker-compose for orchestration.

**Stages**:
1. **Base**: nvidia/cuda:12.1-devel-ubuntu22.04
2. **Builder**: Install dependencies, compile flash-attn
3. **Runtime**: Copy only necessary files, non-root user

**Services**:
- `trainer`: Training with GPU access
- `evaluator`: Evaluation with GPU access
- `inference`: API server (FastAPI/vLLM)
- `tensorboard`: Visualization

---

### 1.10 Testing Strategy

**Decision**: Three-tier testing (unit, integration, smoke) with >90% coverage target.

| Tier | Scope | Speed | Examples |
|------|-------|-------|----------|
| **Unit** | Single functions/classes | <1s each | Config validation, metric computation |
| **Integration** | Pipeline components | 10-60s | Training pipeline end-to-end |
| **Smoke** | Critical paths | <30s total | Import test, config loading, CLI help |

**Coverage Targets**:
- Core modules (config, data, model, metrics): >95%
- Pipeline orchestration: >90%
- Scripts/CLI: >80%

---

## 2. Rejected Alternatives

### 2.1 Framework Alternatives

| Framework | Rejection Reason |
|-----------|------------------|
| **Axolotl** | Opinionated config, less flexible, single-purpose |
| **LlamaFactory** | UI-focused, less programmatic control |
| **Unsloth** | Closed optimization, less transparent |
| **Custom PyTorch** | Reinventing wheel, maintenance burden |

### 2.2 Quantization Alternatives

| Method | Rejection Reason |
|--------|------------------|
| **GPTQ** | Post-training only, no LoRA integration |
| **AWQ** | Requires calibration data, slower quantization |
| **FP8** | Requires H100, not widely supported yet |
| **INT8** | Less memory savings than 4-bit |

### 2.3 Logging Alternatives

| Tool | Rejection Reason |
|------|------------------|
| **MLflow** | Heavy, model-centric, less training-focused |
| **Neptune** | Proprietary, cost |
| **ClearML** | Complex setup, overhead |
| **Sacred** | Abandoned, MongoDB dependency |

---

## 3. Known Limitations & Mitigations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **QLoRA slower than BF16 LoRA** | ~20% slower training | Use Flash Attention 2, compile |
| **Adapter merging required** | Extra step for deployment | Automated merge script |
| **NF4 not on all GPUs** | Requires Ampere+ | Fallback to INT8/FP16 |
| **W&B requires account** | External dependency | Offline mode, TensorBoard fallback |
| **Large datasets need streaming** | Memory pressure | IterableDataset support planned |

---

## 4. Future Extensibility

### 4.1 Planned Extensions
- **DPO/ORPO**: Preference optimization trainers
- **RAG Integration**: Retrieval-augmented evaluation
- **Multi-Node Training**: FSDP/DeepSpeed config
- **Model Merging**: TIES, DARE, SLERP merging
- **Quantization Export**: GGUF, GPTQ, AWQ export

### 4.2 Extension Points (Implemented)
```python
# Custom formatter
class CustomFormatter(PromptFormatter):
    def format(self, example): ...

# Custom metric
class CustomMetric(MetricComputer):
    def compute(self, preds, refs): ...

# Custom callback
class CustomCallback(TrainerCallback):
    def on_step_end(self, args, state, control): ...
```

---

## 5. Decision Log

| Date | Decision | Author | Status |
|------|----------|--------|--------|
| 2026-07-28 | QLoRA with NF4 | Architecture Team | Accepted |
| 2026-07-28 | SFTTrainer over custom loop | Architecture Team | Accepted |
| 2026-07-28 | YAML + Pydantic config | Architecture Team | Accepted |
| 2026-07-28 | Dual TB + W&B logging | Architecture Team | Accepted |
| 2026-07-28 | Multi-stage Docker | Architecture Team | Accepted |
| 2026-07-28 | Three-tier testing | Architecture Team | Accepted |

---

*End of Design Decisions Document*