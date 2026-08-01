# Troubleshooting Guide

This guide covers common issues and solutions for the LLM Fine-Tuning Pipeline.

---

## Quick Diagnostics

### Health Check Commands

```bash
# Check environment
python -c "
import torch, transformers, peft, trl, datasets, accelerate
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'Accelerate: {accelerate.__version__}')
print(f'BitsAndBytes: {__import__(\"bitsandbytes\").__version__}')
try:
    import flash_attn
    print(f'FlashAttn: {flash_attn.__version__}')
except:
    print('FlashAttn: NOT INSTALLED')
"

# Check configs
python -m src.train --config configs --dry-run

# Check data
python -m src.data_pipeline --config configs/data.yaml --dry-run
```

---

## Installation Issues

### BitsAndBytes Compilation Error

```
ERROR: Failed to build bitsandbytes
```

**Solution:**
```bash
# Install with CUDA toolkit
conda install -c nvidia cuda-toolkit=12.1
pip install bitsandbytes --no-binary=bitsandbytes

# Or use pre-built wheel
pip install bitsandbytes==0.43.1 --index-url=https://pypi.org/simple

# Verify
python -c "import bitsandbytes; print('OK')"
```

### Flash Attention Installation

```
ERROR: flash_attn not found
```

**Solution:**
```bash
# Install with specific CUDA
pip install flash-attn==2.5.8 --no-build-isolation

# Requires: ninja, CUDA 11.8+/12.1+, PyTorch 2.0+
pip install ninja
```

### PEFT/Transformers Version Mismatch

```
AttributeError: module 'peft' has no attribute 'LoraConfig'
```

**Solution:**
```bash
# Upgrade to compatible versions
pip install --upgrade peft transformers accelerate trl

# Check versions
python -c "import peft, transformers; print(peft.__version__, transformers.__version__)"
# Need: peft>=0.11, transformers>=4.40
```

---

## Data Pipeline Issues

### Dataset Download Fails

```
ConnectionError: Failed to download dataset
```

**Solutions:**
```bash
# 1. Set HF token
export HF_TOKEN=hf_xxx
# Or in .env file

# 2. Use offline mode (if cached)
export HF_DATASETS_OFFLINE=1

# 3. Increase retries
# In configs/data.yaml:
processing:
  download:
    max_retries: 5

# 4. Manual download
huggingface-cli download tatsu-lab/alpaca --local-dir ./data/raw/alpaca
```

### Column Mapping Error

```
ValueError: Missing required columns: ['instruction']
```

**Solution:**
Check `column_mapping` in `data.yaml` matches dataset columns:
```yaml
datasets:
  - name: "my_dataset"
    path: "org/dataset"
    column_mapping:
      instruction: "prompt"      # Actual column name in dataset
      input: "context"           # Actual column name
      output: "completion"       # Actual column name
```

### Tokenization OOM

```
RuntimeError: CUDA out of memory during tokenization
```

**Solutions:**
```yaml
# In data.yaml
processing:
  tokenization:
    batch_size: 500        # Reduce from 1000
    num_proc: 2            # Reduce parallelism
```

```python
# Or process in chunks
dataset = dataset.map(tokenize_fn, batched=True, batch_size=100)
```

### Split Ratio Error

```
ValueError: Split ratios must sum to 1.0
```

**Solution:**
```yaml
splitting:
  ratios:
    train: 0.8
    validation: 0.1
    test: 0.1  # Must sum to 1.0 exactly
```

---

## Model Loading Issues

### CUDA OOM on Model Load

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```yaml
# In model.yaml - use 8-bit instead of 4-bit
model:
  load_in_4bit: false
  load_in_8bit: true

# Or enable CPU offloading
quantization:
  load_in_4bit: true
  llm_int8_enable_fp32_cpu_offload: true

# Or reduce batch size
trainer:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### Quantization Config Error

```
ValueError: bnb_4bit_compute_dtype must be float16, bfloat16, or float32
```

**Solution:**
```yaml
quantization:
  bnb_4bit_compute_dtype: "bfloat16"  # Not "bf16"
```

### Device Map Issues

```
ValueError: Invalid device_map
```

**Solutions:**
```yaml
# For single GPU
model:
  device_map: "auto"

# For multi-GPU (specify max memory)
model:
  device_map: "auto"
  max_memory:
    0: "20GiB"
    1: "20GiB"
    cpu: "30GiB"

# For CPU-only
model:
  device_map: "cpu"
```

### Tokenizer Mismatch

```
WARNING: Tokenizer pad_token not set
```

**Solution:**
```python
# In code
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Or in config
tokenizer:
  pad_token: "<|end_of_text|>"
```

---

## Training Issues

### Training Loss NaN

```
Loss became NaN at step 100
```

**Solutions:**
```yaml
# 1. Reduce learning rate
trainer:
  learning_rate: 1e-4  # From 2e-4

# 2. Enable gradient clipping
trainer:
  max_grad_norm: 0.5  # From 1.0

# 3. Use FP32 for loss
# In model config
trainer:
  fp16: false
  bf16: true

# 4. Check data for anomalies
# Run data validation
python -m src.data_pipeline --config configs/data.yaml --dataset train
```

### Gradient Exploding

```
Grad norm: 50.2 (expected ~1-5)
```

**Solutions:**
```yaml
trainer:
  max_grad_norm: 0.5
  gradient_accumulation_steps: 4  # Normalize by accum steps
```

### Loss Not Decreasing

```
Step 500: train_loss=2.5, eval_loss=2.4 (no improvement)
```

**Solutions:**
```yaml
# 1. Check learning rate schedule
trainer:
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03  # Increase from 0.01

# 2. Increase LoRA rank
lora:
  r: 128  # From 64

# 3. Check data quality
# - Remove duplicates
# - Filter short/long sequences
# - Verify prompt format

# 4. Increase training time
trainer:
  num_train_epochs: 5  # From 3
```

### Early Stopping Too Early

```
Early stopping triggered at step 200
```

**Solution:**
```yaml
callbacks:
  early_stopping:
    patience: 5      # Increase from 3
    threshold: 0.001 # Decrease sensitivity
```

### Checkpoint Not Saving

```
Best model not saved
```

**Solutions:**
```yaml
trainer:
  save_strategy: "steps"
  save_steps: 100
  save_total_limit: 5
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false

# Check disk space
df -h ./checkpoints
```

### Resume Fails

```
ValueError: Cannot resume from checkpoint
```

**Solutions:**
```bash
# 1. Use correct checkpoint path
python -m src.train --resume-from-checkpoint ./checkpoints/checkpoint-500

# 2. Check checkpoint integrity
ls -la ./checkpoints/checkpoint-500/
# Should have: config.json, trainer_state.json, optimizer.pt, scheduler.pt, model files

# 3. Match configuration exactly
# Config must match when checkpoint was created
```

---

## Memory Issues

### OOM During Training

```
RuntimeError: CUDA out of memory at step 50
```

**Solutions (in order):**
```yaml
# 1. Reduce batch size
trainer:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16

# 2. Enable gradient checkpointing
model:
  gradient_checkpointing: true
trainer:
  gradient_checkpointing: true

# 3. Enable Flash Attention
model:
  attn_implementation: "flash_attention_2"
  use_flash_attention_2: true

# 4. Use 8-bit optimizer
# In code: optim="adamw_bnb_8bit"

# 5. CPU offload optimizer
# In code: optim="adamw_torch", optim_args={"foreach": true}

# 6. Reduce sequence length
sft:
  max_seq_length: 1024  # From 2048

# 7. Packing (if applicable)
sft:
  packing: true
```

### CPU Memory OOM

```
MemoryError: Unable to allocate array
```

**Solutions:**
```yaml
# Reduce dataloader workers
trainer:
  dataloader_num_workers: 0  # From 4
  dataloader_prefetch_factor: 1

# Process data in smaller chunks
processing:
  tokenization:
    batch_size: 100
```

### Fragmentation

```
CUDA memory fragmentation
```

**Solutions:**
```python
# In training loop (via callback)
torch.cuda.empty_cache()

# Or in config
runtime:
  empty_cache_steps: 50
  gc_collect_steps: 100
```

---

## Distributed Training Issues

### NCCL Timeout

```
NCCL timeout: 1800s
```

**Solutions:**
```yaml
trainer:
  ddp_timeout: 3600  # Increase from 1800
```

```bash
# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1  # Disable InfiniBand if issues
export PYTHONFAULTHANDLER=1
```

### Rank Mismatch

```
Rank 0: Expected 4 GPUs, got 2
```

**Solutions:**
```bash
# Check visible GPUs
python -c "import torch; print(torch.cuda.device_count())"

# Set explicitly
export CUDA_VISIBLE_DEVICES=0,1,2,3

# In torchrun
torchrun --nproc_per_node=4 --nnodes=1 ...
```

### Gradient Sync Issues

```
Gradients not synchronized across ranks
```

**Solutions:**
```yaml
trainer:
  ddp_find_unused_parameters: false
  ddp_bucket_cap_mb: 25
```

```python
# In model - ensure all parameters used
model.gradient_checkpointing_enable()
# Or
model.config.use_cache = False
```

---

## Evaluation Issues

### Metrics Import Errors

```
ModuleNotFoundError: No module named 'rouge_score'
```

**Solution:**
```bash
pip install rouge-score nltk bert-score perplexity

# Download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Generation Hangs

```
Generation takes forever / hangs
```

**Solutions:**
```python
# Check eos_token_id
generation_config = GenerationConfig(
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_new_tokens=512,  # Not too large
    do_sample=True,      # Required for early_stopping
)
```

### Benchmark Errors

```
CUDA out of memory during benchmark
```

**Solution:**
```yaml
# In evaluation.yaml
evaluation:
  benchmark:
    num_warmup: 1
    num_runs: 5  # Reduce from 10
```

---

## Configuration Issues

### Config Validation Error

```
ValidationError: 1 validation error for TrainingConfig
```

**Solution:**
```bash
# Check config syntax
python -c "
from src.config import ConfigManager
cm = ConfigManager('configs')
print('All configs valid!')
"

# Or validate specific section
python -c "
from src.config import TrainerConfig
import yaml
with open('configs/training.yaml') as f:
    data = yaml.safe_load(f)
TrainerConfig(**data.get('trainer', {}))
print('Trainer config valid!')
"
```

### Environment Variable Not Resolved

```
ValueError: ${HF_TOKEN} not found
```

**Solution:**
```bash
# 1. Set in shell
export HF_TOKEN=hf_xxx

# 2. Use .env file
echo "HF_TOKEN=hf_xxx" > .env
echo "WANDB_API_KEY=xxx" >> .env

# 3. Check .env loading
python -c "
from src.config import ConfigManager
cm = ConfigManager('configs', '.env')
print(cm.training.trainer.run_name)
"
```

---

## Inference Issues

### Model Load Fails

```
OSError: Cannot load model
```

**Solutions:**
```bash
# 1. Check model files exist
ls -la ./artifacts/merged/
# Should have: config.json, model.safetensors, tokenizer files

# 2. Try with trust_remote_code
model = AutoModelForCausalLM.from_pretrained(
    "./artifacts/merged",
    trust_remote_code=True,
)

# 3. Check PyTorch version compatibility
# Model saved with PT 2.1, loading with PT 2.0 may fail
```

### Slow Inference

```
Generation takes 30s for 100 tokens
```

**Solutions:**
```python
# 1. Use compiled model
model = torch.compile(model, mode="max-autotune")

# 2. Enable Flash Attention
model.config._attn_implementation = "flash_attention_2"

# 3. Use static KV cache
from transformers import StaticCache
cache = StaticCache(config, max_batch_size=1, max_cache_len=2048, device="cuda")
outputs = model.generate(inputs, past_key_values=cache, ...)

# 4. Reduce precision
model = model.half()  # or .to(torch.bfloat16)

# 5. Use vLLM or TGI for production
```

### Wrong Output Format

```
Output doesn't match expected format
```

**Solution:**
```python
# Use same prompt template as training
prompt = """### Instruction:
Your instruction here

### Response:
"""

# NOT:
prompt = "Your instruction here"
```

---

## Logging & Monitoring Issues

### W&B Not Logging

```
W&B run not appearing
```

**Solutions:**
```bash
# 1. Check API key
wandb login

# 2. Check config
trainer:
  report_to: ["wandb"]
  run_name: "my-run"

# 3. Check network
curl https://api.wandb.ai

# 4. Debug mode
export WANDB_DEBUG=true
```

### TensorBoard Not Showing Data

```
No scalar data in TensorBoard
```

**Solutions:**
```bash
# 1. Check log dir
trainer:
  logging_dir: "./logs/tensorboard"
  logging_steps: 10

# 2. Start TensorBoard correctly
tensorboard --logdir ./logs/tensorboard --port 6006

# 3. Check event files exist
ls ./logs/tensorboard/events.out.tfevents.*
```

### Missing GPU Metrics

```
GPU memory not logged
```

**Solution:**
```yaml
callbacks:
  logging:
    enabled: true
    log_steps: 10
    log_gpu_memory: true
```

---

## Performance Tuning Checklist

### Before Training
- [ ] GPU drivers up to date (535+ for CUDA 12)
- [ ] PyTorch compiled with correct CUDA version
- [ ] Flash Attention installed
- [ ] BitsAndBytes compiled for your CUDA
- [ ] Dataset cached locally
- [ ] Config validated with `--dry-run`

### During Training
- [ ] GPU utilization > 80% (nvidia-smi)
- [ ] VRAM usage stable
- [ ] Loss decreasing smoothly
- [ ] Grad norm ~1-5
- [ ] Learning rate following schedule
- [ ] Checkpoints saving correctly

### After Training
- [ ] Eval loss lower than baseline
- [ ] Metrics improved (ROUGE, BLEU, etc.)
- [ ] Model merges without error
- [ ] Inference works correctly
- [ ] Benchmark meets latency targets

---

## Getting Help

### Debug Information to Collect

```bash
# Create debug bundle
mkdir debug_info
python -c "
import torch, transformers, peft, trl, sys
with open('debug_info/versions.txt', 'w') as f:
    f.write(f'Python: {sys.version}\n')
    f.write(f'PyTorch: {torch.__version__}\n')
    f.write(f'CUDA: {torch.version.cuda}\n')
    f.write(f'Transformers: {transformers.__version__}\n')
    f.write(f'PEFT: {peft.__version__}\n')
    f.write(f'TRL: {trl.__version__}\n')
    f.write(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}\n')
    f.write(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\n')
"

# Config dump
python -m src.train --config configs --dry-run 2>&1 | tee debug_info/dry_run.log

# Environment
env | grep -E '(CUDA|PYTORCH|HF|WANDB|NCCL)' > debug_info/env.txt

# Compress
tar -czf debug_info.tar.gz debug_info/
```

### Where to Report Issues

1. **GitHub Issues**: Bug reports, feature requests
2. **Discussions**: Questions, configuration help
3. **Documentation**: Check ARCHITECTURE.md, DEPLOYMENT_GUIDE.md, INFERENCE_GUIDE.md

### Minimal Reproduction

```python
# Create minimal script
# 1. Small dataset (10 samples)
# 2. Small model (TinyLlama)
# 3. Few steps (10)
# 4. Reproduce issue
# 5. Share script + error + debug_info.tar.gz
```

---

## Emergency Recovery

### Corrupted Checkpoint

```bash
# 1. Find last good checkpoint
ls -la checkpoints/
# Use checkpoint-N where N < corrupted

# 2. Resume from it
python -m src.train --resume-from-checkpoint ./checkpoints/checkpoint-1000
```

### Disk Full

```bash
# 1. Clean old checkpoints
cd checkpoints
ls -t | tail -n +4 | xargs rm -rf  # Keep latest 3

# 2. Clean logs
find logs/ -name "*.log" -mtime +7 -delete

# 3. Clean HF cache
rm -rf ~/.cache/huggingface/hub/
```

### Process Killed (OOM)

```bash
# 1. Check dmesg
dmesg -T | grep -i "out of memory"

# 2. Reduce batch size in config
# 3. Resume from last checkpoint
python -m src.train --resume-from-checkpoint ./checkpoints/checkpoint-last
```

---

## Version Compatibility Matrix

| Component | Min Version | Recommended | Notes |
|-----------|-------------|-------------|-------|
| Python | 3.10 | 3.11 | 3.12 may have issues |
| PyTorch | 2.1 | 2.2 | Match CUDA version |
| CUDA | 11.8 | 12.1 | Driver 535+ |
| Transformers | 4.40 | 4.44 | |
| PEFT | 0.11 | 0.13 | |
| TRL | 0.9 | 0.11 | |
| Accelerate | 0.30 | 0.33 | |
| BitsAndBytes | 0.43 | 0.45 | Compile for your CUDA |
| FlashAttn | 2.5 | 2.6 | Match PyTorch/CUDA |
| Datasets | 2.19 | 2.20 | |