# Inference Guide

This guide covers running inference with fine-tuned models, including local inference, API server deployment, batch processing, and production serving.

---

## Overview

After training and merging the LoRA adapter, you have a complete model ready for inference. This guide shows multiple ways to use it.

---

## Model Preparation

### 1. Merge LoRA Adapter

```bash
# Merge and save for inference
python -m src.train \
  --config configs \
  --merge-and-save \
  --merged-output-dir ./artifacts/merged \
  --merged-dtype bfloat16 \
  --safe-serialization \
  --max-shard-size 5GB
```

This creates a standalone model without PEFT dependency.

### 2. Verify Merged Model

```bash
# Quick test
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('./artifacts/merged', device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('./artifacts/merged')

inputs = tokenizer('### Instruction:\nSay hello\n\n### Response:\n', return_tensors='pt').to('cuda')
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
"
```

### 3. Push to Hub (Optional)

```bash
python -m src.train \
  --config configs \
  --merge-and-save \
  --merged-output-dir ./artifacts/merged \
  --push-to-hub \
  --hub-model-id your-org/llama-3-8b-finetuned \
  --hub-token ${HF_TOKEN} \
  --hub-private-repo
```

---

## Local Inference

### 1. Python API

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "./artifacts/merged",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("./artifacts/merged")

# Generation function
def generate(prompt, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    defaults = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "early_stopping": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    defaults.update(kwargs)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **defaults)
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()

# Alpaca format
prompt = """### Instruction:
Write a Python function to calculate fibonacci numbers.

### Response:
"""
print(generate(prompt))

# ChatML format
prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain quantum computing in simple terms.<|im_end|>
<|im_start|>assistant
"""
print(generate(prompt))

# Llama-3 format
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
print(generate(prompt))
```

### 2. Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_generate(prompt, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    defaults = {"max_new_tokens": 512, "temperature": 0.7, "do_sample": True}
    defaults.update(kwargs)
    
    generation_kwargs = dict(inputs, streamer=streamer, **defaults)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for text in streamer:
        yield text
        print(text, end="", flush=True)

# Usage
for _ in stream_generate("### Instruction:\nWrite a poem\n\n### Response:\n"):
    pass
```

### 3. Batch Generation

```python
def batch_generate(prompts, batch_size=4, **kwargs):
    """Process multiple prompts in batches."""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        
        defaults = {"max_new_tokens": 512, "temperature": 0.7, "do_sample": True}
        defaults.update(kwargs)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **defaults)
        
        for j, out in enumerate(outputs):
            response = tokenizer.decode(
                out[inputs.input_ids[j].shape[0]:],
                skip_special_tokens=True,
            )
            results.append(response.strip())
    
    return results

# Usage
prompts = [
    "### Instruction:\nTask 1\n\n### Response:\n",
    "### Instruction:\nTask 2\n\n### Response:\n",
]
responses = batch_generate(prompts)
```

---

## API Server Deployment

### 1. Start Server

```bash
# Basic server
python -m src.inference \
  --model-path ./artifacts/merged \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4

# With custom settings
python -m src.inference \
  --model-path ./artifacts/merged \
  --host 0.0.0.0 \
  --port 8000 \
  --max-batch-size 16 \
  --max-seq-length 4096 \
  --dtype bfloat16
```

### 2. Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | required | Path to merged model |
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8000 | Port |
| `--workers` | 4 | Worker processes |
| `--max-batch-size` | 16 | Max batch size |
| `--max-seq-length` | 4096 | Max sequence length |
| `--dtype` | bfloat16 | Model dtype |
| `--enable-streaming` | true | SSE streaming |

### 3. API Endpoints

#### Completions (OpenAI Compatible)

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "### Instruction:\nWrite a haiku\n\n### Response:\n",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
  }'
```

**Response:**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "llama-3-8b-finetuned",
  "choices": [
    {
      "text": "Silent code flows fast\nGPUs hum in the night\nModels learn and grow",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 23,
    "total_tokens": 38
  }
}
```

#### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain transformers in 3 sentences."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

#### Streaming

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "### Instruction:\nCount to 10\n\n### Response:\n",
    "max_tokens": 50,
    "stream": true
  }'
```

**SSE Response:**
```
data: {"id":"cmpl-xxx","choices":[{"delta":{"content":"One"},"index":0}]}
data: {"id":"cmpl-xxx","choices":[{"delta":{"content":" two"},"index":0}]}
...
data: [DONE]
```

### 4. Python Client

```python
import requests
import json

class InferenceClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def complete(self, prompt, **kwargs):
        payload = {"prompt": prompt, **kwargs}
        resp = requests.post(f"{self.base_url}/v1/completions", json=payload)
        return resp.json()
    
    def chat(self, messages, **kwargs):
        payload = {"messages": messages, **kwargs}
        resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
        return resp.json()
    
    def stream(self, prompt, **kwargs):
        payload = {"prompt": prompt, "stream": True, **kwargs}
        resp = requests.post(f"{self.base_url}/v1/completions", json=payload, stream=True)
        for line in resp.iter_lines():
            if line:
                yield line.decode('utf-8')

# Usage
client = InferenceClient()
result = client.complete("### Instruction:\nHello\n\n### Response:\n", max_tokens=50)
print(result["choices"][0]["text"])

for chunk in client.stream("### Instruction:\nCount\n\n### Response:\n"):
    if chunk.startswith("data: "):
        data = chunk[6:]
        if data != "[DONE]":
            print(json.loads(data)["choices"][0]["delta"].get("content", ""), end="")
```

---

## Batch Inference

### 1. File-Based Batch Processing

```bash
# Prepare input file (JSONL)
cat > prompts.jsonl << EOF
{"prompt": "### Instruction:\nTask 1\n\n### Response:\n", "max_tokens": 100}
{"prompt": "### Instruction:\nTask 2\n\n### Response:\n", "max_tokens": 100}
{"prompt": "### Instruction:\nTask 3\n\n### Response:\n", "max_tokens": 100}
EOF

# Run batch inference
python -m src.inference \
  --model-path ./artifacts/merged \
  --input-file prompts.jsonl \
  --output-file results.jsonl \
  --batch-size 8
```

### 2. Python Batch API

```python
import json

def batch_inference(input_file, output_file, model_path, batch_size=8):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Read prompts
    with open(input_file) as f:
        prompts = [json.loads(line)["prompt"] for line in f]
    
    # Process in batches
    with open(output_file, "w") as f:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            for j, out in enumerate(outputs):
                response = tokenizer.decode(
                    out[inputs.input_ids[j].shape[0]:],
                    skip_special_tokens=True,
                )
                f.write(json.dumps({"prompt": batch[j], "response": response}) + "\n")
```

---

## Production Serving

### 1. vLLM (High Throughput)

```bash
# Install vLLM
pip install vllm

# Run OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
  --model ./artifacts/merged \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

**Features:**
- PagedAttention for memory efficiency
- Continuous batching
- Prefix caching
- Quantization (AWQ, GPTQ)
- Multi-GPU tensor parallelism

### 2. Text Generation Inference (TGI)

```bash
# Using Hugging Face TGI
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $(pwd)/artifacts/merged:/model \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /model \
  --max-input-length 2048 \
  --max-total-tokens 4096 \
  --max-batch-prefill-tokens 4096
```

**Features:**
- Optimized kernels
- Continuous batching
- Tensor parallelism
- Quantization support
- Prometheus metrics

### 3. Triton Inference Server

```bash
# Export to ONNX/TensorRT
python -c "
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained('./artifacts/merged')
model.eval()

# Export to ONNX
torch.onnx.export(
    model,
    (torch.randint(0, 1000, (1, 512)),),
    'model.onnx',
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'logits': {0: 'batch', 1: 'seq'}},
    opset_version=17,
)
"

# Deploy with Triton
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

---

## Performance Optimization

### 1. Generation Parameters

```python
# Fast generation
fast_config = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "use_cache": True,           # KV cache
    "num_beams": 1,              # No beam search
    "early_stopping": True,
}

# High quality
quality_config = {
    "max_new_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.15,
    "do_sample": True,
    "num_beams": 4,              # Beam search
    "early_stopping": True,
    "length_penalty": 1.0,
}

# Deterministic
deterministic_config = {
    "max_new_tokens": 256,
    "temperature": 0.0,
    "do_sample": False,
    "num_beams": 1,
}
```

### 2. KV Cache Optimization

```python
# Enable Flash Attention for faster attention
model.config._attn_implementation = "flash_attention_2"

# Use static cache for fixed-length generation
from transformers import StaticCache

cache = StaticCache(
    config=model.config,
    max_batch_size=8,
    max_cache_len=2048,
    device=model.device,
    dtype=torch.bfloat16,
)

outputs = model.generate(
    inputs,
    past_key_values=cache,
    max_new_tokens=256,
)
```

### 3. Quantization for Inference

```bash
# AWQ Quantization (4-bit)
pip install autoawq
python -m awq.quantize \
  --model_path ./artifacts/merged \
  --quant_path ./artifacts/merged-awq \
  --w_bit 4 \
  --q_group_size 128 \
  --zero_point True \
  --version GEMM

# GPTQ Quantization
pip install auto-gptq
python -m auto_gptq \
  --model_path ./artifacts/merged \
  --quant_path ./artifacts/merged-gptq \
  --bits 4 \
  --group_size 128
```

---

## Monitoring & Scaling

### 1. Health Checks

```python
# /health endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_memory": f"{torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB",
        "device": str(next(model.parameters()).device),
    }
```

### 2. Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter("inference_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Request latency", ["endpoint"])
GPU_MEMORY = Gauge("gpu_memory_used_bytes", "GPU memory used")
ACTIVE_REQUESTS = Gauge("inference_active_requests", "Active requests")

@app.middleware("http")
async def metrics_middleware(request, call_next):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        response = await call_next(request)
        REQUEST_COUNT.labels(endpoint=request.url.path, status=response.status_code).inc()
        return response
    finally:
        ACTIVE_REQUESTS.dec()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.time() - start)
        GPU_MEMORY.set(torch.cuda.memory_allocated())
```

### 3. Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: inference_active_requests
        target:
          type: AverageValue
          averageValue: "10"
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM during generation | Reduce `max_new_tokens`, batch size, or enable CPU offload |
| Slow generation | Enable Flash Attention, use cache, reduce beam size |
| Repetitive output | Increase `repetition_penalty`, `temperature`, `top_p` |
| Cut off responses | Increase `max_new_tokens`, check `eos_token_id` |
| Wrong format | Verify prompt template matches training |
| CUDA OOM on startup | Use `device_map="auto"`, enable 8-bit loading |

### Debug Generation

```python
# Enable debug logging
import logging
logging.getLogger("transformers.generation").setLevel(logging.DEBUG)

# Inspect logits
outputs = model.generate(
    inputs,
    max_new_tokens=10,
    return_dict_in_generate=True,
    output_scores=True,
)

for i, scores in enumerate(outputs.scores):
    print(f"Step {i}: top-5 = {scores[0].topk(5)}")
```

### Performance Profiling

```bash
# Profile with PyTorch profiler
python -m src.inference \
  --model-path ./artifacts/merged \
  --profile \
  --profile-dir ./profile_output
```

---

## Security Considerations

1. **Input Validation**: Sanitize prompts, limit length
2. **Rate Limiting**: Prevent abuse
3. **Authentication**: Add API keys
4. **Content Filtering**: Add safety filters
5. **Audit Logging**: Log all requests

```python
# Rate limiting example
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v1/completions")
@limiter.limit("10/minute")
async def completions(request: Request, ...):
    ...
```