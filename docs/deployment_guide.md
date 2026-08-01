# Deployment Guide

This guide covers deploying the LLM Fine-Tuning Pipeline in various environments: local development, cloud VMs, Kubernetes, and managed services.

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16GB (4-bit, batch=1) | 24GB+ (batch=4+) |
| System RAM | 32GB | 64GB+ |
| Disk | 100GB NVMe | 500GB+ NVMe |
| CPU | 8 cores | 16+ cores |

### Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.10+ |
| PyTorch | 2.1+ (CUDA 11.8/12.1) |
| CUDA | 11.8+ / 12.1+ |
| cuDNN | 8.9+ |
| NVIDIA Driver | 535+ (CUDA 12) / 525+ (CUDA 11) |

---

## Local Development

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with all extras
pip install -e ".[dev,eval,inference]"

# Or minimal install
pip install -e .
pip install -r requirements.txt
```

### 2. GPU Verification

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Check BitsAndBytes
python -c "import bitsandbytes; print('BNB OK')"

# Check Flash Attention
python -c "import flash_attn; print('FlashAttn OK')"
```

### 3. Configuration

```bash
# Copy example env file
cp .env.example .env

# Edit with your tokens
# HF_TOKEN=hf_xxx
# WANDB_API_KEY=xxx
# CUDA_VISIBLE_DEVICES=0,1
```

### 4. Run Pipeline

```bash
# Data pipeline
python -m src.data_pipeline --config configs/data.yaml

# Training
python -m src.train --config configs

# Evaluation
python -m src.evaluate --config configs --base-model meta-llama/Meta-Llama-3-8B-Instruct --finetuned-model ./checkpoints/best
```

---

## Docker Deployment

### 1. Build Image

```bash
# Build training image
docker build -t llm-finetune:latest -f Dockerfile .

# Build with specific CUDA version
docker build --build-arg CUDA_VERSION=12.1 -t llm-finetune:cuda121 .
```

### 2. Run Container

```bash
# Training
docker run --gpus all \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  -e HF_TOKEN=${HF_TOKEN} \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  llm-finetune:latest \
  python -m src.train --config configs

# Data pipeline
docker run --gpus all \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/data:/app/data \
  llm-finetune:latest \
  python -m src.data_pipeline --config configs/data.yaml
```

### 3. Docker Compose (Multi-service)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trainer

# Run training
docker-compose run --rm trainer python -m src.train --config configs

# Run evaluation
docker-compose run --rm evaluator python -m src.evaluate --config configs ...
```

---

## Cloud VM Deployment

### AWS (p3/g5 instances)

```bash
# Launch instance (example with AWS CLI)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type g5.2xlarge \
  --key-name my-key \
  --security-group-ids sg-xxx \
  --subnet-id subnet-xxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=llm-finetune}]' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}'

# Connect and setup
ssh -i my-key.pem ubuntu@<public-ip>
git clone https://github.com/your-org/llm-finetuning-pipeline-lora
cd llm-finetuning-pipeline-lora
./scripts/setup_cloud.sh  # Installs drivers, docker, etc.
```

### GCP (A100/A2 instances)

```bash
# Create instance
gcloud compute instances create llm-finetune \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-2g \
  --accelerator=type=nvidia-a100-40gb,count=2 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --restart-on-failure

# Install NVIDIA drivers
gcloud compute ssh llm-finetune --zone=us-central1-a --command="
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb
  sudo dpkg -i cuda-keyring.deb
  sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-1 nvidia-driver-535
"
```

### Azure (NDv4/NDm-series)

```bash
# Create VM with InfiniBand
az vm create \
  --resource-group myRG \
  --name llm-finetune \
  --image Ubuntu2204 \
  --size Standard_ND96asr_v4 \
  --accelerator-type NVIDIA_A100 \
  --generate-ssh-keys \
  --data-disk-sizes-gb 500
```

---

## Kubernetes Deployment

### 1. Prerequisites

- Kubernetes 1.25+
- NVIDIA GPU Operator installed
- Container runtime with GPU support (containerd/nvidia-container-toolkit)
- StorageClass for persistent volumes

### 2. Namespace & Secrets

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-finetuning
---
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: hf-credentials
  namespace: llm-finetuning
type: Opaque
stringData:
  HF_TOKEN: "hf_xxx"
  WANDB_API_KEY: "xxx"
```

### 3. Persistent Volumes

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data
  namespace: llm-finetuning
spec:
  accessModes: ["ReadWriteMany"]
  storageClassName: "nfs-client"
  resources:
    requests:
      storage: 200Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: checkpoints
  namespace: llm-finetuning
spec:
  accessModes: ["ReadWriteMany"]
  storageClassName: "nfs-client"
  resources:
    requests:
      storage: 500Gi
```

### 4. Training Job

```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-finetune
  namespace: llm-finetuning
spec:
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      runtimeClassName: nvidia
      volumes:
        - name: config
          configMap:
            name: training-config
        - name: data
          persistentVolumeClaim:
            claimName: training-data
        - name: checkpoints
          persistentVolumeClaim:
            claimName: checkpoints
        - name: hf-token
          secret:
            secretName: hf-credentials
      containers:
        - name: trainer
          image: your-registry/llm-finetune:latest
          command: ["python", "-m", "src.train", "--config", "/config"]
          resources:
            limits:
              nvidia.com/gpu: 2
              memory: "64Gi"
              cpu: "16"
            requests:
              nvidia.com/gpu: 2
              memory: "48Gi"
              cpu: "8"
          volumeMounts:
            - name: config
              mountPath: /config
            - name: data
              mountPath: /app/data
            - name: checkpoints
              mountPath: /app/checkpoints
            - name: hf-token
              mountPath: /secrets
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: HF_TOKEN
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: WANDB_API_KEY
            - name: CUDA_VISIBLE_DEVICES
              value: "0,1"
            - name: NCCL_DEBUG
              value: "INFO"
```

### 5. Distributed Training (PyTorchJob)

```yaml
# pytorchjob.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llm-finetune-distributed
  namespace: llm-finetuning
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          runtimeClassName: nvidia
          containers:
            - name: pytorch
              image: your-registry/llm-finetune:latest
              command: ["torchrun", "--nproc_per_node=4", "--nnodes=2", "--node_rank=0", "-m", "src.train", "--config", "configs"]
              resources:
                limits:
                  nvidia.com/gpu: 4
    Worker:
      replicas: 1
      template:
        spec:
          runtimeClassName: nvidia
          containers:
            - name: pytorch
              image: your-registry/llm-finetune:latest
              command: ["torchrun", "--nproc_per_node=4", "--nnodes=2", "--node_rank=1", "-m", "src.train", "--config", "configs"]
              resources:
                limits:
                  nvidia.com/gpu: 4
```

---

## Managed Services

### AWS SageMaker

```python
# sagemaker_train.py
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="src/train.py",
    source_dir=".",
    role="arn:aws:iam::xxx:role/SageMakerRole",
    instance_type="ml.g5.4xlarge",
    instance_count=2,
    framework_version="2.1",
    py_version="py310",
    hyperparameters={
        "config": "configs",
    },
    environment={
        "HF_TOKEN": "hf_xxx",
        "WANDB_API_KEY": "xxx",
    },
    volume_size=200,
    max_run=86400,
)

estimator.fit({"training": "s3://bucket/data", "config": "s3://bucket/configs"})
```

### Google Vertex AI

```bash
# Submit training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=llm-finetune \
  --worker-pool-spec=machine-type=a2-highgpu-2g,replica-count=1,accelerator-type=nvidia-a100-40gb,accelerator-count=2,container-image-uri=gcr.io/project/llm-finetune:latest \
  --worker-pool-spec=machine-type=a2-highgpu-2g,replica-count=1,accelerator-type=nvidia-a100-40gb,accelerator-count=2,container-image-uri=gcr.io/project/llm-finetune:latest \
  --args="--config=configs" \
  --env-vars=HF_TOKEN=hf_xxx,WANDB_API_KEY=xxx
```

### Azure ML

```python
# azureml_train.py
from azure.ai.ml import command
from azure.ai.ml.entities import Environment

job = command(
    code=".",
    command="python -m src.train --config configs",
    environment=Environment(
        image="your-registry/llm-finetune:latest",
        conda_file="conda.yaml"
    ),
    compute="gpu-cluster",
    distribution={"type": "pytorch", "process_count_per_instance": 4},
    environment_variables={
        "HF_TOKEN": "hf_xxx",
        "WANDB_API_KEY": "xxx",
    },
)
```

---

## Inference Deployment

### 1. Model Merging

```bash
# Merge LoRA adapter into base model
python -m src.train \
  --config configs \
  --merge-and-save \
  --merged-output-dir ./artifacts/merged \
  --push-to-hub \
  --hub-model-id your-org/llama-3-8b-finetuned
```

### 2. Local Inference Server

```bash
# Start API server
python -m src.inference \
  --model-path ./artifacts/merged \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### 3. Docker Inference

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/artifacts:/models \
  your-registry/llm-finetune:inference \
  python -m src.inference --model-path /models/merged
```

### 4. Production Inference (vLLM/TGI)

```bash
# Using vLLM for high-throughput serving
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/artifacts/merged:/model \
  vllm/vllm-openai:latest \
  --model /model \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

---

## Monitoring & Observability

### 1. Prometheus Metrics

```yaml
# prometheus.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: llm-finetune-metrics
  namespace: llm-finetuning
spec:
  selector:
    matchLabels:
      app: llm-finetune
  endpoints:
    - port: metrics
      interval: 30s
```

### 2. Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `gpu_memory_used_bytes` | VRAM usage | > 90% |
| `gpu_utilization` | GPU compute % | < 50% (underutilized) |
| `training_loss` | Loss value | NaN or increasing |
| `eval_loss` | Validation loss | Not decreasing |
| `learning_rate` | LR schedule | Stuck at 0 |
| `grad_norm` | Gradient norm | > 10 (exploding) |
| `throughput_tokens_per_sec` | Training speed | Below baseline |

### 3. Grafana Dashboards

Import dashboard JSON for:
- Training loss curves
- GPU utilization heatmap
- Memory usage over time
- Learning rate schedule

### 4. Log Aggregation

```yaml
# fluent-bit-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush 5
        Log_Level info
    [INPUT]
        Name tail
        Path /var/log/containers/*.log
        Parser docker
        Tag kube.*
    [FILTER]
        Name kubernetes
        Match kube.*
    [OUTPUT]
        Name loki
        Match *
        Url http://loki:3100/loki/api/v1/push
```

---

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/train.yml
name: Training Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=src

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ruff
      - run: ruff check src/
      - run: ruff format --check src/

  docker:
    needs: [test, lint]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: your-registry/llm-finetune:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## Security Hardening

### 1. Container Security

```dockerfile
# Use non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Read-only root filesystem
# Security context in K8s:
# securityContext:
#   readOnlyRootFilesystem: true
#   runAsNonRoot: true
#   runAsUser: 1000
```

### 2. Secret Management

- **Never** commit `.env` or tokens
- Use GitHub Secrets / GitLab CI Variables / Vault
- Rotate tokens regularly
- Use short-lived tokens where possible

### 3. Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-finetune-egress
  namespace: llm-finetuning
spec:
  podSelector:
    matchLabels:
      app: llm-finetune
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: TCP
          port: 53  # DNS
    - to: []  # Allow internet for HF Hub, W&B
      ports:
        - protocol: TCP
          port: 443
```

---

## Disaster Recovery

### Backup Strategy

| Asset | Frequency | Retention |
|-------|-----------|-----------|
| Checkpoints | Every 100 steps | 7 days |
| Best model | On improvement | Permanent |
| Configs | On change | Permanent |
| Logs | Continuous | 30 days |
| Datasets | On change | Permanent |

### Restore Procedure

```bash
# 1. Restore checkpoints from backup
aws s3 sync s3://bucket/checkpoints/ ./checkpoints/

# 2. Resume training
python -m src.train --config configs --resume-from-checkpoint ./checkpoints/checkpoint-5000

# 3. Verify
python -m src.evaluate --config configs --finetuned-model ./checkpoints/best
```

---

## Cost Optimization

### Spot/Preemptible Instances

```yaml
# Use spot instances with checkpointing
# Save checkpoint every N steps
# Resume from checkpoint on interruption
# Estimated 60-90% cost reduction
```

### Gradient Accumulation

```yaml
# Reduce GPU count, increase grad accum
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # Effective batch = 16
# Allows training on smaller GPUs
```

### Mixed Precision

```yaml
bf16: true  # Use BF16 on Ampere+
fp16: false # Only if BF16 not supported
tf32: true  # Enable TF32 on Ampere+
```

---

## Troubleshooting Deployment

| Issue | Solution |
|-------|----------|
| OOM on startup | Reduce batch size, enable CPU offload, use 8-bit |
| NCCL timeout | Increase `ddp_timeout`, check network |
| Slow training | Enable Flash Attention, increase workers |
| Checkpoint not found | Verify `output_dir`, check permissions |
| W&B not logging | Check API key, network, `report_to` config |
| Model not saving | Check disk space, `save_strategy` |