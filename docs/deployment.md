# Production Deployment Guide

## Project: LLM Fine-Tuning Pipeline

## Version: 1.0.0

---

## 1. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRODUCTION DEPLOYMENT                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Client     │───▶│  Load        │───▶│  Inference   │                  │
│  │  Applications│    │  Balancer    │    │  Cluster     │                  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                                                  │                          │
│                    ┌─────────────────────────────┼─────────────────────┐   │
│                    ▼                            ▼                     ▼   │
│             ┌──────────────┐             ┌──────────────┐      ┌──────────┐│
│             │  vLLM Pod 1  │             │  vLLM Pod 2  │      │  vLLM    ││
│             │  (8B Model)  │             │  (8B Model)  │  ... │  Pod N   ││
│             └──────────────┘             └──────────────┘      └──────────┘│
│                    │                            │                        │
│                    └────────────────────────────┼────────────────────────┘
│                                                 ▼
│                                        ┌──────────────────┐
│                                        │  Shared Storage  │
│                                        │  (Model Weights) │
│                                        └──────────────────┘
│                                                 │
│                                        ┌────────┴────────┐
│                                        │  Monitoring &   │
│                                        │  Logging Stack  │
│                                        │  (Prometheus,   │
│                                        │   Grafana,      │
│                                        │   Loki)         │
│                                        └─────────────────┘
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pre-deployment Checklist

### 2.1 Model Preparation

```bash
# 1. Merge LoRA adapter
python scripts/merge_adapter.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter_path ./adapters/best \
  --output_path ./artifacts/models/merged/v1.0.0 \
  --dtype bfloat16

# 2. Validate merged model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./artifacts/models/merged/v1.0.0', torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained('./artifacts/models/merged/v1.0.0')
print('Model loaded successfully')
print(f'Params: {model.num_parameters():,}')
"

# 3. Optional: Quantize for faster inference
# AWQ (requires calibration data)
pip install autoawq
python -m awq.quantize \
  --model_path ./artifacts/models/merged/v1.0.0 \
  --quant_path ./artifacts/models/awq/v1.0.0 \
  --w_bit 4 --q_group_size 128 --version GEMM

# GPTQ
pip install auto-gptq
python -m auto_gptq.quantize \
  --model_name_or_path ./artifacts/models/merged/v1.0.0 \
  --quant_path ./artifacts/models/gptq/v1.0.0 \
  --bits 4 --group_size 128 --desc_act False
```

### 2.2 Container Images

```dockerfile
# Production Dockerfile (multi-stage)
# See Dockerfile.prod for full version

# Build
docker build -t llm-inference:v1.0.0 -f Dockerfile.prod .

# Scan
trivy image llm-inference:v1.0.0

# Push
docker tag llm-inference:v1.0.0 registry.example.com/llm-inference:v1.0.0
docker push registry.example.com/llm-inference:v1.0.0
```

---

## 3. Kubernetes Deployment

### 3.1 Namespace & Config

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-inference
  labels:
    name: llm-inference
    environment: production
```

### 3.2 Model Storage (PVC)

```yaml
# k8s/model-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-weights
  namespace: llm-inference
spec:
  accessModes:
    - ReadOnlyMany
  storageClassName: fast-storage
  resources:
    requests:
      storage: 50Gi
```

### 3.3 Inference Deployment (vLLM)

```yaml
# k8s/deployment-vllm.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: llm-inference
  labels:
    app: vllm-inference
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-inference
  template:
    metadata:
      labels:
        app: vllm-inference
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: llm-inference-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      nodeSelector:
        nvidia.com/gpu: "true"
        node-type: "gpu"
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      containers:
        - name: vllm
          image: registry.example.com/llm-inference:v1.0.0
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
              name: http
            - containerPort: 9090
              name: metrics
          env:
            - name: MODEL_PATH
              value: "/models/merged"
            - name: TENSOR_PARALLEL_SIZE
              value: "1"
            - name: GPU_MEMORY_UTILIZATION
              value: "0.90"
            - name: MAX_MODEL_LEN
              value: "4096"
            - name: DTYPE
              value: "bfloat16"
            - name: TRUST_REMOTE_CODE
              value: "true"
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: "32Gi"
              cpu: "8"
            limits:
              nvidia.com/gpu: 1
              memory: "40Gi"
              cpu: "16"
          volumeMounts:
            - name: model-weights
              mountPath: /models
              readOnly: true
            - name: cache
              mountPath: /root/.cache/huggingface
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
      volumes:
        - name: model-weights
          persistentVolumeClaim:
            claimName: model-weights
        - name: cache
          emptyDir:
            sizeLimit: 10Gi
```

### 3.4 Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-inference
  namespace: llm-inference
  labels:
    app: vllm-inference
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
      name: http
    - port: 9090
      targetPort: 9090
      protocol: TCP
      name: metrics
  selector:
    app: vllm-inference
```

### 3.5 Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-inference-hpa
  namespace: llm-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: vllm_requests_per_second
        target:
          type: AverageValue
          averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 50
          periodSeconds: 30
        - type: Pods
          value: 2
          periodSeconds: 30
      selectPolicy: Max
```

### 3.6 Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-inference-ingress
  namespace: llm-inference
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/limit-connections: "100"
    nginx.ingress.kubernetes.io/limit-rps: "50"
spec:
  ingressClassName: nginx
  rules:
    - host: llm-api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: vllm-inference
                port:
                  number: 80
  tls:
    - hosts:
        - llm-api.example.com
      secretName: llm-api-tls
```

---

## 4. Monitoring Stack

### 4.1 Prometheus Rules

```yaml
# k8s/prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: vllm-alerts
  namespace: monitoring
spec:
  groups:
    - name: vllm-alerts
      rules:
        - alert: VLLMHighLatency
          expr: histogram_quantile(0.95, rate(vllm_request_duration_seconds_bucket[5m])) > 10
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High latency on vLLM inference"
            description: "P95 latency > 10s for 5 minutes"

        - alert: VLLMHighErrorRate
          expr: rate(vllm_requests_failed_total[5m]) / rate(vllm_requests_total[5m]) > 0.05
          for: 2m
          labels:
            severity: critical
          annotations:
            summary: "High error rate on vLLM"
            description: "Error rate > 5% for 2 minutes"

        - alert: VLLMGPUOOM
          expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "GPU OOM risk"
            description: "GPU memory > 95% used"

        - alert: VLLMPodNotReady
          expr: kube_pod_status_ready{namespace="llm-inference",pod=~"vllm-.*"} == 0
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "vLLM pod not ready"
            description: "Pod {{ $labels.pod }} not ready for 5 minutes"
```

### 4.2 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "vLLM Inference Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {"expr": "rate(vllm_requests_total[5m])", "legendFormat": "{{method}} {{status}}"}
        ]
      },
      {
        "title": "Latency (P50, P95, P99)",
        "targets": [
          {"expr": "histogram_quantile(0.50, rate(vllm_request_duration_seconds_bucket[5m]))", "legendFormat": "P50"},
          {"expr": "histogram_quantile(0.95, rate(vllm_request_duration_seconds_bucket[5m]))", "legendFormat": "P95"},
          {"expr": "histogram_quantile(0.99, rate(vllm_request_duration_seconds_bucket[5m]))", "legendFormat": "P99"}
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {"expr": "nvidia_gpu_utilization_gpu", "legendFormat": "GPU {{gpu}}"}
        ]
      },
      {
        "title": "GPU Memory",
        "targets": [
          {"expr": "nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100", "legendFormat": "GPU {{gpu}} %"}
        ]
      },
      {
        "title": "Queue Size",
        "targets": [
          {"expr": "vllm_queue_size", "legendFormat": "Queue"}
        ]
      },
      {
        "title": "Tokens/sec",
        "targets": [
          {"expr": "rate(vllm_generated_tokens_total[5m])", "legendFormat": "Tokens/sec"}
        ]
      }
    ]
  }
}
```

---

## 5. Load Testing

### 5.1 Locust Load Test

```python
# loadtest/locustfile.py
from locust import HttpUser, task, between
import json
import random

class LLMUser(HttpUser):
    wait_time = between(1, 3)
    
    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to sort a list:",
        "What are the benefits of renewable energy?",
        "Summarize the plot of Hamlet:",
        "How does photosynthesis work?",
    ]
    
    @task
    def chat_completion(self):
        prompt = random.choice(self.prompts)
        payload = {
            "model": "llama3-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False
        }
        
        with self.client.post("/v1/chat/completions", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(3)
    def completion(self):
        prompt = random.choice(self.prompts)
        payload = {
            "model": "llama3-8b",
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False
        }
        
        with self.client.post("/v1/completions", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
```

```bash
# Run load test
locust -f loadtest/locustfile.py --host=https://llm-api.example.com \
  --users 50 --spawn-rate 5 --run-time 10m --headless \
  --html loadtest/report.html
```

---

## 6. Blue-Green Deployment

```yaml
# k8s/blue-green.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: vllm-inference
  namespace: llm-inference
spec:
  replicas: 3
  strategy:
    blueGreen:
      activeService: vllm-inference-blue
      previewService: vllm-inference-green
      autoPromotionEnabled: false
  selector:
    matchLabels:
      app: vllm-inference
  template:
    metadata:
      labels:
        app: vllm-inference
    spec:
      # ... pod spec from deployment
```

```bash
# Deploy new version (preview)
kubectl argo rollouts set image vllm-inference \
  vllm=registry.example.com/llm-inference:v1.1.0 \
  -n llm-inference

# Test preview
curl https://llm-api-preview.example.com/health

# Promote
kubectl argo rollouts promote vllm-inference -n llm-inference

# Rollback if needed
kubectl argo rollouts abort vllm-inference -n llm-inference
```

---

## 7. Canary Deployment

```yaml
# k8s/canary.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: vllm-inference
  namespace: llm-inference
spec:
  replicas: 10
  strategy:
    canary:
      steps:
        - setWeight: 10
        - pause: {duration: 10m}
        - setWeight: 30
        - pause: {duration: 10m}
        - setWeight: 50
        - pause: {duration: 10m}
        - setWeight: 100
      canaryMetadata:
        labels:
          version: canary
      stableMetadata:
        labels:
          version: stable
      analysis:
        templates:
          - templateName: success-rate
        startingStep: 1
```

---

## 8. Disaster Recovery

### 8.1 Backup Strategy

```bash
# Daily model backup
#!/bin/bash
# backup-model.sh
DATE=$(date +%Y%m%d)
aws s3 sync /models/merged s3://llm-backups/models/merged/$DATE/ \
  --storage-class GLACIER_IR

# Weekly config backup
kubectl get all,configmap,secret,pvc -n llm-inference -o yaml > \
  backups/llm-inference-$(date +%Y%m%d).yaml
```

### 8.2 Recovery Procedure

```bash
# 1. Restore model weights
aws s3 sync s3://llm-backups/models/merged/20260728/ /models/merged/

# 2. Restore Kubernetes resources
kubectl apply -f backups/llm-inference-20260728.yaml

# 3. Verify
kubectl rollout status deployment/vllm-inference -n llm-inference
curl https://llm-api.example.com/health
```

---

## 9. Cost Optimization

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| **Spot Instances** | 70-90% | Node pools with spot, pod disruption budgets |
| **Right-sizing** | 20-40% | HPA + VPA, monitor actual usage |
| **Model Quantization** | 2-4x throughput | AWQ/GPTQ 4-bit, vLLM support |
| **Request Batching** | 2-10x throughput | vLLM continuous batching |
| **Prefix Caching** | 10-50% latency | vLLM automatic prefix caching |
| **GPU Time-slicing** | 2-4x density | NVIDIA GPU sharing (dev only) |

---

## 10. Security Hardening

### 10.1 Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vllm-inference-netpol
  namespace: llm-inference
spec:
  podSelector:
    matchLabels:
      app: vllm-inference
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090
    - to: []
      ports:
        - protocol: TCP
          port: 443  # For model downloads
        - protocol: TCP
          port: 53   # DNS
```

### 10.2 Pod Security Standards

```yaml
# k8s/pod-security.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: llm-inference-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - "configMap"
    - "emptyDir"
    - "projected"
    - "secret"
    - "persistentVolumeClaim"
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
```

---

## 11. Compliance & Auditing

### 11.1 Audit Logging

```yaml
# k8s/audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  - level: Metadata
    resources:
      - group: ""
        resources: ["pods", "services", "configmaps", "secrets"]
  - level: RequestResponse
    resources:
      - group: "apps"
        resources: ["deployments", "rollouts"]
    namespaces: ["llm-inference"]
```

### 11.2 Data Handling

- **No PII in training data**: Validate datasets
- **Model outputs**: Log sampling only, no full conversations
- **Retention**: 30 days for logs, 90 days for metrics
- **Encryption**: At-rest (EBS encryption), in-transit (TLS 1.3)

---

## 12. Runbooks

### 12.1 High Latency

```markdown
## Runbook: High Inference Latency

### Symptoms
- P95 latency > 10s
- Queue size growing
- GPU utilization low

### Diagnosis
1. Check queue size: `vllm_queue_size`
2. Check GPU memory: `nvidia_gpu_memory_used_bytes`
3. Check batch size: `vllm_batch_size`
4. Check for OOM kills: `dmesg | grep -i oom`

### Resolution
1. If queue growing: Scale up HPA replicas
2. If GPU memory high: Reduce max_model_len or enable quantization
3. If GPU util low: Check data loading, increase batch size
4. If OOM: Reduce tensor_parallel_size or model size
```

### 12.2 Model Quality Degradation

```markdown
## Runbook: Model Quality Degradation

### Symptoms
- User complaints about quality
- Automated eval metrics drop
- Increased hallucination reports

### Diagnosis
1. Run evaluation suite: `python -m src.evaluate --model ./artifacts/models/merged/v1.0.0`
2. Compare with baseline metrics
3. Check for data drift in inputs
4. Verify model weights integrity

### Resolution
1. If metrics dropped: Rollback to previous version
2. If data drift: Retrain with new data
3. If weights corrupted: Re-merge from adapter
4. Document findings in incident report
```

---

*End of Deployment Guide*