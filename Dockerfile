# Multi-stage Dockerfile for LLM Fine-Tuning Pipeline

# ==============================================================================
# STAGE 1: Builder
# ==============================================================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /build

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# STAGE 2: Runtime
# ==============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HF_DATASETS_CACHE=/cache/huggingface/datasets \
    TORCH_HOME=/cache/torch \
    DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    git \
    curl \
    ca-certificates \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

# Copy site-packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and configurations
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./

# Create operational directories
RUN mkdir -p /app/data /app/checkpoints /app/adapters /app/logs /app/outputs /app/artifacts /app/cache /cache

# Set up non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app /cache
USER appuser

EXPOSE 8000 6006 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('PyTorch version:', torch.__version__)" || exit 1

ENTRYPOINT ["python", "-m", "src.train", "--config", "configs"]