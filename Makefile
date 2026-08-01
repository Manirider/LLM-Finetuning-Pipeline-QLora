# ==============================================================================
# Makefile for LLM Fine-Tuning Pipeline
# ==============================================================================
# Usage: make <target>
# Run `make help` to see all available targets
# ==============================================================================

.PHONY: help install install-dev install-all lint format typecheck test test-unit test-integration test-smoke test-cov clean build build-dev build-inference docker-build docker-push docker-run docker-run-dev docker-run-eval docker-run-infer docker-run-tensorboard docker-run-jupyter docker-stop docker-logs docker-clean data-download data-process train eval infer benchmark merge-adapter check-config check-secrets security-scan pre-commit ci-local release

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DOCKER := docker
COMPOSE := docker compose
PROJECT_NAME := llm-finetuning-pipeline
VERSION := $(shell grep '^version' pyproject.toml | cut -d'"' -f2)
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_IMAGE_LATEST := $(PROJECT_NAME):latest
CONFIG_DIR := configs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ==============================================================================
# HELP
# ==============================================================================
help: ## Show this help message
	@echo "$(BLUE)=====================================================================$(NC)"
	@echo "$(BLUE)LLM Fine-Tuning Pipeline - Makefile$(NC)"
	@echo "$(BLUE)=====================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make install-dev      # Install development dependencies"
	@echo "  make train            # Run training locally"
	@echo "  make docker-build     # Build Docker image"
	@echo "  make docker-run       # Run training in Docker"
	@echo "  make test-cov         # Run tests with coverage"

# ==============================================================================
# INSTALLATION
# ==============================================================================
install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-all: ## Install all optional dependencies
	@echo "$(GREEN)Installing all dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[all]"

# ==============================================================================
# CODE QUALITY
# ==============================================================================
lint: ## Run linting (ruff)
	@echo "$(GREEN)Running linter...$(NC)"
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Format code (ruff + black)
	@echo "$(GREEN)Formatting code...$(NC)"
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run type checking (mypy)
	@echo "$(GREEN)Running type checker...$(NC)"
	mypy src/

check-all: lint typecheck ## Run all code quality checks

pre-commit: ## Run pre-commit hooks
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

# ==============================================================================
# TESTING
# ==============================================================================
test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests (requires GPU)
	@echo "$(GREEN)Running integration tests...$(NC)"
	pytest tests/integration/ -v -m "integration"

test-smoke: ## Run smoke tests (fast CI checks)
	@echo "$(GREEN)Running smoke tests...$(NC)"
	pytest tests/smoke/ -v -m "smoke"

test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:htmlcov --cov-fail-under=90

test-gpu: ## Run GPU tests (requires GPU)
	@echo "$(GREEN)Running GPU tests...$(NC)"
	pytest tests/ -v -m "gpu"

# ==============================================================================
# CLEANUP
# ==============================================================================
clean: ## Clean build artifacts and cache
	@echo "$(GREEN)Cleaning up...$(NC)"
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	rm -rf .venv/ venv/ env/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage.*" -delete 2>/dev/null || true

clean-data: ## Clean data directories (careful!)
	@echo "$(YELLOW)Cleaning data directories...$(NC)"
	rm -rf data/raw/* data/processed/*

clean-checkpoints: ## Clean checkpoints (careful!)
	@echo "$(YELLOW)Cleaning checkpoints...$(NC)"
	rm -rf checkpoints/*

clean-adapters: ## Clean adapters (careful!)
	@echo "$(YELLOW)Cleaning adapters...$(NC)"
	rm -rf adapters/*

clean-outputs: ## Clean outputs (careful!)
	@echo "$(YELLOW)Cleaning outputs...$(NC)"
	rm -rf outputs/* artifacts/* runs/* logs/*

clean-docker: ## Clean Docker images and containers
	@echo "$(GREEN)Cleaning Docker...$(NC)"
	$(DOCKER) system prune -f
	$(DOCKER) image prune -f

# ==============================================================================
# CONFIGURATION
# ==============================================================================
check-config: ## Validate all configuration files
	@echo "$(GREEN)Validating configurations...$(NC)"
	$(PYTHON) -c "from src.config import ConfigManager; c=ConfigManager(); print('Training:', c.training.num_train_epochs); print('Model:', c.model.model_name_or_path); print('Data:', c.data.dataset_name); print('Logging:', c.logging.log_level)"

check-secrets: ## Check for exposed secrets
	@echo "$(GREEN)Checking for secrets...$(NC)"
	@if grep -r "sk-\|hf_\|wandb_\|ghp_" --include="*.py" --include="*.yaml" --include="*.yml" --include="*.sh" . 2>/dev/null | grep -v ".env" | grep -v ".git" | grep -v "example"; then \
		echo "$(RED)Potential secrets found!$(NC)"; exit 1; \
	else \
		echo "$(GREEN)No secrets found in code.$(NC)"; \
	fi

# ==============================================================================
# DATA PIPELINE
# ==============================================================================
data-download: ## Download datasets
	@echo "$(GREEN)Downloading datasets...$(NC)"
	$(PYTHON) -m src.data_pipeline --config $(CONFIG_DIR)/data.yaml --download-only

data-process: ## Process datasets (download, validate, clean, format, tokenize, split, save)
	@echo "$(GREEN)Processing datasets...$(NC)"
	$(PYTHON) -m src.data_pipeline --config $(CONFIG_DIR)/data.yaml --process

data-stats: ## Show dataset statistics
	@echo "$(GREEN)Dataset statistics:$(NC)"
	$(PYTHON) -m src.data_pipeline --config $(CONFIG_DIR)/data.yaml --stats

data-validate: ## Validate existing processed data
	@echo "$(GREEN)Validating data...$(NC)"
	$(PYTHON) -m src.data_pipeline --config $(CONFIG_DIR)/data.yaml --validate

# ==============================================================================
# TRAINING
# ==============================================================================
train: ## Run training locally
	@echo "$(GREEN)Starting training...$(NC)"
	$(PYTHON) -m src.train --config $(CONFIG_DIR)/training.yaml

train-resume: ## Resume training from latest checkpoint
	@echo "$(GREEN)Resuming training...$(NC)"
	$(PYTHON) -m src.train --config $(CONFIG_DIR)/training.yaml --resume_from_checkpoint $(shell ls -td checkpoints/checkpoint-* | head -1)

train-profile: ## Run training with profiling
	@echo "$(GREEN)Running training with profiler...$(NC)"
	$(PYTHON) -m src.train --config $(CONFIG_DIR)/training.yaml --profile

# ==============================================================================
# EVALUATION
# ==============================================================================
eval: ## Run evaluation
	@echo "$(GREEN)Running evaluation...$(NC)"
	$(PYTHON) -m src.evaluate --config $(CONFIG_DIR)/evaluation.yaml --model_path ./adapters/best

eval-base: ## Evaluate base model only
	@echo "$(GREEN)Evaluating base model...$(NC)"
	$(PYTHON) -m src.evaluate --config $(CONFIG_DIR)/evaluation.yaml --base_model meta-llama/Meta-Llama-3-8B-Instruct --model_path ""

eval-compare: ## Compare base vs fine-tuned
	@echo "$(GREEN)Comparing base vs fine-tuned...$(NC)"
	$(PYTHON) -m src.evaluate --config $(CONFIG_DIR)/evaluation.yaml --base_model meta-llama/Meta-Llama-3-8B-Instruct --model_path ./adapters/best --compare

eval-report: ## Generate evaluation report
	@echo "$(GREEN)Generating evaluation report...$(NC)"
	$(PYTHON) -m src.evaluate --config $(CONFIG_DIR)/evaluation.yaml --model_path ./adapters/best --generate_report

# ==============================================================================
# INFERENCE
# ==============================================================================
infer: ## Run interactive inference
	@echo "$(GREEN)Starting inference...$(NC)"
	$(PYTHON) -m src.inference --model_path ./adapters/best

infer-chat: ## Run chat inference
	@echo "$(GREEN)Starting chat inference...$(NC)"
	$(PYTHON) -m src.inference --model_path ./adapters/best --chat

infer-stream: ## Run streaming inference
	@echo "$(GREEN)Starting streaming inference...$(NC)"
	$(PYTHON) -m src.inference --model_path ./adapters/best --stream

# ==============================================================================
# BENCHMARKING
# ==============================================================================
benchmark: ## Run inference benchmark
	@echo "$(GREEN)Running benchmark...$(NC)"
	$(PYTHON) -m src.inference --model_path ./adapters/best --benchmark

benchmark-full: ## Run full benchmark with multiple batch sizes
	@echo "$(GREEN)Running full benchmark...$(NC)"
	$(PYTHON) -m scripts.benchmark --model_path ./adapters/best --batch-sizes 1,2,4,8,16 --seq-lens 512,1024,2048,4096

# ==============================================================================
# ADAPTER MANAGEMENT
# ==============================================================================
merge-adapter: ## Merge LoRA adapter into base model
	@echo "$(GREEN)Merging adapter...$(NC)"
	$(PYTHON) -m scripts.merge_adapter \
		--base_model meta-llama/Meta-Llama-3-8B-Instruct \
		--adapter_path ./adapters/best \
		--output_path ./artifacts/models/merged/v1.0.0 \
		--dtype bfloat16

merge-adapter-fp16: ## Merge adapter with FP16
	@echo "$(GREEN)Merging adapter (FP16)...$(NC)"
	$(PYTHON) -m scripts.merge_adapter \
		--base_model meta-llama/Meta-Llama-3-8B-Instruct \
		--adapter_path ./adapters/best \
		--output_path ./artifacts/models/merged/v1.0.0-fp16 \
		--dtype float16

# ==============================================================================
# DOCKER
# ==============================================================================
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image: $(DOCKER_IMAGE)$(NC)"
	$(DOCKER) build \
		--build-arg INSTALL_FLASH_ATTN=true \
		-t $(DOCKER_IMAGE) \
		-t $(DOCKER_IMAGE_LATEST) \
		.

docker-build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(NC)"
	$(DOCKER) build \
		--target development \
		-t $(PROJECT_NAME):dev \
		.

docker-build-inference: ## Build inference Docker image
	@echo "$(GREEN)Building inference Docker image...$(NC)"
	$(DOCKER) build \
		--target inference \
		-t $(PROJECT_NAME):inference \
		.

docker-push: ## Push Docker image to registry
	@echo "$(GREEN)Pushing Docker image...$(NC)"
	$(DOCKER) push $(DOCKER_IMAGE)
	$(DOCKER) push $(DOCKER_IMAGE_LATEST)

docker-run: ## Run training in Docker
	@echo "$(GREEN)Running training in Docker...$(NC)"
	$(COMPOSE) up --build trainer

docker-run-dev: ## Run development environment in Docker
	@echo "$(GREEN)Starting development environment...$(NC)"
	$(COMPOSE) --profile dev up --build jupyter

docker-run-eval: ## Run evaluation in Docker
	@echo "$(GREEN)Running evaluation in Docker...$(NC)"
	$(COMPOSE) up --build evaluator

docker-run-infer: ## Run inference server in Docker
	@echo "$(GREEN)Starting inference server...$(NC)"
	$(COMPOSE) up --build -d inference

docker-run-tensorboard: ## Start TensorBoard in Docker
	@echo "$(GREEN)Starting TensorBoard...$(NC)"
	$(COMPOSE) up --build -d tensorboard

docker-run-jupyter: ## Start Jupyter Lab in Docker
	@echo "$(GREEN)Starting Jupyter Lab...$(NC)"
	$(COMPOSE) --profile dev up --build -d jupyter

docker-run-merge: ## Run adapter merge in Docker
	@echo "$(GREEN)Merging adapter in Docker...$(NC)"
	$(COMPOSE) up --build merge-adapter

docker-stop: ## Stop all Docker services
	@echo "$(GREEN)Stopping Docker services...$(NC)"
	$(COMPOSE) down

docker-logs: ## Show Docker logs
	@echo "$(GREEN)Showing Docker logs...$(NC)"
	$(COMPOSE) logs -f

docker-ps: ## Show Docker containers
	@echo "$(GREEN)Docker containers:$(NC)"
	$(COMPOSE) ps

docker-clean: ## Clean Docker resources
	@echo "$(GREEN)Cleaning Docker resources...$(NC)"
	$(COMPOSE) down -v --rmi all --remove-orphans
	$(DOCKER) system prune -f

# ==============================================================================
# SECURITY
# ==============================================================================
security-scan: ## Run security scan on Docker image
	@echo "$(GREEN)Running security scan...$(NC)"
	trivy image --severity HIGH,CRITICAL $(DOCKER_IMAGE_LATEST)

security-audit: ## Run dependency security audit
	@echo "$(GREEN)Running dependency audit...$(NC)"
	pip-audit -r requirements.txt

# ==============================================================================
# CI/CD
# ==============================================================================
ci-local: check-all test-cov security-audit ## Run full CI pipeline locally

# ==============================================================================
# RELEASE
# ==============================================================================
release-patch: ## Bump patch version (0.0.X)
	@echo "$(GREEN)Bumping patch version...$(NC)"
	cz bump --increment PATCH

release-minor: ## Bump minor version (0.X.0)
	@echo "$(GREEN)Bumping minor version...$(NC)"
	cz bump --increment MINOR

release-major: ## Bump major version (X.0.0)
	@echo "$(GREEN)Bumping major version...$(NC)"
	cz bump --increment MAJOR

changelog: ## Generate changelog
	@echo "$(GREEN)Generating changelog...$(NC)"
	cz changelog

# ==============================================================================
# UTILITIES
# ==============================================================================
gpu-info: ## Show GPU information
	@echo "$(GREEN)GPU Information:$(NC)"
	nvidia-smi

gpu-monitor: ## Monitor GPU usage
	@echo "$(GREEN)Monitoring GPU (Ctrl+C to stop)...$(NC)"
	watch -n 1 nvidia-smi

disk-usage: ## Show disk usage
	@echo "$(GREEN)Disk Usage:$(NC)"
	du -sh data/ checkpoints/ adapters/ outputs/ artifacts/ logs/ 2>/dev/null | sort -h

git-status: ## Show git status
	@echo "$(GREEN)Git Status:$(NC)"
	git status --short

# ==============================================================================
# COMPOSITE TARGETS
# ==============================================================================
setup: install-dev pre-commit data-process ## Full project setup

dev-loop: format lint test-unit ## Quick development loop

full-pipeline: data-process train eval merge-adapter benchmark ## Run full pipeline

all: clean install-dev check-all test-cov docker-build ## Full clean build and test

# ==============================================================================
# Print variables (for debugging)
# ==============================================================================
print-vars: ## Print Makefile variables
	@echo "PROJECT_NAME: $(PROJECT_NAME)"
	@echo "VERSION: $(VERSION)"
	@echo "DOCKER_IMAGE: $(DOCKER_IMAGE)"
	@echo "CONFIG_DIR: $(CONFIG_DIR)"
	@echo "PYTHON: $(PYTHON)"
	@echo "PIP: $(PIP)"
	@echo "DOCKER: $(DOCKER)"
	@echo "COMPOSE: $(COMPOSE)"