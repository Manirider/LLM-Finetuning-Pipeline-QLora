#!/usr/bin/env bash
# Merge LoRA adapter into base model

set -euo pipefail

# Default values
BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH="./adapters/best"
OUTPUT_PATH="./artifacts/models/merged/v1.0.0"
DTYPE="bfloat16"
PUSH_TO_HUB=false
HUB_MODEL_ID=""
HUB_TOKEN=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Merge LoRA adapter weights into base model.

Options:
    --base-model MODEL        Base model name or path (default: $BASE_MODEL)
    --adapter-path PATH       Path to LoRA adapter (default: $ADAPTER_PATH)
    --output-path PATH        Output path for merged model (default: $OUTPUT_PATH)
    --dtype DTYPE             Output dtype: float16, bfloat16, float32 (default: $DTYPE)
    --push-to-hub             Push merged model to Hugging Face Hub
    --hub-model-id ID         Hub model ID (required if --push-to-hub)
    --hub-token TOKEN         Hub token (or set HF_TOKEN env var)
    -h, --help                Show this help

Examples:
    $0
    $0 --base-model mistralai/Mistral-7B-Instruct-v0.3 --adapter-path ./adapters/best --output-path ./merged
    $0 --push-to-hub --hub-model-id my-org/llama-3-8b-finetuned --hub-token \$HF_TOKEN
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --adapter-path) ADAPTER_PATH="$2"; shift 2 ;;
        --output-path) OUTPUT_PATH="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --push-to-hub) PUSH_TO_HUB=true; shift ;;
        --hub-model-id) HUB_MODEL_ID="$2"; shift 2 ;;
        --hub-token) HUB_TOKEN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Validate
if [[ ! -d "$ADAPTER_PATH" ]]; then
    log_error "Adapter path does not exist: $ADAPTER_PATH"
    exit 1
fi

if [[ "$PUSH_TO_HUB" == true && -z "$HUB_MODEL_ID" ]]; then
    log_error "--hub-model-id required when --push-to-hub is set"
    exit 1
fi

log_info "Merging LoRA adapter into base model"
log_info "  Base model: $BASE_MODEL"
log_info "  Adapter: $ADAPTER_PATH"
log_info "  Output: $OUTPUT_PATH"
log_info "  Dtype: $DTYPE"

# Run merge
python -m scripts.merge_adapter \
    --base_model "$BASE_MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --output_path "$OUTPUT_PATH" \
    --dtype "$DTYPE" \
    ${PUSH_TO_HUB:+--push_to_hub} \
    ${HUB_MODEL_ID:+--hub_model_id "$HUB_MODEL_ID"} \
    ${HUB_TOKEN:+--hub_token "$HUB_TOKEN"}

if [[ $? -eq 0 ]]; then
    log_success "Model merged successfully!"
    log_info "Merged model saved to: $OUTPUT_PATH"
else
    log_error "Merge failed!"
    exit 1
fi