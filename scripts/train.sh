#!/usr/bin/env bash
# Training launch script

set -euo pipefail

# Default values
CONFIG_DIR="configs"
DATA_CONFIG="configs/data.yaml"
OUTPUT_DIR="./checkpoints"
RESUME_FROM=""
MERGE_AND_SAVE=false
MERGED_OUTPUT_DIR="./artifacts/models/merged"
MERGED_DTYPE="bfloat16"
PUSH_TO_HUB=false
HUB_MODEL_ID=""
HUB_TOKEN=""
SEED=42
DRY_RUN=false
VERBOSE=false

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

Launch LLM fine-tuning with QLoRA.

Options:
    --config DIR              Config directory (default: $CONFIG_DIR)
    --data-config FILE        Data config file (default: $DATA_CONFIG)
    --output-dir DIR          Output directory (default: $OUTPUT_DIR)
    --resume-from PATH        Resume from checkpoint
    --merge-and-save          Merge adapter after training
    --merged-output-dir DIR   Merged model output (default: $MERGED_OUTPUT_DIR)
    --merged-dtype DTYPE      Merged model dtype (default: $MERGED_DTYPE)
    --push-to-hub             Push to Hugging Face Hub
    --hub-model-id ID         Hub model ID
    --hub-token TOKEN         Hub token
    --seed N                  Random seed (default: $SEED)
    --dry-run                 Show config without training
    -v, --verbose             Verbose logging
    -h, --help                Show this help

Examples:
    $0
    $0 --output-dir ./my_checkpoints --epochs 5
    $0 --merge-and-save --push-to-hub --hub-model-id my-org/model --hub-token \$HF_TOKEN
    $0 --resume-from ./checkpoints/checkpoint-500
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG_DIR="$2"; shift 2 ;;
        --data-config) DATA_CONFIG="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --resume-from) RESUME_FROM="$2"; shift 2 ;;
        --merge-and-save) MERGE_AND_SAVE=true; shift ;;
        --merged-output-dir) MERGED_OUTPUT_DIR="$2"; shift 2 ;;
        --merged-dtype) MERGED_DTYPE="$2"; shift 2 ;;
        --push-to-hub) PUSH_TO_HUB=true; shift ;;
        --hub-model-id) HUB_MODEL_ID="$2"; shift 2 ;;
        --hub-token) HUB_TOKEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -v|--verbose) VERBOSE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Validate
if [[ ! -d "$CONFIG_DIR" ]]; then
    log_error "Config directory not found: $CONFIG_DIR"
    exit 1
fi

# Build command
CMD=("python" "-m" "src.train" "--config" "$CONFIG_DIR" "--data-config" "$DATA_CONFIG" "--output-dir" "$OUTPUT_DIR" "--seed" "$SEED")

if [[ -n "$RESUME_FROM" ]]; then
    CMD+=("--resume-from-checkpoint" "$RESUME_FROM")
fi

if [[ "$MERGE_AND_SAVE" == true ]]; then
    CMD+=("--merge-and-save" "--merged-output-dir" "$MERGED_OUTPUT_DIR" "--merged-dtype" "$MERGED_DTYPE")
fi

if [[ "$PUSH_TO_HUB" == true ]]; then
    CMD+=("--push-to-hub")
    [[ -n "$HUB_MODEL_ID" ]] && CMD+=("--hub-model-id" "$HUB_MODEL_ID")
    [[ -n "$HUB_TOKEN" ]] && CMD+=("--hub-token" "$HUB_TOKEN")
fi

if [[ "$DRY_RUN" == true ]]; then
    CMD+=("--dry-run")
fi

if [[ "$VERBOSE" == true ]]; then
    CMD+=("--verbose")
fi

log_info "Starting training..."
log_info "Config: $CONFIG_DIR"
log_info "Output: $OUTPUT_DIR"

if [[ "$DRY_RUN" == true ]]; then
    log_info "DRY RUN - command that would be executed:"
    echo "  ${CMD[*]}"
    exit 0
fi

# Execute
if "${CMD[@]}"; then
    log_success "Training completed successfully!"
else
    log_error "Training failed!"
    exit 1
fi