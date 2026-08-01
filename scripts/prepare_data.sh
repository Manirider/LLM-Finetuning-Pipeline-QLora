#!/usr/bin/env bash
# Data Preparation Script
# Downloads, validates, cleans, formats, tokenizes, and splits datasets

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/configs"
DATA_CONFIG="$CONFIG_DIR/data.yaml"

# Default values
DATASET=""
OUTPUT_DIR=""
FORMATS=()
SKIP_DOWNLOAD=false
SKIP_TOKENIZATION=false
TOKENIZER=""
TEMPLATE=""
MAX_SAMPLES=""
SPLIT_RATIOS=""
SEED=42
VERBOSE=false
DRY_RUN=false

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Data preparation pipeline for LLM fine-tuning.

Options:
    -c, --config PATH         Path to data config YAML (default: $DATA_CONFIG)
    -d, --dataset NAME        Specific dataset to process (default: all)
    -o, --output-dir PATH     Output directory (overrides config)
    -f, --formats LIST        Export formats: jsonl, parquet, arrow (default: from config)
    --skip-download           Skip dataset download, use cached
    --skip-tokenization       Skip tokenization step
    --tokenizer NAME          Tokenizer name or path (default: from model config)
    --template NAME           Prompt template: alpaca, chatml, llama3, vicuna, zephyr, plain, custom
    --max-samples N           Maximum samples per dataset
    --split-ratios RATIOS     Train/val/test ratios as comma-separated (e.g., 0.8,0.1,0.1)
    --seed N                  Random seed (default: 42)
    -v, --verbose             Verbose logging
    --dry-run                 Show what would be done without executing
    -h, --help                Show this help

Examples:
    # Full pipeline with defaults
    $0

    # Process specific dataset with custom template
    $0 --dataset alpaca --template llama3

    # Custom output and formats
    $0 --output-dir ./my_data --formats jsonl,parquet

    # Dry run to see what would happen
    $0 --dry-run

EOF
    exit 1
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            DATA_CONFIG="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--formats)
            IFS=',' read -ra FORMATS <<< "$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-tokenization)
            SKIP_TOKENIZATION=true
            shift
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --split-ratios)
            SPLIT_RATIOS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Build Python command
cd "$PROJECT_ROOT"
CMD=("python" "-m" "src.data_pipeline" "--config" "$DATA_CONFIG")

if [[ -n "$DATASET" ]]; then
    CMD+=("--dataset" "$DATASET")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=("--output-dir" "$OUTPUT_DIR")
fi

if [[ ${#FORMATS[@]} -gt 0 ]]; then
    CMD+=("--formats" "${FORMATS[@]}")
fi

if [[ "$SKIP_TOKENIZATION" == true ]]; then
    CMD+=("--skip-tokenization")
fi

if [[ "$SKIP_DOWNLOAD" == true ]]; then
    log_warn "Skip download not fully implemented, using cached datasets"
fi

if [[ -n "$TOKENIZER" ]]; then
    CMD+=("--tokenizer" "$TOKENIZER")
fi

if [[ -n "$TEMPLATE" ]]; then
    CMD+=("--template" "$TEMPLATE")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    CMD+=("--max-samples" "$MAX_SAMPLES")
fi

if [[ -n "$SPLIT_RATIOS" ]]; then
    CMD+=("--split-ratios" "$SPLIT_RATIOS")
fi

if [[ -n "$SEED" ]]; then
    CMD+=("--seed" "$SEED")
fi

if [[ "$VERBOSE" == true ]]; then
    CMD+=("--verbose")
fi

if [[ "$DRY_RUN" == true ]]; then
    CMD+=("--dry-run")
fi

log_info "Running data pipeline with command:"
echo "  ${CMD[*]}"

if [[ "$DRY_RUN" == true ]]; then
    log_info "DRY RUN - not executing"
    exit 0
fi

# Execute
if "${CMD[@]}"; then
    log_success "Data pipeline completed successfully!"
    exit 0
else
    log_error "Data pipeline failed!"
    exit 1
fi