#!/bin/bash

# LLM Model Batch Evaluation Script
# Supports multiple models: Gemini, GPT, DeepSeek, DashScope, Ark, etc.

set -e

echo "=========================================="
echo "  LLM Model Batch Evaluation Script"
echo "=========================================="
echo ""

# Default configuration
PROCESSED_ROOT="/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed"
OUTPUT_ROOT="/data/data-pool/dingyifan/GeminiEvaluation/workspace/results"
MAX_SAMPLES=10000
CONCURRENCY=32
REASONING_EFFORT="low"
MAX_TOKENS=256

# Show help information
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model MODEL              Model name (required)"
    echo "  --datasets DATASETS        Dataset list, comma-separated (default: all)"
    echo "  --max-samples N            Maximum samples per dataset (default: 1000)"
    echo "  --concurrency N            Concurrency level (default: 5)"
    echo "  --reasoning-effort LEVEL   Reasoning effort: low/medium/high (default: low)"
    echo "  --max-tokens N             Maximum tokens (default: 256)"
    echo "  --processed-root PATH      Processed data root directory"
    echo "  --output-root PATH         Result output root directory"
    echo "  --help                     Show this help message"
    echo ""
    echo "Supported model examples:"
    echo "  Gemini:    gemini-3-pro-preview, gemini-2.5-pro, gemini-2.5-flash"
    echo "  OpenAI:    gpt-5, gpt-5-mini, gpt-4.1"
    echo "  DeepSeek:  deepseek-reasoner, deepseek-chat"
    echo "  DashScope: qwen3-vl-32b-thinking, qwen2.5-vl-7b-instruct"
    echo "  Ark:       doubao-seed-1-6-251015"
    echo ""
    echo "Examples:"
    echo "  $0 --model gemini-3-pro-preview --datasets vljailbreakbench,usb"
    echo "  $0 --model gpt-5-mini --reasoning-effort high --concurrency 10"
    echo "  $0 --model deepseek-reasoner --max-samples 500"
}

# Parse command line arguments
MODEL="grok-4-1-fast-non-reasoning"
DATASETS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --datasets)
            DATASETS_ARG="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --reasoning-effort)
            REASONING_EFFORT="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --processed-root)
            PROCESSED_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see help message"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$MODEL" ]; then
    echo "Error: --model parameter is required"
    echo "Use --help to see help message"
    exit 1
fi

# Create model-specific output directory
MODEL_OUTPUT_DIR="$OUTPUT_ROOT/$MODEL"
mkdir -p "$MODEL_OUTPUT_DIR"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Max samples: $MAX_SAMPLES"
echo "  Concurrency: $CONCURRENCY"
echo "  Reasoning effort: $REASONING_EFFORT"
echo "  Max tokens: $MAX_TOKENS"
echo "  Output directory: $MODEL_OUTPUT_DIR"
echo ""

# Get dataset list
if [ -n "$DATASETS_ARG" ]; then
    # Use specified datasets
    IFS=',' read -ra DATASETS <<< "$DATASETS_ARG"
    echo "Specified evaluation datasets: ${DATASETS[@]}"
else
    # Scan all datasets
    echo "Scanning datasets..."
    DATASETS=($(ls "$PROCESSED_ROOT"/*.jsonl 2>/dev/null | xargs -n1 basename | sed 's/.jsonl$//' | sort))
    
    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo "Error: No datasets found in $PROCESSED_ROOT"
        exit 1
    fi
    
    echo "Found ${#DATASETS[@]} datasets:"
    for dataset in "${DATASETS[@]}"; do
        echo "  - $dataset"
    done
fi
echo ""

# Check .env file
if [ ! -f "/data/data-pool/dingyifan/GeminiEvaluation/.env" ]; then
    echo "Warning: .env file not found"
    echo "Please ensure environment variables are set (e.g., GEMINI_API_KEY, OPENAI_API_KEY, etc.)"
fi

# Ask for user confirmation
read -p "Start evaluation? (y/n): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "=========================================="
echo "Starting batch evaluation..."
echo "=========================================="
echo ""

# Record start time
START_TIME=$(date +%s)

# Success and failure counters
SUCCESS_COUNT=0
FAIL_COUNT=0

# Evaluate each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating dataset: $dataset"
    echo "=========================================="
    
    # Run evaluation
    if python3 /data/data-pool/dingyifan/GeminiEvaluation/evaluation/evaluate.py \
        --dataset "$dataset" \
        --max-samples "$MAX_SAMPLES" \
        --concurrency "$CONCURRENCY" \
        --model "$MODEL" \
        --reasoning-effort "$REASONING_EFFORT" \
        --max-tokens "$MAX_TOKENS" \
        --processed-root "$PROCESSED_ROOT" \
        --output-root "$MODEL_OUTPUT_DIR"; then
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✓ $dataset evaluation successful"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "✗ $dataset evaluation failed"
    fi
    
    echo ""
    echo "Progress: $((SUCCESS_COUNT + FAIL_COUNT))/${#DATASETS[@]}"
    echo ""
    
    # Add brief delay to avoid API rate limiting
    sleep 2
done

# Calculate total time elapsed
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Print summary
echo ""
echo "=========================================="
echo "  Batch evaluation completed!"
echo "=========================================="
echo ""
echo "Statistics:"
echo "  Model: $MODEL"
echo "  Total datasets: ${#DATASETS[@]}"
echo "  Success: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo "  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved in: $MODEL_OUTPUT_DIR"
echo ""

# List all result files
if ls "$MODEL_OUTPUT_DIR"/*.jsonl 1> /dev/null 2>&1; then
    echo "Generated result files:"
    ls -lh "$MODEL_OUTPUT_DIR"/*.jsonl | awk '{print "  " $9 " (" $5 ")"}'
else
    echo "No result files generated"
fi

echo ""
echo "=========================================="
