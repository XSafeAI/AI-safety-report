#!/bin/bash

# Batch compute metrics for all evaluation results
# Uses Qwen3Guard-Gen-8B local model for safety evaluation

set -e

echo "=========================================="
echo "  Batch Metrics Computation Script"
echo "  Evaluation Model: Qwen3Guard-Gen-8B"
echo "=========================================="
echo ""

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Configuration parameters
RESULT_ROOT="/data/data-pool/dingyifan/GeminiEvaluation/workspace/results/grok-4-1-fast-non-reasoning"
MODEL_PATH="/data/data-pool/dingyifan/GeminiEvaluation/external/model/Qwen3Guard-Gen-8B"
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
BATCH_SIZE=1000

# Check model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found $MODEL_PATH"
    echo "Please ensure Qwen3Guard-Gen-8B model is downloaded to this path"
    exit 1
fi

# Check result directory
if [ ! -d "$RESULT_ROOT" ]; then
    echo "Error: Result directory not found $RESULT_ROOT"
    exit 1
fi

# Find all result files (exclude already evaluated files)
echo "Scanning result files..."
RESULT_FILES=($(find "$RESULT_ROOT" -name "*.jsonl" ! -name "*_evaluated.jsonl" | sort))

if [ ${#RESULT_FILES[@]} -eq 0 ]; then
    echo "Warning: No result files found in $RESULT_ROOT"
    exit 1
fi

echo "Found ${#RESULT_FILES[@]} result files:"
for file in "${RESULT_FILES[@]}"; do
    echo "  - $(basename $file)"
done
echo ""

# Change to evaluation directory
cd /data/data-pool/dingyifan/GeminiEvaluation/evaluation

# Success and failure counters
SUCCESS_COUNT=0
FAIL_COUNT=0

# Process each result file
for result_file in "${RESULT_FILES[@]}"; do
    filename=$(basename "$result_file")
    echo ""
    echo "=========================================="
    echo "Processing: $filename"
    echo "=========================================="
    echo "Using Qwen3Guard safety evaluation mode"
    
    # Run evaluation
    if python3 compute_metrics.py \
        --result-file "$result_file" \
        --model-path "$MODEL_PATH" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --batch-size "$BATCH_SIZE"; then
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✓ $filename evaluation completed"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "✗ $filename evaluation failed"
    fi
    
    echo ""
done

# Print summary
echo ""
echo "=========================================="
echo "  Batch evaluation completed!"
echo "=========================================="
echo ""
echo "Statistics:"
echo "  Total files: ${#RESULT_FILES[@]}"
echo "  Success: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo ""

