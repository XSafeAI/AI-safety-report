#!/bin/bash

# 批量计算所有评测结果的指标
# 使用Qwen3Guard-Gen-8B本地模型进行安全性评测

set -e

echo "=========================================="
echo "  批量指标计算脚本"
echo "  评测模型: Qwen3Guard-Gen-8B"
echo "=========================================="
echo ""

export CUDA_VISIBLE_DEVICES=4,5,6,7

# 配置参数
RESULT_ROOT="/data/data-pool/dingyifan/GeminiEvaluation/workspace/results/grok-4-1-fast-non-reasoning"
MODEL_PATH="/data/data-pool/dingyifan/GeminiEvaluation/external/model/Qwen3Guard-Gen-8B"
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
BATCH_SIZE=1000

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 找不到模型路径 $MODEL_PATH"
    echo "请确保Qwen3Guard-Gen-8B模型已下载到该路径"
    exit 1
fi

# 检查结果目录
if [ ! -d "$RESULT_ROOT" ]; then
    echo "错误: 找不到结果目录 $RESULT_ROOT"
    exit 1
fi

# 查找所有结果文件（排除已评测的文件）
echo "扫描结果文件..."
RESULT_FILES=($(find "$RESULT_ROOT" -name "*.jsonl" ! -name "*_evaluated.jsonl" | sort))

if [ ${#RESULT_FILES[@]} -eq 0 ]; then
    echo "警告: 在 $RESULT_ROOT 中没有找到任何结果文件"
    exit 1
fi

echo "找到 ${#RESULT_FILES[@]} 个结果文件:"
for file in "${RESULT_FILES[@]}"; do
    echo "  - $(basename $file)"
done
echo ""

# 切换到评测目录
cd /data/data-pool/dingyifan/GeminiEvaluation/evaluation

# 成功和失败计数
SUCCESS_COUNT=0
FAIL_COUNT=0

# 逐个处理结果文件
for result_file in "${RESULT_FILES[@]}"; do
    filename=$(basename "$result_file")
    echo ""
    echo "=========================================="
    echo "处理: $filename"
    echo "=========================================="
    echo "使用Qwen3Guard安全性评测模式"
    
    # 运行评测
    if python3 compute_metrics.py \
        --result-file "$result_file" \
        --model-path "$MODEL_PATH" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --batch-size "$BATCH_SIZE"; then
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✓ $filename 评测完成"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "✗ $filename 评测失败"
    fi
    
    echo ""
done

# 打印总结
echo ""
echo "=========================================="
echo "  批量评测完成!"
echo "=========================================="
echo ""
echo "统计信息:"
echo "  总文件数: ${#RESULT_FILES[@]}"
echo "  成功: $SUCCESS_COUNT"
echo "  失败: $FAIL_COUNT"
echo ""


