#!/bin/bash

# LLM模型批量评测脚本
# 支持多种模型：Gemini, GPT, DeepSeek, DashScope, Ark等

set -e

echo "=========================================="
echo "  LLM模型批量评测脚本"
echo "=========================================="
echo ""

# 默认配置
PROCESSED_ROOT="/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed"
OUTPUT_ROOT="/data/data-pool/dingyifan/GeminiEvaluation/workspace/results"
MAX_SAMPLES=10000
CONCURRENCY=32
REASONING_EFFORT="low"
MAX_TOKENS=256

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --model MODEL              模型名称（必需）"
    echo "  --datasets DATASETS        数据集列表，逗号分隔（默认：全部）"
    echo "  --max-samples N            每个数据集最大样本数（默认：1000）"
    echo "  --concurrency N            并发数（默认：5）"
    echo "  --reasoning-effort LEVEL   推理强度：low/medium/high（默认：low）"
    echo "  --max-tokens N             最大token数（默认：256）"
    echo "  --processed-root PATH      处理后数据根目录"
    echo "  --output-root PATH         结果输出根目录"
    echo "  --help                     显示此帮助信息"
    echo ""
    echo "支持的模型示例:"
    echo "  Gemini:    gemini-3-pro-preview, gemini-2.5-pro, gemini-2.5-flash"
    echo "  OpenAI:    gpt-5, gpt-5-mini, gpt-4.1"
    echo "  DeepSeek:  deepseek-reasoner, deepseek-chat"
    echo "  DashScope: qwen3-vl-32b-thinking, qwen2.5-vl-7b-instruct"
    echo "  Ark:       doubao-seed-1-6-251015"
    echo ""
    echo "示例:"
    echo "  $0 --model gemini-3-pro-preview --datasets vljailbreakbench,usb"
    echo "  $0 --model gpt-5-mini --reasoning-effort high --concurrency 10"
    echo "  $0 --model deepseek-reasoner --max-samples 500"
}

# 解析命令行参数
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
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL" ]; then
    echo "错误: 必须指定 --model 参数"
    echo "使用 --help 查看帮助"
    exit 1
fi

# 创建模型专属输出目录
MODEL_OUTPUT_DIR="$OUTPUT_ROOT/$MODEL"
mkdir -p "$MODEL_OUTPUT_DIR"

echo "配置信息:"
echo "  模型: $MODEL"
echo "  最大样本数: $MAX_SAMPLES"
echo "  并发数: $CONCURRENCY"
echo "  推理强度: $REASONING_EFFORT"
echo "  最大tokens: $MAX_TOKENS"
echo "  输出目录: $MODEL_OUTPUT_DIR"
echo ""

# 获取数据集列表
if [ -n "$DATASETS_ARG" ]; then
    # 使用指定的数据集
    IFS=',' read -ra DATASETS <<< "$DATASETS_ARG"
    echo "指定评测数据集: ${DATASETS[@]}"
else
    # 扫描所有数据集
    echo "扫描数据集..."
    DATASETS=($(ls "$PROCESSED_ROOT"/*.jsonl 2>/dev/null | xargs -n1 basename | sed 's/.jsonl$//' | sort))
    
    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo "错误: 在 $PROCESSED_ROOT 中没有找到任何数据集"
        exit 1
    fi
    
    echo "找到 ${#DATASETS[@]} 个数据集:"
    for dataset in "${DATASETS[@]}"; do
        echo "  - $dataset"
    done
fi
echo ""

# 检查.env文件
if [ ! -f "/data/data-pool/dingyifan/GeminiEvaluation/.env" ]; then
    echo "警告: 找不到.env文件"
    echo "请确保已设置相应的环境变量（如 GEMINI_API_KEY, OPENAI_API_KEY 等）"
fi

# 询问用户确认
read -p "是否开始评测？(y/n): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "开始批量评测..."
echo "=========================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 成功和失败计数
SUCCESS_COUNT=0
FAIL_COUNT=0

# 逐个评测数据集
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "评测数据集: $dataset"
    echo "=========================================="
    
    # 运行评测
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
        echo "✓ $dataset 评测成功"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "✗ $dataset 评测失败"
    fi
    
    echo ""
    echo "进度: $((SUCCESS_COUNT + FAIL_COUNT))/${#DATASETS[@]}"
    echo ""
    
    # 添加短暂延迟，避免API限流
    sleep 2
done

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# 打印总结
echo ""
echo "=========================================="
echo "  批量评测完成!"
echo "=========================================="
echo ""
echo "统计信息:"
echo "  模型: $MODEL"
echo "  总数据集: ${#DATASETS[@]}"
echo "  成功: $SUCCESS_COUNT"
echo "  失败: $FAIL_COUNT"
echo "  总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "结果保存在: $MODEL_OUTPUT_DIR"
echo ""

# 列出所有结果文件
if ls "$MODEL_OUTPUT_DIR"/*.jsonl 1> /dev/null 2>&1; then
    echo "生成的结果文件:"
    ls -lh "$MODEL_OUTPUT_DIR"/*.jsonl | awk '{print "  " $9 " (" $5 ")"}'
else
    echo "未生成结果文件"
fi

echo ""
echo "=========================================="
