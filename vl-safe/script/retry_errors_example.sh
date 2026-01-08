#!/bin/bash

# 重新评测错误样本的示例脚本

python /data/data-pool/dingyifan/GeminiEvaluation/evaluation/evaluate.py \
    --retry-errors "/data/data-pool/dingyifan/GeminiEvaluation/workspace/results/grok-4-1-fast-non-reasoning/vljailbreakbench_20251222_183605.jsonl" \
    --model "grok-4-1-fast-non-reasoning" \
    --concurrency 32 \
    --max-tokens 256



echo "重新评测完成！"

