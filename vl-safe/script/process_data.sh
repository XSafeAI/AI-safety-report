#!/bin/bash

# 快速开始脚本 - 处理所有数据集

echo "=========================================="
echo "  数据集适配器 - 快速开始"
echo "=========================================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

echo "Python版本:"
python3 --version
echo ""

# 检查依赖
echo "检查依赖包..."
pip install -q pandas pyarrow
echo "✓ 依赖包已安装"
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 显示数据集列表
echo "可用的数据集:"
echo "  1. VLJailbreakBench"
echo "  2. JailbreakV-28K"
echo "  3. MIS (Test)"
echo "  4. USB"
echo "  5. MemeSafetyBench"
echo "  6. MM-SafetyBench"
echo "  7. Argus"
echo "  8. UNICORN"
echo ""

# 询问用户
read -p "是否处理所有数据集? (y/n): " choice

if [[ $choice == "y" || $choice == "Y" ]]; then
    echo ""
    echo "开始处理所有数据集..."
    python3 process_datasets.py
else
    echo ""
    echo "请输入要处理的数据集名称 (例如: vljailbreakbench):"
    read dataset_name
    
    if [ -z "$dataset_name" ]; then
        echo "错误: 数据集名称不能为空"
        exit 1
    fi
    
    echo ""
    echo "开始处理数据集: $dataset_name"
    python3 process_datasets.py --dataset "$dataset_name"
fi

echo ""
echo "=========================================="
echo "  处理完成!"
echo "=========================================="
echo ""
echo "输出目录: /data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed"
echo ""
echo "要测试输出，请运行:"
echo "  python3 test_adapters.py"
echo ""

