#!/bin/bash

# Quick start script - Process all datasets

echo "=========================================="
echo "  Dataset Adapter - Quick Start"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check dependencies
echo "Checking dependencies..."
pip install -q pandas pyarrow
echo "✓ Dependencies installed"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Show dataset list
echo "Available datasets:"
echo "  1. VLJailbreakBench"
echo "  2. JailbreakV-28K"
echo "  3. MIS (Test)"
echo "  4. USB"
echo "  5. MemeSafetyBench"
echo "  6. MM-SafetyBench"
echo "  7. Argus"
echo "  8. UNICORN"
echo ""

# Ask user
read -p "Process all datasets? (y/n): " choice

if [[ $choice == "y" || $choice == "Y" ]]; then
    echo ""
    echo "Starting to process all datasets..."
    python3 process_datasets.py
else
    echo ""
    echo "Please enter the dataset name to process (e.g., vljailbreakbench):"
    read dataset_name
    
    if [ -z "$dataset_name" ]; then
        echo "Error: Dataset name cannot be empty"
        exit 1
    fi
    
    echo ""
    echo "Starting to process dataset: $dataset_name"
    python3 process_datasets.py --dataset "$dataset_name"
fi

echo ""
echo "=========================================="
echo "  Processing completed!"
echo "=========================================="
echo ""
echo "Output directory: /data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed"
echo ""
echo "To test the output, run:"
echo "  python3 test_adapters.py"
echo ""
