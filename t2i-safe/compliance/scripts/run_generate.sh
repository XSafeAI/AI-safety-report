#!/bin/bash

model=$1
prompts_file=$2
results_root="results"

python generate.py \
    --model "$model" \
    --prompts_file "$prompts_file" \
    --results_root "$results_root"