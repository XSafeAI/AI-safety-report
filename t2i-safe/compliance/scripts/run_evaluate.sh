#!/bin/bash

generate_model=$1
evaluate_model="qwen"
images_dir="results/$generate_model/img"
results_json="results/$generate_model/json/results.json"

python3 evaluate.py \
    --model $evaluate_model \
    --images_dir $images_dir \
    --results_json $results_json