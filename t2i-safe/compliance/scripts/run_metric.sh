#!/bin/bash

generate_model=$1
data_dir="results/$generate_model/json"
python metric.py --data_dir $data_dir