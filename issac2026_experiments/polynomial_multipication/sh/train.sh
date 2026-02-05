#!/usr/bin/env bash
# Train polynomial multiplication model.

field_str_list=("ZZ")

for field_str in "${field_str_list[@]}"; do
    config_path="configs/${field_str}/train.yaml"
    python3 train.py --config_path "$config_path" 
done