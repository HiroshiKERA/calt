#!/usr/bin/env bash
# Train arithmetic addition model.

field_str_list=("ZZ") # "GF7" "GF37" "GF97"

for field_str in "${field_str_list[@]}"; do
    config_path="configs/${field_str}/train.yaml"
    python3 train.py --config_path "$config_path" 
done