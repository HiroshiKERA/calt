#!/usr/bin/env bash
# Generate polynomial multiplication dataset.

field_str_list=("GF7" "GF37" "GF97" "ZZ")

for field_str in "${field_str_list[@]}"; do
    config_path="configs/${field_str}/data.yaml"
    python3 generate_dataset.py --config_path "$config_path"
done
