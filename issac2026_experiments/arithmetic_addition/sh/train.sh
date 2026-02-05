#!/usr/bin/env bash
# Train arithmetic addition model.

field_str_list=("ZZ" "GF7" "GF37" "GF97") # "ZZ" "GF7" "GF37" "GF97"
target_mode_list=("full" "last_element")

for field_str in "${field_str_list[@]}"; do
    for target_mode in "${target_mode_list[@]}"; do
        config_path="configs/${field_str}/train.yaml"
        python3 train.py --config_path "$config_path" --target_mode "$target_mode" 

        echo "Finished training for $field_str with $target_mode"
        echo "--------------------------------"
    done
done