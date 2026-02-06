#!/usr/bin/env bash
# Train polynomial multiplication model.

field_str_list=("ZZ" "GF7" "GF31" "GF97")
target_mode_list=("full" "last_element")

for field_str in "${field_str_list[@]}"; do
    for target_mode in "${target_mode_list[@]}"; do
        train_config_path="configs/${field_str}/train.yaml"
        data_config_path="configs/${field_str}/data.yaml"
        python3 train.py --train_config_path "$train_config_path" --data_config_path "$data_config_path" --target_mode "$target_mode"

        echo "Finished training for $field_str with $target_mode"
        echo "--------------------------------"
    done
done