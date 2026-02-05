#!/usr/bin/env bash
# Generate integer factorization dataset.

config_path="configs/data.yaml"
python3 generate_dataset.py --config_path "$config_path"