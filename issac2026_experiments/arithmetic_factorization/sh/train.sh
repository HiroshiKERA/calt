#!/usr/bin/env bash
# Train arithmetic factorization model.

config_path="configs/train.yaml"

python3 train.py --config_path "$config_path" 