#!/usr/bin/env bash
# Train arithmetic factorization model.
# Two runs: default (n -> factors) and target_reversed. Both under results/.

config_path="configs/train.yaml"
mkdir -p results results/reversed

# Default: factorize n -> factors
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --config_path "$config_path" > results/train.log 2>&1 &
# Target reversed: n -> factors のまま、target の factor の並びだけ反転 (2,3,5 -> 5,3,2)
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --config_path "$config_path" --target_reversed > results/reversed/train.log 2>&1 &
