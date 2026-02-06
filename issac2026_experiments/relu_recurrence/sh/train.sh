#!/usr/bin/env bash
# Train relu_recurrence (ReLU, L=10) model. Both runs under results/.

mkdir -p results results/reversed
# Default: target as-is (y_1,...,y_L)
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --config_path configs/train.yaml > results/train.log 2>&1 &
# Target reversed (for reverse vs non-reverse performance evaluation)
CUDA_VISIBLE_DEVICES=5 nohup python3 train.py --config_path configs/train.yaml --target_reversed > results/reversed/train.log 2>&1 &
