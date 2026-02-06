#!/usr/bin/env bash
# Train arithmetic factorization model.
# Single GPU per run; uses GPU 0 (train_all.sh order: add 0-7, factor 0, poly 0-7).

config_path="configs/train.yaml"

CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --config_path "$config_path" > train.log 2>&1 &
