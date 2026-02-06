#!/usr/bin/env bash
# Train arithmetic addition model.
# Single GPU per run; GPUs 0-7 assigned in order.

# ZZ
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --config_path configs/ZZ/train.yaml --target_mode full > train_ZZ_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --config_path configs/ZZ/train.yaml --target_mode last_element > train_ZZ_last_element.log 2>&1 &

# GF7
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --config_path configs/GF7/train.yaml --target_mode full > train_GF7_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --config_path configs/GF7/train.yaml --target_mode last_element > train_GF7_last_element.log 2>&1 &

# GF31
CUDA_VISIBLE_DEVICES=4 nohup python3 train.py --config_path configs/GF31/train.yaml --target_mode full > train_GF31_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 train.py --config_path configs/GF31/train.yaml --target_mode last_element > train_GF31_last_element.log 2>&1 &

# GF97
CUDA_VISIBLE_DEVICES=6 nohup python3 train.py --config_path configs/GF97/train.yaml --target_mode full > train_GF97_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 train.py --config_path configs/GF97/train.yaml --target_mode last_element > train_GF97_last_element.log 2>&1 &
