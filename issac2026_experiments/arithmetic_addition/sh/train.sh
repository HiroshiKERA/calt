#!/usr/bin/env bash
# Train arithmetic addition model.
# Single GPU per run; GPUs 0-7 assigned in order.
# Logs are written under results/<field>/.

mkdir -p results/ZZ results/ZZ_dg1 results/ZZ_dg3 results/GF7 results/GF31 results/GF97

# ZZ (digit_group=0)
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --config_path configs/ZZ/train.yaml --target_mode full --wandb_runname_postfix full > results/ZZ/train_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --config_path configs/ZZ/train.yaml --target_mode last_element --wandb_runname_postfix last_element > results/ZZ/train_last_element.log 2>&1 &

# ZZ (digit_group=1)
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --config_path configs/ZZ_dg1/train.yaml --target_mode full --wandb_runname_postfix full > results/ZZ_dg1/train_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --config_path configs/ZZ_dg1/train.yaml --target_mode last_element --wandb_runname_postfix last_element > results/ZZ_dg1/train_last_element.log 2>&1 &

# # ZZ (digit_group=3)
# CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --config_path configs/ZZ_dg3/train.yaml --target_mode full --wandb_runname_postfix full > results/ZZ_dg3/train_full.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --config_path configs/ZZ_dg3/train.yaml --target_mode last_element --wandb_runname_postfix last_element > results/ZZ_dg3/train_last_element.log 2>&1 &


# GF7
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --config_path configs/GF7/train.yaml --target_mode full --wandb_runname_postfix full > results/GF7/train_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --config_path configs/GF7/train.yaml --target_mode last_element --wandb_runname_postfix last_element > results/GF7/train_last_element.log 2>&1 &

# GF31
CUDA_VISIBLE_DEVICES=4 nohup python3 train.py --config_path configs/GF31/train.yaml --target_mode full --wandb_runname_postfix full > results/GF31/train_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 train.py --config_path configs/GF31/train.yaml --target_mode last_element --wandb_runname_postfix last_element > results/GF31/train_last_element.log 2>&1 &

# GF97
CUDA_VISIBLE_DEVICES=6 nohup python3 train.py --config_path configs/GF97/train.yaml --target_mode full --wandb_runname_postfix full > results/GF97/train_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 train.py --config_path configs/GF97/train.yaml --target_mode last_element --wandb_runname_postfix last_element > results/GF97/train_last_element.log 2>&1 &
