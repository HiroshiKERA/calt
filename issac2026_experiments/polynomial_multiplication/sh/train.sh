#!/usr/bin/env bash
# Train polynomial multiplication model.
# Single GPU per run; GPUs 0-7 assigned in order (after arithmetic tasks in train_all.sh).
# Logs are written under results/<field>_<mode>/.

mkdir -p \
  results/ZZ_full results/ZZ_last_element \
  results/GF7_full results/GF7_last_element \
  results/GF31_full results/GF31_last_element \
  results/GF97_full results/GF97_last_element

# ZZ
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --train_config_path configs/ZZ/train.yaml --data_config_path configs/ZZ/data.yaml --target_mode full --wandb_runname_postfix full > results/ZZ_full/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --train_config_path configs/ZZ/train.yaml --data_config_path configs/ZZ/data.yaml --target_mode last_element --wandb_runname_postfix last_element > results/ZZ_last_element/train.log 2>&1 &

# GF7
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --train_config_path configs/GF7/train.yaml --data_config_path configs/GF7/data.yaml --target_mode full --wandb_runname_postfix full > results/GF7_full/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --train_config_path configs/GF7/train.yaml --data_config_path configs/GF7/data.yaml --target_mode last_element --wandb_runname_postfix last_element > results/GF7_last_element/train.log 2>&1 &

# GF31
CUDA_VISIBLE_DEVICES=4 nohup python3 train.py --train_config_path configs/GF31/train.yaml --data_config_path configs/GF31/data.yaml --target_mode full --wandb_runname_postfix full > results/GF31_full/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 train.py --train_config_path configs/GF31/train.yaml --data_config_path configs/GF31/data.yaml --target_mode last_element --wandb_runname_postfix last_element > results/GF31_last_element/train.log 2>&1 &

# GF97
CUDA_VISIBLE_DEVICES=6 nohup python3 train.py --train_config_path configs/GF97/train.yaml --data_config_path configs/GF97/data.yaml --target_mode full --wandb_runname_postfix full > results/GF97_full/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 train.py --train_config_path configs/GF97/train.yaml --data_config_path configs/GF97/data.yaml --target_mode last_element --wandb_runname_postfix last_element > results/GF97_last_element/train.log 2>&1 &
