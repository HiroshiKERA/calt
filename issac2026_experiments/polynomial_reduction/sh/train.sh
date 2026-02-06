#!/usr/bin/env bash
# Train polynomial reduction model.
# Single GPU per run; GPUs 0-7 assigned in order.
# Logs are written under results/<field>_<mode>/.

mkdir -p \
  results/ZZ_full results/ZZ_last_element \
  results/GF7_full results/GF7_last_element \
  results/GF31_full results/GF31_last_element \
  results/GF97_full results/GF97_last_element \
  results/ZZ_dg1_full results/ZZ_dg1_last_element

# ZZ (GPUs 0-1)
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --train_config_path configs/ZZ/train.yaml --data_config_path configs/ZZ/data.yaml --target_mode full --wandb_runname_postfix full > results/ZZ_full/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --train_config_path configs/ZZ/train.yaml --data_config_path configs/ZZ/data.yaml --target_mode last_element --wandb_runname_postfix last_element > results/ZZ_last_element/train.log 2>&1 &

# ZZ_dg1 (GPUs 4-5)
CUDA_VISIBLE_DEVICES=4 nohup python3 train.py --train_config_path configs/ZZ_dg1/train.yaml --data_config_path configs/ZZ_dg1/data.yaml --target_mode full --wandb_runname_postfix full > results/ZZ_dg1_full/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 train.py --train_config_path configs/ZZ_dg1/train.yaml --data_config_path configs/ZZ_dg1/data.yaml --target_mode last_element --wandb_runname_postfix last_element > results/ZZ_dg1_last_element/train.log 2>&1 &

# GF7 (GPUs 2-3)
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --train_config_path configs/GF7/train.yaml --data_config_path configs/GF7/data.yaml --target_mode full --wandb_runname_postfix full > results/GF7_full/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --train_config_path configs/GF7/train.yaml --data_config_path configs/GF7/data.yaml --target_mode last_element --wandb_runname_postfix last_element > results/GF7_last_element/train.log 2>&1 &

# GF31 (GPUs 4-5)
CUDA_VISIBLE_DEVICES=6 nohup python3 train.py --train_config_path configs/GF31/train.yaml --data_config_path configs/GF31/data.yaml --target_mode full --wandb_runname_postfix full > results/GF31_full/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 train.py --train_config_path configs/GF31/train.yaml --data_config_path configs/GF31/data.yaml --target_mode last_element --wandb_runname_postfix last_element > results/GF31_last_element/train.log 2>&1 &

# GF97 (GPUs 6-7)
CUDA_VISIBLE_DEVICES=7 nohup python3 train.py --train_config_path configs/GF97/train.yaml --data_config_path configs/GF97/data.yaml --target_mode full --wandb_runname_postfix full > results/GF97_full/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 train.py --train_config_path configs/GF97/train.yaml --data_config_path configs/GF97/data.yaml --target_mode last_element --wandb_runname_postfix last_element > results/GF97_last_element/train.log 2>&1 &
