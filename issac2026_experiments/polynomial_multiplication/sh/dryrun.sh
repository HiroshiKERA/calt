#!/usr/bin/env bash
# Run one dryrun training for polynomial_multiplication (GF7, full).
# Use from polynomial_multiplication with calt-env active:
#   cd issac2026_experiments/polynomial_multiplication && bash sh/dryrun.sh

set -e
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --train_config_path configs/GF7/train.yaml \
  --data_config_path configs/GF7/data.yaml \
  --target_mode full \
  --dryrun \
  --wandb_runname_postfix dryrun
