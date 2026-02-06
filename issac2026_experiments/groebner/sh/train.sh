#!/usr/bin/env bash
# Train Groebner model for QQ and GF7.
# - training_order=degrevlex: データセットと同じ順序
# - training_order=lex      : ロード時に F を lex 環に移し、lex Groebner 基底を再計算

mkdir -p \
  results_QQ_degrevlex results_QQ_lex \
  results_GF7_degrevlex results_GF7_lex

########################
# QQ (field_str = QQ)  #
########################

# QQ, degrevlex
CUDA_VISIBLE_DEVICES=5 nohup python3 train.py \
  --config_path configs/train.yaml \
  --data_config_path configs/data.yaml \
  --training_order degrevlex \
  > results_QQ_degrevlex/train.log 2>&1 &

# QQ, lex
CUDA_VISIBLE_DEVICES=5 nohup python3 train.py \
  --config_path configs/train.yaml \
  --data_config_path configs/data.yaml \
  --training_order lex \
  > results_QQ_lex/train.log 2>&1 &

########################
# GF7 (field_str = GF7)#
########################

# GF7, degrevlex
CUDA_VISIBLE_DEVICES=6 nohup python3 train.py \
  --config_path configs/train_GF7.yaml \
  --data_config_path configs/data_GF7.yaml \
  --training_order degrevlex \
  > results_GF7_degrevlex/train.log 2>&1 &

# GF7, lex
CUDA_VISIBLE_DEVICES=7 nohup python3 train.py \
  --config_path configs/train_GF7.yaml \
  --data_config_path configs/data_GF7.yaml \
  --training_order lex \
  > results_GF7_lex/train.log 2>&1 &

