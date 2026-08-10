#!/usr/bin/env bash
# A-Multiplication-DL (Prod), decoder-only, forward and reverse.
#
#   bash sh/train_decoder.sh
#
# Same data, tokenizer, optimizer and 64-epoch budget as sh/train.sh; only the
# model differs. Both orders run with identical settings — nothing is tuned per
# order — so the loss curves are directly comparable.
set -u
cd "$(dirname "$0")/.."

SAGE_BIN="${SAGE_BIN:-$HOME/micromamba/envs/sage/bin}"
CALT_SRC="${CALT_SRC:-$HOME/calt_issac_rerun/src}"
export PATH="$SAGE_BIN:$PATH"
export PYTHONPATH="$CALT_SRC"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p results_decoder logs

CUDA_VISIBLE_DEVICES=0 nohup "$SAGE_BIN/python" train.py \
    --config_path configs/train_decoder.yaml \
    > logs/decoder_forward.log 2>&1 &
echo "forward: pid $!  -> logs/decoder_forward.log"

CUDA_VISIBLE_DEVICES=1 nohup "$SAGE_BIN/python" train.py \
    --config_path configs/train_decoder.yaml \
    --target_reversed \
    > logs/decoder_reverse.log 2>&1 &
echo "reverse: pid $!  -> logs/decoder_reverse.log"

wait
