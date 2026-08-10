#!/usr/bin/env bash
# The arithmetic tasks under the decoder-only model.
#
#   bash sh/train_arithmetic_decoder.sh
#
# Only the A-* tasks: the arithmetic community trains decoder-only models, so
# these are the rows where the architecture is the natural one. The polynomial
# and Groebner tasks keep the encoder-decoder of the paper.
#
# Nine runs here, plus digit_product and relu_recurrence which have their own
# sh/train_decoder.sh. Everything else — data, tokenizer, optimizer, 64-epoch
# budget, seed — is what the seq2seq configs use; only the model differs.
#
# Results land in <task>/results_decoder/, leaving the seq2seq results in place.
set -u
cd "$(dirname "$0")/.."

SAGE_BIN="${SAGE_BIN:-$HOME/micromamba/envs/sage/bin}"
CALT_SRC="${CALT_SRC:-$HOME/calt_issac_rerun/src}"
GPUS="${GPUS:-0 1}"
export PATH="$SAGE_BIN:$PATH"
export PYTHONPATH="$CALT_SRC"
export WANDB_MODE="${WANDB_MODE:-offline}"

# task | config | extra args | log name
JOBS=(
    "arithmetic_addition|configs/ZZ/train_decoder.yaml|--target_mode last_element --wandb_runname_postfix last_element|ZZ_last_element"
    "arithmetic_addition|configs/ZZ/train_decoder.yaml|--target_mode full --wandb_runname_postfix full|ZZ_full"
    "arithmetic_addition|configs/GF7/train_decoder.yaml|--target_mode last_element --wandb_runname_postfix last_element|GF7_last_element"
    "arithmetic_addition|configs/GF7/train_decoder.yaml|--target_mode full --wandb_runname_postfix full|GF7_full"
    "arithmetic_addition|configs/GF31/train_decoder.yaml|--target_mode last_element --wandb_runname_postfix last_element|GF31_last_element"
    "arithmetic_addition|configs/GF31/train_decoder.yaml|--target_mode full --wandb_runname_postfix full|GF31_full"
    "arithmetic_addition|configs/GF97/train_decoder.yaml|--target_mode last_element --wandb_runname_postfix last_element|GF97_last_element"
    "arithmetic_addition|configs/GF97/train_decoder.yaml|--target_mode full --wandb_runname_postfix full|GF97_full"
    "arithmetic_factorization|configs/train_decoder.yaml||factorization"
)

run_job() {
    local gpu="$1" spec="$2"
    IFS='|' read -r task config extra name <<< "$spec"
    mkdir -p "$task/logs"
    echo "[gpu $gpu] $task $name"
    (cd "$task" && CUDA_VISIBLE_DEVICES="$gpu" "$SAGE_BIN/python" train.py \
        --config_path "$config" $extra > "logs/decoder_${name}.log" 2>&1)
    echo "[gpu $gpu] done: $task $name"
}

# One worker per GPU, each pulling the next job off the list.
next=0
lock="$(mktemp -d)"
claim() {
    local i
    exec 9>"$lock/lock"
    flock 9
    i=$(cat "$lock/next" 2>/dev/null || echo 0)
    echo $((i + 1)) > "$lock/next"
    flock -u 9
    echo "$i"
}

for gpu in $GPUS; do
    (
        while :; do
            i=$(claim)
            [ "$i" -ge "${#JOBS[@]}" ] && break
            run_job "$gpu" "${JOBS[$i]}"
        done
    ) &
done
wait
rm -rf "$lock"

echo "all arithmetic decoder-only runs finished"
