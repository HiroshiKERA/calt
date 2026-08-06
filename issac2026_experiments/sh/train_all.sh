#!/usr/bin/env bash
# Run training for all ISSAC2026 experiments.
#
# Execute from issac2026_experiments/:  bash sh/train_all.sh
#
# Each task's train.sh starts its runs with `nohup ... &`, so this script
# returns immediately and every job runs in parallel. That is 50 runs at once,
# each pinned to a CUDA_VISIBLE_DEVICES index written into the task's train.sh:
# on a machine with fewer GPUs than that, edit those assignments first, or run
# the tasks one at a time by calling each `bash sh/train.sh` yourself.
set -eu
cd "$(dirname "$0")/.."

# Number of runs each task launches, for sizing the machine.
#   arithmetic_addition        8 (ZZ, GF7, GF31, GF97 x full/last_element)
#   arithmetic_factorization   1
#   polynomial_multiplication  8
#   polynomial_reduction       8
#   digit_product              2 (forward, reverse)
#   relu_recurrence            2 (forward, reverse)
#   groebner                   4 (QQ, GF7 x lex, degrevlex)
for task in \
    arithmetic_addition \
    arithmetic_factorization \
    polynomial_multiplication \
    polynomial_reduction \
    digit_product \
    relu_recurrence \
    groebner
do
    echo "=== launching: $task"
    (cd "$task" && bash sh/train.sh)
done

echo "all runs launched; watch <task>/results*/train.log"
