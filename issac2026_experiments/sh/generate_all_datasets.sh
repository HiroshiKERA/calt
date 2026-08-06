#!/usr/bin/env bash
# Generate the datasets for every experiment in the paper.
#
# Execute from issac2026_experiments/:  bash sh/generate_all_datasets.sh
#
# Generation is deterministic: each task seeds SageMath per sample from the
# root_seed in its data config, so the same command reproduces the same data.
# Everything lands in <task>/data/, which is gitignored — the datasets are meant
# to be regenerated rather than shipped.
set -eu
cd "$(dirname "$0")/.."

for task in \
    arithmetic_addition \
    arithmetic_factorization \
    polynomial_multiplication \
    polynomial_reduction \
    digit_product \
    relu_recurrence \
    groebner
do
    echo "=== generating: $task"
    (cd "$task" && bash sh/generate_dataset.sh)
done

echo "all datasets generated"
