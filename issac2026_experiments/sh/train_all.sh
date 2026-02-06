#!/usr/bin/env bash
# Run training for all ISSAC2026 experiments.
# Execute from issac2026_experiments/:  bash sh/train_all.sh
#
# Each task's train.sh starts background jobs (nohup ... &). This script
# invokes them in order; all jobs run in parallel across tasks.

# Arithmetic addition (ZZ, GF7, GF31, GF97 × full/last_element → 8 jobs)
cd arithmetic_addition
bash sh/train.sh
cd ..

# Arithmetic factorization (1 job)
cd arithmetic_factorization
bash sh/train.sh
cd ..

# Polynomial multiplication (ZZ, GF7, GF31, GF97 × full/last_element → 8 jobs)
cd polynomial_multiplication
bash sh/train.sh
cd ..
