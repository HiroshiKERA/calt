# Generate dataset for all experiments.

# # Arithmetic addition
# cd arithmetic_addition
# bash sh/generate_dataset.sh
# cd ..

# # Arithmetic factorization
# cd arithmetic_factorization
# bash sh/generate_dataset.sh
# cd ..

# # Polynomial multiplication
# cd polynomial_multiplication
# bash sh/generate_dataset.sh
# cd ..

# # Polynomial reduction
# cd polynomial_reduction
# bash sh/generate_dataset.sh
# cd ..

# Digit product (Prod, L=10)
cd digit_product
bash sh/generate_dataset.sh
cd ..

# ReLU recurrence (L=10)
cd relu_recurrence
bash sh/generate_dataset.sh
cd ..

