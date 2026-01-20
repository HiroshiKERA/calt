cd eigvec_3x3
# python generate_dataset.py
CUDA_VISIBLE_DEVICES=0 nohup python train.py > train.log 2>&1 &
cd ..

cd gf17_addition
# python generate_dataset_sage.py
CUDA_VISIBLE_DEVICES=1 nohup python train.py > train.log 2>&1 &
cd ..

cd integer_polynomial_factorization
# python generate_dataset.py
CUDA_VISIBLE_DEVICES=2 nohup python train.py > train.log 2>&1 &
cd ..

# cd rational_factorization
# python generate_dataset.py
# CUDA_VISIBLE_DEVICES=3 nohup python train.py > train.log 2>&1 &
# cd ..


