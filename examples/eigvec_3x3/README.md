# 3x3 Eigenvector Prediction (Largest Eigenvalue)

Predict the first eigenvector (largest |λ|) of a 3x3 real matrix.

## Task
- Input: 3x3 matrix flattened as rows with ',' and rows separated by ';' (rounded to 2 decimals)
- Output: eigenvector components rounded to 2 decimals, comma-separated

## Usage
```bash
python generate_dataset.py
python train.py
```

## Files
- `configs/train.yaml`: model/training config (bart)
- `configs/lexer.yaml`: lexer config (sign=attach, digit_group=3, allow_float)
- `generate_dataset.py`: dataset generation
- `train.py`: training
