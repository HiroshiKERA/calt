# Integer Polynomial Factorization

Factor an expanded quadratic with integer coefficients into factored form.

## Task
- Input: expanded quadratic (e.g., `1*x^2+3*x+2`)
- Output: factored form (e.g., `(x+1)*(x+2)`)

## Usage
```bash
python generate_dataset.py
python train.py
```

## Files
- `configs/train.yaml`: model/training config (generic transformer)
- `configs/lexer.yaml`: lexer config (sign=attach, digit_group=3)
- `generate_dataset.py`: dataset generation (Sage backend + PolynomialSampler with ZZ)
- `train.py`: training

