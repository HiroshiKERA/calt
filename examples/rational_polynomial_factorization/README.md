# Rational Polynomial Factorization

Factor an expanded quadratic with rational coefficients into factored form.

## Task
- Input: expanded quadratic (e.g., `1*x^2+13/24*x+5/12`)
- Output: factored form (e.g., `(x+-13/24)*(x+-5/12)`)

## Usage
```bash
python generate_dataset.py
python train.py
```

## Files
- `configs/train.yaml`: model/training config (generic transformer)
- `configs/lexer.yaml`: lexer config (sign=attach, digit_group=3)
- `generate_dataset.py`: dataset generation
- `train.py`: training
