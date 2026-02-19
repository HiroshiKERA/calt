# Polynomial reduction

Task: given polynomials f and g1,...,gt, predict the result of reducing f by the Gröbner basis of g1,...,gt.

- **Coefficients**: integers ZZ
- **Variables**: 3 variables (x, y, z)
- **Data**: f, g1, g2, g3 are generated with 2--5 terms each; the grevlex Gröbner basis G of g1,g2,g3 is computed; only the remainder r = I.reduce(f) is saved (pickle). Quotients are not stored (for pattern 2 they are computed at load time via lift).

## Data generation

```bash
cd calt/examples/polynomial_reduction
python generate_dataset.py
```

## Training

- **order**: `grevlex` (as-is) or `lex` (f is moved to lex ring; G is converted to lex Gröbner basis via FGLM)
- **pattern**: `1` = target is remainder only; `2` = target is quotients w.r.t. g1..gt and remainder

```bash
# grevlex, remainder only
python train.py

# lex, remainder only
python train.py --order lex

# grevlex, quotients + remainder
python train.py --pattern 2

# lex, quotients + remainder
python train.py --order lex --pattern 2
```

## Load-time conversion

When training with lex order on data saved in grevlex, use `PolynomialReductionLoadPreprocessor(order="lex", ...)`. f is moved to the lex ring and G is converted to a lex Gröbner basis via FGLM (for zero-dimensional ideals).
