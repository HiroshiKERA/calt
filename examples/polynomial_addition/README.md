# Polynomial Addition

Predict cumulative sums of a sequence of polynomials (e.g. p1, p1+p2, p1+p2+p3).

## Task
- Input: sequence of polynomials (e.g. `poly1 | poly2 | poly3` in text)
- Output: cumulative sums (e.g. `poly1 | poly1+poly2 | poly1+poly2+poly3`)

Data is loaded from text (`train_raw.txt`), parsed to SageMath with `TextToSageLoadPreprocessor`, then converted to C/E expanded form with `ExpandedFormLoadPreprocessor` for training.

## Usage
```bash
python generate_dataset.py
python train.py
```

Optional: `python train.py --dryrun` for a quick test run.

## Files
- `configs/train.yaml`: model/training config; data paths point to `train_raw.txt` / `test_raw.txt`
- `configs/lexer.yaml`: C/E vocab (coefficients, exponents), misc `+`, `|`
- `generate_dataset.py`: dataset generation (SageMath, saves text + pickle)
- `train.py`: training with ChainLoadPreprocessor(TextToSage, ExpandedForm)
