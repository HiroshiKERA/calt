# GF(17) Modulo Addition Task

Sequential prediction of cumulative sums modulo 17 from integer sequences.

## Task

Given an integer sequence, predict the cumulative sum modulo 17 at each position.

Example:
- Input: `1,6,13,15,0,3,6,7`
- Output: `1,7,3,1,1,4,10,0`

## Usage

```bash
# Generate dataset
python generate_dataset.py

# Train model
python train.py
```

## Files

- `configs/train.yaml`: Model and training configuration
- `configs/lexer.yaml`: Lexer configuration for tokenization
- `generate_dataset.py`: Dataset generation script
- `train.py`: Training script
