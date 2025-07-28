# Quick Start

This tutorial provides a minimal introduction on how to use the CALT library for learning arithmetic and symbolic computation with Transformer models.　

We recommend the users to use [CALT codebase](https://github.com/HiroshiKERA/calt-codebase) as a comprehensive template repository and its documents.

## Installation

CALT can be installed via pip:

```bash
pip install calt-x
```

### Requirements

- Python ≥ 3.10

## Basic Usage

CALT is designed to be simple to use. You only need to implement an instance generator for your task. Here's a basic example for polynomial addition:

### 1. Define Your Problem Generator

```python
import random
from typing import List, Tuple
from calt.data_loader import PolynomialSampler
from calt.data_loader.utils import PolyElement

class SumProblemGenerator:
    """Generator for polynomial addition task.
    
    Task: input F=[f_1, ..., f_s], target g=f_1+...+f_s
    """
    def __init__(
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        self.sampler = sampler
        self.max_polynomials = max_polynomials  
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> Tuple[List[PolyElement], PolyElement]:
        random.seed(seed)  # Set random seed for reproducibility
        num_polys = random.randint(self.min_polynomials, self.max_polynomials) 

        # Generate input polynomials
        F = self.sampler.sample(num_samples=num_polys)
        
        # Compute target (sum of polynomials)
        g = sum(F)

        return F, g
```

### 2. Set Up Data Generation

```python
from calt.data_loader import PolynomialSampler, DatasetGenerator

# Create a polynomial sampler
sampler = PolynomialSampler(
    num_variables=2,
    max_degree=3,
    max_coefficient=10
)

# Create your problem generator
generator = SumProblemGenerator(
    sampler=sampler,
    max_polynomials=5,
    min_polynomials=2
)

# Create dataset generator
dataset_generator = DatasetGenerator(
    generator=generator,
    num_train_samples=1000,
    num_test_samples=100,
    num_workers=4
)
```

### 3. Generate Dataset

```python
# Generate training and test datasets
train_dataset, test_dataset = dataset_generator.generate()

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

### 4. Train a Transformer Model

```python
from calt.trainer import CALTTrainer
from calt.models import TransformerConfig

# Configure the Transformer model
config = TransformerConfig(
    vocab_size=1000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_seq_length=512
)

# Create trainer
trainer = CALTTrainer(
    model_config=config,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=10
)

# Start training
trainer.train()
```

## Advanced Example: Custom Problem

For more complex tasks, you can define your own problem generator. Here's an example for polynomial multiplication:

```python
class MultiplicationProblemGenerator:
    """Generator for polynomial multiplication task."""
    
    def __init__(self, sampler: PolynomialSampler):
        self.sampler = sampler

    def __call__(self, seed: int) -> Tuple[List[PolyElement], PolyElement]:
        random.seed(seed)
        
        # Generate two polynomials
        f1, f2 = self.sampler.sample(num_samples=2)
        
        # Compute product
        product = f1 * f2
        
        return [f1, f2], product
```

## Key Concepts

### Problem Generator Interface

Your problem generator must implement the `__call__` method with the following signature:

```python
def __call__(self, seed: int) -> Tuple[InputType, OutputType]:
    # Your implementation here
    pass
```

- `seed`: Random seed for reproducibility
- Returns: Tuple of (input, target) for your task

### Data Types

CALT supports various data types through the `PolyElement` class and other utilities. The library handles the conversion to token sequences for the Transformer model.

### Training Process

CALT automatically:
1. Converts your data to token sequences
2. Handles batching and data loading
3. Manages the training loop
4. Provides logging and evaluation metrics

## Next Steps

- Check out the [API Reference](../api/) for detailed documentation
- Explore the [CALT codebase](https://github.com/HiroshiKERA/calt-codebase) for more advanced examples
- See our [research papers](https://arxiv.org/abs/2506.08600) for applications in computer algebra

## Getting Help

- [GitHub Issues](https://github.com/HiroshiKERA/calt/issues) - Report bugs or request features
- [GitHub Discussions](https://github.com/HiroshiKERA/calt/discussions) - Ask questions and share ideas
- [Demo Notebook](https://colab.research.google.com/github/HiroshiKERA/calt/blob/dev/notebooks/demo.ipynb) - Interactive examples 