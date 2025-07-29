# CALT: Computer ALgebra with Transformer

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://hiroshikera.github.io/calt/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-View%20Documentation-blue.svg)](https://hiroshikera.github.io/calt/)

> 📖 **📚 [View Full Documentation](https://hiroshikera.github.io/calt/)**

## Overview

CALT is a simple Python library for learning arithmetic and symbolic computation with a Transformer model (a deep neural model to realize sequence-to-sequence functions). 

It offers a basic Transformer model and training, and non-experts of deep learning (e.g., mathematicians) can focus on constructing datasets to train and evaluate the model. Particularly, users only need to implement an instance generator for their own task.

For example, users define the following code for the polynomial addition task.

```python
class SumProblemGenerator:
    # Task - input: F=[f_1, ..., f_s], target: G=[g:= f_1+...+f_s]
    def __init__(
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        self.sampler = sampler
        self.max_polynomials = max_polynomials  
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> Tuple[List[PolyElement], PolyElement]:
        random.seed(seed) # Set random seed
        num_polys = random.randint(self.min_polynomials, self.max_polynomials) 

        F = self.sampler.sample(num_samples=num_polys)
        g = sum(F)

        return F, g
```

CALT automatically calls this generator in parallel to efficiently construct large datasets and trains a Transformer model to learn the computation. The sample generation process itself can reveal unexplored mathematical problems, enabling researchers to study their theoretical and algorithmic solutions. The following is a small list of such studies from our group. 

- ["Learning to Compute Gröbner Bases," Kera et al., 2024](https://arxiv.org/abs/2311.12904)
- ["Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms," Kera and Pelleriti et al., 2025](https://arxiv.org/abs/2505.23696)
- ["Geometric Generality of Transformer-Based Gröbner Basis Computation," Kambe et al., 2025](https://arxiv.org/abs/2504.12465)

Refer to our paper ["CALT: A Library for Computer Algebra with Transformer," Kera et al., 2025](https://arxiv.org/abs/2506.08600) for a comprehensive overview.

## 🚀 Quick Start

### Basic Installation

```bash
pip install calt-x
```

### For Research Projects

For researchers and developers who want to start experiments with CALT, we provide the [CALT codebase](https://github.com/HiroshiKERA/calt-codebase) - a comprehensive template repository with pre-configured environment and development tools.

```bash
git clone https://github.com/HiroshiKERA/calt-codebase.git
cd calt-codebase
conda env create -f environment.yml 
```

## 📖 Documentation & Resources

- **📚 [Full Documentation](https://hiroshikera.github.io/calt/)** - Complete guide with quickstart and project organization tips
- **⚡ [Quickstart Guide](https://hiroshikera.github.io/calt/quickstart/)** - Get up and running quickly
- **📓 [Demo Notebook](https://colab.research.google.com/github/HiroshiKERA/calt/blob/dev/notebooks/demo.ipynb)** - Interactive examples

## 🔗 Links

- [📖 Documentation](https://hiroshikera.github.io/calt/)
- [🐛 Issues](https://github.com/HiroshiKERA/calt/issues)
- [💬 Discussions](https://github.com/HiroshiKERA/calt/discussions)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{kera2025calt,
  title={CALT: A Library for Computer Algebra with Transformer},
  author={Hiroshi Kera and Shun Arawaka and Yuta Sato},
  year={2025},
  archivePrefix={arXiv},
  eprint={2506.08600}
}
```

